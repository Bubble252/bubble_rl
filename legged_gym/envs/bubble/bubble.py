# ==============================================================================
# Description: Bubble 双轮足机器人环境
# 继承自 LeggedRobot，参考 B2W 的轮式处理方式：
#   - 标准 P 控制，轮子 dof_err=0 (不位置锁)
#   - 观测中 dof_pos[wheel]=0 (continuous joint 位置无意义)
#   - 保留 Diablo 的 no_moonwalk 对称性奖励
# ==============================================================================

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from legged_gym.utils.math import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from .bubble_config import BubbleFlatCfg


class Bubble(LeggedRobot):
    """Bubble 双轮足机器人环境。
    
    关键设计（参考 B2W）:
    - _compute_torques: 标准 P 模式，轮子 dof_err=0, dof_vel 可选偏置
    - compute_observations: dof_pos[wheel]=0, dof_err[wheel]=0
    - 不重写 compute_reward / check_termination，使用父类默认逻辑
    """

    def _reward_base_height(self):
        """重载: 指数型高度奖励——越接近目标越好。
        exp(-40 * error^2): 
          error=0    → reward=1.0 (满分)
          error=0.01 → reward=0.96
          error=0.02 → reward=0.85
          error=0.05 → reward=0.37
          error=0.10 → reward=0.02 (几乎为零)
        配合 scales.base_height > 0 使用（正向奖励）。
        """
        base_height = self.root_states[:, 2]
        error = base_height - self.cfg.rewards.base_height_target
        return torch.exp(-40.0 * error ** 2)

    def _reward_no_fly(self):
        """奖励双轮保持与地面接触"""
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        rew = 1.0 * (torch.sum(1.0 * contacts, dim=1) == 2)
        return rew

    def _reward_wheel_vel(self):
        """只惩罚轮子速度（不影响腿部），防止轮子空转。
        返回 mean(wheel_vel²)，配合负权重使用。
        """
        wheel_vel = self.dof_vel[:, self.wheel_indices]  # (N, 2)
        return torch.sum(wheel_vel ** 2, dim=1)

    def _reward_no_moonwalk(self):
        """惩罚左右腿不对称（"太空步"），参考 Diablo"""
        joints = list(self.cfg.init_state.default_joint_angles.keys())
        thigh_left = joints.index("left_thigh_joint")
        knee_left = joints.index("left_knee_joint")
        thigh_right = joints.index("right_thigh_joint")
        knee_right = joints.index("right_knee_joint")

        theta_right = self.dof_pos[:, knee_right]
        theta_left = self.dof_pos[:, knee_left]

        # 两腿在 x 轴方向投影之和应相等
        r = torch.sin(theta_right-0.8) + torch.sin(3.14159265/2+
            self.dof_pos[:, thigh_right] + theta_right-0.8
        )
        l = torch.sin(theta_left+0.8) + torch.sin(-3.14159265/2+
            self.dof_pos[:, thigh_left] + theta_left+0.8
        )
        return torch.square(r + l)

    def _process_rigid_shape_props(self, props, env_id):
        """重写: 在父类摩擦随机化基础上，添加弹性系数(restitution)随机化。"""
        props = super()._process_rigid_shape_props(props, env_id)
        if getattr(self.cfg.domain_rand, 'randomize_restitution', False):
            if env_id == 0:
                min_rest, max_rest = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch.rand(
                    self.num_envs, dtype=torch.float, device='cpu',
                ) * (max_rest - min_rest) + min_rest
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id].item()
        return props

    def _compute_torques(self, actions):
        """根据 wheel_drive_mode 选择轮子驱动方式：

        "bubble" — 油门踏板模式:
            轮子 dof_err=0, action 直接控制力矩方向
            torque = Kp * action_scaled - Kd * vel

        "diablo" — 位置追踪模式:
            轮子保留 dof_err（不清零），角度差 → 持续驱动力
            torque = Kp * (action_scaled + default - pos) - Kd * vel
            网络不断给出目标偏移，轮子永远追不上 → 持续有力

        "b2w" — 恒速驱动模式:
            轮子 dof_err=0, dof_vel 覆盖为 -wheel_speed（常数）
            D 项变成 +Kd*wheel_speed = 恒定正向推力
            torque = Kp * action_scaled + Kd * wheel_speed
        """
        actions_scaled = actions * self.cfg.control.action_scale
        mode = self.cfg.control.wheel_drive_mode

        # 位置误差（腿部始终正常计算）
        dof_err = self.default_dof_pos - self.dof_pos

        if mode == "bubble":
            # 轮子位置误差清零 → 不追踪角度
            dof_err[:, self.wheel_indices] = 0
            torques = self.p_gains * (actions_scaled + dof_err) - self.d_gains * self.dof_vel

        elif mode == "diablo":
            # 轮子保留 dof_err → 角度差产生持续驱动力（像 Diablo）
            # 不做任何清零，所有关节统一 PD
            torques = self.p_gains * (actions_scaled + dof_err) - self.d_gains * self.dof_vel

        elif mode == "b2w":
            # 轮子位置误差清零
            dof_err[:, self.wheel_indices] = 0
            # 覆盖轮子速度为 -wheel_speed → D 项变成恒定正向推力
            dof_vel_modified = self.dof_vel.clone()
            dof_vel_modified[:, self.wheel_indices] = -self.cfg.control.wheel_speed
            torques = self.p_gains * (actions_scaled + dof_err) - self.d_gains * dof_vel_modified

        else:
            raise ValueError(f"Unknown wheel_drive_mode: {mode}")

        return torch.clip(torques * self.torques_scale, -self.torque_limits, self.torque_limits)

    def step(self, actions):
        """重写 step: 添加 action delay FIFO 支持。
        
        当 randomize_action_delay=True 时，每个 env 有不同的延迟帧数。
        新动作推入 FIFO 头部，从 action_delay_idx 位置取出延迟后的动作。
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        use_delay = getattr(self.cfg.domain_rand, 'randomize_action_delay', False)

        for _ in range(self.cfg.control.decimation):
            if use_delay:
                # 将当前 action 推入 FIFO 头部，旧的往后移
                self.action_fifo = torch.cat(
                    (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
                )
                # 每个 env 从自己的延迟位置取动作
                delayed_actions = self.action_fifo[
                    torch.arange(self.num_envs, device=self.device), self.action_delay_idx, :
                ]
                self.torques = self._compute_torques(delayed_actions).view(self.torques.shape)
            else:
                self.torques = self._compute_torques(self.actions).view(self.torques.shape)

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _reset_dofs(self, env_ids):
        """重写: 默认角度全为0时, 父类 0*rand=0 没有随机化探索。
        改为 default + rand 而非 default * rand。
        注意: default_dof_pos 可能是 (1, num_dof) 或 (num_envs, num_dof)。"""
        # 取对应 env 的默认角度（2D 则 per-env 已含零位偏差随机化）
        if self.default_dof_pos.shape[0] == 1:
            default_pos = self.default_dof_pos  # (1, num_dof) 广播
        else:
            default_pos = self.default_dof_pos[env_ids]  # (len(env_ids), num_dof)
        # 加性随机化: 在默认角度基础上 ±0.05 rad (0.5Nm电机弱，不能偏太多)
        self.dof_pos[env_ids] = default_pos + torch_rand_float(-0.05, 0.05, 
            (len(env_ids), self.num_dof), device=self.device)
        # 轮子角度随机无意义，清零
        self.dof_pos[env_ids, self.wheel_indices[0]] = 0.
        self.dof_pos[env_ids, self.wheel_indices[1]] = 0.
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _init_buffers(self):
        super()._init_buffers()

        # 找到轮子关节索引 (left_wheel_joint=2, right_wheel_joint=5)
        self.wheel_indices = []
        for i, name in enumerate(self.dof_names):
            if "wheel" in name:
                self.wheel_indices.append(i)
        self.wheel_indices = torch.tensor(self.wheel_indices, device=self.device, dtype=torch.long)

        # ★ 关键：continuous 关节在 URDF 没有 effort limit，IsaacGym 读到 inf
        #   必须手动覆盖为真实电机力矩上限 (2 N·m)
        self.torque_limits[self.wheel_indices] = 0.5
        print(f"[Bubble] ★ Wheel torque limits overridden to 0.5 N·m")
        print(f"[Bubble] ★ Wheel drive mode: {self.cfg.control.wheel_drive_mode}")

        # ====================================================================
        # ★ 域随机化: 扩展 p_gains/d_gains 为 2D (num_envs, num_dof)
        #   + 添加 torques_scale / action_delay / default_dof_pos 随机化
        #   参考 TRON1A base_task.py 实现
        # ====================================================================

        # 父类 p_gains/d_gains 是 1D (num_actions,)，扩展为 2D (num_envs, num_dof)
        p_gains_1d = self.p_gains.clone()  # (num_actions,)
        d_gains_1d = self.d_gains.clone()
        self.p_gains = p_gains_1d.unsqueeze(0).repeat(self.num_envs, 1)  # (num_envs, num_dof)
        self.d_gains = d_gains_1d.unsqueeze(0).repeat(self.num_envs, 1)

        # 力矩缩放因子 (num_envs, num_dof)，默认全1
        self.torques_scale = torch.ones(
            self.num_envs, self.num_dof,
            dtype=torch.float, device=self.device, requires_grad=False,
        )

        # --- Kp 随机化 ---
        if getattr(self.cfg.domain_rand, 'randomize_Kp', False):
            kp_min, kp_max = self.cfg.domain_rand.randomize_Kp_range
            self.p_gains *= torch_rand_float(
                kp_min, kp_max, self.p_gains.shape, device=self.device,
            )
            print(f"[Bubble] ★ Kp randomized: range [{kp_min}, {kp_max}]")

        # --- Kd 随机化 ---
        if getattr(self.cfg.domain_rand, 'randomize_Kd', False):
            kd_min, kd_max = self.cfg.domain_rand.randomize_Kd_range
            self.d_gains *= torch_rand_float(
                kd_min, kd_max, self.d_gains.shape, device=self.device,
            )
            print(f"[Bubble] ★ Kd randomized: range [{kd_min}, {kd_max}]")

        # --- 电机力矩缩放随机化 ---
        if getattr(self.cfg.domain_rand, 'randomize_motor_torque', False):
            t_min, t_max = self.cfg.domain_rand.randomize_motor_torque_range
            self.torques_scale *= torch_rand_float(
                t_min, t_max, self.torques_scale.shape, device=self.device,
            )
            print(f"[Bubble] ★ Motor torque scale randomized: range [{t_min}, {t_max}]")

        # --- 默认关节角度随机化 (模拟零位偏差) ---
        # 父类 default_dof_pos 是 (1, num_dof)，先扩展为 (num_envs, num_dof)
        if self.default_dof_pos.shape[0] == 1:
            self.default_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        if getattr(self.cfg.domain_rand, 'randomize_default_dof_pos', False):
            dof_min, dof_max = self.cfg.domain_rand.randomize_default_dof_pos_range
            self.default_dof_pos += torch_rand_float(
                dof_min, dof_max, (self.num_envs, self.num_dof), device=self.device,
            )
            # 轮子默认角度保持 0（continuous joint）
            self.default_dof_pos[:, self.wheel_indices] = 0.
            print(f"[Bubble] ★ Default DOF pos randomized: range [{dof_min}, {dof_max}] rad")

        # --- 动作延迟随机化 (action delay FIFO) ---
        delay_max_ms = getattr(self.cfg.domain_rand, 'delay_ms_range', [0, 0])[1]
        delay_max_steps = int(np.ceil(delay_max_ms / 1000.0 / self.sim_params.dt)) if delay_max_ms > 0 else 1
        delay_max_steps = max(delay_max_steps, 1)  # 至少 1 帧 FIFO

        self.action_delay_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device, requires_grad=False,
        )
        self.action_fifo = torch.zeros(
            (self.num_envs, delay_max_steps, self.num_actions),
            dtype=torch.float, device=self.device, requires_grad=False,
        )

        if getattr(self.cfg.domain_rand, 'randomize_action_delay', False):
            delay_min_ms, delay_max_ms = self.cfg.domain_rand.delay_ms_range
            action_delay_idx = torch.round(
                torch_rand_float(
                    delay_min_ms / 1000.0 / self.sim_params.dt,
                    delay_max_ms / 1000.0 / self.sim_params.dt,
                    (self.num_envs, 1), device=self.device,
                )
            ).squeeze(-1)
            self.action_delay_idx = action_delay_idx.long().clamp(0, delay_max_steps - 1)
            print(f"[Bubble] ★ Action delay randomized: range [{delay_min_ms}, {delay_max_ms}] ms, max FIFO depth={delay_max_steps}")
        else:
            print(f"[Bubble] Action delay disabled (FIFO depth={delay_max_steps})")

        # ====================================================================

        # ★ 关节状态历史 buffer（参考 Diablo）
        if getattr(self.cfg.env, 'enable_joint_state_history', False):
            hist_len = self.cfg.env.joint_state_history_length
            self.dof_pos_error_history = torch.zeros(
                self.num_envs, hist_len, self.num_actions,
                dtype=torch.float, device=self.device, requires_grad=False,
            )
            self.dof_vel_history = torch.zeros(
                self.num_envs, hist_len, self.num_actions,
                dtype=torch.float, device=self.device, requires_grad=False,
            )
            print(f"[Bubble] ★ Joint state history enabled: {hist_len} frames × {self.num_actions} DOFs = {hist_len * self.num_actions * 2} extra obs dims")

        print(f"[Bubble] Wheel DOF indices: {self.wheel_indices.tolist()}")
        print(f"[Bubble] DOF names: {self.dof_names}")
        print(f"[Bubble] Torque limits: {self.torque_limits}")
        print(f"[Bubble] P gains (env0): {self.p_gains[0]}")
        print(f"[Bubble] D gains (env0): {self.d_gains[0]}")
        print(f"[Bubble] Default DOF pos (env0): {self.default_dof_pos[0]}")
        print(f"[Bubble] DOF pos limits: {self.dof_pos_limits}")

    def compute_observations(self):
        """观测根据 wheel_drive_mode 调整轮子位置信息:

        "bubble" / "b2w" — dof_pos[wheel]=0 (continuous 位置无意义)
        "diablo"         — 保留轮子 dof_pos (网络需要看到轮子角度来做位置追踪)

        布局: lin_vel(3) + ang_vel(3) + gravity(3) + commands(3) + dof_pos(6) + dof_vel(6) + actions(6) = 30
        若启用 history: + dof_pos_err_history(10*6=60) + dof_vel_history(10*6=60) = 150
        """
        mode = self.cfg.control.wheel_drive_mode

        # 位置误差
        dof_pos_obs = (self.dof_pos - self.default_dof_pos).clone()
        if mode != "diablo":
            # bubble / b2w: 轮子位置清零
            dof_pos_obs[:, self.wheel_indices] = 0

        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,            # 3
                self.base_ang_vel * self.obs_scales.ang_vel,            # 3
                self.projected_gravity,                                  # 3
                self.commands[:, :3] * self.commands_scale,             # 3
                dof_pos_obs * self.obs_scales.dof_pos,                  # 6
                self.dof_vel * self.obs_scales.dof_vel,                 # 6
                self.actions,                                            # 6
            ),
            dim=-1,
        )  # 总维度: 30

        # ★ 拼接关节状态历史（参考 Diablo）
        if getattr(self.cfg.env, 'enable_joint_state_history', False):
            self.obs_buf = torch.cat(
                (
                    self.obs_buf,
                    self.dof_pos_error_history.view(self.num_envs, -1) * self.obs_scales.dof_pos,   # 10*6=60
                    self.dof_vel_history.view(self.num_envs, -1) * self.obs_scales.dof_vel,         # 10*6=60
                ),
                dim=-1,
            )  # 总维度: 150

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """噪声向量，与观测维度对齐（30 或 150）"""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel       # lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel      # ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level                                 # gravity
        noise_vec[9:12] = 0.0                                                               # commands
        noise_vec[12:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos    # dof_pos
        noise_vec[18:24] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel    # dof_vel
        noise_vec[24:30] = 0.0                                                              # actions

        # ★ 关节状态历史噪声（参考 Diablo）
        if getattr(self.cfg.env, 'enable_joint_state_history', False):
            hist_len = self.cfg.env.joint_state_history_length
            n_dof = self.num_actions  # 6
            # dof_pos_error history: [30 : 30+60]
            noise_vec[30:30 + hist_len * n_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            # dof_vel history: [90 : 90+60]
            noise_vec[30 + hist_len * n_dof:30 + 2 * hist_len * n_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel

        if self.cfg.terrain.measure_heights:
            base_idx = 30
            if getattr(self.cfg.env, 'enable_joint_state_history', False):
                base_idx = 150
            noise_vec[base_idx:] = (
                noise_scales.height_measurements
                * noise_level
                * self.obs_scales.height_measurements
            )
        return noise_vec

    def _init_extras_diag(self):
        """初始化诊断累计 buffer（每个 episode 累加，reset 时取均值写入 extras）"""
        n = self.num_envs
        dev = self.device
        self._diag_sums = {
            "base_height":     torch.zeros(n, device=dev),
            "pitch":           torch.zeros(n, device=dev),
            "roll":            torch.zeros(n, device=dev),
            "wheel_vel_abs":   torch.zeros(n, device=dev),
            "leg_torque_abs":  torch.zeros(n, device=dev),
            "wheel_torque_abs":torch.zeros(n, device=dev),
            "penalized_contact_ratio": torch.zeros(n, device=dev),
            "actual_vx":       torch.zeros(n, device=dev),
            "cmd_vx":          torch.zeros(n, device=dev),
        }
        # 逐关节角度
        self._leg_indices_list = [i for i in range(self.num_dof) if i not in self.wheel_indices.tolist()]
        for idx in self._leg_indices_list:
            self._diag_sums[f"joint_{self.dof_names[idx]}"] = torch.zeros(n, device=dev)
        self._diag_count = torch.zeros(n, device=dev)

    def post_physics_step(self):
        """重写: 每步累计诊断指标 + 更新关节状态历史"""
        super().post_physics_step()

        # ★ 关节状态历史滑动窗口更新（参考 Diablo）
        if getattr(self.cfg.env, 'enable_joint_state_history', False):
            mode = self.cfg.control.wheel_drive_mode
            # 位置误差 history（与观测一致：bubble/b2w 轮子清零）
            dof_pos_err = (self.dof_pos - self.default_dof_pos).clone()
            if mode != "diablo":
                dof_pos_err[:, self.wheel_indices] = 0
            self.dof_pos_error_history = torch.roll(self.dof_pos_error_history, shifts=-1, dims=1)
            self.dof_pos_error_history[:, -1, :] = dof_pos_err
            # 速度 history
            self.dof_vel_history = torch.roll(self.dof_vel_history, shifts=-1, dims=1)
            self.dof_vel_history[:, -1, :] = self.dof_vel

        # 第一次调用时初始化
        if not hasattr(self, '_diag_sums'):
            self._init_extras_diag()

        # ---- 每步累计 ----
        with torch.no_grad():
            self._diag_sums["base_height"] += self.root_states[:, 2]
            self._diag_sums["pitch"] += self.projected_gravity[:, 0]
            self._diag_sums["roll"] += self.projected_gravity[:, 1]
            self._diag_sums["wheel_vel_abs"] += self.dof_vel[:, self.wheel_indices].abs().mean(dim=1)
            self._diag_sums["leg_torque_abs"] += self.torques[:, self._leg_indices_list].abs().mean(dim=1)
            self._diag_sums["wheel_torque_abs"] += self.torques[:, self.wheel_indices].abs().mean(dim=1)
            penalized = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
            self._diag_sums["penalized_contact_ratio"] += (penalized > 0.1).any(dim=1).float()
            self._diag_sums["actual_vx"] += self.base_lin_vel[:, 0]
            self._diag_sums["cmd_vx"] += self.commands[:, 0]
            for idx in self._leg_indices_list:
                self._diag_sums[f"joint_{self.dof_names[idx]}"] += self.dof_pos[:, idx]
            self._diag_count += 1

    def reset_idx(self, env_ids):
        """重写: reset 后把诊断均值写入 extras['episode'] + 清零关节历史"""
        if len(env_ids) > 0 and hasattr(self, '_diag_sums'):
            # 先保存诊断值（super 会清空 extras["episode"]）
            diag_snapshot = {}
            count = self._diag_count[env_ids].clamp(min=1)
            for key, buf in self._diag_sums.items():
                diag_snapshot["diag_" + key] = (buf[env_ids] / count).mean().item()
            # 清零已 reset 的 env
            for buf in self._diag_sums.values():
                buf[env_ids] = 0.
            self._diag_count[env_ids] = 0.

            # 调用父类（会做 self.extras["episode"] = {} 然后填入 rew_xxx）
            super().reset_idx(env_ids)

            # 在父类之后追加诊断数据
            self.extras["episode"].update(diag_snapshot)
        else:
            super().reset_idx(env_ids)

        # ★ 清零关节状态历史（参考 Diablo）
        if getattr(self.cfg.env, 'enable_joint_state_history', False) and len(env_ids) > 0:
            self.dof_pos_error_history[env_ids].zero_()
            self.dof_vel_history[env_ids].zero_()

        # ★ 清零 action delay FIFO
        if hasattr(self, 'action_fifo') and len(env_ids) > 0:
            self.action_fifo[env_ids] = 0.
