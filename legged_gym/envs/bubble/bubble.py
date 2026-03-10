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
        r = torch.sin(theta_right) + torch.sin(
            self.dof_pos[:, thigh_right] + theta_right
        )
        l = torch.sin(theta_left) + torch.sin(
            self.dof_pos[:, thigh_left] + theta_left
        )
        return torch.square(r + l)

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

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """重写: 默认角度全为0时, 父类 0*rand=0 没有随机化探索。
        改为 default + rand 而非 default * rand。"""
        # 加性随机化: 在默认角度基础上 ±0.05 rad (0.5Nm电机弱，不能偏太多)
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.05, 0.05, 
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
        print(f"[Bubble] P gains: {self.p_gains}")
        print(f"[Bubble] D gains: {self.d_gains}")
        print(f"[Bubble] Default DOF pos: {self.default_dof_pos}")
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
