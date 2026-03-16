# ==============================================================================
# Description: Bubble 点足模式环境
# 方案 A: Config 软锁定 — 轮子超大 Kp/Kd 锁死 + 步态生成器行走
#
# 继承自 Bubble，核心改动:
#   - _compute_torques: "pointfoot" 模式，轮子 PD 锁定，action 只映射到腿部
#   - _init_buffers: 初始化步态相关 buffer (gaits, gait_indices, clock_inputs, ...)
#   - compute_observations: 观测中加入步态时钟 + 步态参数 + 仅腿部关节
#   - _step_contact_targets: 步态相位推进 + desired_contact_states
#   - 全套步态感知奖励函数 (移植自 TRON1A solefoot)
#   - compute_reward: 支持 clip_single_reward
# ==============================================================================

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from legged_gym.utils.math import *
from legged_gym.utils.helpers import class_to_dict
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs.bubble.bubble import Bubble
from .bubble_pointfoot_config import BubblePointfootCfg


class BubblePointfoot(Bubble):
    """Bubble 点足模式环境 — 轮子软锁定 + 步态行走。

    关键设计:
    - 轮子通过超大 Kp=1000/Kd=50 软锁定在 0 度
    - 网络输出 6D action, 但 wheel 分量在 _compute_torques 中强制归零
    - 步态生成器产生 desired_contact_states 驱动交替行走
    - 步态感知奖励 (移植自 TRON1A solefoot_flat)
    """

    def __init__(self, cfg: BubblePointfootCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ======================== 力矩计算 ========================

    def _compute_torques(self, actions):
        """pointfoot 模式: 轮子硬锁定, action 只作用于腿部关节。

        轮子: 每步物理仿真前强制 pos=0, vel=0 (见 step 重写)
              torque 也强制为 0 — 不产生任何驱动
        腿部: torque = Kp_leg * (action_scaled + default - pos) - Kd_leg * vel
        """
        actions_scaled = actions * self.cfg.control.action_scale

        # 完整 PD 控制 (所有 DOF)
        dof_err = self.default_dof_pos - self.dof_pos

        # ★ 轮子 action 强制归零 — 网络不能控制轮子
        actions_scaled_full = actions_scaled.clone()
        actions_scaled_full[:, self.wheel_indices] = 0.0

        torques = self.p_gains * (actions_scaled_full + dof_err) - self.d_gains * self.dof_vel

        # ★★ 轮子 torque 也归零 — 完全不给轮子施加任何力
        torques[:, self.wheel_indices] = 0.0

        return torch.clip(torques * self.torques_scale, -self.torque_limits, self.torque_limits)

    def step(self, actions):
        """重写 step: 在每个物理子步中强制锁死轮子 DOF 状态。

        原理: 仅靠 PD 锁 (Kp=1000) 仍会产生微振荡。
        最可靠的方式是每个 substep 后直接覆盖 dof_state，
        让轮子在物理引擎中彻底静止。
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

            # ★★ 每个子步后强制锁死轮子: pos=0, vel=0
            self.dof_pos[:, self.wheel_indices] = 0.0
            self.dof_vel[:, self.wheel_indices] = 0.0

        # ★ 写回 DOF state 到物理引擎 (确保下一帧从锁死状态开始)
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(all_env_ids),
            len(all_env_ids),
        )

        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ======================== 初始化 ========================

    def _init_buffers(self):
        """扩展: 初始化步态相关 buffer"""
        super()._init_buffers()

        # ★ 轮子现在通过 step() 中强制覆盖 dof_state 来锁死
        #   torque 归零，不需要高扭矩上限
        self.torque_limits[self.wheel_indices] = 0.0  # 轮子不产生任何力矩
        print(f"[BubblePointfoot] ★ Wheel torques disabled (locked via dof_state override)")
        
        # ★ 轮子 PD 增益也归零 — 不参与力矩计算
        self.p_gains[:, self.wheel_indices] = 0.0
        self.d_gains[:, self.wheel_indices] = 0.0
        print(f"[BubblePointfoot] ★ Wheel PD gains zeroed (Kp=0, Kd=0)")

        # 找到腿部关节索引 (排除轮子)
        self.leg_indices = torch.tensor(
            [i for i in range(self.num_dof) if i not in self.wheel_indices.tolist()],
            device=self.device, dtype=torch.long
        )
        print(f"[BubblePointfoot] Leg DOF indices: {self.leg_indices.tolist()}")

        # --- 步态相关 buffer ---
        self.gaits = torch.zeros(
            self.num_envs, self.cfg.gait.num_gait_params,
            dtype=torch.float, device=self.device, requires_grad=False,
        )
        self.desired_contact_states = torch.zeros(
            self.num_envs, len(self.feet_indices),
            dtype=torch.float, device=self.device, requires_grad=False,
        )
        self.gait_indices = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False,
        )
        self.clock_inputs_sin = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False,
        )
        self.clock_inputs_cos = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False,
        )

        # 步态参数范围 (用于采样)
        self.gaits_ranges = class_to_dict(self.cfg.gait.ranges)

        # 足端状态 buffer
        self.foot_positions = torch.zeros(
            self.num_envs, len(self.feet_indices), 3,
            dtype=torch.float, device=self.device, requires_grad=False,
        )
        self.last_foot_positions = torch.zeros_like(self.foot_positions)
        self.foot_velocities = torch.zeros_like(self.foot_positions)
        self.foot_heights = torch.zeros(
            self.num_envs, len(self.feet_indices),
            dtype=torch.float, device=self.device, requires_grad=False,
        )

        # 上一帧接触状态 (步态力/速度奖励需要)
        self.contact_filt = torch.zeros(
            self.num_envs, len(self.feet_indices),
            dtype=torch.bool, device=self.device,
        )
        self.last_contacts_pf = torch.zeros_like(self.contact_filt)

        # obs history buffer
        obs_hist_len = getattr(self.cfg.env, 'obs_history_length', 1)
        self.obs_history = torch.zeros(
            self.num_envs, self.num_obs * obs_hist_len,
            dtype=torch.float, device=self.device, requires_grad=False,
        )

        # 上一帧 action (二阶平滑需要两帧)
        self.last_actions_pf = torch.zeros(
            self.num_envs, self.num_actions, 2,
            dtype=torch.float, device=self.device, requires_grad=False,
        )

        # commands scale (3D: vx, vy, wz)
        self.commands_scale_pf = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            device=self.device, requires_grad=False,
        )

        print(f"[BubblePointfoot] ★ Pointfoot mode initialized")
        print(f"[BubblePointfoot]   Gait params: {self.cfg.gait.num_gait_params}")
        print(f"[BubblePointfoot]   Obs dim: {self.num_obs}, History: {obs_hist_len}")
        print(f"[BubblePointfoot]   Feet indices: {self.feet_indices.tolist()}")

    # ======================== 足端状态 ========================

    def _compute_foot_state(self):
        """计算足端位置、速度、高度"""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rigid_body_state = gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        ).view(self.num_envs, -1, 13)

        self.foot_positions = rigid_body_state[:, self.feet_indices, 0:3]
        self.foot_velocities = (self.foot_positions - self.last_foot_positions) / self.dt

        # 足端高度 = 足端 z 坐标 - 地面高度 (平地 = 0)
        self.foot_heights = self.foot_positions[:, :, 2]

        # 接触滤波
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 2.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts_pf)
        self.last_contacts_pf = contact

    # ======================== 步态生成器 ========================

    def _step_contact_targets(self):
        """推进步态相位, 计算 desired_contact_states。
        移植自 TRON1A base_task._step_contact_targets。
        """
        frequencies = self.gaits[:, 0]
        offsets = self.gaits[:, 1]
        durations = torch.cat(
            [
                self.gaits[:, 2].view(self.num_envs, 1),
                self.gaits[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        # 推进相位
        self.gait_indices = torch.remainder(
            self.gait_indices + self.dt * frequencies, 1.0
        )

        # 时钟信号
        self.clock_inputs_sin = torch.sin(2 * np.pi * self.gait_indices)
        self.clock_inputs_cos = torch.cos(2 * np.pi * self.gait_indices)

        # von mises 平滑接触概率
        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        foot_indices = torch.remainder(
            torch.cat(
                [
                    self.gait_indices.view(self.num_envs, 1),
                    (self.gait_indices + offsets + 1).view(self.num_envs, 1),
                ],
                dim=1,
            ),
            1.0,
        )
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (
            0.5 / durations[stance_idxs]
        )
        foot_indices[swing_idxs] = 0.5 + (
            torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]
        ) * (0.5 / (1 - durations[swing_idxs]))

        self.desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (
            1 - smoothing_cdf_start(foot_indices - 1.5)
        )

    def _resample_gaits(self, env_ids):
        """随机采样步态参数"""
        if len(env_ids) == 0:
            return
        self.gaits[env_ids, 0] = torch_rand_float(
            self.gaits_ranges["frequencies"][0],
            self.gaits_ranges["frequencies"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.gaits[env_ids, 1] = torch_rand_float(
            self.gaits_ranges["offsets"][0],
            self.gaits_ranges["offsets"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.gaits[env_ids, 1] = 0.5  # 固定 180° 相位差
        self.gaits[env_ids, 2] = torch_rand_float(
            self.gaits_ranges["durations"][0],
            self.gaits_ranges["durations"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.gaits[env_ids, 3] = torch_rand_float(
            self.gaits_ranges["swing_height"][0],
            self.gaits_ranges["swing_height"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)

    # ======================== 观测 ========================

    def compute_observations(self):
        """观测布局:
        ang_vel(3) + gravity(3) + dof_pos_leg(4) + dof_vel_leg(4) +
        actions(6) + clock_sin(1) + clock_cos(1) + gait_params(4) +
        commands(3) = 29
        """
        # 仅腿部关节的位置误差和速度
        dof_pos_leg = (self.dof_pos[:, self.leg_indices] - self.default_dof_pos[:, self.leg_indices])
        dof_vel_leg = self.dof_vel[:, self.leg_indices]

        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,                    # 3
                self.projected_gravity,                                          # 3
                dof_pos_leg * self.obs_scales.dof_pos,                          # 4
                dof_vel_leg * self.obs_scales.dof_vel,                          # 4
                self.actions,                                                    # 6
                self.clock_inputs_sin.view(self.num_envs, 1),                   # 1
                self.clock_inputs_cos.view(self.num_envs, 1),                   # 1
                self.gaits,                                                      # 4
                self.commands[:, :3] * self.commands_scale_pf,                  # 3
            ),
            dim=-1,
        )  # 总维度: 29

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

        # obs history
        obs_hist_len = getattr(self.cfg.env, 'obs_history_length', 1)
        if obs_hist_len > 1:
            self.obs_history = torch.cat(
                (self.obs_history[:, self.num_obs:], self.obs_buf), dim=-1
            )

    def _get_noise_scale_vec(self, cfg):
        """噪声向量, 与观测维度对齐 (29D)"""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        idx = 0
        noise_vec[idx:idx+3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel   # ang_vel
        idx += 3
        noise_vec[idx:idx+3] = noise_scales.gravity * noise_level                              # gravity
        idx += 3
        noise_vec[idx:idx+4] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos   # dof_pos_leg
        idx += 4
        noise_vec[idx:idx+4] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel   # dof_vel_leg
        idx += 4
        noise_vec[idx:idx+6] = 0.0                                                             # actions
        idx += 6
        noise_vec[idx:] = 0.0  # clock_inputs, gaits, commands — 无噪声
        return noise_vec

    # ======================== 物理步后处理 ========================

    def post_physics_step(self):
        """重写: 计算足端状态 + 推进步态 + 更新 action 历史"""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # 更新基座状态
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 计算足端状态
        self._compute_foot_state()

        # 推进步态
        self._step_contact_targets()

        # 命令重采样
        env_ids = (
            (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False).flatten()
        )
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        # 零指令
        if hasattr(self.cfg.commands, 'zero_command_prob'):
            zero_prob = self.cfg.commands.zero_command_prob
            min_norm = getattr(self.cfg.commands, 'min_norm', 0.05)
            small_cmd = torch.norm(self.commands[:, :3], dim=1) < min_norm
            self.commands[small_cmd, :3] = 0.0

        # 高度测量 (如果需要)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # 终止 + 奖励 + 重置
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        # 更新 action 历史
        self.last_actions_pf[:, :, 1] = self.last_actions_pf[:, :, 0]
        self.last_actions_pf[:, :, 0] = self.actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_foot_positions[:] = self.foot_positions[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    # ======================== 重置 ========================

    def reset_idx(self, env_ids):
        """重写: 增加步态 buffer 重置"""
        if len(env_ids) == 0:
            return

        # 课程
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # 重置物理状态
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)

        # 重置 buffer
        self.last_actions[env_ids] = 0.0
        self.last_actions_pf[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.gait_indices[env_ids] = 0
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_contacts_pf[env_ids] = False
        self.contact_filt[env_ids] = False

        # obs history 清零
        obs_hist_len = getattr(self.cfg.env, 'obs_history_length', 1)
        if obs_hist_len > 1:
            self.obs_history[env_ids] = 0

        # episode sums
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        """重写: 加性随机化, 轮子锁定在 0"""
        self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(
            -0.1, 0.1, (len(env_ids), self.num_dof), device=self.device
        )
        # 轮子锁定在 0
        self.dof_pos[env_ids, self.wheel_indices[0]] = 0.0
        self.dof_pos[env_ids, self.wheel_indices[1]] = 0.0
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _resample_commands(self, env_ids):
        """重写: 3D 指令采样 (vx, vy, wz)"""
        if len(env_ids) == 0:
            return
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1), device=self.device,
        ).squeeze(1)

        # 零指令概率
        zero_prob = getattr(self.cfg.commands, 'zero_command_prob', 0.3)
        zero_mask = torch.rand(len(env_ids), device=self.device) < zero_prob
        self.commands[env_ids[zero_mask], :3] = 0.0

    # ======================== 奖励计算 ========================

    def compute_reward(self):
        """重写: 支持 clip_single_reward + only_positive_rewards=False"""
        self.rew_buf[:] = 0.0
        clip_single = getattr(self.cfg.rewards, 'clip_single_reward', None)

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if clip_single is not None:
                rew = torch.clip(rew, -clip_single * self.dt, clip_single * self.dt)
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

        # clip total reward
        clip_reward = getattr(self.cfg.rewards, 'clip_reward', None)
        if clip_reward is not None:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], -clip_reward * self.dt, clip_reward * self.dt)

        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    # ======================== 步态感知奖励 (移植自 TRON1A) ========================

    def _reward_keep_balance(self):
        """存活奖励: 只要不摔倒就给 1.0"""
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_tracking_contacts_shaped_force(self):
        """步态接触力跟踪: 摆动相脚应离地 (接触力应为 0)"""
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        for i in range(len(self.feet_indices)):
            swing_phase = 1 - desired_contact[:, i]
            if self.reward_scales.get("tracking_contacts_shaped_force", 0) > 0:
                reward += swing_phase * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma
                )
            else:
                reward += swing_phase * (
                    1 - torch.exp(-foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma)
                )

        return reward / max(len(self.feet_indices), 1)

    def _reward_tracking_contacts_shaped_vel(self):
        """步态速度跟踪: 支撑相脚应静止, 摆动相脚应有垂直速度"""
        foot_vel_norm = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        for i in range(len(self.feet_indices)):
            stand_phase = desired_contact[:, i]
            swing_phase = 1 - desired_contact[:, i]

            if self.reward_scales.get("tracking_contacts_shaped_vel", 0) > 0:
                reward += stand_phase * torch.exp(
                    -foot_vel_norm[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
                reward += swing_phase * torch.exp(
                    -(self.foot_velocities[:, i, 2] ** 2) / self.cfg.rewards.gait_vel_sigma
                )
            else:
                reward += stand_phase * (
                    1 - torch.exp(-foot_vel_norm[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)
                )
                reward += swing_phase * (
                    1 - torch.exp(-(self.foot_velocities[:, i, 2] ** 2) / self.cfg.rewards.gait_vel_sigma)
                )

        return reward / max(len(self.feet_indices), 1)

    def _reward_tracking_contacts_shaped_height(self):
        """步态高度跟踪: 摆动相脚应达到目标高度, 支撑相脚应贴地"""
        foot_heights = self.foot_heights
        desired_contact = self.desired_contact_states
        des_height = self.gaits[:, 3]  # swing_height
        reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        for i in range(len(self.feet_indices)):
            swing_phase = 1 - desired_contact[:, i]
            stand_phase = desired_contact[:, i]

            if self.reward_scales.get("tracking_contacts_shaped_height", 0) > 0:
                reward += swing_phase * torch.exp(
                    -(foot_heights[:, i] - des_height) ** 2 / self.cfg.rewards.gait_height_sigma
                )
                reward += stand_phase * torch.exp(
                    -(foot_heights[:, i]) ** 2 / self.cfg.rewards.gait_height_sigma
                )
            else:
                reward += swing_phase * (
                    1 - torch.exp(-(foot_heights[:, i] - des_height) ** 2 / self.cfg.rewards.gait_height_sigma)
                )
                reward += stand_phase * (
                    1 - torch.exp(-(foot_heights[:, i]) ** 2 / self.cfg.rewards.gait_height_sigma)
                )

        return reward / max(len(self.feet_indices), 1)

    def _reward_feet_distance(self):
        """惩罚双脚距离过近, 防止交叉绊倒"""
        feet_dist = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        return torch.clip(self.cfg.rewards.min_feet_distance - feet_dist, 0, 1)

    def _reward_feet_regulation(self):
        """接近地面时惩罚水平速度 (脚越低越不该有水平速度)"""
        feet_height_ref = self.cfg.rewards.base_height_target * 0.025
        reward = torch.sum(
            torch.exp(-self.foot_heights / max(feet_height_ref, 0.001))
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)),
            dim=1,
        )
        return reward

    def _reward_foot_landing_vel(self):
        """惩罚着地时的垂直速度 (保护硬件)"""
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        threshold = self.cfg.rewards.about_landing_threshold
        about_to_land = (self.foot_heights < threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        return torch.sum(torch.square(landing_z_vels), dim=1)

    def _reward_action_smooth(self):
        """二阶动作平滑: (a_t - 2*a_{t-1} + a_{t-2})²"""
        return torch.sum(
            torch.square(
                self.actions
                - 2 * self.last_actions_pf[:, :, 0]
                + self.last_actions_pf[:, :, 1]
            ),
            dim=1,
        )

    def _reward_base_height(self):
        """重载: 惩罚高度偏离目标 (负权重, 与 TRON1A 一致)"""
        base_height = self.root_states[:, 2]
        return torch.abs(base_height - self.cfg.rewards.base_height_target)

    def _reward_tracking_lin_vel(self):
        """速度跟踪 (高斯核)"""
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """角速度跟踪 (高斯核)"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_orientation(self):
        """惩罚非水平姿态"""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_lin_vel_z(self):
        """惩罚 z 轴线速度"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """惩罚 xy 轴角速度"""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_torques(self):
        """惩罚力矩平方和 (仅腿部)"""
        return torch.sum(torch.square(self.torques[:, self.leg_indices]), dim=1)

    def _reward_dof_acc(self):
        """惩罚关节加速度"""
        dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        return torch.sum(torch.square(dof_acc[:, self.leg_indices]), dim=1)

    def _reward_action_rate(self):
        """惩罚动作变化率"""
        return torch.sum(torch.square(self.actions - self.last_actions_pf[:, :, 0]), dim=1)

    def _reward_collision(self):
        """惩罚非法碰撞"""
        return torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
            dim=1,
        )

    def _reward_dof_pos_limits(self):
        """惩罚关节接近极限"""
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_contact_forces(self):
        """惩罚足底接触力过大"""
        return torch.sum(
            (self.contact_forces[:, self.feet_indices, 2] - self.cfg.rewards.max_contact_force).clip(min=0.0),
            dim=1,
        )

    def _reward_termination(self):
        """终止惩罚"""
        return self.reset_buf * ~self.time_out_buf
