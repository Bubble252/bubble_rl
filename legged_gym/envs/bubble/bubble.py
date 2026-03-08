# ==============================================================================
# Description: Bubble 双轮足机器人环境
# 继承自 LeggedRobot，重载观测和奖励函数
# 参考 Diablo 的实现，支持跳跃和腿长调节
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

    def compute_reward(self):
        """重写奖励计算：让 collision 惩罚也绕过 only_positive_rewards 截断。
        这样 shank/idler 碰地的惩罚不会被截断为0，策略能真正感受到痛。
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # collision 和 termination 留到截断之后再加
            if name == "collision":
                continue
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # 正奖励截断
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # termination: 截断后加（父类逻辑）
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        # collision: 也在截断后加，确保 shank/idler 碰地惩罚不被吃掉
        if "collision" in self.reward_scales:
            rew = self._reward_collision() * self.reward_scales["collision"]
            self.rew_buf += rew
            self.episode_sums["collision"] += rew

    def _reward_base_height(self):
        """重载: 直接用root z坐标计算高度（不依赖measured_heights，因为measure_heights=False）"""
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_no_fly(self):
        """奖励轮子保持与地面接触，跳跃时反转"""
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        rew = 1.0 * (torch.sum(1.0 * contacts, dim=1) == 2)
        rew[self.jump] *= -1  # 跳跃时惩罚贴地（鼓励离地）
        return rew

    def _reward_tracking_lin_vel(self):
        """重载: 指数型速度跟踪奖励，鼓励精确跟踪而非保守不动"""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / 0.25)  # sigma=0.25, 精度0.5m/s时奖励~0.37

    def _reward_stand_symmetric(self):
        """新增: 惩罚左右腿不对称角度，防止单腿跪地"""
        # 左右thigh应相同, 左右knee应相同
        thigh_diff = torch.square(self.dof_pos[:, 0] - self.dof_pos[:, 3])  # left-right thigh
        knee_diff = torch.square(self.dof_pos[:, 1] - self.dof_pos[:, 4])   # left-right knee
        return thigh_diff + knee_diff

    def _reward_wheel_vel_tracking(self):
        """奖励轮子转速匹配线速度命令。
        目标轮速 = cmd_vel_x / wheel_radius。
        用指数型奖励，轮速越接近目标越高分。
        """
        wheel_radius = 0.033  # bubble 轮子半径
        # 目标轮速 (rad/s): 前进速度 / 半径
        target_wheel_vel = self.commands[:, 0] / wheel_radius  # shape: (num_envs,)
        # 左右轮实际角速度
        left_wheel_vel = self.dof_vel[:, self.wheel_indices[0]]
        right_wheel_vel = self.dof_vel[:, self.wheel_indices[1]]
        avg_wheel_vel = (left_wheel_vel + right_wheel_vel) / 2.0
        # 指数型奖励
        error = torch.square(avg_wheel_vel - target_wheel_vel)
        return torch.exp(-error / 25.0)  # sigma=25, 5 rad/s 误差时奖励~0.37

    def _reward_no_moonwalk(self):
        """惩罚左右腿不对称（"太空步"）"""
        joints = list(self.cfg.init_state.default_joint_angles.keys())
        thigh_left = joints.index("left_thigh_joint")
        knee_left = joints.index("left_knee_joint")
        thigh_right = joints.index("right_thigh_joint")
        knee_right = joints.index("right_knee_joint")

        # 记录 theta 供 z_adjust_leg 使用
        self.theta_right = self.dof_pos[:, knee_right]
        self.theta_left = self.dof_pos[:, knee_left]

        # 两腿在 x 轴方向投影之和应相等
        r = torch.sin(self.theta_right) + torch.sin(
            self.dof_pos[:, thigh_right] + self.theta_right
        )
        l = torch.sin(self.theta_left) + torch.sin(
            self.dof_pos[:, thigh_left] + self.theta_left
        )
        rew = torch.square(r + l)
        return rew

    def _reward_encourage_jump(self):
        """奖励跳跃动作，非跳跃时反转为惩罚"""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        fly = torch.sum(1.0 * contact, dim=1) == 0  # 处于空中
        first_contact = (self.base_air_time > 0.0) * ~fly
        self.base_air_time += self.dt * torch.clip(
            self.root_states[:, 2],
            torch.tensor(0.0, device=self.device),
            self.commands[:, 4],
        )

        rew_airTime = (self.base_air_time - 5e-5) * first_contact

        # 奖励向上的速度
        l, r = self.command_ranges["jump_height"]
        rew_airTime += (
            torch.maximum(torch.tensor(0.0), self.root_states[:, 9])
            * (self.commands[:, 4] - l)
            / (r - l)
        )

        self.base_air_time *= ~fly
        rew_airTime[~self.jump] *= -1  # 非跳跃时反转为惩罚
        return rew_airTime

    def _reward_z_adjust_leg(self):
        """将膝盖弯曲至指定角度以实现腿长调节"""
        # bubble 的左右 knee 轴方向相同（都是 0,-1,0），所以不需要取反
        rew = torch.square(self.commands[:, 5] - self.theta_right)
        rew += torch.square(self.commands[:, 5] - self.theta_left)
        return (self.adjust_leg) * rew

    def _reward_lin_vel_z(self):
        """惩罚z轴线速度，跳跃时反转"""
        rew = torch.square(self.root_states[:, 9])
        rew[self.jump] *= -1
        return (~self.adjust_leg) * rew

    def _reward_feet_air_time(self):
        """奖励长步幅，非跳跃时反转为惩罚"""
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )
        rew_airTime *= (
            torch.norm(self.commands[:, :2], dim=1) > 0.1
        )  # 零命令时不奖励
        self.feet_air_time *= ~contact_filt
        rew_airTime[~self.jump] *= -1  # 非跳跃时反转为惩罚离地
        return rew_airTime

    def _compute_torques(self, actions):
        """混合控制: 腿部用 P 位置控制，轮子用直接力矩控制。
        
        腿部 (thigh, knee): torque = Kp*(action_scaled + default_pos - dof_pos) - Kd*dof_vel
        轮子 (wheel):       torque = action * wheel_action_scale  (直接力矩，简单有效)
        
        网络直接输出轮子力矩方向和大小，无需经过 PD 环节。
        """
        actions_scaled = actions * self.cfg.control.action_scale
        
        # 先全部按 P 模式计算（腿部）
        torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        
        # 轮子覆盖为直接力矩控制: action * scale = torque (N·m)
        wheel_torques = actions[:, self.wheel_indices] * self.cfg.control.wheel_action_scale
        torques[:, self.wheel_indices] = wheel_torques
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _init_buffers(self):
        super()._init_buffers()
        
        # 找到轮子关节的索引 (left_wheel_joint=2, right_wheel_joint=5)
        self.wheel_indices = []
        for i, name in enumerate(self.dof_names):
            if "wheel" in name:
                self.wheel_indices.append(i)
        self.wheel_indices = torch.tensor(self.wheel_indices, device=self.device, dtype=torch.long)
        
        # 轮子 URDF continuous 关节 effort=INF, 设合理上限防止物理爆炸
        wheel_torque_limit = getattr(self.cfg.control, 'wheel_torque_limit', 2.0)
        for idx in self.wheel_indices:
            if self.torque_limits[idx] > 100.0:
                self.torque_limits[idx] = wheel_torque_limit
                print(f"[Bubble] Clamped torque_limit for DOF {idx} ({self.dof_names[idx]}) to {wheel_torque_limit} N·m")
        
        print(f"[Bubble] Wheel DOF indices: {self.wheel_indices.tolist()}")
        print(f"[Bubble] DOF names: {self.dof_names}")
        print(f"[Bubble] Torque limits: {self.torque_limits}")
        print(f"[Bubble] Hybrid control: legs=P position, wheels=V velocity")
        
        self.base_air_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.jump = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.adjust_leg = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )

    def check_termination(self):
        """翻倒或角速度过大时立刻终止，防止物理爆炸"""
        super().check_termination()
        # 倾斜超过约60度 → 终止
        self.reset_buf |= self.projected_gravity[:, 2] > -0.5
        # 角速度超过 10 rad/s → 终止（正常行走不会超过这个值）
        ang_vel_magnitude = torch.norm(self.base_ang_vel, dim=1)
        self.reset_buf |= ang_vel_magnitude > 10.0
        # 线速度 z 超过 1.5 m/s → 终止（被弹飞）
        self.reset_buf |= torch.abs(self.base_lin_vel[:, 2]) > 1.5
        # 高度过低 → 终止（跪地/塌陷）
        self.reset_buf |= self.root_states[:, 2] < 0.08

    def _post_physics_step_callback(self):
        env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                1.5 * wrap_to_pi(self.commands[:, 3] - heading),
                self.cfg.commands.ranges.ang_vel_yaw[0],
                self.cfg.commands.ranges.ang_vel_yaw[1],
            )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (
            self.common_step_counter % self.cfg.domain_rand.push_interval == 0
        ):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """重新采样命令，包括跳跃和腿长调节"""
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        # 清除之前的跳跃/腿长命令
        self.commands[env_ids, 4:7] = 0.0

        # 采样跳跃高度命令
        l, r = self.command_ranges["jump_height"]
        jump_heights = torch.rand(len(env_ids), 1, device=self.device) * (r - l) + l
        mask = (
            torch.rand(len(env_ids), 1, device=self.device)
            < self.cfg.commands.threshold
        )
        jump_heights[mask] = 0.0
        self.commands[env_ids, 4] = jump_heights.squeeze(1)
        self.jump = self.commands[:, 4] != 0

        # 采样腿长命令
        l, r = self.command_ranges["knee_angle"]
        knee_angles = torch.rand(len(env_ids), 2, device=self.device) * (r - l) + l
        # 跳跃和腿长调节不能同时执行
        knee_angles[self.jump[env_ids], :] = 0.0
        mask = (
            torch.rand(len(env_ids), 2, device=self.device)
            < self.cfg.commands.threshold
        )
        knee_angles[mask] = 0.0
        self.commands[env_ids, 5:7] = knee_angles
        self.adjust_leg = torch.all(self.commands[:, 5:7] != 0, dim=1)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        # 过小命令置零
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

    def compute_observations(self):
        """计算观测值 (33维)
        轮子 dof_pos 是累积值（continuous joint），替换为 sin/cos 编码避免无界。
        """
        # 处理轮子 dof_pos：用 sin(pos) 替代原始值
        dof_pos_obs = (self.dof_pos - self.default_dof_pos).clone()
        # 轮子位置用 sin 编码（有界 [-1,1]，且保留方向信息）
        dof_pos_obs[:, self.wheel_indices] = torch.sin(self.dof_pos[:, self.wheel_indices])
        
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,           # 3
                self.base_ang_vel * self.obs_scales.ang_vel,           # 3
                self.projected_gravity,                                 # 3
                self.commands[:, :3] * self.commands_scale,            # 3
                (self.commands[:, 4] * self.obs_scales.height_measurements).unsqueeze(-1),  # 1 跳跃高度
                (self.commands[:, 5:7] * self.obs_scales.dof_pos),     # 2 腿长命令
                dof_pos_obs * self.obs_scales.dof_pos,                 # 6
                self.dof_vel * self.obs_scales.dof_vel,                # 6
                self.actions,                                           # 6
            ),
            dim=-1,
        )  # 总维度: 3+3+3+3+1+2+6+6+6 = 33

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
        """设置观测噪声向量"""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        x = 3  # 额外命令数（jump_height + knee_angle×2）
        noise_vec[9:12 + x] = 0.0  # commands 不加噪声
        noise_vec[12 + x:18 + x] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[18 + x:24 + x] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[24 + x:30 + x] = 0.0  # previous actions

        if self.cfg.terrain.measure_heights:
            noise_vec[30 + x:] = (
                noise_scales.height_measurements
                * noise_level
                * self.obs_scales.height_measurements
            )
        return noise_vec
