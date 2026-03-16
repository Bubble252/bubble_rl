# ==============================================================================
# Description: Bubble Phase 3 — 跳跃 + 调腿技能环境
# 继承自 Bubble (Phase 2)，参考 Diablo flat 的技能实现:
#   - _resample_commands: 互斥的 walk / jump / adjust_leg 模式
#   - _reward_encourage_jump: 跳跃奖励 + rew[~jump] *= -1 翻转
#   - _reward_z_adjust_leg: 调腿惩罚
#   - _reward_no_fly: 跳跃时翻转 (奖励离地)
#   - _reward_lin_vel_z: 跳跃时翻转 (奖励向上速度)
#   - check_termination: fall_recovery 逻辑
#   - compute_observations: 扩展到包含新命令维度
#
# Bubble 与 Diablo 的关键差异:
#   - Bubble 左右腿对称, 用 1 个 leg_length 命令 (不分左右膝)
#   - Bubble 站立高度 0.14m, 力矩 0.5Nm, 跳跃范围 2~8cm
#   - 命令维度: 6 (vx, vy, yaw, heading, jump_height, leg_length)
# ==============================================================================

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from legged_gym.utils.math import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs.bubble.bubble import Bubble
from .bubble_skill_config import BubbleSkillCfg


class BubbleSkill(Bubble):
    """Bubble Phase 3: 跳跃 + 调腿技能"""

    # ======================== 质心高度计算 ========================

    def _get_com_world_pos(self):
        """计算 base_link 质心的世界坐标 (3D)
        
        com_world = base_pos + R(quat) @ com_offset_local
        其中 R(quat) 将局部偏移旋转到世界坐标系。
        """
        # base_quat: [num_envs, 4] (x, y, z, w)
        # com_offset_local: [3] → expand to [num_envs, 3]
        com_offset_expanded = self.com_offset_local.unsqueeze(0).expand(self.num_envs, -1)
        # quat_rotate: 将局部向量旋转到世界坐标系
        com_offset_world = quat_rotate(self.base_quat, com_offset_expanded)
        # base_pos (世界) + 旋转后的偏移
        com_world = self.root_states[:, :3] + com_offset_world
        return com_world

    def _get_com_height(self):
        """计算 base_link 质心的世界 z 坐标（高度）"""
        return self._get_com_world_pos()[:, 2]

    # ======================== 重写 base_height 奖励 ========================

    def _reward_base_height(self):
        """重载: 基于质心高度的惩罚型奖励 (参考 TRON1A wheelfoot)
        
        return abs(com_height - target)
        scale 为负值 → 偏离目标越远惩罚越大。
        
        与 exp(-200*error²) 正向奖励的区别:
        - 正向奖励: 弯腿蹲低如果更接近target反而拿更多奖励 → 鼓励弯腿!
        - 惩罚型: 偏离就罚, 没有"拿不到奖励"的下限 → 强约束!
        
        TRON1A wheelfoot: scale=-20, solefoot: scale=-10
        """
        com_height = self._get_com_height()
        return torch.abs(com_height - self.cfg.rewards.base_height_target)

    # ======================== 奖励函数 ========================

    def _reward_feet_air_time(self):
        """奖励空中时间 — 跳跃时翻转: 奖励离地滞空
        
        参考 Diablo: feet_air_time 在非跳跃时惩罚离地过久,
        跳跃时翻转为奖励离地 (配合 encourage_jump 提供跳跃激励)
        """
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
        )  # 零命令时无奖励
        self.feet_air_time *= ~contact_filt
        rew_airTime[~self.jump] *= -1  # 非跳跃: 惩罚离地; 跳跃: 奖励离地
        return rew_airTime

    def _reward_no_fly(self):
        """奖励双轮贴地 — 跳跃时翻转为惩罚贴地 (鼓励离地)"""
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        rew = 1.0 * (torch.sum(1.0 * contacts, dim=1) == 2)
        rew[self.jump] *= -1  # 跳跃时: 贴地 → 惩罚
        return rew

    def _reward_lin_vel_z(self):
        """惩罚 z 轴速度 — 跳跃时翻转为奖励向上速度"""
        # 使用全局速度 (root_states[:, 9])，防止机器人通过改变姿态骗奖励
        rew = torch.square(self.root_states[:, 9])
        rew[self.jump] *= -1  # 跳跃时: z 速度大 → 奖励
        return (~self.adjust_leg) * rew

    def _reward_encourage_jump(self):
        """跳跃奖励: 空中停留时间 + 向上速度
        
        参考 Diablo 实现, 适配 Bubble:
        - 累积空中时间 × 高度 (clip 到目标高度)
        - 额外奖励向上速度 (按目标高度归一化)
        - 非跳跃模式: 翻转为惩罚 (抑制无命令跳跃)
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        fly = torch.sum(1.0 * contact, dim=1) == 0  # 处于空中
        first_contact = (self.base_air_time > 0.0) * ~fly  # 刚着地

        # 累积: dt × clip(com_height, 0, target_jump_height)
        self.base_air_time += self.dt * torch.clip(
            self._get_com_height(),
            torch.tensor(0.0, device=self.device),
            self.commands[:, 4],
        )

        rew_airTime = (self.base_air_time - 5e-5) * first_contact

        # 额外奖励向上速度 (按跳跃高度范围归一化)
        l, r = self.command_ranges["jump_height"]
        rew_airTime += (
            torch.maximum(torch.tensor(0.0, device=self.device), self.root_states[:, 9])
            * (self.commands[:, 4] - l) / (r - l)
        )

        self.base_air_time *= ~fly  # 落地后清零

        rew_airTime[~self.jump] *= -1  # 非跳跃模式: 翻转 → 惩罚无命令跳跃
        return rew_airTime

    def _reward_z_adjust_leg(self):
        """调腿惩罚: 仅在 adjust_leg 模式下惩罚膝盖偏离目标角度
        
        参考 Diablo 实现:
        - 只在 adjust_leg=True 时生效 (非调腿时不管腿部姿态)
        - 只惩罚 knee, 不约束 thigh
        - commands[:, 5] = 目标腿长偏移 (对称应用到左右膝盖)
        
        roll 约束由独立的 _reward_roll 负责, 不在这里处理。
        """
        target_knee_left = self.default_dof_pos[:, self.knee_left_idx] + self.commands[:, 5]
        target_knee_right = self.default_dof_pos[:, self.knee_right_idx] - self.commands[:, 5]
        rew = torch.square(target_knee_left - self.dof_pos[:, self.knee_left_idx])
        rew += torch.square(target_knee_right - self.dof_pos[:, self.knee_right_idx])
        return self.adjust_leg.float() * rew

    def _reward_roll(self):
        """直接惩罚 roll 倾斜
        
        projected_gravity[:, 1] = 重力在 base y 轴的投影 = roll 分量
        机体水平时 = 0, 侧倾时 ≠ 0
        
        优点:
        - 直接惩罚目标问题 (roll), 不限制任何关节自由度
        - 不管 roll 来源 (thigh/knee 不对称都会被惩罚)
        - 不影响 pitch (加减速前后倾正常)
        """
        return torch.square(self.projected_gravity[:, 1])

    # ======================== 命令采样 ========================

    def _resample_commands(self, env_ids):
        """重写: 互斥采样 walk / jump / adjust_leg 命令
        
        流程:
        1. 采样基本运动命令 (vx, vy, heading)
        2. 清零技能命令
        3. 以 threshold 概率采样跳跃高度
        4. 以 threshold 概率采样腿长 (与跳跃互斥)
        5. 小速度命令清零
        """
        # 基本运动命令
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

        # 清零技能命令
        self.commands[env_ids, 4:6] = 0.0

        # --- 采样跳跃高度 ---
        l, r = self.command_ranges["jump_height"]
        jump_heights = torch.rand(len(env_ids), 1, device=self.device) * (r - l) + l
        # threshold 概率不跳
        mask_no_jump = torch.rand(len(env_ids), 1, device=self.device) < self.cfg.commands.threshold
        jump_heights[mask_no_jump] = 0.0
        self.commands[env_ids, 4] = jump_heights.squeeze(1)
        # 更新全局 jump 标志
        self.jump = self.commands[:, 4] != 0

        # --- 采样腿长 (与跳跃互斥) ---
        l_leg, r_leg = self.command_ranges["leg_length"]
        leg_lengths = torch.rand(len(env_ids), 1, device=self.device) * (r_leg - l_leg) + l_leg
        # 跳跃中的 env 不调腿
        leg_lengths[self.jump[env_ids].unsqueeze(1)] = 0.0
        # threshold 概率不调腿
        mask_no_leg = torch.rand(len(env_ids), 1, device=self.device) < self.cfg.commands.threshold
        leg_lengths[mask_no_leg] = 0.0
        self.commands[env_ids, 5] = leg_lengths.squeeze(1)
        # 更新全局 adjust_leg 标志
        self.adjust_leg = self.commands[:, 5] != 0

        # heading 命令
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1), device=self.device,
            ).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
                (len(env_ids), 1), device=self.device,
            ).squeeze(1)

        # 小速度清零
        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
        ).unsqueeze(1)

    # ======================== 终止检查 ========================

    def check_termination(self):
        """重写: 加入 fall_recovery 逻辑
        
        当 tracking_lin_vel 奖励 > 80% 最大值时, 启用 fall_recovery,
        此时接触终止条件被跳过 (允许跌倒恢复)。
        Bubble 更脆弱, 用 80% 阈值 (Diablo 用 85%)。
        """
        reset_condition = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            ) > 1.0,
            dim=1,
        )
        self.reset_buf = torch.logical_and(
            reset_condition, torch.logical_not(self.fall_recovery)
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    # ======================== 观测 ========================

    def compute_observations(self):
        """重写: 在 Phase 2 基础上增加 jump_height(1) + leg_length(1) 命令观测
        
        布局:
          lin_vel(3) + ang_vel(3) + gravity(3) + commands_walk(3)
          + jump_height(1) + leg_length(1)                         ← 新增
          + dof_pos(6) + dof_vel(6) + actions(6) = 32
          + history(120) = 152
        """
        mode = self.cfg.control.wheel_drive_mode

        # 位置误差
        dof_pos_obs = (self.dof_pos - self.default_dof_pos).clone()
        if mode != "diablo":
            dof_pos_obs[:, self.wheel_indices] = 0

        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,            # 3
                self.base_ang_vel * self.obs_scales.ang_vel,            # 3
                self.projected_gravity,                                  # 3
                self.commands[:, :3] * self.commands_scale,             # 3
                (self.commands[:, 4] * self.obs_scales.height_measurements).unsqueeze(-1),  # 1: jump_height
                (self.commands[:, 5] * self.obs_scales.dof_pos).unsqueeze(-1),              # 1: leg_length
                dof_pos_obs * self.obs_scales.dof_pos,                  # 6
                self.dof_vel * self.obs_scales.dof_vel,                 # 6
                self.actions,                                            # 6
            ),
            dim=-1,
        )  # 总维度: 32

        # 关节状态历史
        if getattr(self.cfg.env, 'enable_joint_state_history', False):
            self.obs_buf = torch.cat(
                (
                    self.obs_buf,
                    self.dof_pos_error_history.view(self.num_envs, -1) * self.obs_scales.dof_pos,
                    self.dof_vel_history.view(self.num_envs, -1) * self.obs_scales.dof_vel,
                ),
                dim=-1,
            )  # 总维度: 152

        # 地形高度观测 (使用质心高度)
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self._get_com_height().unsqueeze(1) - 0.5 - self.measured_heights,
                    -1, 1.0,
                ) * self.obs_scales.height_measurements
            )
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # 噪声
        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """噪声向量, 与 152 维观测对齐"""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0    # walk commands
        noise_vec[12:14] = 0.0   # jump_height + leg_length (命令无噪声)
        noise_vec[14:20] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[20:26] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[26:32] = 0.0   # actions

        # 关节状态历史噪声
        if getattr(self.cfg.env, 'enable_joint_state_history', False):
            hist_len = self.cfg.env.joint_state_history_length
            n_dof = self.num_actions  # 6
            noise_vec[32:32 + hist_len * n_dof] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[32 + hist_len * n_dof:32 + 2 * hist_len * n_dof] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel

        if self.cfg.terrain.measure_heights:
            base_idx = 32
            if getattr(self.cfg.env, 'enable_joint_state_history', False):
                base_idx = 152
            noise_vec[base_idx:] = (
                noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
            )
        return noise_vec

    # ======================== 初始化 ========================

    def _init_buffers(self):
        super()._init_buffers()
        # Phase 3 专用 buffer
        self.base_air_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.jump = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.adjust_leg = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.fall_recovery = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        # 预计算膝关节索引
        joints = list(self.cfg.init_state.default_joint_angles.keys())
        self.knee_left_idx = joints.index("left_knee_joint")
        self.knee_right_idx = joints.index("right_knee_joint")
        self.thigh_left_idx = joints.index("left_thigh_joint")
        self.thigh_right_idx = joints.index("right_thigh_joint")

        # ★ Phase 3: 腿部电机力矩上限从 URDF 的 0.5Nm → 4Nm (跳跃/站高需要更大力矩)
        self.torque_limits[self.knee_left_idx] = 4.0
        self.torque_limits[self.knee_right_idx] = 4.0
        self.torque_limits[self.thigh_left_idx] = 4.0
        self.torque_limits[self.thigh_right_idx] = 4.0
        print(f"[BubbleSkill] ★ Knee + Thigh torque limits overridden to 4.0 N·m")

        # ★ base_link 质心在局部坐标系中的偏移 (来自 URDF <inertial><origin>)
        # xyz="0.005 -0.050238963083457974 -0.03448810722272746"
        self.com_offset_local = torch.tensor(
            [0.005, -0.050239, -0.034488], dtype=torch.float, device=self.device
        )
        print(f"[BubbleSkill] ★ COM offset (local): {self.com_offset_local.tolist()}")
        print(f"[BubbleSkill] ★ base_height_target 现在参考质心高度")

        print(f"[BubbleSkill] Phase 3 buffers initialized: jump, adjust_leg, fall_recovery, base_air_time")
        print(f"[BubbleSkill] Knee indices: left={self.knee_left_idx}, right={self.knee_right_idx}")
        print(f"[BubbleSkill] Torque limits: {self.torque_limits}")

    def reset_idx(self, env_ids):
        """重写: reset 时清零 Phase 3 buffer"""
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            self.base_air_time[env_ids] = 0.0
            self.jump[env_ids] = False
            self.adjust_leg[env_ids] = False

    def post_physics_step(self):
        """重写: 将 diag_base_height 改为使用质心高度"""
        super().post_physics_step()
        # 修正: 父类用 root_states[:,2] (base_link 原点), 这里改为质心高度
        if hasattr(self, '_diag_sums') and hasattr(self, 'com_offset_local'):
            with torch.no_grad():
                # 撤销父类的累加 (root_states[:,2]), 换成质心高度
                self._diag_sums["base_height"] -= self.root_states[:, 2]
                self._diag_sums["base_height"] += self._get_com_height()

    def update_command_curriculum(self, env_ids):
        """命令课程: 当 tracking 奖励足够高时启用 fall_recovery
        
        Bubble 用 80% 阈值 (比 Diablo 的 85% 更早启用, 因为 Bubble 更脆弱)
        """
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids])
            / self.max_episode_length
            > 0.80 * self.reward_scales["tracking_lin_vel"]
        ):
            self.fall_recovery[env_ids] = True
        else:
            self.fall_recovery[env_ids] = False
