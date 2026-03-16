# ==============================================================================
# Description: Bubble Phase 3 — 跳跃 + 调腿技能训练配置
# 基于 Phase 2 的地形训练，新增:
#   - 6 维命令: lin_vel_x, lin_vel_y, ang_vel_yaw, heading, jump_height, leg_length
#   - 跳跃奖励 encourage_jump
#   - 调腿奖励 z_adjust_leg
#   - fall_recovery 逻辑
# Bubble 左右腿对称，只用 1 个 leg_length 命令（不像 Diablo 分左右膝）
# ==============================================================================

from legged_gym.envs.bubble.bubble_config import BubbleFlatCfg, BubbleFlatCfgPPO


class BubbleSkillCfg(BubbleFlatCfg):
    """Phase 3: 在 Phase 2 基础上增加跳跃 + 调腿技能"""

    class env(BubbleFlatCfg.env):
        num_envs = 2048
        num_observations = 152    # 150(Phase2) + jump_height(1) + leg_length(1) = 152
        num_actions = 6
        num_privileged_obs = None
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        enable_joint_state_history = True
        joint_state_history_length = 10

    class terrain(BubbleFlatCfg.terrain):
        # Phase 3 + 地形: trimesh + curriculum
        mesh_type = "plane"            # ← 平地训练
        measure_heights = False
        curriculum = False
        max_init_terrain_level = 3
        add_perlin_noise = False
        perlin_zScale = 0.03

    class init_state(BubbleFlatCfg.init_state):
        pass  # 继承 Phase 2 的弯腿初始姿态

    class control(BubbleFlatCfg.control):
        # Phase 3: 小腿电机力矩上限从 0.5Nm → 4Nm (Kp×action_scale=16×0.25=4Nm)
        stiffness = {
            "thigh": 6.0,            # ← 6×0.25=1.5Nm, 足够姿态调整, 不至于猛推劈叉
            "knee": 16.0,            # ← 4Nm: 16.0×0.25=4.0Nm, 跳跃需要更大力矩
            "wheel": 1.2,            # ← 保持不变
        }
        damping = {
            "thigh": 0.3,            # ← Kd/Kp=0.083, 接近knee比例(0.094), 适度抑振
            "knee": 2.0,             # ← 折中阻尼: Kd/Kp=0.094, 平衡抑振与探索
            "wheel": 0.0,            # ← 保持不变
        }
        decimation = 4                # ← 50Hz策略频率, 每个动作保持4个物理步, 天然低通滤波抗抖

    class asset(BubbleFlatCfg.asset):
        # Phase 3: 跳跃时 shank/idler 可能碰地, 不能终止
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/bubble/urdf/bubble.urdf"
        name = "bubble"
        foot_name = "drive_wheel"
        penalize_contacts_on = ["base", "thigh", "shank", "idler"]
        terminate_after_contacts_on = ["base"]  # 只有机体碰地终止
        flip_visual_attachments = False
        self_collisions = 0
        replace_cylinder_with_capsule = False

    class domain_rand(BubbleFlatCfg.domain_rand):
        # 继承基类域随机化 (7s/0.5m/s)，技能训练阶段微调:
        push_interval_s = 5            # ← 更频繁! 跳跃落地后需要快速恢复平衡
        max_push_vel_xy = 0.5          # ← 与基类一致, 跳跃时Kp=16有更强恢复力
        delay_ms_range = [0, 15]       # ← 技能动作对延迟更敏感，范围稍大

    class rewards(BubbleFlatCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.0
        only_positive_rewards = True
        tracking_sigma = 0.15
        base_height_target = 0.13     # ← 实际弯腿站立高度 0.14m

        class scales(BubbleFlatCfg.rewards.scales):
            # === 正向奖励 ===
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            no_fly = 1.0               # 跳跃时会被翻转为 -1 (惩罚贴地)
            # === 惩罚项 ===
            termination = -0.8
            lin_vel_z = -1.0           # 跳跃时翻转为奖励向上速度
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = -20.0        # ← 惩罚型! 参考TRON1A wheelfoot(-20), 偏离目标高度越远惩罚越大
            feet_air_time = 0.0
            collision = -30.0          # ← 降低! 跳跃落地碰撞多，不能太狠
            action_rate = -0.1
            feet_stumble = 0.0
            stumble = -0.0
            stand_still = 0.0
            feet_contact_forces = 0.0
            dof_pos_limits = -1.0
            dof_vel_limits = 0.0
            torque_limits = 0.0
            no_moonwalk = -5.0
            wheel_vel = -0.0001
            # === Phase 3 新增 ===
            feet_air_time = 0.0        # ← 跳跃时翻转: 奖励空中滞留 (Diablo=1.0)
            encourage_jump = 2.0       # ← 跳跃奖励 (Diablo=1.5, Bubble保守)
            z_adjust_leg = -10.0       # ← 调腿惩罚: 仅 adjust_leg 模式下跟踪膝盖目标
            roll = -15.0               # ← 直接惩罚 roll 倾斜 (projected_gravity y²)

    class commands(BubbleFlatCfg.commands):
        curriculum = True              # ← 开启命令课程
        max_curriculum = 2.0
        num_commands = 6               # ← 6维: vx, vy, yaw, heading, jump_height, leg_length
        resampling_time = 20           # ← 延长采样周期 (跳跃/调腿需要更长执行时间)
        heading_command = True
        threshold = 0.1                # ← 10% 概率触发技能 (Diablo=0.5, Bubble 多练走)

        class ranges(BubbleFlatCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-3.14, 3.14]
            jump_height = [0.02, 0.08]   # ← Bubble 只有 0.17m 高, 2Nm 力矩, 跳 2~8cm
            leg_length = [-0.4, 0.4]     # ← 腿部弯曲角度偏移量 (对称, 单个值控制双腿)

    class normalization(BubbleFlatCfg.normalization):
        pass  # 继承

    class noise(BubbleFlatCfg.noise):
        pass  # 继承

    class sim(BubbleFlatCfg.sim):
        pass  # 继承


class BubbleSkillCfgPPO(BubbleFlatCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy(BubbleFlatCfgPPO.policy):
        pass  # 继承 [512, 256, 128] ELU

    class algorithm(BubbleFlatCfgPPO.algorithm):
        pass  # 继承

    class runner(BubbleFlatCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 4000          # ← Phase 3: 技能学习需要更多迭代

        save_interval = 50
        experiment_name = "skill_bubble"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
