# ==============================================================================
# Description: Bubble 点足模式训练配置
# 方案 A: Config 软锁定（零 URDF 改动）
#   - 轮子 Kp=1000, Kd=50, 锁死在 0 度
#   - num_actions=4, 网络只控制 4 个腿部关节
#   - 轮子力矩上限提升到 10 N·m (抵抗外力)
#   - 步态生成器 + 步态感知奖励（移植自 TRON1A solefoot）
#   - only_positive_rewards = False (需要负反馈学步态)
# ==============================================================================

from legged_gym.envs.bubble.bubble_config import BubbleFlatCfg, BubbleFlatCfgPPO


class BubblePointfootCfg(BubbleFlatCfg):
    """Bubble 点足模式配置 — 轮子软锁定 + 步态行走"""

    class env(BubbleFlatCfg.env):
        num_envs = 4096
        # 观测: ang_vel(3) + gravity(3) + dof_pos_leg(4) + dof_vel_leg(4) +
        #        actions(6) + clock_sin(1) + clock_cos(1) + gait_params(4) +
        #        commands(3) = 29
        # 注意: actions 仍为 6D (与 num_dof 对齐), 但 wheel 部分恒为 0
        num_observations = 29
        num_actions = 6             # ← 保持 6 与 num_dof 对齐, wheel action 在环境中强制归零
        num_privileged_obs = None
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 10       # ← 初期容易摔, 缩短 episode
        enable_joint_state_history = False  # ← 用 obs_history 替代
        obs_history_length = 5      # ← 历史帧数

    class terrain(BubbleFlatCfg.terrain):
        mesh_type = "plane"         # ← 先在平地训练
        measure_heights = False
        curriculum = False
        static_friction = 1.0       # ← 提高! 点足接触面积小, 需要大摩擦
        dynamic_friction = 1.0
        restitution = 0.0           # ← 无弹性, 减少弹跳

    class viewer(BubbleFlatCfg.viewer):
        ref_env = 0
        pos = [0.8, -0.5, 0.5]
        lookat = [0.0, 0.0, 0.15]

    class init_state(BubbleFlatCfg.init_state):
        pos = [0.0, 0.0, 0.18]     # ← 接近站立高度
        default_joint_angles = {
            # 弯腿站立姿态
            "left_thigh_joint": 0.3,
            "left_knee_joint": -0.6,
            "left_wheel_joint": 0.0,    # 锁定在 0
            "right_thigh_joint": -0.3,
            "right_knee_joint": 0.6,
            "right_wheel_joint": 0.0,   # 锁定在 0
        }

    class gait:
        """步态生成器参数 — 移植自 TRON1A solefoot"""
        num_gait_params = 4     # frequencies, offsets, durations, swing_height
        resampling_time = 5     # [s]
        touch_down_vel = 0.0

        class ranges:
            frequencies = [1.0, 1.5]     # 步频 1~1.5 Hz
            offsets = [0.5, 0.5]         # 左右腿相位差 180° (标准行走)
            durations = [0.5, 0.5]       # 50% 占空比
            swing_height = [0.02, 0.05]  # ← 远小于 TRON1A (0.10~0.20), Bubble 仅 14cm 高

    class control(BubbleFlatCfg.control):
        wheel_drive_mode = "pointfoot"  # ← 新模式: 轮子锁定

        control_type = 'P'
        stiffness = {
            "thigh": 6.0,       # ← Phase3 级别, 跳跃/行走需要更大力矩
            "knee": 16.0,       # ← 4Nm: 16.0×0.25=4.0Nm
            "wheel": 0.0,      # ★ 轮子不参与 PD, 由 step() 硬锁
        }
        damping = {
            "thigh": 0.5,
            "knee": 1.5,
            "wheel": 0.0,      # ★ 轮子不参与 PD, 由 step() 硬锁
        }
        action_scale = 0.25
        decimation = 4           # ← 50 Hz 策略频率

        wheel_torque_limit = 0.0   # ← 轮子不产生力矩, step() 直接锁 dof_state

    class asset(BubbleFlatCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/bubble/urdf/bubble.urdf"
        name = "bubble_pointfoot"
        foot_name = "drive_wheel"       # 足端 body 名 (轮子锁定后变成 "脚")
        penalize_contacts_on = ["base", "thigh", "shank", "idler"]
        terminate_after_contacts_on = ["base"]
        flip_visual_attachments = False
        self_collisions = 0
        replace_cylinder_with_capsule = False  # ← 暂不改碰撞体
        fix_base_link = False           # ← 正式训练: 自由体

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-0.2, 0.3]
        push_robots = False             # ← 初期不推, 先学会站/走
        push_interval_s = 15
        max_push_vel_xy = 0.1

    class commands(BubbleFlatCfg.commands):
        curriculum = False
        num_commands = 3                # ← vx, vy, ang_vel_yaw (暂不含 height/stand 模式)
        resampling_time = 5
        heading_command = False
        min_norm = 0.05
        zero_command_prob = 0.01         # ← 降低, 让机器人多练走路 (TRON1A-PF=0.0)

        class ranges(BubbleFlatCfg.commands.ranges):
            lin_vel_x = [-0.15, 0.2]   # ← 极小速度范围, 2 DOF 点足能力有限
            lin_vel_y = [0.0, 0.0]      # ← Bubble 无侧向移动能力
            ang_vel_yaw = [-0.2, 0.2]   # ← 极小转弯
            heading = [-3.14, 3.14]

    class rewards(BubbleFlatCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.8
        max_contact_force = 100.0
        only_positive_rewards = False   # ★ 关键! 需要负反馈学步态
        clip_reward = 100.0
        clip_single_reward = 5.0
        tracking_sigma = 0.3            # ← 放宽, 2 DOF 跟踪精度有限
        ang_tracking_sigma = 0.3
        base_height_target = 0.14       # ← Bubble 弯腿站立高度
        feet_height_target = 0.03       # ← 摆动高度目标 (很小)
        min_feet_distance = 0.06        # ← 双脚最小间距 (Bubble 轮距约 0.12m)

        # 步态奖励 sigma
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005
        height_tracking_sigma = 0.01
        about_landing_threshold = 0.02  # ← 着陆检测阈值

        class scales(BubbleFlatCfg.rewards.scales):
            # === 正向奖励 (对齐 TRON1A-PF) ===
            keep_balance = 1.0              # ← 降回 TRON1A 级别, 不鼓励摆烂
            tracking_lin_vel = 1.0          # ← 对齐 TRON1A-PF
            tracking_ang_vel = 0.5          # ← 对齐 TRON1A-PF

            # === 步态感知奖励 ★★ 核心改动 ===
            tracking_contacts_shaped_force = -2.0   # ★ 对齐 TRON1A! 强制学步态
            tracking_contacts_shaped_vel = -2.0     # ★ 对齐 TRON1A!
            tracking_contacts_shaped_height = -2.0  # ★ 对齐 TRON1A-SF!

            # === 身体姿态约束 (对齐 TRON1A-PF 轻约束) ===
            base_height = -2.0              # ← 对齐 TRON1A-PF (原 -10)
            orientation = -10.0             # ← 对齐 TRON1A-PF
            lin_vel_z = -0.5                # ← 对齐 TRON1A-PF (原 -2.0)
            ang_vel_xy = -0.05              # ★ 对齐 TRON1A-PF! 允许晃动探索

            # === 动作平滑 (对齐 TRON1A) ===
            torques = -0.00008              # ← 对齐 TRON1A
            dof_acc = -2.5e-7
            action_rate = -0.01             # ← 对齐 TRON1A
            action_smooth = -0.01

            # === 安全约束 ===
            collision = -1.0                # ★ 对齐 TRON1A-PF! (原 -10)
            dof_pos_limits = -2.0
            feet_distance = 0.0             # ← 禁用! Bubble 双脚距离由 URDF 刚性决定, 不会变
            foot_landing_vel = -0.15        # ← 对齐 TRON1A-PF (原 -5.0)
            feet_contact_forces = -0.001

            # === 正则化 ===
            feet_regulation = -0.05
            termination = -0.8

            # === 禁用轮式奖励 ===
            no_fly = 0.0                    # ★ 禁用! 与抬腿行走矛盾
            no_moonwalk = 0.0               # ★ 禁用! 基于轮子速度, 点足不适用
            wheel_vel = 0.0                 # ★ 禁用! 轮子已锁定
            stumble = 0.0
            stand_still = 0.0
            feet_air_time = 0.0
            feet_stumble = 0.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim(BubbleFlatCfg.sim):
        dt = 0.005
        substeps = 1


class BubblePointfootCfgPPO(BubbleFlatCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 5000       # ← 点足需要更长训练

        save_interval = 100
        experiment_name = "pointfoot_bubble"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
