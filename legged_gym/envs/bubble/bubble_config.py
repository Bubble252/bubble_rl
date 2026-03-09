# ==============================================================================
# Description: Bubble 双轮足机器人训练配置
# 回归 B2W/Diablo 已验证风格：
#   - 标准 P 控制 + 轮子 dof_err=0
#   - only_positive_rewards = True
#   - 4 命令 (无 jump/knee_angle)
#   - 精简奖励集
# ==============================================================================

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BubbleFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2048           # ↑ 从1024提升，参考 Diablo=2048
        num_observations = 150    # 30 + 10*6(dof_pos_err) + 10*6(dof_vel) = 150
        num_actions = 6           # left/right: thigh, knee, wheel
        num_privileged_obs = None
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        enable_joint_state_history = True   # ← 关节状态历史，参考 Diablo
        joint_state_history_length = 10     # ← 前 10 帧

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        selected = False
        terrain_kwargs = None

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [0.5, -0.3, 0.4]
        lookat = [0.0, 0.0, 0.1]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.20]   # 略高于站立高度让机器人自然落地
        default_joint_angles = {
            "left_thigh_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_thigh_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_wheel_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # ===================== 轮子驱动模式选择 =====================
        # "bubble"  — 油门踏板模式：dof_err=0, 低Kd, action≈力矩方向
        # "diablo"  — 位置追踪模式：保留dof_err, 高Kp, 角度差产生持续驱动力
        # "b2w"     — 恒速驱动模式：dof_err=0, dof_vel覆盖为常数, D项恒定推力
        wheel_drive_mode = "b2w"      # ← 切换这里！
        wheel_speed = 3.0             # ← b2w 模式：恒定推力 = Kd × wheel_speed = 0.3 N·m (降低推力减少空转)

        control_type = 'P'
        stiffness = {
            "thigh": 3.0,            # ← 降低刚度，减少过冲 (5→3)
            "knee": 3.0,             # ← 同上
            "wheel": 4.0,
        }  # [N*m/rad]
        damping = {
            "thigh": 1.0,            # ← 加倍阻尼，抑制振荡 (0.5→1.0)
            "knee": 1.0,             # ← 同上
            "wheel": 0.1,
        }  # [N*m*s/rad]
        action_scale = 0.15       # ← 限制单步最大调整幅度 (0.25→0.15)
        decimation = 2            # ← 100 Hz 策略频率 

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/bubble/urdf/bubble.urdf"
        name = "bubble"
        foot_name = "drive_wheel"
        penalize_contacts_on = ["base", "thigh", "shank", "idler"]  # idler 改为惩罚而非终止
        terminate_after_contacts_on = ["base", "shank"]  # base 或小腿触地 → 立即终止（防跪）
        flip_visual_attachments = False
        self_collisions = 0
        replace_cylinder_with_capsule = False  # 保持轮子碰撞体为圆柱

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.0
        only_positive_rewards = True   # ← 回归！B2W/Diablo 都是 True
        tracking_sigma = 0.15          # ← 缩小sigma，提高跟踪精度 (0.25→0.15)
        base_height_target = 0.15      # ← 顺应物理最优高度 (0.15→0.17)改回去了

        class scales(LeggedRobotCfg.rewards.scales):
            # === 正向奖励 ===
            tracking_lin_vel = 3.0     # ← 提高跟踪权重 (2.0→3.0)
            tracking_ang_vel = 0.5     # ← 参考 Diablo=0.5
            no_fly = 1.0               # ← 参考 Diablo=1.0，轮子贴地
            # === 惩罚项 ===
            termination = -0.8         # ← 参考 B2W=-0.8
            lin_vel_z = -1.0           # ← 参考 Diablo=-1.0
            ang_vel_xy = -0.5          # ← 抑制 pitch 角速度
            orientation = -20.0        # ← 逼迫站直
            torques = -0.00001         # ← 参考 Diablo
            dof_vel = 0.0              # ← 不惩罚，靠物理阻尼解决
            dof_acc = 0.0
            base_height = 0          # ← 正向奖励 exp(-40*err²)
            feet_air_time = 0.0
            collision = -50.0
            action_rate = -0.3         # ← 适当放松，物理阻尼已够 (0.5→0.3)
            feet_stumble = 0.0
            stand_still = 0.0
            feet_contact_forces = 0.0
            dof_pos_limits = -1.0      # ← 参考 Diablo=-1.0
            dof_vel_limits = 0.0
            torque_limits = 0.0
            no_moonwalk = -2.0         # ← 参考 Diablo=-2.0，防太空步
            wheel_vel = -0.0005        # ← 惩罚轮速过大

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 2.0
        num_commands = 4              # ← 回归! lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 10
        heading_command = False       # ← 关闭！否则会覆盖ang_vel_yaw=0，产生转向命令

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]   # 扩大速度范围
            lin_vel_y = [0.0, 0.0]    # 双轮足无侧向
            ang_vel_yaw = [0.0, 0.0]  # 先不转弯，专注直线
            heading = [-3.14, 3.14]

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

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class BubbleFlatCfgPPO(LeggedRobotCfgPPO):
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
        max_iterations = 1200

        save_interval = 50
        experiment_name = "flat_bubble"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
