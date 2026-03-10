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
        num_envs = 2048           # ← Phase2: 地形训练需要更多并行环境 (2048→4096)
        num_observations = 150    # 30 + 10*6(dof_pos_err) + 10*6(dof_vel) = 150
        num_actions = 6           # left/right: thigh, knee, wheel
        num_privileged_obs = None
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20
        enable_joint_state_history = True   # ← 关节状态历史，参考 Diablo
        joint_state_history_length = 10     # ← 前 10 帧

    class terrain(LeggedRobotCfg.terrain):
        # ===================== 地形训练 =====================
        mesh_type = "trimesh"          # ← 开启地形训练!
        measure_heights = False        # ← 纯本体感知，不用地形高度观测
        curriculum = True              # ← 课程学习：从易到难
        max_init_terrain_level = 3     # ← 起始最高难度级别（保守）
        selected = False
        terrain_kwargs = None
        # 地形类型分布: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.4, 0.4, 0.0, 0.0, 0.2]  # ← 仅斜坡+少量离散障碍，无台阶
        # ===== Bubble 专属地形缩放（轮径仅 0.063m，力矩仅 2Nm）=====
        slope_scale = 0.15             # ← 最大坡度 ~8.5° (默认0.4→22°太陡)
        step_height_base = 0.01        # ← 最小台阶 1cm (默认5cm)
        step_height_scale = 0.05       # ← 最大台阶 ~6cm (默认18cm)
        obstacle_height_base = 0.01    # ← 最小障碍 1cm
        obstacle_height_scale = 0.04   # ← 最大障碍 ~5cm (< 轮径)
        rough_noise_range = 0.02       # ← 粗糙噪声 ±2cm (默认±5cm)
        wave_amplitude = 0.03          # ← 波浪幅度 3cm (默认10cm)
        wave_amplitude_scale = 0.05    # ← 渐进波浪振幅缩放
        add_perlin_noise = True        # ← 重新开启 perlin 噪声
        perlin_zScale = 0.03           # ← Bubble专属! 默认0.2→0.03, 幅度≈±3cm (站立高度14cm)
        track_test = False             # ← 不使用跑道测试模式

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [0.5, -0.3, 0.4]
        lookat = [0.0, 0.0, 0.1]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.18]   # 接近弯腿站立高度
        default_joint_angles = {
            # 弯腿姿态: thigh向前弯(+), knee向后弯(-) → 降低重心 + 增加腿部可调范围
            # 左腿: thigh正=向前, knee负=向后弯
            "left_thigh_joint": 0.2,
            "left_knee_joint": -0.2,
            "left_wheel_joint": 0.0,
            # 右腿: 因axis反向, thigh负=向前, knee正=向后弯
            "right_thigh_joint": -0.2,
            "right_knee_joint": 0.2,
            "right_wheel_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # ===================== 轮子驱动模式选择 =====================
        # "bubble"  — 油门踏板模式：dof_err=0, 低Kd, action≈力矩方向
        # "diablo"  — 位置追踪模式：保留dof_err, 高Kp, 角度差产生持续驱动力
        # "b2w"     — 恒速驱动模式：dof_err=0, dof_vel覆盖为常数, D项恒定推力
        wheel_drive_mode = "b2w"      # ← 切换这里！
        wheel_speed = 2.0             # ← 降低! 基线推力=0.05×1.0=0.05Nm, 可调范围[-0.20,+0.30]更对称

        control_type = 'P'
        stiffness = {
            "thigh": 2.0,            # ← Kp×action_scale=2.0×0.25=0.50 刚好极限
            "knee": 2.0,             # ← 同上
            "wheel": 1.2,            # ← 轮子Kp: 1.2×0.25=±0.3Nm 对称范围
        }  # [N*m/rad]
        damping = {
            "thigh": 0.08,            # ← 微调: 0.05→0.08, 总阻尼=URDF(0.02)+Kd(0.08)=0.10
            "knee": 0.08,             # ← 同上
            "wheel": 0.0,             # ← Kd=0: 去掉D项偏置, 轮子力矩完全对称 ±0.3Nm
        }  # [N*m*s/rad]
        action_scale = 0.25       # ← Kp×0.25=0.5Nm极限, 实际action<1所以有余量
        decimation = 2            # ← 100 Hz 策略频率 

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/bubble/urdf/bubble.urdf"
        name = "bubble"
        foot_name = "drive_wheel"
        penalize_contacts_on = ["base", "thigh", "shank", "idler"]  # idler 改为惩罚而非终止
        terminate_after_contacts_on = ["base"]  # 只有机体碰地终止, shank暂时允许(0.5Nm太弱)
        flip_visual_attachments = False
        self_collisions = 0
        replace_cylinder_with_capsule = False  # 保持轮子碰撞体为圆柱

    class domain_rand:
        randomize_friction = True
        friction_range = [0.3, 1.5]    # ← Phase2: 扩大摩擦范围，适应不同地面 (0.5~1.25→0.3~1.5)
        randomize_base_mass = True     # ← 开启! 地形训练阶段增强鲁棒性
        added_mass_range = [-0.2, 0.3] # ← Bubble 仅 2.17kg，不能加太多
        push_robots = True             # ← 开启! 随机外力推扰, sim2real 必需
        push_interval_s = 15           # ← 每15秒推一次 (与 Diablo 一致)
        max_push_vel_xy = 0.2          # ← 保持低推力, 0.5Nm电机恢复能力有限

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.0
        only_positive_rewards = True   # ← 关键! 截断负奖励→0, 让"活着"永远比"死"好
        tracking_sigma = 0.15          # ← 放宽sigma, 0.5Nm电机跟踪精度有限
        base_height_target = 0.14     # ← 弯腿后目标高度降低

        class scales(LeggedRobotCfg.rewards.scales):
            # === 正向奖励 ===
            tracking_lin_vel = 2.0     # ← 主奖励
            tracking_ang_vel = 1.0     # ← 开启! 让策略跟踪yaw命令，不再自由转圈
            no_fly = 1.0               # ← 轮子贴地
            # === 惩罚项 (only_positive_rewards=True 会截断, 但仍影响梯度方向) ===
            termination = -0.8
            lin_vel_z = -1.0
            ang_vel_xy = -0.05         # ← 恢复! 压制pitch/roll振荡
            orientation = -1.0         # ← 大幅降低! 5.0→1.0, 0.5Nm电机维持姿态能力有限
            torques = -0.00001
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0          # ← 开启! 奖励保持站立高度(指数型)
            feet_air_time = 0.0
            collision = -100.0           # ← 大幅降低! 50→5, 之前是第二大杀手(-0.025/step)
            action_rate = -0.05        # ← 降低! 0.3→0.05
            feet_stumble = 0.0
            stumble = -0.5
            stand_still = 0.0
            feet_contact_forces = 0.0
            dof_pos_limits = -1.0
            dof_vel_limits = 0.0
            torque_limits = 0.0
            no_moonwalk = -8.0         # ← 增大! -2→-8, 之前惩罚(-0.015)远小于正奖励(~0.22)
            wheel_vel = -0.0005

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 2.0
        num_commands = 4              # ← 回归! lin_vel_x, lin_vel_y, ang_vel_yaw, heading
        resampling_time = 10
        heading_command = True        # ← 开启！从heading误差计算ang_vel_yaw命令

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]   # ← 0.5Nm电机: 降低速度期望
            lin_vel_y = [0.0, 0.0]    # 双轮足无侧向
            ang_vel_yaw = [-0.5, 0.5]  # ← 0.5Nm电机: 降低转弯速度
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
        max_iterations = 2000         # ← Phase2: 地形训练需要更多迭代 (1200→2000)

        save_interval = 50
        experiment_name = "flat_bubble"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
