# ==============================================================================
# Description: Bubble 双轮足机器人训练配置
# 基于 Diablo 配置修改，适配 bubble 机器人的关节结构
# Bubble DOF: left_thigh, left_knee, left_wheel, right_thigh, right_knee, right_wheel (6 DOF)
# idler_wheel 关节设为 fixed，不参与控制
# ==============================================================================

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BubbleFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 1024  # 1024个环境，适配单机训练资源
        num_observations = 33  # 3+3+3+3+1(jump)+2(knee_angle)+6+6+6 = 33
        num_actions = 6  # left/right: thigh, knee, wheel
        num_privileged_obs = None
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"  # 先用平地训练
        measure_heights = False
        selected = False
        terrain_kwargs = None

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [0.5, -0.3, 0.4]    # 相机距机器人很近，适配小型机器人(0.16m高)
        lookat = [0.0, 0.0, 0.1]   # 看向机器人脚部附近

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.20]  # x,y,z [m] 站立高度约0.16m，生成时略高一点让机器人自然落地
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "left_thigh_joint": 0.0,     # ↑ -0.2→0.0, 回到垂直，不前倾
            "left_knee_joint": 0.0,      # ↑ -0.15→0.0, 膝盖伸直=站立最高
            "left_wheel_joint": 0.0,
            "right_thigh_joint": 0.0,    # ↑ -0.2→0.0
            "right_knee_joint": 0.0,     # ↑ -0.15→0.0
            "right_wheel_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        # 腿部: PD 位置控制 (P mode)
        # 轮子: 直接力矩控制 — 网络输出直接乘以 scale 得到力矩
        stiffness = {
            "thigh": 5.0,
            "knee": 5.0,
            "wheel": 0.0,   # 直接力矩模式不需要 Kp
        }  # [N*m/rad]
        damping = {
            "thigh": 0.5,
            "knee": 0.5,
            "wheel": 0.0,  # 直接力矩模式不需要 Kd
        }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle (legs)
        action_scale = 0.5  # 腿部: 每步最大 0.5 rad 位置偏移
        wheel_action_scale = 2.0  # 轮子: 网络输出 ±1 → 力矩 ±2.0 N·m (直接力矩)
        wheel_torque_limit = 2.0  # 轮子力矩上限 [N·m]，URDF effort=2
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/bubble/urdf/bubble.urdf"
        name = "bubble"
        foot_name = "drive_wheel"  # 匹配 left_drive_wheel_link 和 right_drive_wheel_link，不匹配 idler_wheel
        penalize_contacts_on = ["base", "thigh", "shank", "idler"]
        terminate_after_contacts_on = ["base"]  # 只base碰地终止，shank用collision惩罚处理
        flip_visual_attachments = False
        self_collisions = 0  # 0 = enable
        replace_cylinder_with_capsule = False  # 保持轮子碰撞体为圆柱体，不替换为胶囊

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
        only_positive_rewards = False  # 关闭正奖励截断，让惩罚信号直接影响策略
        base_height_target = 0.17  # 腿伸直时更高: base→thigh(0.046)→shank(0.046)→wheel(r≈0.033) + base自身偏移

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -5.0       # 惩罚死亡
            tracking_lin_vel = 6.0   # 指数奖励 → 精确跟踪
            tracking_ang_vel = 1.5
            lin_vel_z = -2.0         # ↓ 关掉截断后降低惩罚力度
            ang_vel_xy = -0.5        # ↓ 关掉截断后降低
            orientation = -2.0       # ↓ 保持水平
            torques = -0.0001
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = -10.0      # ↓ 降低避免速死
            feet_air_time = 0.0
            collision = -10.0        # ↓ 降低，不再需要绕过截断
            action_rate = -0.05
            feet_stumble = 0.0
            stand_still = 0.0
            feet_contact_forces = 0.0
            dof_pos_limits = -0.5
            dof_vel_limits = 0.0
            torque_limits = 0.0
            no_fly = 2.0             # 轮子贴地奖励
            no_moonwalk = -0.5
            stand_symmetric = -2.0   # ↓ 降低
            encourage_jump = 0.0
            z_adjust_leg = 0.0
            wheel_vel_tracking = 3.0  # ↑ 增大轮速跟踪奖励

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 2.0
        num_commands = 7  # lin_vel_x, lin_vel_y, ang_vel_yaw, heading, jump_height, knee_angle(2)
        resampling_time = 10
        heading_command = True
        threshold = 1.0  # 1.0=关闭跳跃和腿长命令（全部置零），先学站稳走路

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]  # 先从小速度范围开始
            lin_vel_y = [0.0, 0.0]  # 双轮足无法侧向移动
            ang_vel_yaw = [-1.5, 1.5]
            heading = [-3.14, 3.14]
            jump_height = [0.05, 0.25]  # 跳跃高度
            knee_angle = [-0.4, 0.45]  # 膝关节弯曲角度（匹配URDF限位 -0.436~0.489）

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
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 0.1
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
        max_iterations = 3000

        save_interval = 50
        experiment_name = "flat_bubble"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
