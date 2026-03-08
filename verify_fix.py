"""
验证修改后的 URDF 和控制模式
1. 圆柱体碰撞体
2. 直接力矩控制
"""
from isaacgym import gymapi, gymutil, gymtorch
import torch
import numpy as np

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = 0.005
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.physx.bounce_threshold_velocity = 0.5
sim_params.physx.max_depenetration_velocity = 0.1
sim_params.use_gpu_pipeline = False

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 加载资产 — 注意 replace_cylinder_with_capsule=False
asset_options = gymapi.AssetOptions()
asset_options.collapse_fixed_joints = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
asset_options.replace_cylinder_with_capsule = False  # 保持圆柱体！

asset = gym.load_asset(sim, "resources/robots/bubble/urdf", "bubble.urdf", asset_options)

num_bodies = gym.get_asset_rigid_body_count(asset)
num_shapes = gym.get_asset_rigid_shape_count(asset)
num_dofs = gym.get_asset_dof_count(asset)
body_names = gym.get_asset_rigid_body_names(asset)
dof_names = gym.get_asset_dof_names(asset)

print(f"Bodies: {num_bodies}, Shapes: {num_shapes}, DOFs: {num_dofs}")
print(f"Body names: {body_names}")
print(f"DOF names: {dof_names}")

# 添加地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
gym.add_ground(sim, plane_params)

env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
start_pose = gymapi.Transform()
start_pose.p = gymapi.Vec3(0, 0, 0.20)
start_pose.r = gymapi.Quat(0, 0, 0, 1)
actor = gym.create_actor(env, asset, start_pose, "bubble", 0, 0)

# 所有 DOF effort 模式
dof_props = gym.get_actor_dof_properties(env, actor)
for i in range(num_dofs):
    dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
    dof_props['stiffness'][i] = 0.0
    dof_props['damping'][i] = 0.0
gym.set_actor_dof_properties(env, actor, dof_props)

gym.prepare_sim(sim)

# 稳定
print("Stabilizing...")
for _ in range(100):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

# 检查初始状态
dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
print(f"\nAfter stabilizing:")
print(f"  Base Z: {body_states['pose']['p'][0][2]:.4f}")
for i in range(num_dofs):
    print(f"  DOF {i} ({dof_names[i]}): pos={dof_states['pos'][i]:.6f}, vel={dof_states['vel'][i]:.4f}")

# 测试: 只给轮子施加 2 N·m 力矩，腿部施加 P 控制保持站立
TORQUE = 2.0  # wheel_action_scale = 2.0, action = 1.0
STEPS = 200
Kp_leg = 5.0
Kd_leg = 0.5

print(f"\nApplying {TORQUE} N·m to wheels, P control on legs for {STEPS} steps...")

for step in range(STEPS):
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    
    efforts = np.zeros(num_dofs, dtype=np.float32)
    
    # 腿部 P 控制: torque = Kp * (0 - pos) - Kd * vel  (保持默认位置)
    for i in [0, 1, 3, 4]:  # thigh & knee joints
        efforts[i] = Kp_leg * (0.0 - dof_states['pos'][i]) - Kd_leg * dof_states['vel'][i]
        efforts[i] = np.clip(efforts[i], -2.0, 2.0)
    
    # 轮子直接力矩
    efforts[2] = TORQUE   # left wheel
    efforts[5] = TORQUE   # right wheel
    
    for i in range(num_dofs):
        gym.apply_dof_effort(env, i, float(efforts[i]))
    
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    if step % 50 == 0 or step == STEPS - 1:
        dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
        body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL)
        base_x = body_states['pose']['p'][0][0]
        base_z = body_states['pose']['p'][0][2]
        print(f"  Step {step}: Base X={base_x:.4f}, Z={base_z:.4f}")
        print(f"    L_wheel: pos={dof_states['pos'][2]:.4f}, vel={dof_states['vel'][2]:.2f}")
        print(f"    R_wheel: pos={dof_states['pos'][5]:.4f}, vel={dof_states['vel'][5]:.2f}")
        print(f"    L_thigh: pos={dof_states['pos'][0]:.4f}, vel={dof_states['vel'][0]:.2f}")

print("\nDone!")
gym.destroy_sim(sim)
