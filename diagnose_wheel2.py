"""
诊断脚本：深入检查为什么轮子无法旋转
1. 检查碰撞体形状类型
2. 悬空测试（无地面接触）
3. 放大轮子质量测试
"""
from isaacgym import gymapi, gymutil, gymtorch
import torch
import numpy as np

gym = gymapi.acquire_gym()

# 创建 sim
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

# 加载资产
asset_options = gymapi.AssetOptions()
asset_options.collapse_fixed_joints = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
asset_options.replace_cylinder_with_capsule = True

asset = gym.load_asset(sim, "resources/robots/bubble/urdf", "bubble.urdf", asset_options)

# 打印资产信息
num_bodies = gym.get_asset_rigid_body_count(asset)
num_shapes = gym.get_asset_rigid_shape_count(asset)
num_dofs = gym.get_asset_dof_count(asset)
body_names = gym.get_asset_rigid_body_names(asset)
dof_names = gym.get_asset_dof_names(asset)

print(f"Bodies: {num_bodies}, Shapes: {num_shapes}, DOFs: {num_dofs}")
print(f"Body names: {body_names}")
print(f"DOF names: {dof_names}")

# 获取 rigid shape properties
shape_props = gym.get_asset_rigid_shape_properties(asset)
print(f"\n=== Rigid Shape Properties ({len(shape_props)} shapes) ===")
for i, sp in enumerate(shape_props):
    print(f"  Shape {i}: friction={sp.friction:.3f}, restitution={sp.restitution:.3f}, "
          f"compliance={sp.compliance:.6f}, thickness={sp.thickness:.6f}")

# 添加地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
gym.add_ground(sim, plane_params)

# ==================== 测试 1: 正常站立 + 施加力矩 ====================
print("\n" + "="*60)
print("TEST 1: 正常站立 + 施加力矩 (2 N·m)")
print("="*60)

env1 = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
start_pose = gymapi.Transform()
start_pose.p = gymapi.Vec3(0, 0, 0.20)
start_pose.r = gymapi.Quat(0, 0, 0, 1)
actor1 = gym.create_actor(env1, asset, start_pose, "bubble1", 0, 0)

# 设置 DOF 属性 - 全部 effort mode
dof_props = gym.get_actor_dof_properties(env1, actor1)
for i in range(num_dofs):
    dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
    dof_props['stiffness'][i] = 0.0
    dof_props['damping'][i] = 0.0
gym.set_actor_dof_properties(env1, actor1, dof_props)

# ==================== 测试 2: 悬空（fix base）+ 施加力矩 ====================
print("\n" + "="*60)
print("TEST 2: 悬空固定 + 施加力矩 (2 N·m)")
print("="*60)

env2 = gym.create_env(sim, gymapi.Vec3(2, -1, 0), gymapi.Vec3(4, 1, 1), 1)
start_pose2 = gymapi.Transform()
start_pose2.p = gymapi.Vec3(3, 0, 0.30)  # 更高，确保悬空
start_pose2.r = gymapi.Quat(0, 0, 0, 1)
actor2 = gym.create_actor(env2, asset, start_pose2, "bubble2", 1, 0)

# fix base
body_props2 = gym.get_actor_rigid_body_properties(env2, actor2)
body_props2[0].flags = gymapi.RIGID_BODY_DISABLE_GRAVITY  # disable gravity on base
gym.set_actor_rigid_body_properties(env2, actor2, body_props2)

dof_props2 = gym.get_actor_dof_properties(env2, actor2)
for i in range(num_dofs):
    dof_props2['driveMode'][i] = gymapi.DOF_MODE_EFFORT
    dof_props2['stiffness'][i] = 0.0
    dof_props2['damping'][i] = 0.0
gym.set_actor_dof_properties(env2, actor2, dof_props2)

gym.prepare_sim(sim)

# 先让机器人稳定几步
print("\nStabilizing...")
for _ in range(100):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

# 施加力矩并观察
TORQUE = 2.0
STEPS = 200

print(f"\nApplying {TORQUE} N·m to wheel DOFs for {STEPS} steps...")

for step in range(STEPS):
    # Test 1: normal standing
    efforts1 = np.zeros(num_dofs, dtype=np.float32)
    efforts1[2] = TORQUE  # left wheel
    efforts1[5] = TORQUE  # right wheel
    gym.apply_dof_effort(env1, 0, efforts1[0])  # left_thigh
    gym.apply_dof_effort(env1, 1, efforts1[1])  # left_knee
    gym.apply_dof_effort(env1, 2, efforts1[2])  # left_wheel
    gym.apply_dof_effort(env1, 3, efforts1[3])  # right_thigh
    gym.apply_dof_effort(env1, 4, efforts1[4])  # right_knee
    gym.apply_dof_effort(env1, 5, efforts1[5])  # right_wheel

    # Test 2: suspended
    efforts2 = np.zeros(num_dofs, dtype=np.float32)
    efforts2[2] = TORQUE
    efforts2[5] = TORQUE
    gym.apply_dof_effort(env2, 0, efforts2[0])
    gym.apply_dof_effort(env2, 1, efforts2[1])
    gym.apply_dof_effort(env2, 2, efforts2[2])
    gym.apply_dof_effort(env2, 3, efforts2[3])
    gym.apply_dof_effort(env2, 4, efforts2[4])
    gym.apply_dof_effort(env2, 5, efforts2[5])

    gym.simulate(sim)
    gym.fetch_results(sim, True)

    if step % 50 == 0 or step == STEPS - 1:
        # Get DOF states
        dof_states1 = gym.get_actor_dof_states(env1, actor1, gymapi.STATE_ALL)
        dof_states2 = gym.get_actor_dof_states(env2, actor2, gymapi.STATE_ALL)

        print(f"\n  Step {step}:")
        print(f"    [Standing] L_wheel: pos={dof_states1['pos'][2]:.6f}, vel={dof_states1['vel'][2]:.4f}")
        print(f"    [Standing] R_wheel: pos={dof_states1['pos'][5]:.6f}, vel={dof_states1['vel'][5]:.4f}")
        print(f"    [Suspend ] L_wheel: pos={dof_states2['pos'][2]:.6f}, vel={dof_states2['vel'][2]:.4f}")
        print(f"    [Suspend ] R_wheel: pos={dof_states2['pos'][5]:.6f}, vel={dof_states2['vel'][5]:.4f}")

        # Also print base state
        body_states1 = gym.get_actor_rigid_body_states(env1, actor1, gymapi.STATE_ALL)
        body_states2 = gym.get_actor_rigid_body_states(env2, actor2, gymapi.STATE_ALL)
        base_pos1 = body_states1['pose']['p'][0]
        base_pos2 = body_states2['pose']['p'][0]
        print(f"    [Standing] Base Z: {base_pos1[2]:.4f}")
        print(f"    [Suspend ] Base Z: {base_pos2[2]:.4f}")

# ==================== 测试 3: 检查碰撞接触 ====================
print("\n" + "="*60)
print("TEST 3: 检查 net contact force (standing)")
print("="*60)

# Get contact forces
contact_forces = gym.get_env_rigid_contact_forces(env1)
print(f"Contact forces type: {type(contact_forces)}")
if contact_forces is not None:
    print(f"Contact forces: {contact_forces}")

# Check rigid body shape count per body
print(f"\n=== Shape count per body ===")
for i, name in enumerate(body_names):
    # get shape indices for this body
    print(f"  Body {i} ({name})")

print("\nDone!")

gym.destroy_sim(sim)
