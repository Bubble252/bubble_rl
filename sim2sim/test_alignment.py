#!/usr/bin/env python3
"""诊断 sim2sim 对齐问题"""
import mujoco
import numpy as np
import sys

xml_path = "/home/bubble/bubble_wheel_rl/Wheel_legged_gym/sim2sim/bubble.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

print("=" * 60)
print("1. 关节名称和轴向")
print("=" * 60)
for i in range(model.njnt):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  Joint {i}: {n:30s}  axis={model.jnt_axis[i]}")

print("\n" + "=" * 60)
print("2. 执行器映射")
print("=" * 60)
for i in range(model.nu):
    jid = model.actuator_trnid[i, 0]
    an = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    gear = model.actuator_gear[i, 0]
    cr = model.actuator_ctrlrange[i]
    print(f"  Act {i}: {an:25s} → Joint {jid} ({jn:25s})  gear={gear}  ctrlrange={cr}")

print("\n" + "=" * 60)
print("3. 重力投影测试 (10° pitch)")
print("=" * 60)

def quat_rotate_inverse_np(q, v):
    """q: xyzw, v: 3D"""
    q_w = q[3]; q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

data.qpos[:] = 0
data.qpos[2] = 0.18
angle = np.radians(10)
data.qpos[3:7] = [np.cos(angle/2), 0, np.sin(angle/2), 0]  # wxyz, pitch around Y
data.qpos[7:13] = [0.2, -0.2, 0.0, -0.2, 0.2, 0.0]
mujoco.mj_forward(model, data)

q_mj = data.qpos[3:7]  # wxyz
q_isaac = np.array([q_mj[1], q_mj[2], q_mj[3], q_mj[0]])  # xyzw
pg = quat_rotate_inverse_np(q_isaac, np.array([0, 0, -1.0]))
print(f"  quat_mj(wxyz):     {q_mj}")
print(f"  quat_isaac(xyzw):  {q_isaac}")
print(f"  projected_gravity: {pg}")
pitch_from_pg = np.degrees(np.arcsin(np.clip(-pg[0], -1, 1)))
print(f"  pitch from pg:     {pitch_from_pg:.1f}° (expect ~10°)")

print("\n" + "=" * 60)
print("4. 力矩方向测试")
print("=" * 60)
print("  测试: 给左大腿正力矩(ctrl[0]>0), 看关节角度变化方向")

data.qpos[:] = 0
data.qpos[2] = 0.5  # 高一点防止碰地
data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[7:13] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
mujoco.mj_forward(model, data)

# 施加正力矩到左大腿
data.ctrl[:] = 0
data.ctrl[0] = 0.3  # left_thigh_motor, 正力矩

for step in range(100):
    mujoco.mj_step(model, data)

print(f"  施加 ctrl[0]=+0.3 (left_thigh_motor)")
print(f"  left_thigh_joint pos after 100 steps: {data.qpos[7]:.4f}")
print(f"  (axis=[0,1,0], 正力矩 → 正角度 = 大腿向前)")
print(f"  IsaacGym axis=[0,1,0], 正角度 = 向前弯(thigh=+0.2 default)")

# 测试轮子
data.qpos[:] = 0
data.qpos[2] = 0.5
data.qpos[3:7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)
data.ctrl[:] = 0
data.ctrl[2] = 0.3  # left_wheel_motor

for step in range(100):
    mujoco.mj_step(model, data)

print(f"\n  施加 ctrl[2]=+0.3 (left_wheel_motor)")
print(f"  left_wheel vel after 100 steps: {data.qvel[8]:.4f}")
print(f"  (axis=[0,-1,0], 正力矩 → 轮子转动方向)")

print("\n" + "=" * 60)
print("5. 接触模型参数")
print("=" * 60)
print(f"  solver:    {['PGS','CG','Newton'][model.opt.solver]}")
print(f"  timestep:  {model.opt.timestep}")
print(f"  iterations: {model.opt.iterations}")
print(f"  gravity:   {model.opt.gravity}")

# 地面摩擦
floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
print(f"\n  Floor geom friction: {model.geom_friction[floor_id]}")

# 轮子摩擦
for name in ["left_wheel_col", "right_wheel_col"]:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    print(f"  {name} friction: {model.geom_friction[gid]}")

print("\n" + "=" * 60)
print("6. 关节阻尼 (damping)")
print("=" * 60)
for i in range(model.njnt):
    if i == 0:
        continue  # skip freejoint
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    dof_adr = model.jnt_dofadr[i]
    damp = model.dof_damping[dof_adr]
    fric = model.dof_frictionloss[dof_adr]
    armature = model.dof_armature[dof_adr]
    print(f"  {n:30s}  damping={damp:.4f}  frictionloss={fric:.4f}  armature={armature:.4f}")

print("\n" + "=" * 60)
print("7. 关键问题: MuJoCo ctrl 是否被限幅?")
print("=" * 60)
print(f"  actuator_ctrllimited: {model.actuator_ctrllimited}")
print(f"  actuator_ctrlrange:")
for i in range(model.nu):
    an = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"    {an}: {model.actuator_ctrlrange[i]}")

# 测试: ctrl 超出范围会被限幅吗?
data.qpos[:] = 0; data.qpos[2] = 0.5; data.qpos[3:7] = [1,0,0,0]
mujoco.mj_forward(model, data)
data.ctrl[0] = 1.0  # 超出 [-0.5, 0.5]
mujoco.mj_step(model, data)
print(f"\n  设置 ctrl[0]=1.0, 实际 actuator_force[0] = {data.actuator_force[0]:.4f}")
print(f"  (如果被限幅到 0.5, 说明 ctrllimited 生效)")

print("\n" + "=" * 60)
print("8. 双重阻尼问题检查")
print("=" * 60)
print("  MuJoCo joint damping 是被动阻尼力, 自动施加")
print("  我们的 PD 控制已经包含了 Kd 项 (D gains)")
print("  如果 MJCF 中 damping>0, 则实际阻尼 = PD的Kd + MJCF的damping")
print("  IsaacGym URDF damping=0.02, 但PD控制使用的是cfg中的Kd=0.08")
print("")
print("  当前 MJCF damping 值 (见上面第6项)")
print("  IsaacGym 中: 实际力矩 = PD控制力矩 (已含Kd*vel)")
print("               URDF damping 由 PhysX 物理引擎额外施加")
print("  MuJoCo 中:   实际力矩 = ctrl (我们计算的PD力矩, 含Kd*vel)")
print("               + MJCF damping 由 MuJoCo 额外施加")
print("")
print("  ⚠️ 这意味着 MuJoCo 有双重阻尼!")
print("  解决方案: MJCF 中 damping 应该设为 0 (因为PD已经处理了)")
print("  或者保留 MJCF damping 但减小 PD 的 Kd")
