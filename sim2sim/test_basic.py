#!/usr/bin/env python3
"""测试 b2w 基础力矩能否让机器人站稳"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('/home/bubble/bubble_wheel_rl/Wheel_legged_gym/sim2sim/bubble.xml')
data = mujoco.MjData(model)

default_dof_pos = np.array([0.2, -0.2, 0.0, -0.2, 0.2, 0.0])
Kp = np.array([2.0, 2.0, 1.0, 2.0, 2.0, 1.0])
Kd = np.array([0.08, 0.08, 0.05, 0.08, 0.08, 0.05])
wheel_speed = 2.0
wheel_idx = [2, 5]

def qri(q, v):
    w = q[3]; u = q[:3]
    return v * (2*w**2-1) - np.cross(u,v)*w*2 + u*np.dot(u,v)*2

# Test: zero action PD control
data.qpos[0:3] = [0, 0, 0.18]
data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[7:13] = default_dof_pos.copy()
data.qvel[:] = 0
mujoco.mj_forward(model, data)

print('=== Zero-action b2w test ===')
for step in range(2000):
    dof_pos = data.qpos[7:13]
    dof_vel = data.qvel[6:12]
    
    action_scaled = np.zeros(6)
    dof_err = default_dof_pos - dof_pos
    dof_err[wheel_idx] = 0.0
    dof_vel_mod = dof_vel.copy()
    dof_vel_mod[wheel_idx] = -wheel_speed
    
    torques = Kp * (action_scaled + dof_err) - Kd * dof_vel_mod
    torques = np.clip(torques, -0.5, 0.5)
    
    data.ctrl[:] = torques
    mujoco.mj_step(model, data)
    
    if step % 400 == 0:
        qm = data.qpos[3:7]
        qi = np.array([qm[1], qm[2], qm[3], qm[0]])
        pg = qri(qi, np.array([0, 0, -1.0]))
        pitch = np.degrees(np.arcsin(np.clip(-pg[0], -1, 1)))
        print(f'  t={step*0.005:.1f}s h={data.qpos[2]:.4f} pitch={pitch:.1f}deg wheel_torq=[{torques[2]:.3f},{torques[5]:.3f}] leg_torq=[{torques[0]:.3f},{torques[1]:.3f},{torques[3]:.3f},{torques[4]:.3f}]')

print(f'\nFinal height: {data.qpos[2]:.4f} (target: 0.14)')
if data.qpos[2] < 0.05:
    print('FAILED: Robot fell')
else:
    print('OK: Robot standing')
