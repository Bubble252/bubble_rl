#!/usr/bin/env python3
"""
逐步诊断 sim2sim 策略推理过程
打印前几步的观测、动作、力矩，与 IsaacGym 对比
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_mujoco import (build_actor, load_policy, Cfg, 
                         quat_rotate_inverse_np, mujoco_quat_to_isaac,
                         compute_torques_b2w, ObservationBuilder)

# 加载模型
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(project_root, "logs", "flat_bubble")

# 用最新的地形训练模型
import glob
runs = sorted([d for d in glob.glob(os.path.join(log_dir, "Mar*")) if os.path.isdir(d)])
latest_run = os.path.basename(runs[-1])
print(f"Using run: {latest_run}")

actor = load_policy(log_dir, latest_run)

# 加载 MuJoCo
xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bubble.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 初始化
data.qpos[0:3] = [0, 0, 0.18]
data.qpos[3:7] = [1, 0, 0, 0]
data.qpos[7:13] = Cfg.default_dof_pos
data.qvel[:] = 0
mujoco.mj_forward(model, data)

obs_builder = ObservationBuilder()
commands = np.array([0.0, 0.0, 0.0])

print("\n" + "=" * 80)
print("逐步诊断: 前 20 个策略步 (每步 0.01s)")
print("=" * 80)

action = np.zeros(6)
for policy_step in range(20):
    # 构建观测
    obs = obs_builder.build(data, commands)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        action_tensor = actor(obs_tensor)
    action = action_tensor.squeeze(0).numpy()
    action = np.clip(action, -Cfg.clip_actions, Cfg.clip_actions)
    obs_builder.last_actions = action.copy()
    
    # 当前状态
    qm = data.qpos[3:7]
    qi = mujoco_quat_to_isaac(qm)
    pg = quat_rotate_inverse_np(qi, np.array([0, 0, -1.0]))
    bv = quat_rotate_inverse_np(qi, data.qvel[0:3])
    bw = quat_rotate_inverse_np(qi, data.qvel[3:6])
    dof_pos = data.qpos[7:13]
    dof_vel = data.qvel[6:12]
    
    # 力矩
    torques = compute_torques_b2w(action, dof_pos, dof_vel)
    
    pitch_deg = np.degrees(np.arcsin(np.clip(-pg[0], -1, 1)))
    
    if policy_step < 5 or policy_step % 5 == 0:
        print(f"\n--- Policy Step {policy_step} (t={policy_step*0.01:.2f}s) ---")
        print(f"  Height: {data.qpos[2]:.4f}m  Pitch: {pitch_deg:.1f}°")
        print(f"  Base lin vel (body): [{bv[0]:.3f}, {bv[1]:.3f}, {bv[2]:.3f}]")
        print(f"  Base ang vel (body): [{bw[0]:.3f}, {bw[1]:.3f}, {bw[2]:.3f}]")
        print(f"  Proj gravity: [{pg[0]:.3f}, {pg[1]:.3f}, {pg[2]:.3f}]")
        print(f"  DOF pos: [{', '.join(f'{x:.3f}' for x in dof_pos)}]")
        print(f"  DOF vel: [{', '.join(f'{x:.3f}' for x in dof_vel)}]")
        print(f"  Action:  [{', '.join(f'{x:.3f}' for x in action)}]")
        print(f"  Torques: [{', '.join(f'{x:.3f}' for x in torques)}]")
        print(f"  Obs[0:12]: [{', '.join(f'{x:.3f}' for x in obs[:12])}]")
        print(f"  Obs[12:30]: [{', '.join(f'{x:.3f}' for x in obs[12:30])}]")
    
    # 执行 decimation 步
    for _ in range(Cfg.decimation):
        dof_pos = data.qpos[7:13]
        dof_vel = data.qvel[6:12]
        torques = compute_torques_b2w(action, dof_pos, dof_vel)
        data.ctrl[:] = torques
        mujoco.mj_step(model, data)

print(f"\n\n=== Final State (t=0.20s) ===")
print(f"  Height: {data.qpos[2]:.4f}m")
qm = data.qpos[3:7]
qi = mujoco_quat_to_isaac(qm)
pg = quat_rotate_inverse_np(qi, np.array([0, 0, -1.0]))
print(f"  Pitch: {np.degrees(np.arcsin(np.clip(-pg[0], -1, 1))):.1f}°")
if data.qpos[2] < 0.05:
    print("  ⚠️ FELL!")
