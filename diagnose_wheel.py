#!/usr/bin/env python3
"""诊断脚本：检查轮子为什么不转"""
import sys
sys.path.insert(0, '.')

from legged_gym.envs import *
from legged_gym.utils import task_registry
from isaacgym import gymapi
import torch

# 注册并创建环境
env_cfg, train_cfg = task_registry.get_cfgs(name="bubble")
env_cfg.env.num_envs = 4

# 尝试用替代碰撞形状（用凸分解避免凸包问题）
env_cfg.asset.mesh_normal_mode = 0  # gymapi.COMPUTE_PER_FACE
env, _ = task_registry.make_env(name="bubble", args=None, env_cfg=env_cfg)

print("="*70)
print("DIAGNOSTIC: Testing wheel torque response")
print("="*70)
print(f"wheel_indices: {env.wheel_indices}")
print(f"p_gains: {env.p_gains}")
print(f"d_gains: {env.d_gains}")
print(f"torque_limits: {env.torque_limits}")
print()

# 测试不同力矩大小的响应
for test_torque_limit in [2.0, 5.0, 10.0]:
    # 临时修改力矩限制
    env.torque_limits[env.wheel_indices] = test_torque_limit
    
    # 重置所有环境
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    
    print(f"--- torque_limit = {test_torque_limit} N·m ---")
    for step in range(100):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        actions[:, env.wheel_indices] = 1.0  # 轮子满力
        obs, _, rew, done, info = env.step(actions)
    
    e = 0
    print(f"  After 100 steps (2s):")
    print(f"    wheel dof_pos: L={env.dof_pos[e, env.wheel_indices[0]].item():.4f}  R={env.dof_pos[e, env.wheel_indices[1]].item():.4f}")
    print(f"    wheel dof_vel: L={env.dof_vel[e, env.wheel_indices[0]].item():.4f}  R={env.dof_vel[e, env.wheel_indices[1]].item():.4f}")
    print(f"    wheel torque:  L={env.torques[e, env.wheel_indices[0]].item():.4f}  R={env.torques[e, env.wheel_indices[1]].item():.4f}")
    print(f"    base_lin_vel:  x={env.base_lin_vel[e, 0].item():.4f}")
    print(f"    base height:   {env.root_states[e, 2].item():.4f}")
    print()

# 检查轮子碰撞体形状
print("--- Checking wheel rigid body shapes ---")
body_names = env.gym.get_asset_rigid_body_names(env.gym.get_actor_asset(env.envs[0], 0))
print(f"Rigid body names: {body_names}")

# 打印每个刚体的 shape 数量和类型
actor = env.gym.get_actor_handle(env.envs[0], 0)
num_bodies = env.gym.get_actor_rigid_body_count(env.envs[0], actor)
for i in range(num_bodies):
    body_props = env.gym.get_actor_rigid_body_properties(env.envs[0], actor)
    num_shapes = env.gym.get_actor_rigid_body_shape_count(env.envs[0], actor)
    print(f"  Body {i} ({body_names[i]}): mass={body_props[i].mass:.6f} kg")

print()
print("DONE")
