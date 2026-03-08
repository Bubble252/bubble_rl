#!/usr/bin/env python3
"""Check DOF properties for Bubble and Diablo robots in IsaacGym"""
import sys

from isaacgym import gymapi
import numpy as np

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.physx.use_gpu = False
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

print("=" * 70, flush=True)
print("BUBBLE ROBOT DOF PROPERTIES", flush=True)
print("=" * 70, flush=True)
asset_options = gymapi.AssetOptions()
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
bubble_asset = gym.load_asset(sim, "resources/robots/bubble/urdf", "bubble.urdf", asset_options)
n = gym.get_asset_dof_count(bubble_asset)
props = gym.get_asset_dof_properties(bubble_asset)
names = gym.get_asset_dof_names(bubble_asset)
for i in range(n):
    print(f"  [{i}] {names[i]:25s} lower={props['lower'][i]:15.6f}  upper={props['upper'][i]:15.6f}  effort={props['effort'][i]:15.6f}  velocity={props['velocity'][i]:15.6f}  hasLimits={props['hasLimits'][i]}", flush=True)

print(flush=True)
print("=" * 70, flush=True)
print("DIABLO ROBOT DOF PROPERTIES", flush=True)
print("=" * 70, flush=True)
diablo_asset = gym.load_asset(sim, "resources/robots/diablo/urdf", "diablo.urdf", asset_options)
n2 = gym.get_asset_dof_count(diablo_asset)
props2 = gym.get_asset_dof_properties(diablo_asset)
names2 = gym.get_asset_dof_names(diablo_asset)
for i in range(n2):
    print(f"  [{i}] {names2[i]:25s} lower={props2['lower'][i]:15.6f}  upper={props2['upper'][i]:15.6f}  effort={props2['effort'][i]:15.6f}  velocity={props2['velocity'][i]:15.6f}  hasLimits={props2['hasLimits'][i]}", flush=True)

gym.destroy_sim(sim)
print(flush=True)
print("DONE", flush=True)
