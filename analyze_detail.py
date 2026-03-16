#!/usr/bin/env python3
"""Detailed analysis of hirght2 transition zone."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('logs/skill_bubble/Mar11_02-17-27_hirght2')
ea.Reload()

key_tags = [
    'Train/mean_reward', 'Train/mean_episode_length',
    'Episode/rew_base_height', 'Episode/rew_collision',
    'Episode/rew_tracking_lin_vel', 'Episode/rew_orientation',
    'Episode/rew_no_fly', 'Episode/rew_action_rate',
    'Episode/diag_pitch', 'Episode/diag_base_height',
    'Episode/diag_joint_left_knee_joint', 'Episode/diag_joint_left_thigh_joint',
    'Episode/terrain_level', 'Episode/diag_penalized_contact_ratio',
    'Policy/mean_noise_std',
]

print('=== hirght2: iter 50~800 转折区间 ===')
for tag in key_tags:
    events = ea.Scalars(tag)
    selected = [(e.step, round(e.value, 4)) for e in events if 50 <= e.step <= 800]
    if len(selected) > 12:
        step = max(1, len(selected) // 12)
        selected = selected[::step][:12]
    print(f'{tag}:')
    print(f'  steps={[s[0] for s in selected]}')
    print(f'  vals ={[s[1] for s in selected]}')
