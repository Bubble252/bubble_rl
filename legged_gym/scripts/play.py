# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    global SET_CAMERA_FOR_SPECIFIC_ROBOT
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging (按 [ ] 切换)
    num_robots = env.num_envs
    joint_index = 1 # which joint is used for logging
    stop_state_log = 2000 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    # 多视角预设（适配 bubble 小型机器人 ~0.16m 高）
    # [相机偏移x, y, z] 相对于机器人位置
    camera_presets = [
        {"name": "side",      "offset": [0.0,  -0.4, 0.15]},  # 侧面平视
        {"name": "front",     "offset": [0.5,   0.0, 0.15]},  # 正前方
        {"name": "back",      "offset": [-0.4,  0.0, 0.15]},  # 正后方
        {"name": "top-close",  "offset": [0.15, -0.15, 0.4]}, # 近距俯视
        {"name": "overview",  "offset": [0.6,  -0.4, 0.5]},   # 远景概览
        {"name": "low-angle", "offset": [0.25, -0.2, 0.05]},  # 低角度（几乎贴地）
    ]
    cam_preset_idx = 0
    print(f"\n[Camera] 按 1-6 切换视角 (当前: {camera_presets[cam_preset_idx]['name']})")
    print(f"  1=侧面  2=正前  3=正后  4=俯视  5=远景  6=低角度")
    print(f"  [ ] = 切换观察的机器人 (当前: robot {robot_index}/{num_robots-1})")
    print(f"  IsaacGym自带: 鼠标左键拖拽旋转, 中键平移, 滚轮缩放\n")

    # 注册数字键事件
    if env.viewer is not None:
        for key_val, key_name in [(gymapi.KEY_1, "CAM1"), (gymapi.KEY_2, "CAM2"),
                                   (gymapi.KEY_3, "CAM3"), (gymapi.KEY_4, "CAM4"),
                                   (gymapi.KEY_5, "CAM5"), (gymapi.KEY_6, "CAM6")]:
            env.gym.subscribe_viewer_keyboard_event(env.viewer, key_val, key_name)
        # [ ] 切换观察的机器人
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_LEFT_BRACKET, "PREV_ROBOT")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_RIGHT_BRACKET, "NEXT_ROBOT")

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', 'measure_points',f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if SET_CAMERA_FOR_SPECIFIC_ROBOT:
            # 检查键盘事件切换视角 / 切换机器人
            if env.viewer is not None:
                events = env.gym.query_viewer_action_events(env.viewer)
                for evt in events:
                    for ci in range(len(camera_presets)):
                        if evt.action == f"CAM{ci+1}" and evt.value > 0:
                            cam_preset_idx = ci
                            print(f"[Camera] 切换到: {camera_presets[cam_preset_idx]['name']}")
                    if evt.action == "PREV_ROBOT" and evt.value > 0:
                        robot_index = (robot_index - 1) % num_robots
                        print(f"[Camera] 观察机器人: {robot_index}/{num_robots-1}")
                    if evt.action == "NEXT_ROBOT" and evt.value > 0:
                        robot_index = (robot_index + 1) % num_robots
                        print(f"[Camera] 观察机器人: {robot_index}/{num_robots-1}")

            robot_pos = env.root_states[robot_index, 0:3].cpu().numpy()
            d = camera_presets[cam_preset_idx]["offset"]
            env.set_camera(robot_pos + np.array(d), robot_pos)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    'base_height': env.root_states[robot_index, 2].item(),
                    # 'command_base_height': env.commands[robot_index, 4].item(),
                    # 'knee_angle': env.theta_right[robot_index].item(),
                    # 'command_knee_angle': env.commands[robot_index, 5].item(),
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    SET_CAMERA_FOR_SPECIFIC_ROBOT = True   # 开启跟随相机
    args = get_args()
    play(args)
