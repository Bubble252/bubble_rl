#!/usr/bin/env python3
"""
Bubble Sim2Sim: IsaacGym → MuJoCo
加载 IsaacGym 训练的策略，在 MuJoCo 中推理并可视化。

用法:
  python sim2sim/run_mujoco.py --load_run <run_name> [--checkpoint <iter>] [--cmd_vx 0.3]

关键对齐项:
  1. 关节顺序: left_thigh, left_knee, left_wheel, right_thigh, right_knee, right_wheel (与IsaacGym一致)
  2. 控制逻辑: 精确复现 b2w 模式 torque = Kp*(action_scaled+dof_err) - Kd*dof_vel_modified
  3. 观测空间: 150维 = 30(base) + 60(dof_pos_err_history) + 60(dof_vel_history)
  4. 坐标系: MuJoCo quat=wxyz → 转换为 IsaacGym xyzw
  5. decimation=2: 每2个物理步执行1次策略
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer
import time
import threading
from collections import deque

# ========================== 配置 (与 bubble_config.py 对齐) ==========================

class Cfg:
    """从 bubble_config.py 提取的关键参数"""
    # 物理
    dt = 0.005
    decimation = 2
    policy_dt = dt * decimation  # 0.01s = 100Hz
    
    # 控制
    action_scale = 0.25
    wheel_drive_mode = "b2w"
    wheel_speed = 2.0  # 注意: 需要与训练时一致! 检查你的 config
    
    # PD 增益 (索引顺序: left_thigh, left_knee, left_wheel, right_thigh, right_knee, right_wheel)
    Kp = np.array([2.0, 2.0, 1.0, 2.0, 2.0, 1.0])
    Kd = np.array([0.08, 0.08, 0.05, 0.08, 0.08, 0.05])
    
    # 轮子索引
    wheel_indices = [2, 5]
    leg_indices = [0, 1, 3, 4]
    
    # 默认关节角度 (与 IsaacGym 训练一致)
    default_dof_pos = np.array([0.2, -0.2, 0.0, -0.2, 0.2, 0.0])
    
    # 力矩限幅
    torque_limit = 0.5
    
    # 观测缩放
    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    
    # 命令缩放 (与 legged_robot.py 一致: commands_scale = [lin_vel, lin_vel, ang_vel])
    commands_scale = np.array([2.0, 2.0, 0.25])
    
    # 观测裁剪
    clip_obs = 100.0
    clip_actions = 100.0
    
    # 网络
    num_obs = 150
    num_actions = 6
    actor_hidden_dims = [512, 256, 128]
    
    # 历史
    history_length = 10


# ========================== 网络定义 ==========================

def build_actor(num_obs, num_actions, hidden_dims, activation='elu'):
    """重建 ActorCritic 的 actor 网络 (与 rsl_rl 一致)"""
    act_fn = nn.ELU() if activation == 'elu' else nn.ReLU()
    layers = []
    layers.append(nn.Linear(num_obs, hidden_dims[0]))
    layers.append(act_fn)
    for i in range(len(hidden_dims)):
        if i == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dims[i], num_actions))
        else:
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(act_fn)
    return nn.Sequential(*layers)


def load_policy(log_dir, run_name, checkpoint=-1):
    """加载训练好的策略权重"""
    run_dir = os.path.join(log_dir, run_name)
    
    if checkpoint == -1:
        # 找最新的 model_*.pt
        models = [f for f in os.listdir(run_dir) if f.startswith('model_') and f.endswith('.pt')]
        if not models:
            raise FileNotFoundError(f"No model files found in {run_dir}")
        # 按迭代数排序
        models.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        model_file = models[-1]
    else:
        model_file = f"model_{checkpoint}.pt"
    
    model_path = os.path.join(run_dir, model_file)
    print(f"[Sim2Sim] Loading policy from: {model_path}")
    
    # 加载 checkpoint
    loaded = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 重建 actor
    actor = build_actor(Cfg.num_obs, Cfg.num_actions, Cfg.actor_hidden_dims)
    
    # 提取 actor 权重 (checkpoint 中 key 格式: "actor.0.weight", "actor.0.bias", ...)
    actor_state = {}
    for key, val in loaded['model_state_dict'].items():
        if key.startswith('actor.'):
            actor_state[key.replace('actor.', '')] = val
    
    actor.load_state_dict(actor_state)
    actor.eval()
    
    print(f"[Sim2Sim] Policy loaded: {model_file}")
    print(f"[Sim2Sim] Actor: {actor}")
    return actor


# ========================== MuJoCo 工具函数 ==========================

def quat_rotate_inverse_np(q, v):
    """将向量从世界坐标系旋转到 body 坐标系
    q: (4,) quaternion in xyzw format (IsaacGym convention)
    v: (3,) vector in world frame
    Returns: (3,) vector in body frame
    """
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def mujoco_quat_to_isaac(q_mj):
    """MuJoCo quat (wxyz) → IsaacGym quat (xyzw)"""
    return np.array([q_mj[1], q_mj[2], q_mj[3], q_mj[0]])


# ========================== B2W 控制器 ==========================

def compute_torques_b2w(action, dof_pos, dof_vel, cfg=Cfg):
    """精确复现 IsaacGym 的 b2w 控制模式
    
    torques = Kp * (action_scaled + dof_err) - Kd * dof_vel_modified
    
    其中:
      - dof_err = default_pos - dof_pos, 但 wheel 部分清零
      - dof_vel_modified = dof_vel, 但 wheel 部分覆盖为 -wheel_speed
    """
    action_scaled = action * cfg.action_scale
    
    # 位置误差 (腿部)
    dof_err = cfg.default_dof_pos - dof_pos
    # 轮子位置误差清零
    dof_err[cfg.wheel_indices] = 0.0
    
    # 速度 (轮子覆盖为常数)
    dof_vel_mod = dof_vel.copy()
    dof_vel_mod[cfg.wheel_indices] = -cfg.wheel_speed
    
    # PD
    torques = cfg.Kp * (action_scaled + dof_err) - cfg.Kd * dof_vel_mod
    
    # 限幅
    torques = np.clip(torques, -cfg.torque_limit, cfg.torque_limit)
    return torques


# ========================== 观测构建 ==========================

class ObservationBuilder:
    """精确复现 IsaacGym 的 150 维观测空间"""
    
    def __init__(self, cfg=Cfg):
        self.cfg = cfg
        self.history_len = cfg.history_length
        self.num_dof = cfg.num_actions  # 6
        
        # 历史 buffer
        self.dof_pos_err_history = deque(
            [np.zeros(self.num_dof) for _ in range(self.history_len)],
            maxlen=self.history_len
        )
        self.dof_vel_history = deque(
            [np.zeros(self.num_dof) for _ in range(self.history_len)],
            maxlen=self.history_len
        )
        
        self.last_actions = np.zeros(cfg.num_actions)
    
    def reset(self):
        self.dof_pos_err_history = deque(
            [np.zeros(self.num_dof) for _ in range(self.history_len)],
            maxlen=self.history_len
        )
        self.dof_vel_history = deque(
            [np.zeros(self.num_dof) for _ in range(self.history_len)],
            maxlen=self.history_len
        )
        self.last_actions = np.zeros(self.cfg.num_actions)
    
    def build(self, data, commands):
        """
        构建 150 维观测向量
        
        data: mujoco.MjData
        commands: np.array([cmd_vx, cmd_vy, cmd_yaw])  (3,)
        
        布局:
          [0:3]   base_lin_vel (body frame) * lin_vel_scale
          [3:6]   base_ang_vel (body frame) * ang_vel_scale
          [6:9]   projected_gravity
          [9:12]  commands[:3] * commands_scale
          [12:18] dof_pos_err * dof_pos_scale (wheel=0)
          [18:24] dof_vel * dof_vel_scale
          [24:30] last_actions
          [30:90] dof_pos_err_history (10*6) * dof_pos_scale
          [90:150] dof_vel_history (10*6) * dof_vel_scale
        """
        cfg = self.cfg
        
        # 1. 四元数: MuJoCo (wxyz) → IsaacGym (xyzw)
        quat_mj = data.qpos[3:7]  # wxyz
        quat_isaac = mujoco_quat_to_isaac(quat_mj)
        
        # 2. 基座线速度 (world → body frame)
        base_lin_vel_world = data.qvel[0:3]
        base_lin_vel_body = quat_rotate_inverse_np(quat_isaac, base_lin_vel_world)
        
        # 3. 基座角速度 (world → body frame)
        base_ang_vel_world = data.qvel[3:6]
        base_ang_vel_body = quat_rotate_inverse_np(quat_isaac, base_ang_vel_world)
        
        # 4. 投影重力 (重力在body frame中的方向)
        gravity_world = np.array([0.0, 0.0, -1.0])
        projected_gravity = quat_rotate_inverse_np(quat_isaac, gravity_world)
        
        # 5. 关节状态
        dof_pos = data.qpos[7:13]  # 去掉 freejoint 的 7 维
        dof_vel = data.qvel[6:12]  # 去掉 freejoint 的 6 维
        
        # 6. 位置误差 (wheel 清零)
        dof_pos_err = (dof_pos - cfg.default_dof_pos).copy()
        dof_pos_err[cfg.wheel_indices] = 0.0
        
        # 7. 更新历史
        self.dof_pos_err_history.append(dof_pos_err.copy())
        self.dof_vel_history.append(dof_vel.copy())
        
        # 8. 拼接观测
        obs = np.concatenate([
            base_lin_vel_body * cfg.lin_vel_scale,          # 3
            base_ang_vel_body * cfg.ang_vel_scale,          # 3
            projected_gravity,                               # 3
            commands[:3] * cfg.commands_scale,               # 3
            dof_pos_err * cfg.dof_pos_scale,                # 6
            dof_vel * cfg.dof_vel_scale,                    # 6
            self.last_actions,                               # 6
        ])  # 30
        
        # 历史
        pos_err_hist = np.concatenate(list(self.dof_pos_err_history)) * cfg.dof_pos_scale  # 60
        vel_hist = np.concatenate(list(self.dof_vel_history)) * cfg.dof_vel_scale          # 60
        
        obs = np.concatenate([obs, pos_err_hist, vel_hist])  # 150
        
        # 裁剪
        obs = np.clip(obs, -cfg.clip_obs, cfg.clip_obs)
        
        return obs


# ========================== 键盘控制器 ==========================

class KeyboardCommand:
    """
    方向键控制速度命令 (非阻塞, 终端原始模式)
    
    操作:
      ↑/↓: 前进/后退
      ←/→: 左转/右转
      1:   灵敏度档1 (低)  vx±0.05  yaw±0.1
      2:   灵敏度档2 (中)  vx±0.1   yaw±0.3   [默认]
      3:   灵敏度档3 (高)  vx±0.2   yaw±0.5
      4:   灵敏度档4 (极高) vx±0.5  yaw±1.0
      X:   停止 (速度归零)
      Q:   退出
    """
    # 灵敏度档位: (vx_step, yaw_step)
    SENSITIVITY = {
        '1': (0.05, 0.1,  '低'),
        '2': (0.1,  0.3,  '中'),
        '3': (0.2,  0.5,  '高'),
        '4': (0.5,  1.0,  '极高'),
    }
    
    def __init__(self, initial_vx=0.0, initial_yaw=0.0):
        self.cmd_vx = initial_vx
        self.cmd_vy = 0.0
        self.cmd_yaw = initial_yaw
        
        # 速度限幅
        self.vx_max = 1.0
        self.vx_min = -1.0
        self.yaw_max = 2.0
        self.yaw_min = -2.0
        
        # 默认档位2
        self.vx_step = 0.1
        self.yaw_step = 0.3
        self.gear = '2'
        
        self.running = True
        self._thread = None
        self._old_settings = None
    
    def start(self):
        """启动键盘监听线程"""
        try:
            import termios, tty
            self._old_settings = termios.tcgetattr(sys.stdin)
        except (ImportError, termios.error):
            print("[KeyboardCommand] 终端不支持原始模式, 键盘控制不可用")
            return
        
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        print("\n[KeyboardCommand] 键盘控制已启动:")
        print("  ↑/↓: 前进/后退    ←/→: 左转/右转")
        print("  1-4: 灵敏度档位    X: 停止    Q: 退出")
        print(f"  当前: vx={self.cmd_vx:.2f}, yaw={self.cmd_yaw:.2f}, 档位={self.gear}(中)\n")
    
    def _listen(self):
        """后台线程: 读取键盘输入 (支持方向键 ESC 序列)"""
        import termios, tty
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                ch = sys.stdin.read(1)
                
                # 方向键: ESC [ A/B/C/D
                if ch == '\x1b':
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'A':    # ↑ 前进
                            self.cmd_vx = min(self.cmd_vx + self.vx_step, self.vx_max)
                        elif ch3 == 'B':  # ↓ 后退
                            self.cmd_vx = max(self.cmd_vx - self.vx_step, self.vx_min)
                        elif ch3 == 'D':  # ← 左转
                            self.cmd_yaw = min(self.cmd_yaw + self.yaw_step, self.yaw_max)
                        elif ch3 == 'C':  # → 右转
                            self.cmd_yaw = max(self.cmd_yaw - self.yaw_step, self.yaw_min)
                # 灵敏度档位
                elif ch in self.SENSITIVITY:
                    self.gear = ch
                    self.vx_step, self.yaw_step, name = self.SENSITIVITY[ch]
                    print(f"\r  [档位] {ch}({name}): vx±{self.vx_step}  yaw±{self.yaw_step}                    ", end='', flush=True)
                    continue
                elif ch in ('x', 'X'):
                    self.cmd_vx = 0.0
                    self.cmd_yaw = 0.0
                elif ch in ('q', 'Q'):
                    self.running = False
                    break
                else:
                    continue
                
                print(f"\r  [CMD] vx={self.cmd_vx:+.2f} m/s | yaw={self.cmd_yaw:+.2f} rad/s | 档{self.gear}    ", end='', flush=True)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
    
    def get_commands(self):
        """返回当前命令 [vx, vy, yaw]"""
        return np.array([self.cmd_vx, self.cmd_vy, self.cmd_yaw])
    
    def stop(self):
        """停止键盘监听"""
        self.running = False
        if self._old_settings is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except:
                pass


# ========================== 主程序 ==========================

def main():
    parser = argparse.ArgumentParser(description="Bubble Sim2Sim: IsaacGym → MuJoCo")
    parser.add_argument('--load_run', type=str, required=True,
                        help='训练 run 的文件夹名 (在 logs/flat_bubble/ 下)')
    parser.add_argument('--checkpoint', type=int, default=-1,
                        help='checkpoint 迭代数, -1=最新')
    parser.add_argument('--cmd_vx', type=float, default=0.0,
                        help='前进速度命令 (m/s)')
    parser.add_argument('--cmd_yaw', type=float, default=0.0,
                        help='偏航角速度命令 (rad/s)')
    parser.add_argument('--duration', type=float, default=20.0,
                        help='仿真时长 (s)')
    parser.add_argument('--no_render', action='store_true',
                        help='无可视化模式')
    parser.add_argument('--record', action='store_true',
                        help='录制数据到 csv')
    parser.add_argument('--wheel_speed', type=float, default=None,
                        help='覆盖 wheel_speed (如果训练时不同)')
    parser.add_argument('--keyboard', action='store_true',
                        help='启用键盘 WASD 控制速度命令')
    args = parser.parse_args()
    
    # ---- 加载策略 ----
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "logs", "flat_bubble")
    actor = load_policy(log_dir, args.load_run, args.checkpoint)
    
    # ---- 覆盖配置 ----
    if args.wheel_speed is not None:
        Cfg.wheel_speed = args.wheel_speed
        print(f"[Sim2Sim] wheel_speed overridden to {Cfg.wheel_speed}")
    
    # ---- 加载 MuJoCo 模型 ----
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bubble.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 设置初始姿态
    data.qpos[0:3] = [0.0, 0.0, 0.18]   # 位置
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 四元数 wxyz (直立)
    data.qpos[7:13] = Cfg.default_dof_pos  # 关节角度
    mujoco.mj_forward(model, data)
    
    # ---- 初始化 ----
    obs_builder = ObservationBuilder()
    commands = np.array([args.cmd_vx, 0.0, args.cmd_yaw])
    
    # ---- 键盘控制 ----
    kb_ctrl = None
    if args.keyboard:
        kb_ctrl = KeyboardCommand(initial_vx=args.cmd_vx, initial_yaw=args.cmd_yaw)
    
    total_steps = int(args.duration / Cfg.dt)
    policy_steps = int(args.duration / Cfg.policy_dt)
    
    # 键盘模式: 无限时长
    if kb_ctrl:
        total_steps = int(1e9)  # 实际上靠 Q 键或关闭 viewer 退出
        args.duration = float('inf')
    
    print(f"\n[Sim2Sim] === 配置摘要 ===")
    print(f"  物理 dt: {Cfg.dt}s, 策略 dt: {Cfg.policy_dt}s (decimation={Cfg.decimation})")
    print(f"  命令: vx={args.cmd_vx}, yaw={args.cmd_yaw}")
    print(f"  wheel_speed: {Cfg.wheel_speed}")
    print(f"  仿真时长: {args.duration}s ({policy_steps} policy steps)")
    print(f"  关节顺序: left_thigh, left_knee, left_wheel, right_thigh, right_knee, right_wheel")
    print()
    
    # ---- 数据记录 ----
    log_data = {
        'time': [], 'base_height': [], 'pitch': [], 'roll': [],
        'vx': [], 'vy': [], 'yaw_rate': [],
        'dof_pos': [], 'dof_vel': [], 'torques': [],
    }
    
    # ---- 仿真循环 ----
    action = np.zeros(Cfg.num_actions)
    step_count = 0
    
    def sim_step():
        """在 viewer callback 或 headless 循环中调用"""
        nonlocal action, step_count
        
        if step_count >= total_steps:
            return
        
        # 键盘模式: 实时更新命令
        if kb_ctrl:
            if not kb_ctrl.running:
                return  # Q 键退出
            commands[:] = kb_ctrl.get_commands()
        
        # 每 decimation 步执行一次策略
        if step_count % Cfg.decimation == 0:
            # 构建观测
            obs = obs_builder.build(data, commands)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                action_tensor = actor(obs_tensor)
            action = action_tensor.squeeze(0).numpy()
            action = np.clip(action, -Cfg.clip_actions, Cfg.clip_actions)
            
            # 更新 last_actions
            obs_builder.last_actions = action.copy()
        
        # 计算力矩 (每个物理步都施加)
        dof_pos = data.qpos[7:13]
        dof_vel = data.qvel[6:12]
        torques = compute_torques_b2w(action, dof_pos, dof_vel)
        
        # 施加力矩
        data.ctrl[:] = torques
        
        # 物理步进
        mujoco.mj_step(model, data)
        
        # 记录
        if step_count % Cfg.decimation == 0:
            quat_mj = data.qpos[3:7]
            quat_isaac = mujoco_quat_to_isaac(quat_mj)
            base_lin_vel_body = quat_rotate_inverse_np(quat_isaac, data.qvel[0:3])
            base_ang_vel_body = quat_rotate_inverse_np(quat_isaac, data.qvel[3:6])
            proj_grav = quat_rotate_inverse_np(quat_isaac, np.array([0, 0, -1.0]))
            
            log_data['time'].append(step_count * Cfg.dt)
            log_data['base_height'].append(data.qpos[2])
            log_data['pitch'].append(np.arcsin(-proj_grav[0]))  # 近似
            log_data['roll'].append(np.arcsin(proj_grav[1]))
            log_data['vx'].append(base_lin_vel_body[0])
            log_data['vy'].append(base_lin_vel_body[1])
            log_data['yaw_rate'].append(base_ang_vel_body[2])
            log_data['dof_pos'].append(dof_pos.copy())
            log_data['dof_vel'].append(dof_vel.copy())
            log_data['torques'].append(torques.copy())
        
        step_count += 1
    
    if args.no_render:
        # ---- 无渲染模式 ----
        print("[Sim2Sim] Running headless...")
        t0 = time.time()
        for _ in range(total_steps):
            sim_step()
        elapsed = time.time() - t0
        print(f"[Sim2Sim] Done in {elapsed:.2f}s (real-time ratio: {args.duration/elapsed:.1f}x)")
    else:
        # ---- 可视化模式 ----
        print("[Sim2Sim] Launching MuJoCo viewer...")
        print("  按 ESC 退出, 空格暂停")
        print("  鼠标右键拖动旋转, 滚轮缩放")
        if kb_ctrl:
            print("  [键盘控制] W/S前后 A/D转向 X停止 Q退出")
        
        # 启动键盘监听 (必须在 viewer 前启动)
        if kb_ctrl:
            kb_ctrl.start()
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 设置相机
            viewer.cam.distance = 0.6
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 160
            viewer.cam.lookat[:] = [0.0, 0.0, 0.12]
            
            # 渲染质量设置
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
            
            # 渲染帧率控制: 目标 60fps
            render_interval = max(1, int(1.0 / (60.0 * Cfg.dt)))  # 每N物理步渲染一次
            
            while viewer.is_running() and step_count < total_steps:
                # 键盘模式检查退出
                if kb_ctrl and not kb_ctrl.running:
                    break
                
                step_start = time.time()
                
                sim_step()
                
                # 相机跟随机器人
                viewer.cam.lookat[0] = data.qpos[0]
                viewer.cam.lookat[1] = data.qpos[1]
                viewer.cam.lookat[2] = 0.12
                
                # 高帧率渲染
                if step_count % render_interval == 0:
                    viewer.sync()
                
                # 实时同步
                elapsed = time.time() - step_start
                sleep_time = Cfg.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
    
    # ---- 清理键盘 ----
    if kb_ctrl:
        kb_ctrl.stop()
    
    # ---- 结果分析 ----
    print(f"\n[Sim2Sim] === 结果统计 ===")
    times = np.array(log_data['time'])
    heights = np.array(log_data['base_height'])
    pitches = np.array(log_data['pitch'])
    vxs = np.array(log_data['vx'])
    
    # 取稳态 (后半段)
    half = len(times) // 2
    print(f"  base_height: {heights[half:].mean():.4f} ± {heights[half:].std():.4f} m  (target: 0.14)")
    print(f"  pitch:       {pitches[half:].mean():.4f} ± {pitches[half:].std():.4f} rad")
    print(f"  vx:          {vxs[half:].mean():.4f} ± {vxs[half:].std():.4f} m/s  (cmd: {args.cmd_vx})")
    print(f"  存活:        {times[-1]:.1f}s / {args.duration}s")
    
    # 检查是否倒了
    if heights[-1] < 0.05:
        print(f"  ⚠️  机器人可能已经倒下 (最终高度: {heights[-1]:.4f}m)")
    
    # ---- 保存数据 & 画图 ----
    if args.record or True:  # 默认画图
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
            fig.suptitle(f'Sim2Sim: {args.load_run} | cmd_vx={args.cmd_vx} | wheel_speed={Cfg.wheel_speed}', fontsize=12)
            
            # Base Height
            axes[0, 0].plot(times, heights, label='measured')
            axes[0, 0].axhline(y=0.14, color='r', linestyle='--', label='target')
            axes[0, 0].set_title('Base Height')
            axes[0, 0].set_ylabel('Height (m)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Pitch
            axes[0, 1].plot(times, pitches)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_title('Pitch')
            axes[0, 1].set_ylabel('Angle (rad)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Velocity Tracking
            axes[1, 0].plot(times, vxs, label='actual_vx')
            axes[1, 0].axhline(y=args.cmd_vx, color='r', linestyle='--', label=f'cmd_vx={args.cmd_vx}')
            axes[1, 0].set_title('Velocity Tracking')
            axes[1, 0].set_ylabel('Velocity (m/s)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Torques
            torques_arr = np.array(log_data['torques'])
            joint_names = ['L_thigh', 'L_knee', 'L_wheel', 'R_thigh', 'R_knee', 'R_wheel']
            for i in range(6):
                axes[1, 1].plot(times, torques_arr[:, i], label=joint_names[i], alpha=0.7)
            axes[1, 1].set_title('Torques')
            axes[1, 1].set_ylabel('Torque (N·m)')
            axes[1, 1].legend(fontsize=7)
            axes[1, 1].grid(True, alpha=0.3)
            
            # DOF Positions
            dof_pos_arr = np.array(log_data['dof_pos'])
            for i in [0, 1, 3, 4]:  # 只画腿部
                axes[2, 0].plot(times, dof_pos_arr[:, i], label=joint_names[i], alpha=0.7)
            axes[2, 0].set_title('Joint Positions (legs)')
            axes[2, 0].set_ylabel('Angle (rad)')
            axes[2, 0].set_xlabel('Time (s)')
            axes[2, 0].legend(fontsize=7)
            axes[2, 0].grid(True, alpha=0.3)
            
            # Yaw Rate
            yaw_rates = np.array(log_data['yaw_rate'])
            axes[2, 1].plot(times, yaw_rates, label='actual_yaw')
            axes[2, 1].axhline(y=args.cmd_yaw, color='r', linestyle='--', label=f'cmd_yaw={args.cmd_yaw}')
            axes[2, 1].set_title('Yaw Rate')
            axes[2, 1].set_ylabel('Angular vel (rad/s)')
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     f"sim2sim_{args.load_run.replace('/', '_')}.png")
            plt.savefig(save_path, dpi=150)
            print(f"\n[Sim2Sim] 图表已保存: {save_path}")
            
        except ImportError:
            print("[Sim2Sim] matplotlib not found, skipping plots")


if __name__ == '__main__':
    main()
