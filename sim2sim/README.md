# Bubble Sim2Sim: IsaacGym → MuJoCo

## 快速开始

```bash
# 1. 无渲染测试 (最快验证策略能否在 MuJoCo 中站立)
python sim2sim/run_mujoco.py --load_run Mar10_14-56-35_small_torque_dixing --no_render --cmd_vx 0.0

# 2. 可视化 (观察机器人行为)
python sim2sim/run_mujoco.py --load_run Mar10_14-56-35_small_torque_dixing --cmd_vx 0.3

# 3. 指定 checkpoint
python sim2sim/run_mujoco.py --load_run Mar10_14-56-35_small_torque_dixing --checkpoint 1000 --cmd_vx 0.3

# 4. 覆盖 wheel_speed (如果训练时用的不同值)
python sim2sim/run_mujoco.py --load_run Mar10_14-56-35_small_torque_dixing --wheel_speed 1.0 --cmd_vx 0.3
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--load_run` | 必填 | logs/flat_bubble/ 下的 run 文件夹名 |
| `--checkpoint` | -1 | checkpoint 迭代数, -1=最新 |
| `--cmd_vx` | 0.0 | 前进速度命令 (m/s) |
| `--cmd_yaw` | 0.0 | 偏航角速度命令 (rad/s) |
| `--duration` | 20.0 | 仿真时长 (s) |
| `--no_render` | False | 无可视化模式 |
| `--wheel_speed` | None | 覆盖 wheel_speed |

## ⚠️ 注意事项

1. **wheel_speed 必须与训练时一致**：`run_mujoco.py` 中默认 `wheel_speed=2.0`，如果你训练时改成了 1.0，请用 `--wheel_speed 1.0`
2. **输出**：自动保存 `sim2sim/sim2sim_<run_name>.png` 对比图
3. **Sim2Sim Gap 来源**：接触模型差异是主要来源，可通过调整 `bubble.xml` 中的 `friction` 参数缩小
