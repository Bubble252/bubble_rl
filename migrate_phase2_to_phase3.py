"""
Phase 2 → Phase 3 权重迁移脚本

Phase 2: 150 维观测 (30 base + 120 history)
Phase 3: 152 维观测 (30 base + 2 skill_commands + 120 history)

新增的 2 维 (jump_height, leg_length) 插入在 base(30维) 之后、history(120维) 之前
即 index 12~13 位置 (commands之后)

策略: 对 actor.0.weight 和 critic.0.weight:
  - 前 12 列 (lin_vel 3 + ang_vel 3 + gravity 3 + walk_commands 3) 保持不变
  - 插入 2 列零 (jump_height + leg_length 的新权重)
  - 后 138 列 (dof_pos 6 + dof_vel 6 + actions 6 + history 120) 保持不变
"""

import torch
import os
import shutil

# === 配置 ===
SRC_DIR = "logs/flat_bubble/Mar10_19-46-18_symmetric_wheel_from500"
SRC_CKPT = "model_500.pt"
DST_DIR = "logs/skill_bubble/migrated_from_phase2"  # Phase 3 的 experiment_name 是 skill_bubble
OLD_OBS = 150
NEW_OBS = 152
INSERT_IDX = 12  # 在 index 12 处插入 2 列 (commands 之后)
INSERT_DIM = 2

src_path = os.path.join(SRC_DIR, SRC_CKPT)
print(f"Loading: {src_path}")
d = torch.load(src_path, map_location="cpu", weights_only=True)

sd = d["model_state_dict"]
modified = []

for key in ["actor.0.weight", "critic.0.weight"]:
    w = sd[key]  # (out_features, 150)
    assert w.shape[1] == OLD_OBS, f"{key} shape {w.shape} != expected (*, {OLD_OBS})"
    
    # 拆分 → 插入零列 → 拼接
    w_before = w[:, :INSERT_IDX]         # (512, 12)
    w_after = w[:, INSERT_IDX:]          # (512, 138)
    w_new = torch.zeros(w.shape[0], INSERT_DIM)  # (512, 2) 零初始化
    
    sd[key] = torch.cat([w_before, w_new, w_after], dim=1)  # (512, 152)
    modified.append(f"  {key}: {w.shape} → {sd[key].shape}")

# 重置优化器状态 (维度变了, 旧的 Adam state 不兼容)
d["optimizer_state_dict"] = {}

# 重置迭代计数 (Phase 3 从 0 开始计)
d["iter"] = 0

# 保存
os.makedirs(DST_DIR, exist_ok=True)
dst_path = os.path.join(DST_DIR, "model_0.pt")
torch.save(d, dst_path)

print(f"\n=== 迁移完成 ===")
for m in modified:
    print(m)
print(f"\n输出: {dst_path}")
print(f"\n训练命令:")
print(f"  python legged_gym/scripts/train.py --task=bubble_skill \\")
print(f"    --resume --load_run=migrated_from_phase2 --checkpoint=0")
