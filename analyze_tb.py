#!/usr/bin/env python3
"""Read TensorBoard logs and print key metrics for analysis."""
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_base = "logs/skill_bubble"

# Analyze recent important runs
runs = [
    "Mar11_01-20-42_com_height016",
    "Mar11_01-54-38_kd2",
    "Mar11_02-17-27_hirght2",
]

for run_name in runs:
    run_path = os.path.join(log_base, run_name)
    if not os.path.exists(run_path):
        print(f"SKIP: {run_name}")
        continue

    print(f"\n{'='*70}")
    print(f"RUN: {run_name}")
    print(f"{'='*70}")

    ea = EventAccumulator(run_path)
    ea.Reload()
    available_tags = ea.Tags().get("scalars", [])

    for tag in sorted(available_tags):
        events = ea.Scalars(tag)
        n = len(events)
        if n == 0:
            continue
        idx = sorted(set([min(i, n - 1) for i in [0, n // 10, n // 4, n // 2, 3 * n // 4, n - 1]]))
        vals = [(events[i].step, round(events[i].value, 4)) for i in idx]
        print(f"  {tag}: (n={n})")
        print(f"    steps = {[v[0] for v in vals]}")
        print(f"    vals  = {[v[1] for v in vals]}")
