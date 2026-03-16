#!/usr/bin/env python3
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator('logs/skill_bubble/Mar11_14-15-03_trimesh_exp200')
ea.Reload()
tags = sorted(ea.Tags()['scalars'])

for tag in tags:
    evts = ea.Scalars(tag)
    if evts:
        n = len(evts)
        idxs = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
        parts = []
        for i in idxs:
            parts.append(f"s{evts[i].step}={evts[i].value:.4f}")
        print(f"{tag}: {' | '.join(parts)}")
