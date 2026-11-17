#!/usr/bin/env python3
"""把 TensorBoard event 的全部 scalar 导入 wandb(一次性,跑完即退)。"""
import os, sys, collections
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import wandb

TB_DIR = sys.argv[1]
PROJECT = sys.argv[2]
ENTITY = sys.argv[3]
RUN_NAME = sys.argv[4]
# 逗号分隔的前缀,命中则跳过(默认排除 router/* 的高基数专家 dump,避免 payload 过大传不上)
EXCLUDE = [p for p in os.environ.get("EXCLUDE_PREFIX", "router/").split(",") if p]

print(f"[load] {TB_DIR}", flush=True)
ea = EventAccumulator(TB_DIR, size_guidance={'scalars': 0})
ea.Reload()
all_tags = ea.Tags().get('scalars', [])
tags = [t for t in all_tags if not any(t.startswith(p) for p in EXCLUDE)]
print(f"[load] {len(all_tags)} tags total, keep {len(tags)} (exclude prefixes: {EXCLUDE})", flush=True)

# 按 step 归并:step -> {tag: value}
by_step = collections.defaultdict(dict)
for t in tags:
    for s in ea.Scalars(t):
        by_step[s.step][t] = s.value

steps = sorted(by_step)
print(f"[load] steps {steps[0]}..{steps[-1]} ({len(steps)} points)", flush=True)

run = wandb.init(project=PROJECT, entity=ENTITY, name=RUN_NAME,
                 settings=wandb.Settings(_disable_stats=True))
for st in steps:
    wandb.log(by_step[st], step=st)
print(f"[done] logged {len(steps)} steps -> {run.url}", flush=True)
run.finish()
print("[finish] ok", flush=True)
