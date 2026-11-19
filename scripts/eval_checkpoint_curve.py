#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估各训练 checkpoint 的成功率 —— 绘制"成功率随训练步数变化"曲线。

 对 ACT(chunk40) 与 BC 各自的 1000/2000/3000/4000 四个 checkpoint，在完全相同的
 50 个固定种子下评估（与 BC_VS_ACT_COMPARISON 同一测试集），得到成功率随训练步数
 的变化。rollout 逻辑与主评估脚本逐字一致，保证口径统一。

 用法（lerobot 环境）
--------------------
    python scripts/eval_checkpoint_curve.py --out results/metrics/CHECKPOINT_CURVE.json
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STUDIO_LAB = os.environ.get("STUDIO_LAB_ROOT", os.path.join(os.path.dirname(REPO), "mujoco-studio-lab"))
for p in (REPO, STUDIO_LAB, os.path.join(STUDIO_LAB, "envs")):
    if p not in sys.path:
        sys.path.insert(0, p)

import lerobot_push_cube_env as lc  # noqa: E402,F401
from lerobot.envs import make_env_config  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from push_cube_env import PushCubeEnv, TARGET_XY  # noqa: E402

from eval_bc_comparison import FIXED_SEEDS, load_policy_from_checkpoint, run_episode  # noqa: E402

OUT_ROOT = os.environ.get("OUTPUTS_DIR", os.path.join(os.path.dirname(REPO), "outputs"))
MODELS = {
    "act": os.path.join(OUT_ROOT, "act_chunk40", "checkpoints"),
    "bc": os.path.join(OUT_ROOT, "bc_baseline", "checkpoints"),
}
STEPS = [1000, 2000, 3000, 4000]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/metrics/CHECKPOINT_CURVE.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=FIXED_SEEDS)
    args = ap.parse_args()

    out_path = os.path.join(REPO, "results/metrics", args.out) \
        if not (os.path.isabs(args.out) or args.out.startswith("results/metrics")) \
        else (args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out))

    env = PushCubeEnv(image_obs=True, max_episode_steps=50)
    result = {"protocol": {"n_seeds": len(args.seeds), "max_steps": 50,
                           "success_threshold_m": 0.03}, "models": {}}
    for label, ckpt_root in MODELS.items():
        result["models"][label] = {}
        for step in STEPS:
            ckpt = os.path.join(ckpt_root, "%06d" % step)
            if not os.path.isdir(ckpt):
                print(f"[curve] skip {label}@{step}: no dir")
                continue
            policy, pre, post, cfg = load_policy_from_checkpoint(ckpt)
            device = next(policy.parameters()).device
            successes, dists = 0, []
            for seed in args.seeds:
                if hasattr(policy, "reset"):
                    policy.reset()
                rec = run_episode(env, policy, pre, post, device, seed, 50)
                successes += int(rec["success"])
                dists.append(rec["dist"])
            rate = successes / len(args.seeds)
            result["models"][label][str(step)] = {
                "success_rate": rate, "n_success": successes,
                "n_rollouts": len(args.seeds),
                "avg_distance_error": float(np.mean(dists)),
            }
            print(f"[curve] {label:4s} @{step:5d}  rate={rate:.3f} "
                  f"({successes}/{len(args.seeds)})  dist={np.mean(dists):.4f}m", flush=True)
            del policy, pre, post
            # 每完成一个 checkpoint 就落盘，防止中断丢失
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2)
    env.close()
    print("[curve] wrote", out_path)
    print("CURVE DONE")


if __name__ == "__main__":
    main()
