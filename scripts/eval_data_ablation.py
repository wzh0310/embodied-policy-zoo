#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估数据量消融三档（20/40/63）最终 checkpoint 的成功率，同一 50 固定种子。

 与 CHECKPOINT_CURVE 共用评估逻辑与测试集，产出 data_ablation 的成功率/步数/距离，
 供绘制"数据量-成功率"折线图。
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STUDIO_LAB = os.environ.get("STUDIO_LAB_ROOT", os.path.join(os.path.dirname(REPO), "mujoco-studio-lab"))
for p in (REPO, os.path.join(REPO, "scripts"), STUDIO_LAB,
          os.path.join(STUDIO_LAB, "envs")):
    if p not in sys.path:
        sys.path.insert(0, p)

from push_cube_env import PushCubeEnv  # noqa: E402
from eval_bc_comparison import FIXED_SEEDS, load_policy_from_checkpoint, run_episode  # noqa: E402

OUT_ROOT = os.environ.get("OUTPUTS_DIR", os.path.join(os.path.dirname(REPO), "outputs"))
CKPTS = {
    20: os.path.join(OUT_ROOT, "act_data20", "checkpoints", "004000"),
    40: os.path.join(OUT_ROOT, "act_data40", "checkpoints", "004000"),
    63: os.path.join(OUT_ROOT, "act_chunk40", "checkpoints", "004000"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/metrics/DATA_ABLATION.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=FIXED_SEEDS)
    args = ap.parse_args()
    out_path = os.path.join(REPO, "results/metrics", args.out) \
        if not (os.path.isabs(args.out) or args.out.startswith("results/metrics")) \
        else (args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out))

    env = PushCubeEnv(image_obs=True, max_episode_steps=50)
    result = {"protocol": {"n_seeds": len(args.seeds), "max_steps": 50,
                           "note": "同一 50 固定种子；63 档复用最优 ACT checkpoint"},
              "models": {}}
    for n, ckpt in CKPTS.items():
        if not os.path.isdir(ckpt):
            print(f"[data] skip n={n}: no checkpoint {ckpt}")
            continue
        policy, pre, post, cfg = load_policy_from_checkpoint(ckpt)
        device = next(policy.parameters()).device
        succ, dists, steps_s = 0, [], []
        for seed in args.seeds:
            if hasattr(policy, "reset"):
                policy.reset()
            rec = run_episode(env, policy, pre, post, device, seed, 50)
            succ += int(rec["success"])
            dists.append(rec["dist"])
            if rec["success"]:
                steps_s.append(rec["steps"])
        result["models"][str(n)] = {
            "n_demos": n, "checkpoint": ckpt,
            "success_rate": succ / len(args.seeds), "n_success": succ,
            "n_rollouts": len(args.seeds),
            "avg_distance_error": float(np.mean(dists)),
            "avg_completion_steps": float(np.mean(steps_s)) if steps_s else None,
        }
        print(f"[data] n={n:3d}  rate={succ/len(args.seeds):.3f} "
              f"({succ}/{len(args.seeds)})  dist={np.mean(dists):.4f}m", flush=True)
        del policy, pre, post
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
    env.close()
    print("[data] wrote", out_path)
    print("DATA ABLATION EVAL DONE")


if __name__ == "__main__":
    main()
