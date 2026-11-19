#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量录制多个成功回合的演示片段（第三人称 + 第一人称双视角面板）。

 在候选种子上逐个 rollout，收集前 N 个成功回合的完整帧序列（真实策略推理），
 每个回合单独存为临时 mp4 到 results/videos/.clips/，供后续 ffmpeg 合成为
 约 15 秒的演示短视频。双视角：左 global_cam（第三人称全景）、右 top_cam
（喂给策略的第一人称 RGB 观测）。

 用法（lerobot 环境）
--------------------
    python scripts/record_demo_clips.py \
        --checkpoint /path/to/outputs/act_chunk40/checkpoints/004000 \
        --num-clips 4 --out-dir results/videos/.clips
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STUDIO_LAB = os.environ.get("STUDIO_LAB_ROOT", os.path.join(os.path.dirname(REPO), "mujoco-studio-lab"))
for p in (REPO, STUDIO_LAB, os.path.join(STUDIO_LAB, "envs")):
    if p not in sys.path:
        sys.path.insert(0, p)

from record_video import load_policy, rollout_dual  # noqa: E402
from push_cube_env import PushCubeEnv  # noqa: E402
from eval_bc_comparison import FIXED_SEEDS  # noqa: E402

CANDIDATES = FIXED_SEEDS  # 复用评估同分布的 50 个种子，确保找到成功回合


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-clips", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--out-dir", default="results/videos/.clips")
    ap.add_argument("--min-steps", type=int, default=15,
                    help="过滤太短的回合（帧数不足则不选）")
    args = ap.parse_args()

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    pol, pre, post = load_policy(args.checkpoint)
    device = next(pol.parameters()).device
    env = PushCubeEnv(image_obs=True, max_episode_steps=args.max_steps)
    from lerobot.utils.io_utils import write_video

    clips = []
    for seed in CANDIDATES:
        if len(clips) >= args.num_clips:
            break
        if hasattr(pol, "reset"):
            pol.reset()
        frames, dist, success, steps = rollout_dual(env, pol, pre, post, device, seed, args.max_steps)
        if not success or len(frames) < args.min_steps:
            continue
        idx = len(clips)
        out = os.path.join(out_dir, "clip_%02d_seed%d.mp4" % (idx, seed))
        write_video(out, np.stack(frames), fps=50)
        clips.append((out, seed, steps, dist))
        print("[clips] saved clip %d: seed=%d steps=%d dist=%.4f -> %s" % (
            idx, seed, steps, dist, out), flush=True)
    env.close()

    with open(os.path.join(out_dir, "clips_manifest.json"), "w") as f:
        import json
        json.dump([{"path": c[0], "seed": c[1], "steps": c[2], "dist": c[3]} for c in clips],
                  f, indent=2)
    print("[clips] total clips =", len(clips))
    print("CLIPS DONE")


if __name__ == "__main__":
    main()
