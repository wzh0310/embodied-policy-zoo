#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用推理视频录制工具（面向 SB3 状态策略）。

加载训练好的 SB3 模型，rollout 完整回合并录制 MP4，支持两种视角：
    global  = 全局第三人称视角（global_cam）
    task    = 主观测相机视角（top_cam，与策略输入画面一致）
    both    = 左右并排（默认）

脚本在候选 seed/随机初始位姿上逐个 rollout，选出成功的回合录制，确保交付的是
策略真实推理推动立方体的过程。用 cv2.VideoWriter 编码，不依赖 lerobot。

用法:
    python utils/record_video.py --model <final_model.zip> --view both \
        --seeds 0 1 2 --out-dir assets/videos/ppo
"""
import argparse
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from envs.paths import setup_sys_path  # noqa: E402

setup_sys_path()
import gymnasium as gym  # noqa: E402
import pushcube_envs  # noqa: E402,F401
from stable_baselines3 import PPO  # noqa: E402

GYM_ID = "PushCube-v0"
TARGET_XY = np.array([0.82, -0.12])


def render_frame(model, data, view, r_global, r_task):
    if view == "global":
        r_global.update_scene(data, camera="global_cam")
        return r_global.render()
    r_task.update_scene(data, camera="top_cam")
    if view == "task":
        return r_task.render()
    r_global.update_scene(data, camera="global_cam")
    r_task.update_scene(data, camera="top_cam")
    return np.concatenate([r_global.render(), r_task.render()], axis=1)


def rollout(model, env, seed, max_steps, stochastic, randomize_start, view, fps, set_model_seed=False):
    """返回 (mp4路径或None, success, steps, final_dist)。"""
    if set_model_seed:
        model.set_random_seed(seed)
    obs, _ = env.reset(seed=seed)
    unwrapped = env.unwrapped
    if randomize_start:
        rng = np.random.default_rng(seed)
        unwrapped.data.qpos[9] += float(rng.uniform(-0.03, 0.03))
        unwrapped.data.qpos[10] += float(rng.uniform(-0.035, 0.035))
        unwrapped.data.qvel[9:15] = 0.0
        mujoco.mj_forward(unwrapped.model, unwrapped.data)
        obs = unwrapped._get_obs()
        unwrapped._prev_dist = unwrapped._cube_target_dist()

    import cv2
    r_global = mujoco.Renderer(unwrapped.model, height=224, width=224)
    r_task = mujoco.Renderer(unwrapped.model, height=224, width=224)
    frames, rewards = [], []
    for t in range(max_steps):
        frames.append(render_frame(unwrapped.model, unwrapped.data, view, r_global, r_task))
        action, _ = model.predict(obs, deterministic=not stochastic)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            break
    r_global.close()
    r_task.close()

    cube = unwrapped.data.xpos[unwrapped._cube_body]
    dist = float(np.linalg.norm(cube[:2] - TARGET_XY))
    success = bool(info.get("success", False))
    steps = int(info.get("steps", t + 1))
    return frames, rewards, success, steps, dist


def write_mp4(frames, path, fps):
    import cv2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    # mp4v 编码；宽高需为偶数
    w -= w % 2
    h -= h % 2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f[:h, :w], cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="已保存的 SB3 模型 .zip")
    ap.add_argument("--view", choices=["global", "task", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--stochastic", action="store_true", help="用随机动作采样（增加轨迹多样性）")
    ap.add_argument("--randomize-start", action="store_true", help="随机化立方体初始位姿")
    ap.add_argument("--out-dir", default="assets/videos/ppo_state")
    ap.add_argument("--num-videos", type=int, default=3, help="最多录制几条成功轨迹")
    ap.add_argument("--set-model-seed", action="store_true", help="每个 seed 重置模型 RNG，使随机动作可复现")
    args = ap.parse_args()

    model = PPO.load(args.model)
    env = gym.make(GYM_ID, obs_mode="state", disable_env_checker=True)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir

    saved = []
    for seed in args.seeds:
        if len(saved) >= args.num_videos:
            break
        frames, rewards, success, steps, dist = rollout(
            model, env, seed, args.max_steps, args.stochastic,
            args.randomize_start, args.view, args.fps, args.set_model_seed)
        print("[record] seed=%d success=%s steps=%d dist=%.4f reward=%.3f" % (
            seed, success, steps, dist, sum(rewards)))
        if success:
            out = out_dir / ("success_seed%d.mp4" % seed)
            write_mp4(frames, out, args.fps)
            print("  -> wrote %s" % out)
            saved.append(str(out.relative_to(REPO)))
    env.close()

    if not saved:
        print("[record] 未找到成功回合，未录制。可增加 --seeds 或启用 --randomize-start。")
    else:
        print("[record] 完成，共 %d 条成功视频:" % len(saved))
        for s in saved:
            print("  ", s)


if __name__ == "__main__":
    main()
