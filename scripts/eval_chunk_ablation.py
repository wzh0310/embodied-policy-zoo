#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分块长度消融评估 —— 在 eval.py 基础上增加动作平滑度指标。

复用与 eval.py 完全一致的策略加载、预处理、rollout 逻辑，仅在 rollout 中
逐帧记录下发的 raw action，用于计算动作平滑度：
  * mean_jerk       : 相邻动作差分的平均 L2 范数（越小越平滑，单位 m/step）
  * direction_flip  : 相邻动作方向点积<0 的比例（越小越平滑）
  * mean_action_norm: 平均动作幅值（辅助判断 chunk 大小对步幅的影响）
其余指标（success_rate / steps / dist）与 eval.py 定义完全相同，确保可比。

同一测试集：固定 seeds [100,200,300] × 10 = 30 rollouts，初始偏移注入逻辑
与 eval.py 完全一致，保证四组模型在完全相同的 30 个初始态上评估。
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

import lerobot_push_cube_env as lc  # noqa
from lerobot.envs import make_env_config  # noqa
from lerobot.policies.act.configuration_act import ACTConfig  # noqa
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa
from push_cube_env import PushCubeEnv, TARGET_XY  # noqa


def load_policy(ckpt_dir):
    pretrained = os.path.join(ckpt_dir, "pretrained_model")
    if not os.path.isdir(pretrained):
        pretrained = ckpt_dir
    cfg = ACTConfig.from_pretrained(pretrained)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = make_env_config("push_cube")
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, pretrained_path=pretrained)
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=pretrained)
    return policy, pre, post, cfg


def run_episode(env, policy, pre, post, device, seed, max_steps):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    env.data.qpos[9] += float(rng.uniform(-0.030, 0.030))
    env.data.qpos[10] += float(rng.uniform(-0.035, 0.035))
    env.data.qvel[9:15] = 0.0
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    env._prev_dist = env._cube_target_dist()
    actions = []
    for t in range(max_steps):
        img_chw = np.transpose(obs["image"], (2, 0, 1)).astype(np.float32) / 255.0
        canon = {
            "observation.images.main": torch.from_numpy(
                np.ascontiguousarray(img_chw)[None]).to(device),
            "observation.state": torch.from_numpy(
                np.ascontiguousarray(obs["state"]).astype(np.float32)[None]).to(device),
        }
        with torch.inference_mode():
            processed = pre(canon)
            action = policy.select_action(processed)
            action = post(action)
        if isinstance(action, dict):
            action = action["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        a = np.asarray(action).reshape(-1)[:4]
        raw = np.clip(a, env.action_space.low, env.action_space.high)
        actions.append(raw[:3].copy())  # 只取 xyz 位移维度算平滑度（夹爪维度恒 0）
        obs, reward, term, trunc, info = env.step(raw)
        if term or trunc:
            break
    cube = env.data.xpos[env._cube_body]
    dist = float(np.linalg.norm(cube[:2] - TARGET_XY))
    success = bool(dist < 0.03)
    steps = int(info.get("steps", t + 1))

    # 动作平滑度
    arr = np.asarray(actions)  # (T,3)
    if len(arr) >= 2:
        d1 = np.diff(arr, axis=0)                       # 一阶差分 = 相邻动作变化
        mean_disp = float(np.mean(np.linalg.norm(d1, axis=1)))
        norms = np.linalg.norm(arr, axis=1)
        mean_norm = float(np.mean(norms))
        flip = 0.0
        if len(arr) >= 2:
            dots = []
            for i in range(len(arr) - 1):
                na, nb = norms[i], norms[i + 1]
                if na > 1e-6 and nb > 1e-6:
                    dots.append(float(np.dot(arr[i], arr[i + 1]) / (na * nb)))
            flip = float(np.mean([1.0 if d < 0 else 0.0 for d in dots])) if dots else 0.0
    else:
        mean_disp, mean_norm, flip = 0.0, 0.0, 0.0
    return {"success": success, "steps": steps, "dist": dist,
            "mean_jerk": mean_disp, "direction_flip": flip,
            "mean_action_norm": mean_norm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True, help="e.g. chunk4 / chunk20")
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--episodes-per-seed", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--out", default="chunk_ablation_metrics.json")
    args = ap.parse_args()

    out_path = args.out if os.path.isabs(args.out) else os.path.join(
        REPO, "results/metrics", args.out)
    policy, pre, post, cfg = load_policy(args.checkpoint)
    device = next(policy.parameters()).device
    print("[eval] label", args.label, "| chunk_size", getattr(cfg, "chunk_size", "?"),
          "| n_action_steps", getattr(cfg, "n_action_steps", "?"), "| device", device)

    env = PushCubeEnv(image_obs=True, max_episode_steps=args.max_steps)
    results = []
    for seed in args.seeds:
        for ep in range(args.episodes_per_seed):
            if hasattr(policy, "reset"):
                policy.reset()
            m = run_episode(env, policy, pre, post, device,
                            seed=seed * 1000 + ep, max_steps=args.max_steps)
            m["seed"] = seed
            m["episode"] = ep
            results.append(m)
    env.close()

    succ = [r for r in results if r["success"]]
    summary = {
        "label": args.label,
        "chunk_size": getattr(cfg, "chunk_size", None),
        "n_action_steps": getattr(cfg, "n_action_steps", None),
        "checkpoint": args.checkpoint,
        "n_rollouts": len(results),
        "success_rate": len(succ) / len(results),
        "avg_completion_steps": float(np.mean([r["steps"] for r in succ])) if succ else None,
        "avg_distance_error": float(np.mean([r["dist"] for r in results])),
        "mean_jerk": float(np.mean([r["mean_jerk"] for r in results])),
        "direction_flip": float(np.mean([r["direction_flip"] for r in results])),
        "mean_action_norm": float(np.mean([r["mean_action_norm"] for r in results])),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("[RESULT] %-8s succ=%.3f (%d/%d) dist=%.4f steps=%s jerk=%.5f flip=%.3f anim=%.5f" % (
        args.label, summary["success_rate"], len(succ), len(results),
        summary["avg_distance_error"], summary["avg_completion_steps"],
        summary["mean_jerk"], summary["direction_flip"], summary["mean_action_norm"]))
    print("EVAL_DONE", args.label)


if __name__ == "__main__":
    main()
