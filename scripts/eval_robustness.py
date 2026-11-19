#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACT 自动化评估脚本 —— embodied-policy-zoo 统一评估入口。

评估协议（固定种子，量化对比）
------------------------------
* 3 个固定种子 × 每种子 10 回合 = 30 次离屏 rollout。
* 关键指标：
    1) success_rate         —— 立方体进入目标区域的回合占比（核心指标）
    2) avg_completion_steps —— 成功回合的平均执行步数
    3) avg_distance_error   —— 回合结束立方体到目标的平均距离（米）
* 全程离屏渲染（MUJOCO_GL=egl），不弹出可视化窗口。
* 可选录制若干成功回合的 MP4。

策略加载方式与 LeRobot 官方评估完全一致：从 checkpoint 读取 config，
make_policy 加载权重，并使用其自带的 pre/post processor，确保预处理与训练
严格一致（不做任何手工二次归一化）。

关键参数说明
------------
--checkpoint       checkpoint 目录（含 pretrained_model/）。
--seeds            固定随机种子列表，评估可复现性的基础。
--episodes-per-seed 每种子的回合数。
--max-steps        单回合最大步数（环境 episode_length=50）。
--out              指标 JSON 输出路径（默认本仓 results/metrics/）。
--record-success   最多录制 N 个成功回合到 results/videos/。

用法（在 lerobot conda 环境）
---------------------------
    python scripts/eval.py \
        --checkpoint /data/wzh/robot/outputs/act_push_cube_full/checkpoints/004000 \
        --out results/metrics/baseline_metrics.json
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np
import torch

# ---- 跨仓路径解析 ----
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STUDIO_LAB = os.environ.get("STUDIO_LAB_ROOT", "/data/wzh/robot/mujoco-studio-lab")
if not os.path.isdir(STUDIO_LAB):
    raise SystemExit(
        f"[eval] 找不到兄弟仓库 mujoco-studio-lab: {STUDIO_LAB}\n"
        "请设置环境变量 STUDIO_LAB_ROOT 指向其绝对路径。"
    )
for p in (REPO, STUDIO_LAB, os.path.join(STUDIO_LAB, "envs")):
    if p not in sys.path:
        sys.path.insert(0, p)

import lerobot_push_cube_env as lc  # noqa: E402,F401
from lerobot.envs import make_env_config  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from push_cube_env import PushCubeEnv, TARGET_XY  # noqa: E402


def load_policy_from_checkpoint(ckpt_dir):
    """从 checkpoint 加载 ACT 策略及其 pre/post 处理器。"""
    pretrained = os.path.join(ckpt_dir, "pretrained_model")
    if not os.path.isdir(pretrained):
        pretrained = ckpt_dir
    cfg = ACTConfig.from_pretrained(pretrained)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = make_env_config("push_cube")
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, pretrained_path=pretrained)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=pretrained,
    )
    return policy, preprocessor, postprocessor, cfg


def run_episode(env, policy, pre, post, device, seed, max_steps, render_frames=False):
    """单个 rollout：返回指标与可选帧序列。

    注意：基础 env.reset() 使用固定 home keyframe（立方体位置固定）。为让 30 次
    测试真正独立并匹配训练数据分布，这里按 seed 注入与数据生成同分布的立方体
    初始偏移（x ±0.03 / y ±0.035），再重新 mj_forward。
    """
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    env.data.qpos[9] += float(rng.uniform(-0.050, 0.050))
    env.data.qpos[10] += float(rng.uniform(-0.065, 0.065))
    env.data.qvel[9:15] = 0.0
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    env._prev_dist = env._cube_target_dist()
    frames = []
    for t in range(max_steps):
        # LeRobot 数据加载时图像为 float32 [0,1]（uint8/255）；processor 的
        # normalizer 从输入张量 dtype 推导 stats，必须传 float32 否则 uint8 溢出。
        # 图像布局为 CHW (B,C,H,W)：normalizer stats 形状 (C,) 作用在 dim 1。
        img_chw = np.transpose(obs["image"], (2, 0, 1)).astype(np.float32) / 255.0
        canon = {
            "observation.images.main": torch.from_numpy(
                np.ascontiguousarray(img_chw)[None]).to(device),
            "observation.state": torch.from_numpy(
                np.ascontiguousarray(obs["state"]).astype(np.float32)[None]).to(device),
        }
        # processor -> policy -> postprocessor。本 LeRobot 构建中 select_action /
        # postprocessor 直接返回 tensor (B, action_dim)，不是 dict。
        with torch.inference_mode():
            processed = pre(canon)
            action = policy.select_action(processed)
            action = post(action)
        if isinstance(action, dict):
            action = action["action"]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        a = np.asarray(action).reshape(-1)[:4]
        # postprocessor（反归一化）输出的已经是环境 action_space 内的 raw 动作。
        # 切勿再做 a*std+mean —— 会把 dy/dz 压向数据均值导致推偏。
        raw = np.clip(a, env.action_space.low, env.action_space.high)
        obs, reward, term, trunc, info = env.step(raw)
        if render_frames:
            frames.append(np.ascontiguousarray(env.render()))
        if term or trunc:
            break
    cube = env.data.xpos[env._cube_body]
    dist = float(np.linalg.norm(cube[:2] - TARGET_XY))
    success = bool(dist < 0.03)
    steps = int(info.get("steps", t + 1))
    return {"success": success, "steps": steps, "dist": dist,
            "total_reward": float(reward)}, frames


def resolve(path, default_subdir):
    """相对路径基于本仓库根解析。"""
    if os.path.isabs(path):
        return path
    return os.path.join(REPO, default_subdir, path) if not path.startswith(default_subdir) \
        else os.path.join(REPO, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--episodes-per-seed", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--out", default="baseline_metrics.json")
    ap.add_argument("--record-success", type=int, default=0)
    args = ap.parse_args()

    out_path = os.path.join(REPO, "results/metrics", args.out) \
        if not os.path.isabs(args.out) else args.out
    video_dir = os.path.join(REPO, "results/videos")

    print("=" * 64)
    print("[eval] checkpoint =", args.checkpoint)
    print("[eval] studio_root =", STUDIO_LAB)
    policy, pre, post, cfg = load_policy_from_checkpoint(args.checkpoint)
    device = next(policy.parameters()).device
    print("[eval] device", device, "| seeds", args.seeds,
          "| ep/seed", args.episodes_per_seed)

    env = PushCubeEnv(image_obs=True, max_episode_steps=args.max_steps)
    results = []
    saved_videos = 0
    for seed in args.seeds:
        for ep in range(args.episodes_per_seed):
            if hasattr(policy, "reset"):
                policy.reset()  # 清空时序集成队列，避免跨回合状态泄漏
            rec = saved_videos < args.record_success
            metrics, frames = run_episode(
                env, policy, pre, post, device,
                seed=seed * 1000 + ep, max_steps=args.max_steps,
                render_frames=rec)
            metrics["seed"] = seed
            metrics["episode"] = ep
            results.append(metrics)
            if rec and metrics["success"] and frames:
                from lerobot.utils.io_utils import write_video
                os.makedirs(video_dir, exist_ok=True)
                vp = os.path.join(video_dir, "eval_success_%d_%d.mp4" % (seed, ep))
                write_video(vp, np.stack(frames), fps=50)
                print("[eval] wrote video", vp)
                saved_videos += 1
            print("[eval] seed=%d ep=%d success=%s steps=%d dist=%.4f" % (
                seed, ep, metrics["success"], metrics["steps"], metrics["dist"]))
    env.close()

    successes = [r for r in results if r["success"]]
    n_total = len(results)
    summary = {
        "checkpoint": args.checkpoint,
        "n_rollouts": n_total,
        "success_rate": len(successes) / n_total if n_total else 0.0,
        "avg_completion_steps": float(np.mean([r["steps"] for r in successes])) if successes else None,
        "avg_distance_error": float(np.mean([r["dist"] for r in results])),
        "avg_distance_error_success": float(np.mean([r["dist"] for r in successes])) if successes else None,
        "avg_total_reward": float(np.mean([r["total_reward"] for r in results])),
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes_per_seed,
        "per_episode": results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("=" * 64)
    print("[eval] SUCCESS_RATE = %.3f (%d/%d)" % (
        summary["success_rate"], len(successes), n_total))
    print("[eval] avg_completion_steps =", summary["avg_completion_steps"])
    print("[eval] avg_distance_error = %.4f m" % summary["avg_distance_error"])
    print("[eval] metrics ->", out_path)
    print("EVAL DONE")


if __name__ == "__main__":
    main()
