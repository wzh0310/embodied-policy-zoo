#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对齐评估脚本 —— BC 基线 vs 最优 ACT, 完全相同的测试集与指标。

公平对比设计
------------
* 两个(checkpoint)模型在同一进程、同一环境实例、同一 50 个固定种子下评估,
  消除环境/渲染/随机性差异, 唯一变量是策略权重本身。
* 每个种子 1 回合 = 50 次离屏 rollout / 模型。立方体初始偏移由种子确定性注入
  (x ±0.030 / y ±0.035), 与数据生成同分布。
* 指标完全统一:
    1) success_rate         —— 立方体进入目标区域 (dist < 0.03) 的回合占比
    2) avg_completion_steps —— 成功回合的平均执行步数
    3) avg_distance_error   —— 回合结束立方体到目标的平均距离 (米)
  另附 avg_total_reward / 成功回合距离, 便于交叉核对。

策略加载与 LeRobot 官方评估一致: checkpoint 自带 config + pre/post processor。
BC 基线是 ACT 关闭 CVAE/分块后的配置 (policy.type 仍为 act), 故共用 ACTConfig 加载路径。

用法（lerobot conda 环境）
-------------------------
    python scripts/eval_bc_comparison.py \
        --checkpoint act=/data/wzh/robot/outputs/act_chunk40/checkpoints/004000 \
        --checkpoint bc=/data/wzh/robot/outputs/bc_baseline/checkpoints/004000 \
        --out results/metrics/BC_VS_ACT_COMPARISON.json
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STUDIO_LAB = os.environ.get("STUDIO_LAB_ROOT", "/data/wzh/robot/mujoco-studio-lab")
if not os.path.isdir(STUDIO_LAB):
    raise SystemExit(f"[eval] 找不到兄弟仓库 mujoco-studio-lab: {STUDIO_LAB}")
for p in (REPO, STUDIO_LAB, os.path.join(STUDIO_LAB, "envs")):
    if p not in sys.path:
        sys.path.insert(0, p)

import lerobot_push_cube_env as lc  # noqa: E402,F401
from lerobot.envs import make_env_config  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from push_cube_env import PushCubeEnv, TARGET_XY  # noqa: E402

# 固定 50 个随机初始种子 (default_rng(4242) 在 [1000,99999] 生成, 无重复)。
FIXED_SEEDS = [
    47988, 78442, 85728, 67476, 11226, 24500, 71546, 18766, 85176, 35315,
    3887, 16058, 14088, 31831, 94471, 24661, 92082, 54811, 93553, 91852,
    3944, 44981, 54321, 76267, 73478, 57952, 65895, 51676, 5258, 66058,
    82694, 94725, 45003, 91614, 6601, 72302, 83026, 13152, 27511, 91023,
    61753, 71614, 32068, 35976, 43265, 22915, 94002, 91492, 70899, 81752,
]


def load_policy_from_checkpoint(ckpt_dir):
    pretrained = os.path.join(ckpt_dir, "pretrained_model")
    if not os.path.isdir(pretrained):
        pretrained = ckpt_dir
    cfg = ACTConfig.from_pretrained(pretrained)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = make_env_config("push_cube")
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, pretrained_path=pretrained)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=pretrained)
    return policy, preprocessor, postprocessor, cfg


def run_episode(env, policy, pre, post, device, seed, max_steps):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    env.data.qpos[9] += float(rng.uniform(-0.030, 0.030))
    env.data.qpos[10] += float(rng.uniform(-0.035, 0.035))
    env.data.qvel[9:15] = 0.0
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    env._prev_dist = env._cube_target_dist()
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
        obs, reward, term, trunc, info = env.step(raw)
        if term or trunc:
            break
    cube = env.data.xpos[env._cube_body]
    dist = float(np.linalg.norm(cube[:2] - TARGET_XY))
    steps = int(info.get("steps", t + 1))
    return {"success": bool(dist < 0.03), "steps": steps,
            "dist": dist, "total_reward": float(reward)}


def summarize(results, label, checkpoint, seeds):
    succ = [r for r in results if r["success"]]
    n = len(results)
    return {
        "label": label,
        "checkpoint": checkpoint,
        "n_rollouts": n,
        "success_rate": len(succ) / n if n else 0.0,
        "n_success": len(succ),
        "avg_completion_steps": float(np.mean([r["steps"] for r in succ])) if succ else None,
        "avg_distance_error": float(np.mean([r["dist"] for r in results])),
        "avg_distance_error_success": float(np.mean([r["dist"] for r in succ])) if succ else None,
        "avg_total_reward": float(np.mean([r["total_reward"] for r in results])),
        "seeds": seeds,
        "per_episode": results,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="LABEL=PATH, 可重复。例: act=/path/to/ckpt")
    ap.add_argument("--seeds", type=int, nargs="+", default=FIXED_SEEDS,
                    help="固定种子列表; 默认即上述 50 个")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--out", default="BC_VS_ACT_COMPARISON.json")
    args = ap.parse_args()

    checkpoints = {}
    for item in args.checkpoint:
        if "=" not in item:
            raise SystemExit(f"[eval] --checkpoint 格式应为 LABEL=PATH, 得到: {item}")
        label, path = item.split("=", 1)
        checkpoints[label] = path

    if os.path.isabs(args.out):
        out_path = args.out
    elif args.out.startswith("results/metrics"):
        out_path = os.path.join(REPO, args.out)
    else:
        out_path = os.path.join(REPO, "results/metrics", args.out)
    seeds = args.seeds
    print("=" * 70)
    print("[eval] aligned evaluation | seeds=%d | studio=%s" % (len(seeds), STUDIO_LAB))
    for label, path in checkpoints.items():
        print("[eval]   %-8s -> %s" % (label, path))

    env = PushCubeEnv(image_obs=True, max_episode_steps=args.max_steps)
    summaries = {}
    for label, path in checkpoints.items():
        policy, pre, post, cfg = load_policy_from_checkpoint(path)
        device = next(policy.parameters()).device
        print("-" * 70)
        print("[eval] evaluating %-8s on device %s ..." % (label, device))
        results = []
        for seed in seeds:
            if hasattr(policy, "reset"):
                policy.reset()
            rec = run_episode(env, policy, pre, post, device, seed, args.max_steps)
            rec["seed"] = seed
            results.append(rec)
        summaries[label] = summarize(results, label, path, seeds)
        s = summaries[label]
        print("[eval] %-8s SUCCESS_RATE=%.3f (%d/%d) steps=%s dist=%.4fm" % (
            label, s["success_rate"], s["n_success"], s["n_rollouts"],
            s["avg_completion_steps"], s["avg_distance_error"]))
        del policy, pre, post
    env.close()

    comparison = {
        "protocol": {
            "n_seeds": len(seeds),
            "episodes_per_seed": 1,
            "n_rollouts_per_model": len(seeds),
            "max_steps": args.max_steps,
            "success_threshold_m": 0.03,
            "cube_init_offset": {"x": 0.030, "y": 0.035},
            "note": "两个模型在同一环境同一 50 固定种子下评估, 唯一变量为策略权重。",
        },
        "models": summaries,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(comparison, fh, indent=2)
    print("=" * 70)
    print("[eval] comparison ->", out_path)
    print("EVAL DONE")


if __name__ == "__main__":
    main()
