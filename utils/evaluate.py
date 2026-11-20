#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一策略评估：固定回合数，输出成功率 / 平均回合步数 / 平均回合奖励。

可被训练脚本作为函数复用（rollout_metrics），也可命令行独立评估一个已保存模型：

    python utils/evaluate.py --model outputs/ppo_state/final_model.zip --episodes 50
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from envs.paths import setup_sys_path  # noqa: E402

setup_sys_path()
import gymnasium as gym  # noqa: E402
import pushcube_envs  # noqa: E402,F401  触发 gym 注册

GYM_ID = "PushCube-v0"


def make_eval_env(obs_mode="state"):
    return gym.make(GYM_ID, obs_mode=obs_mode, disable_env_checker=True)


def rollout_metrics(model, env, n_episodes=50, seed=0, deterministic=True):
    """跑 n_episodes 个回合，统计成功率/平均步数/平均奖励。"""
    rewards, lengths, successes = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward, steps, done = 0.0, 0, False
        ep_success = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            steps += 1
            ep_success = bool(info.get("success", info.get("is_success", False)))
            done = bool(terminated or truncated)
        rewards.append(ep_reward)
        lengths.append(steps)
        successes.append(ep_success)
    return {
        "n_episodes": n_episodes,
        "deterministic": deterministic,
        "success_rate": round(float(np.mean(successes)), 4),
        "success_count": int(np.sum(successes)),
        "mean_episode_steps": round(float(np.mean(lengths)), 2),
        "std_episode_steps": round(float(np.std(lengths)), 2),
        "mean_episode_reward": round(float(np.mean(rewards)), 4),
        "std_episode_reward": round(float(np.std(rewards)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="已保存的 SB3 模型 .zip 路径")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true", help="用随机动作评估（默认确定性）")
    ap.add_argument("--out", default=None, help="量化结果 JSON 输出路径")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model)
    env = make_eval_env("state")
    metrics = rollout_metrics(model, env, args.episodes, args.seed,
                              deterministic=not args.stochastic)
    env.close()

    print("=" * 60)
    print("评估: %s | %d 回合 (%s)" % (
        Path(args.model).name, args.episodes,
        "随机" if args.stochastic else "确定性"))
    print("  成功率        : %.0f%% (%d/%d)" % (
        metrics["success_rate"] * 100, metrics["success_count"], args.episodes))
    print("  平均回合步数  : %.2f +/- %.2f" % (
        metrics["mean_episode_steps"], metrics["std_episode_steps"]))
    print("  平均回合奖励  : %.4f +/- %.4f" % (
        metrics["mean_episode_reward"], metrics["std_episode_reward"]))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print("  结果已写入: %s" % out)


if __name__ == "__main__":
    main()
