#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""视觉 PPO (CnnPolicy + NatureCNN + 帧堆叠) 专用评估工具。

复现训练时的观测管线 (obs_mode=rgb + VecFrameStack n_stack)，但不加训练侧数据
增强，保证评估确定性。统计成功率 / 平均回合步数 / 平均回合奖励。

用法:
    python utils/evaluate_vision.py --model <model.zip> --episodes 50 --device cuda
    python utils/evaluate_vision.py --model <model.zip> --episodes 50 --stochastic
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack  # noqa: E402

from envs.paths import setup_sys_path  # noqa: E402
setup_sys_path()
import pushcube_envs  # noqa: E402,F401

GYM_ID = "PushCube-v0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--frame-stack", type=int, default=4)
    ap.add_argument("--rgb-preset", default="train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--stochastic", dest="deterministic", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def _f():
        return gym.make(GYM_ID, obs_mode="rgb", rgb_preset=args.rgb_preset,
                        disable_env_checker=True)
    venv = VecFrameStack(DummyVecEnv([_f]), n_stack=args.frame_stack,
                         channels_order="first")
    model = PPO.load(args.model, device=args.device)

    rewards, lengths, successes, cube_dists = [], [], [], []
    for ep in range(args.episodes):
        obs = venv.reset()
        ep_reward, steps, done, succ = 0.0, 0, False, False
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done_vec, info = venv.step(action)
            ep_reward += float(reward[0])
            steps += 1
            succ = bool(info[0].get("success", info[0].get("is_success", False)))
            done = bool(done_vec[0])
        rewards.append(ep_reward)
        lengths.append(steps)
        successes.append(succ)
        u = venv.envs[0].unwrapped
        cube = u.data.xpos[u._cube_body]
        cube_dists.append(float(np.linalg.norm(cube[:2] - np.array([0.82, -0.12]))))
    venv.close()

    result = {
        "model": args.model,
        "episodes": args.episodes,
        "deterministic": args.deterministic,
        "frame_stack": args.frame_stack,
        "success_rate": round(float(np.mean(successes)), 4),
        "n_success": int(np.sum(successes)),
        "mean_episode_reward": round(float(np.mean(rewards)), 4),
        "std_episode_reward": round(float(np.std(rewards)), 4),
        "mean_episode_steps": round(float(np.mean(lengths)), 2),
        "mean_final_cube_dist": round(float(np.mean(cube_dists)), 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
