#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""视觉输入 PPO 训练脚本（SB3 CnnPolicy，84x84 灰度观测）。

复用 PushCubeEnv 的 obs_mode='rgb'：离屏渲染、几何代理加速、CHW float32 [0,1]，
与 SB3 CNN 策略输入格式对齐。动作空间仍是末端笛卡尔增量（±0.02m），与模仿学习
和 state PPO 完全一致，便于后续视觉/状态策略的公平对比。

关键工程约束（勿删）：
  1) policy_kwargs={'normalize_images': False}：环境已做 CHW + [0,1] 归一化，
     必须关闭 SB3 自带的 /255 与 HWC->CHW，否则重复预处理。
  2) 必须先构造 PPO，再（由 SB3 learn 内部）触发环境 reset/render。若在构造
     PPO 前手动 reset 出图，MuJoCo 的 EGL 上下文会与 torch 卷积初始化冲突而
     段错误（软件 EGL 主机）。OffscreenRenderer 已做懒初始化配合此约束。

用法:
    python algos/ppo/train_rgb_ppo.py --total-timesteps 50000 --eval-freq 10000
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402  先于环境渲染导入 torch，降低 EGL 冲突风险
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

from envs.paths import OUTPUTS, setup_sys_path  # noqa: E402
setup_sys_path()
import pushcube_envs  # noqa: E402,F401  触发 gym 注册

GYM_ID = "PushCube-v0"
# 环境与 SB3 的视觉输入契约（见模块 docstring）
CNN_KWARGS = dict(normalize_images=False)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-timesteps", type=int, default=500_000)
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--rgb-preset", type=str, default="train",
                    choices=["train", "demo"])
    ap.add_argument("--randomize-start", action="store_true")
    ap.add_argument("--start-xy-range", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-freq", type=int, default=10000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--save-freq", type=int, default=20000)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    return ap.parse_args()


def make_env(rgb_preset, randomize_start, start_xy_range, seed):
    def _f():
        return gym.make(GYM_ID, obs_mode="rgb", rgb_preset=rgb_preset,
                        randomize_start=randomize_start,
                        start_xy_range=start_xy_range,
                        disable_env_checker=True)
    return _f


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        OUTPUTS / "ppo_rgb" / f"rgb_{args.rgb_preset}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    env_fn = make_env(args.rgb_preset, args.randomize_start,
                      args.start_xy_range, args.seed)
    env = env_fn()
    # 先构造策略（此时 env 尚未 reset / 渲染，GL 上下文未建立），再 learn。
    model = PPO("CnnPolicy", env, n_steps=args.n_steps,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                ent_coef=args.ent_coef, gamma=args.gamma,
                gae_lambda=args.gae_lambda, policy_kwargs=CNN_KWARGS,
                device=args.device, verbose=1, seed=args.seed,
                tensorboard_log=str(out_dir / "logs"))
    logger = configure(str(out_dir / "logs"), ["csv", "stdout"])
    model.set_logger(logger)

    ckpt = CheckpointCallback(save_freq=args.save_freq,
                              save_path=str(out_dir / "checkpoints"),
                              name_prefix="rgb_ppo")
    model.learn(total_timesteps=args.total_timesteps, callback=[ckpt])
    model.save(str(out_dir / "final_model"))
    env.close()

    meta = dict(vars(args))
    meta["cnn_kwargs"] = CNN_KWARGS
    meta["obs_mode"] = "rgb"
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
