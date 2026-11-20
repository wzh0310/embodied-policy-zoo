#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端视觉 PPO 训练脚本（SB3 CnnPolicy + NatureCNN + 4 帧堆叠）。

输入：PushCubeEnv 的 obs_mode='rgb'（84x84 灰度、CHW、float32 [0,1]，几何代理离屏
渲染 ~482fps）。用 VecFrameStack 堆叠 4 帧提供运动时序信息，观测形状由 (1,84,84)
变为 (4,84,84)，NatureCNN 首层卷积自动适配 4 通道。

默认超参数面向视觉任务（样本效率更低）：总步数 2,000,000 / 学习率 2e-4 /
n_steps 2048 / batch 256 / ent_coef 0.01 / gamma 0.99。动作空间仍是末端笛卡尔
增量（±0.02m），与模仿学习、状态 PPO 完全一致，便于公平对比。

关键工程约束（勿删）：
  1) policy_kwargs={'normalize_images': False}：环境已做 CHW + [0,1] 归一化，
     必须关闭 SB3 自带 /255 与 HWC->CHW，避免重复预处理。
  2) features_extractor_class=NatureCNN：显式指定视觉主干（SB3 CnnPolicy 默认
     即 NatureCNN，此处固化以明确架构、便于后续替换 ResNet 等）。
  3) 先构造 PPO，再由 learn 内部触发 reset/render。若构造前手动 reset 出图，
     MuJoCo EGL 上下文会与 torch 卷积初始化冲突而段错误（软件 EGL 主机）。
     OffscreenRenderer 懒初始化 + torch 先行导入共同规避此问题。

用法:
    python algos/ppo/train_vision_ppo.py                          # 默认 200 万步
    python algos/ppo/train_vision_ppo.py --total-timesteps 10000  # 小批量验证
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
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.torch_layers import NatureCNN  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack  # noqa: E402

from envs.paths import OUTPUTS, setup_sys_path  # noqa: E402
setup_sys_path()
import pushcube_envs  # noqa: E402,F401  触发 gym 注册

GYM_ID = "PushCube-v0"
# 环境与 SB3 的视觉输入契约（见模块 docstring）
CNN_KWARGS = dict(
    normalize_images=False,           # 环境已 CHW + [0,1]
    features_extractor_class=NatureCNN,
    features_extractor_kwargs=dict(features_dim=256),
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--frame-stack", type=int, default=4,
                    help="number of stacked frames (时序信息)")
    ap.add_argument("--rgb-preset", type=str, default="train",
                    choices=["train", "demo"])
    ap.add_argument("--randomize-start", action="store_true")
    ap.add_argument("--start-xy-range", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-freq", type=int, default=50_000)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    return ap.parse_args()


def make_vec_env(args):
    def _f():
        return gym.make(GYM_ID, obs_mode="rgb", rgb_preset=args.rgb_preset,
                        randomize_start=args.randomize_start,
                        start_xy_range=args.start_xy_range,
                        disable_env_checker=True)
    venv = DummyVecEnv([_f])
    # 帧堆叠：CHW 灰度 (1,84,84) x4 -> (4,84,84)，提供运动时序
    venv = VecFrameStack(venv, n_stack=args.frame_stack, channels_order="first")
    return venv


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        OUTPUTS / "ppo_vision" / f"vision_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    venv = make_vec_env(args)
    print(f"[obs] raw (1,84,84) -> stacked {venv.observation_space.shape} "
          f"({args.frame_stack} frames, NatureCNN)")

    # 先构造策略（venv 尚未 reset / 渲染，GL 上下文未建立），再 learn。
    model = PPO("CnnPolicy", venv, n_steps=args.n_steps,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                ent_coef=args.ent_coef, gamma=args.gamma,
                gae_lambda=args.gae_lambda, policy_kwargs=CNN_KWARGS,
                device=args.device, verbose=1, seed=args.seed,
                tensorboard_log=str(out_dir / "logs"))
    logger = configure(str(out_dir / "logs"), ["csv", "stdout"])
    model.set_logger(logger)

    ckpt = CheckpointCallback(save_freq=args.save_freq,
                              save_path=str(out_dir / "checkpoints"),
                              name_prefix="vision_ppo")
    model.learn(total_timesteps=args.total_timesteps, callback=[ckpt])
    model.save(str(out_dir / "final_model"))
    venv.close()

    meta = dict(vars(args))
    meta["cnn_kwargs"] = {
        "normalize_images": False,
        "features_extractor": "NatureCNN",
        "features_dim": 256,
    }
    meta["obs_mode"] = "rgb"
    meta["stacked_obs_shape"] = list(venv.observation_space.shape)
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] saved to {out_dir}")


if __name__ == "__main__":
    main()
