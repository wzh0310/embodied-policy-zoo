#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SB3 环境二次验证 + PPO(MLP) 最小交互闭环。

1) 用 SB3 自带 env_checker 对注册环境做二次验证（格式合规）。
2) 初始化 PPO + MlpPolicy，与环境交互 100 步，打印观测/动作/奖励，
   确认「环境 - 算法」最小闭环无格式错误与崩溃。

只用低维 state 观测（MLP 策略），不涉及图像渲染，headless 安全。
用法: python scripts/verify_sb3_ppo_closedloop.py
"""
import json
import os
from pathlib import Path

import gymnasium as gym

# state 模式不渲染，无需 EGL；保留变量以便切换到图像观测时直接可用
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results" / "metrics"

import pushcube_envs  # noqa: E402,F401  触发 gym 注册
from stable_baselines3.common.env_checker import check_env  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

report = {}
print("=" * 64)
print("[1] SB3 env_checker 二次验证")
check_env(gym.make("PushCube-v0", obs_mode="state", disable_env_checker=True).unwrapped,
          warn=True, skip_render_check=True)
report["sb3_env_checker"] = "pass"
print("  check_env (SB3) PASS")

print("=" * 64)
print("[2] PPO(MlpPolicy) 最小闭环: 交互 100 步")
env = gym.make("PushCube-v0", obs_mode="state", disable_env_checker=True)
model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0, seed=0)
print("  policy:", type(model.policy).__name__, "| obs_space:", env.observation_space)

obs, _ = env.reset(seed=0)
total_reward = 0.0
n_steps = 100
for t in range(n_steps):
    action, _state = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += float(reward)
    if t < 3:  # 打印前 3 步的观测/动作/奖励作为格式样例
        print("  step %2d | obs[%s] a[%s] r=%.4f done=%s" % (
            t + 1,
            ", ".join("%.3f" % v for v in obs[:3]),
            ", ".join("%+.4f" % v for v in action),
            reward, bool(terminated or truncated)))
    if terminated or truncated:
        obs, _ = env.reset()
env.close()

report.update({
    "algorithm": "PPO",
    "policy": "MlpPolicy",
    "obs_mode": "state",
    "interaction_steps": n_steps,
    "total_reward": round(total_reward, 4),
    "no_errors": True,
})
print("  完成 %d 步交互, 累计奖励 %.4f, 无报错" % (n_steps, total_reward))

out = RESULTS / "SB3_PPO_SMOKE.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print("报告已写入: %s" % out.relative_to(ROOT))
