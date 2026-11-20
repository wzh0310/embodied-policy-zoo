#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SB3 环境适配验证 + 随机策略基线。

验证三项:
  1. SubprocVecEnv x4 的观测 / 奖励 / 终止信号形状是否符合 SB3 预期, 并实测采样吞吐
  2. 随机策略 10 回合基线: 平均奖励 / 成功率 / 平均步数
  3. 依赖版本与离屏渲染 (MUJOCO_GL=egl) 的 RGB 图像生成

注意: 离屏渲染验证刻意放在最后。MuJoCo + EGL 下, 若父进程先调用 render 建立 GL
上下文, 之后 fork 的子进程再渲染会死锁; 因此所有 SubprocVecEnv(fork) 操作必须在
父进程产生任何 GL 上下文之前完成。

用法 (容器内 mujoco 环境, 无需显示窗口):
    python scripts/verify_sb3_env.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# 容器无显示窗口: 离屏渲染环境变量必须在导入 mujoco 前设置
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym  # noqa: E402
import stable_baselines3 as sb3  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results" / "metrics"
RESULTS.mkdir(parents=True, exist_ok=True)

N_ENVS = 4
N_EPISODES = 10
SEED = 20241120

report = {
    "env_id": "PushCube-v0",
    "obs_mode_baseline": "state",
    "n_parallel_envs": N_ENVS,
    "n_eval_episodes": N_EPISODES,
    "seed": SEED,
}


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


import pushcube_envs  # noqa: E402,F401  触发 gym 注册
assert "PushCube-v0" in gym.registry, "PushCube-v0 未注册到 gym.registry"

# ---------------------------------------------------------------- 1. 向量化适配
section("[1/3] SubprocVecEnv x%d 向量化适配验证" % N_ENVS)
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402


def make_env(rank, seed, obs_mode="state"):
    def _f():
        e = gym.make("PushCube-v0", obs_mode=obs_mode, disable_env_checker=True)
        e.reset(seed=seed + rank)
        return e
    return _f


venv = SubprocVecEnv([make_env(i, SEED) for i in range(N_ENVS)], start_method="fork")
checks = {}
assert venv.num_envs == N_ENVS
obs = venv.reset()
assert obs.shape == (N_ENVS, 13) and obs.dtype == np.float32
checks["obs_shape"] = list(obs.shape)
checks["obs_dtype"] = str(obs.dtype)

rng = np.random.default_rng(SEED)


def vec_sample():
    low, high = venv.action_space.low, venv.action_space.high
    return (low + rng.random(low.shape) * (high - low)).astype(np.float32)


obs, rewards, dones, infos = venv.step(vec_sample())
assert obs.shape == (N_ENVS, 13)
assert rewards.shape == (N_ENVS,)
assert dones.shape == (N_ENVS,) and dones.dtype == np.bool_
checks["reward_shape"] = list(rewards.shape)
checks["reward_dtype"] = str(rewards.dtype)
checks["done_shape"] = list(dones.shape)
checks["done_dtype"] = str(dones.dtype)
checks["observation_space"] = str(venv.observation_space)
checks["action_space"] = str(venv.action_space)
print("  obs      : %s %s" % (obs.shape, obs.dtype))
print("  reward   : %s %s" % (rewards.shape, rewards.dtype))
print("  done     : %s %s" % (dones.shape, dones.dtype))
print("  obs_space: %s" % venv.observation_space)


def measure_throughput(vec, single_obs_mode, n_steps):
    t0 = time.perf_counter()
    for _ in range(n_steps):
        vec.step(vec_sample())
    t_vec = time.perf_counter() - t0
    single = gym.make("PushCube-v0", obs_mode=single_obs_mode, disable_env_checker=True)
    single.reset(seed=SEED)
    slo, shi = single.action_space.low, single.action_space.high
    t0 = time.perf_counter()
    for _ in range(n_steps):
        single.step((slo + rng.random(slo.shape) * (shi - slo)).astype(np.float32))
    t_single = time.perf_counter() - t0
    single.close()
    vec_fps = N_ENVS * n_steps / t_vec
    single_fps = n_steps / t_single
    return {
        "steps_per_batch": n_steps,
        "vec_transitions_per_sec": round(vec_fps, 1),
        "single_transitions_per_sec": round(single_fps, 1),
        "speedup_x": round(vec_fps / single_fps, 2),
    }, vec_fps, single_fps


tp_state, vf_s, sf_s = measure_throughput(venv, "state", 400)
checks["throughput_state_only"] = tp_state
print("  [state ] 并行 %.1f vs 单环境 %.1f trans/s => %.2fx" % (vf_s, sf_s, tp_state["speedup_x"]))
venv.close()

# 含 RGB 渲染的 both 模式: 渲染是主要开销, 多进程收益明显
venv_img = SubprocVecEnv([make_env(i, SEED, "both") for i in range(N_ENVS)], start_method="fork")
venv_img.reset()
tp_both, vf_b, sf_b = measure_throughput(venv_img, "both", 60)
checks["throughput_with_render"] = tp_both
print("  [both  ] 并行 %.1f vs 单环境 %.1f trans/s => %.2fx" % (vf_b, sf_b, tp_both["speedup_x"]))
venv_img.close()
report["vector_env_checks"] = checks

# ---------------------------------------------------------------- 2. 随机基线
section("[2/3] 随机策略基线 (%d 回合)" % N_EPISODES)
base = gym.make("PushCube-v0", obs_mode="state", disable_env_checker=True)
act_rng = np.random.default_rng(SEED)
low, high = base.action_space.low, base.action_space.high
episodes = []
for ep in range(N_EPISODES):
    base.reset(seed=SEED + ep)
    ep_reward, steps, success, truncated = 0.0, 0, False, False
    done_flag = False
    while not done_flag:
        action = (low + act_rng.random(low.shape) * (high - low)).astype(np.float32)
        obs, reward, terminated, truncated, info = base.step(action)
        ep_reward += float(reward)
        steps += 1
        success = bool(info.get("success", False))
        if terminated or truncated:
            done_flag = True
    episodes.append({
        "episode": ep + 1,
        "reward": round(ep_reward, 4),
        "steps": steps,
        "success": success,
        "truncated": bool(truncated),
    })
    print("  ep %2d: reward=%9.3f  steps=%3d  success=%s" %
          (ep + 1, ep_reward, steps, success))
base.close()

rewards = [e["reward"] for e in episodes]
steps_arr = [e["steps"] for e in episodes]
n_success = sum(e["success"] for e in episodes)
summary = {
    "mean_reward": round(float(np.mean(rewards)), 4),
    "std_reward": round(float(np.std(rewards)), 4),
    "success_rate": round(n_success / N_EPISODES, 4),
    "success_count": n_success,
    "mean_steps": round(float(np.mean(steps_arr)), 2),
}
report["episodes"] = episodes
report["summary"] = summary

section("基线汇总")
print("  平均奖励 : %.4f +/- %.4f" % (summary["mean_reward"], summary["std_reward"]))
print("  成功率   : %.0f%% (%d/%d)" % (summary["success_rate"] * 100, n_success, N_EPISODES))
print("  平均步数 : %.2f" % summary["mean_steps"])

# ---------------------------------------------------------------- 3. 渲染/版本 (最后)
section("[3/3] 依赖版本与离屏渲染验证")
versions = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "torch": torch.__version__,
    "gymnasium": gym.__version__,
    "stable_baselines3": sb3.__version__,
    "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
}
for k, v in versions.items():
    print("  %-20s %s" % (k, v))
report["versions"] = versions

render_env = gym.make("PushCube-v0", obs_mode="both", render_mode="rgb_array")
obs, info = render_env.reset(seed=0)
assert isinstance(obs, dict) and "image" in obs and "state" in obs
img = obs["image"]
rendered = render_env.render()
assert img.shape == (224, 224, 3) and img.dtype == np.uint8
assert rendered is not None and rendered.shape == (224, 224, 3)
render_env.close()
print("  离屏 RGB: obs.image=%s %s, render()=%s  [OK]" % (img.shape, img.dtype, rendered.shape))
report["offscreen_render"] = {
    "image_shape": list(img.shape),
    "dtype": str(img.dtype),
    "render_shape": list(rendered.shape),
    "ok": True,
}

out = RESULTS / "RANDOM_POLICY_BASELINE.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n报告已写入: %s" % out.relative_to(ROOT))
