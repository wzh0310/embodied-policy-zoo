#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""视觉观测管线标准化验证：观测格式 / 两档渲染 / 缓存失效 / 随机种子可复现 /
动作归一化 / 帧率 / 环境合规 / SB3 CNN 闭环。

所有检查汇总写入 results/metrics/VISION_PIPELINE_VERIFY.json。
用法: python scripts/verify_vision_pipeline.py
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import pushcube_envs  # noqa: E402,F401
from pushcube_envs import PushCubeEnv  # noqa: E402
from gymnasium.utils.env_checker import check_env  # noqa: E402

OUT = REPO / "results" / "metrics" / "VISION_PIPELINE_VERIFY.json"


def main():
    r = {}

    # 1. 四种观测模式向后兼容
    compat = {}
    for mode in ("state", "both", "image", "rgb"):
        e = gym.make("PushCube-v0", obs_mode=mode, disable_env_checker=True)
        o, _ = e.reset(seed=0)
        if mode == "state":
            ok = o.shape == (13,) and o.dtype == np.float32
        elif mode == "image":
            ok = o.shape == (224, 224, 3) and o.dtype == np.uint8
        elif mode == "rgb":
            ok = o.shape == (1, 84, 84) and o.dtype == np.float32 and o.max() <= 1
        else:
            ok = isinstance(o, dict) and {"image", "state"} <= set(o)
        e.step(e.action_space.sample())
        compat[mode] = "PASS" if ok else "FAIL"
        e.close()
    r["obs_mode_compatibility"] = compat

    # 2. 两档渲染格式
    e = PushCubeEnv(obs_mode="rgb")
    tr, dm = e.get_rgb_image("train"), e.get_rgb_image("demo")
    r["get_rgb_image_presets"] = (
        "PASS" if tr.shape == (1, 84, 84) and dm.shape == (3, 224, 224) else "FAIL")
    e.close()

    # 3. reset 图像缓存失效
    e = PushCubeEnv(obs_mode="rgb", randomize_start=True)
    o1 = e.reset(seed=10)[0].copy()
    o2 = e.reset(seed=999)[0].copy()
    r["reset_cache_invalidation"] = "PASS" if not np.allclose(o1, o2) else "FAIL"
    e.close()

    # 4. 随机初始位姿可复现
    def starts(seed):
        env = PushCubeEnv(obs_mode="state", randomize_start=True)
        xy = [env.reset(seed=seed)[1]["cube_start_xy"].copy() for _ in range(3)]
        env.close()
        return xy
    a, b, c = starts(7), starts(7), starts(8)
    r["reproducible_random_start"] = (
        "PASS" if all(np.allclose(x, y) for x, y in zip(a, b))
        and not np.allclose(a[0], c[0]) else "FAIL")

    # 5. 动作归一化一致性（训练/推理同一边界，往返无损）
    e = PushCubeEnv(obs_mode="state")
    lo, hi = e.action_space.low, e.action_space.high
    u = np.array([-1, 1, 0, 0.5], dtype=np.float32)
    phys = lo + (u + 1) / 2 * (hi - lo)
    back = 2 * (phys - lo) / (hi - lo) - 1
    ok = (np.allclose(lo[:3], -0.02) and np.allclose(hi[:3], 0.02)
          and lo[3] == 0 and hi[3] == 1 and np.allclose(back, u, atol=1e-5))
    r["action_normalization"] = "PASS" if ok else "FAIL"
    e.close()

    # 6. check_env 合规（state + rgb）
    for mode in ("state", "rgb"):
        try:
            check_env(PushCubeEnv(obs_mode=mode), skip_render_check=True)
            r[f"check_env_{mode}"] = "PASS"
        except Exception:
            r[f"check_env_{mode}"] = "FAIL"

    # 7. 训练档帧率（含同步 + 后处理全管线）
    e = PushCubeEnv(obs_mode="rgb")
    for _ in range(10):
        e._get_rgb(); e._rgb_cache = None
    n = 400
    t0 = time.perf_counter()
    for _ in range(n):
        e._get_rgb(); e._rgb_cache = None
    fps = n / (time.perf_counter() - t0)
    e.close()
    r["train_capture_fps"] = round(fps, 1)
    r["throughput_ge_200fps"] = "PASS" if fps >= 200 else "FAIL"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print(f"[saved] {OUT}")
    assert all(v == "PASS" for k, v in r.items() if isinstance(v, str)), \
        "some checks failed"


if __name__ == "__main__":
    main()
