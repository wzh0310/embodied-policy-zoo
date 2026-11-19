#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键入口：用统一命令指定算法与配置，完成训练 / 评估 / 录制 / 绘图。

所有子命令都通过算法名解析到对应脚本，屏蔽底层脚本差异；路径全部相对仓库根，
不需要记忆容器内的绝对路径。

用法（在 lerobot 环境、仓库根目录执行）：

    # 训练：--algo 选择 act / bc，--config 指定 YAML
    python scripts/run.py train --algo act \
        --config configs/act_push_cube/ablation/chunk40.yaml --steps 4000
    python scripts/run.py train --algo bc \
        --config configs/act_push_cube/bc_baseline.yaml --steps 4000

    # 评估：单 checkpoint 标准评估
    python scripts/run.py eval --checkpoint /path/to/checkpoints/004000 \
        --out results/metrics/my_eval.json

    # 录制演示短视频
    python scripts/run.py record --checkpoint /path/to/checkpoints/004000

    # 可视化（重新生成全部对比曲线）
    python scripts/run.py plot
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(REPO, "scripts")

SUBCMD_SCRIPT = {
    "train": "train.py",
    "eval": "eval.py",
    "record": "record_video.py",
    "record_clips": "record_demo_clips.py",
    "compare": "eval_bc_comparison.py",
    "mse": "eval_action_mse.py",
    "curve": "eval_checkpoint_curve.py",
    "data_ablation": "eval_data_ablation.py",
    "plot": "plot_comparison_curves.py",
    "plot_data": "plot_data_and_success.py",
}


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=sorted(SUBCMD_SCRIPT),
                    help="要执行的操作（train/eval/record/...）")
    ap.add_argument("--algo", choices=["act", "bc"], default="act",
                    help="算法名（train 时用于选择默认配置，仅为提示，不强制）")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="透传给底层脚本的参数（例如 --config / --steps / --checkpoint）")
    return ap


def main():
    args = build_parser().parse_args()
    script = os.path.join(SCRIPTS, SUBCMD_SCRIPT[args.command])
    if not os.path.isfile(script):
        raise SystemExit(f"[run] 脚本不存在: {script}")

    cmd = [sys.executable, script] + args.rest
    print(f"[run] algorithm={args.algo}  command={args.command}")
    print("[run] " + " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("TQDM_DISABLE", "1")
    raise SystemExit(subprocess.call(cmd, cwd=REPO, env=env))


if __name__ == "__main__":
    main()
