#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从视觉 PPO 的 SB3 progress.csv 绘制完整训练曲线。

画 4 个子图：rollout 成功率 / 策略+价值损失 / explained_variance / fps。
用英文标签（容器 matplotlib 无中文字体）。
用法: python scripts/plot_vision_curve.py --csv <progress.csv> --out <curve.png>
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in row.items() if v != ""})
            except ValueError:
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Vision PPO (CnnPolicy) Training Curve")
    args = ap.parse_args()
    rows = load_csv(args.csv)
    steps = [r.get("time/total_timesteps", 0) for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

    # 1) rollout success rate
    if any("rollout/success_rate" in r for r in rows):
        succ = [r.get("rollout/success_rate", 0) * 100 for r in rows]
        axes[0].plot(steps, succ, color="#d62728", lw=1.5)
        # 滑动平均（20 点）
        if len(succ) >= 20:
            sm = [sum(succ[max(0, i-19):i+1]) / min(i+1, 20) for i in range(len(succ))]
            axes[0].plot(steps, sm, color="#d62728", lw=2.5, label="success % (MA20)")
            axes[0].legend(loc="upper left")
        axes[0].set_ylabel("Success rate (%)")
        axes[0].set_ylim(0, 100)
    axes[0].set_title(args.title)

    # 2) losses
    for key, color, label in [
        ("train/loss", "#1f77b4", "total loss"),
        ("train/value_loss", "#ff7f0e", "value loss"),
    ]:
        if any(key in r for r in rows):
            axes[1].plot(steps, [r.get(key, 0) for r in rows], color=color, lw=1.2, label=label)
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")

    # 3) explained variance
    if any("train/explained_variance" in r for r in rows):
        axes[2].plot(steps, [r.get("train/explained_variance", 0) for r in rows],
                     color="#2ca02c", lw=1.2)
        axes[2].axhline(0, color="gray", ls="--", lw=0.8)
    axes[2].set_ylabel("Explained var")

    # 4) fps
    if any("time/fps" in r for r in rows):
        axes[3].plot(steps, [r.get("time/fps", 0) for r in rows], color="#9467bd", lw=1.2)
    axes[3].set_ylabel("FPS")
    axes[3].set_xlabel("Timesteps")

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("saved", args.out, "points", len(rows))


if __name__ == "__main__":
    main()
