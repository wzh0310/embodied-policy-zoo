#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 SB3 progress.csv 绘制训练曲线（奖励 / 成功率 / 价值损失）。

用法: python scripts/plot_ppo_curve.py --csv <progress.csv> --out <curve.png>
"""
import argparse
import csv
from pathlib import Path

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


def plot(rows, out, title="PPO State-Policy Training Curve"):
    steps = [r.get("time/total_timesteps", 0) for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # 评估奖励（训练中确定性回合）
    er = [(r["time/total_timesteps"], r["eval/mean_episode_reward"])
          for r in rows if "eval/mean_episode_reward" in r]
    if er:
        axes[0].plot([x for x, _ in er], [y for _, y in er], "o-", color="#d62728", label="Eval episode reward")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title(title)
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    # 成功率
    sr = [(r["time/total_timesteps"], r.get("eval/success_rate", r.get("rollout/success_rate", 0)))
          for r in rows if "eval/success_rate" in r or "rollout/success_rate" in r]
    if sr:
        axes[1].plot([x for x, _ in sr], [y for _, y in sr], "s-", color="#1f77b4", label="Success rate")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    # 价值损失 / explained variance
    vl = [(r["time/total_timesteps"], r["train/value_loss"])
          for r in rows if "train/value_loss" in r]
    if vl:
        axes[2].plot([x for x, _ in vl], [y for _, y in vl], "-", color="#2ca02c", label="Value loss")
    ev = [(r["time/total_timesteps"], r["train/explained_variance"])
          for r in rows if "train/explained_variance" in r]
    if ev:
        ax2 = axes[2].twinx()
        ax2.plot([x for x, _ in ev], [y for _, y in ev], "--", color="#9467bd", alpha=0.7, label="Explained var")
        ax2.set_ylabel("explained var")
        ax2.legend(loc="upper right")
    axes[2].set_xlabel("Timesteps")
    axes[2].set_ylabel("Value loss")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out, "points=", len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="PPO 状态策略训练曲线")
    args = ap.parse_args()
    plot(load_csv(args.csv), args.out, args.title)


if __name__ == "__main__":
    main()
