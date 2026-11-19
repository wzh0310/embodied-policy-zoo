#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制成功率曲线 + 数据量消融曲线（仅 Pillow），复用 plot_comparison_curves.panel。"""
import json
import os
import sys

from PIL import Image, ImageDraw

REPO = "/data/wzh/robot/embodied-policy-zoo"
sys.path.insert(0, os.path.join(REPO, "scripts"))
from plot_comparison_curves import panel, font, C_ACT, C_BC, BG, TXT, SUB  # noqa: E402

M = os.path.join(REPO, "results/metrics")


def plot_success_curve():
    with open(os.path.join(M, "CHECKPOINT_CURVE.json")) as f:
        data = json.load(f)
    series = []
    for label, name, color in [("act", "ACT (chunk40)", C_ACT), ("bc", "BC", C_BC)]:
        pts = [(int(s), v["success_rate"] * 100)
               for s, v in sorted(data["models"].get(label, {}).items(), key=lambda t: int(t[0]))]
        if pts:
            series.append((name, color, pts))
    W, H = 1040, 620
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 16), "Success Rate vs Training Steps", fill=TXT, font=font(22, True), anchor="ma")
    d.text((W / 2, 46), "identical 50 fixed seeds / model; success = cube within 3 cm of target",
           fill=SUB, font=font(13), anchor="ma")
    best = panel(d, 20, 72, W - 40, 480, series,
                 "Task success rate on identical test set", "success rate (%)",
                 4000, "max", ymin=0, ymax=100, xticks=[1000, 2000, 3000, 4000])
    d.text((W / 2, 570), "Peak: " + "  |  ".join(
        f"{lb} {by:.1f}% @ step {int(bx)}" for lb, bx, by, *_ in best),
        fill=TXT, font=font(13), anchor="ma")
    out = os.path.join(M, "SUCCESS_CURVE.png")
    img.save(out)
    print("saved", out)


def plot_data_ablation():
    with open(os.path.join(M, "DATA_ABLATION.json")) as f:
        data = json.load(f)
    pts = [(int(n), v["success_rate"] * 100)
           for n, v in sorted(data["models"].items(), key=lambda t: int(t[0]))]
    W, H = 900, 600
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 16), "Data-Scale Ablation (ACT)", fill=TXT, font=font(22, True), anchor="ma")
    d.text((W / 2, 46), "identical ACT config / 4000 steps / 50 fixed seeds; only demo count varies",
           fill=SUB, font=font(13), anchor="ma")
    # 专用面板：x=数据量
    pad_l, pad_r, pad_t, pad_b = 70, 30, 60, 60
    px0, py0 = pad_l, pad_t
    pw, ph = W - pad_l - pad_r, H - pad_t - pad_b
    d.rectangle([20, 74, W - 20, H - 90], fill=(255, 255, 255), outline=(225, 228, 233))
    ymax = 100
    for i in range(5):
        yy = py0 + ph - ph * i / 4
        val = ymax * i / 4
        d.line([px0, yy, px0 + pw, yy], fill=(225, 228, 233), width=1)
        d.text((px0 - 8, yy - 7), f"{val:.0f}%", fill=SUB, font=font(12), anchor="ra")
    xmax = 63
    for xv in [20, 40, 63]:
        xx = px0 + pw * xv / xmax
        d.line([xx, py0, xx, py0 + ph], fill=(225, 228, 233), width=1)
        d.text((xx, py0 + ph + 6), str(xv), fill=SUB, font=font(12), anchor="ma")
    d.line([px0, py0, px0, py0 + ph], fill=(60, 60, 70), width=2)
    d.line([px0, py0 + ph, px0 + pw, py0 + ph], fill=(60, 60, 70), width=2)
    xy = [(px0 + pw * x / xmax, py0 + ph - ph * y / ymax) for x, y in pts]
    if len(xy) > 1:
        d.line(xy, fill=C_ACT, width=3, joint="curve")
    for (x, y), (bx, by) in zip(pts, xy):
        d.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=C_ACT, outline=(255, 255, 255), width=2)
        d.text((bx, by - 16), f"{y:.1f}%", fill=C_ACT, font=font(13, True), anchor="ma")
    d.text((W / 2, 92), "Task success rate vs number of expert demonstrations",
           fill=TXT, font=font(16, True), anchor="ma")
    d.text((px0 + pw / 2, py0 + ph + 26), "Number of demonstrations", fill=SUB, font=font(12), anchor="ma")
    note = "100 demos unavailable: merged dataset has only 63 successful episodes (v2=40 + v3b success=23)."
    d.text((W / 2, H - 24), note, fill=SUB, font=font(11), anchor="ma")
    out = os.path.join(M, "DATA_ABLATION_CURVE.png")
    img.save(out)
    print("saved", out)


if __name__ == "__main__":
    if os.path.isfile(os.path.join(M, "CHECKPOINT_CURVE.json")):
        plot_success_curve()
    if os.path.isfile(os.path.join(M, "DATA_ABLATION.json")):
        plot_data_ablation()
