#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制 ACT vs BC 对比曲线（仅依赖 Pillow，无 matplotlib）。

 输出两张图：
  1) loss_curve_comparison.png —— 上: 训练 L1 loss；下: 验证 loss（均归一化空间，
     两模型同口径）。ACT 的 total train loss 含 CVAE 的 KL 项、BC 无 KL，故训练曲线
     统一用 L1 重建损失以保证公平；eval_loss 两边均为归一化空间 L1。
  2) success_curve.png —— 评估成功率随训练 checkpoint 步数变化（数据来自
     CHECKPOINT_CURVE.json，同一 50 固定种子）。

 每张图统一风格：浅色背景、网格、粗轴线、图例、最优点高亮标注。
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from plot_training_curve import parse_log  # noqa: E402

C_ACT = (31, 119, 180)     # 蓝
C_BC = (214, 39, 40)       # 红
BG = (250, 250, 252)
GRID = (225, 228, 233)
AXIS = (60, 60, 70)
TXT = (35, 35, 45)
SUB = (110, 115, 125)

OUT_ROOT = os.environ.get("OUTPUTS_DIR", os.path.join(os.path.dirname(REPO), "outputs"))
ACT_LOG = os.path.join(OUT_ROOT, "train_chunk40.log")
BC_LOG = os.path.join(OUT_ROOT, "bc_baseline_train.log")


def font(sz, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for fp in ["/usr/share/fonts/truetype/dejavu/" + name,
               "/usr/share/fonts/truetype/liberation2/" + name]:
        try:
            return ImageFont.truetype(fp, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def panel(draw, x0, y0, w, h, series, title, ylabel, xmax, best="min",
          ymin=None, ymax=None, xticks=None):
    """在矩形区域内画一个折线子图。series=[(label,color,[(x,y),...]),...]"""
    pad_l, pad_r, pad_t, pad_b = 66, 20, 34, 40
    px0, py0 = x0 + pad_l, y0 + pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    # 背景
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255), outline=GRID)
    # y 范围
    all_y = [y for _, _, pts in series for _, y in pts]
    lo = ymin if ymin is not None else min(all_y)
    hi = ymax if ymax is not None else max(all_y)
    if lo == hi:
        hi = lo + 1e-6
    # 仅在未显式给定范围时加 margin；显式范围（如 0~100%）严格使用
    if ymin is None:
        lo -= (hi - lo) * 0.12
    if ymax is None:
        hi += (hi - lo) * 0.12
    # 网格 + y 轴刻度
    f_tick = font(13)
    for i in range(5):
        yy = py0 + ph - ph * i / 4
        val = lo + (hi - lo) * i / 4
        draw.line([px0, yy, px0 + pw, yy], fill=GRID, width=1)
        draw.text((px0 - 8, yy - 7), f"{val:.3f}" if hi < 5 else f"{val:.1f}",
                  fill=SUB, font=f_tick, anchor="ra")
    # x 轴刻度
    xt = xticks if xticks else [0, xmax * 0.25, xmax * 0.5, xmax * 0.75, xmax]
    for xv in xt:
        xx = px0 + pw * xv / xmax
        draw.line([xx, py0, xx, py0 + ph], fill=GRID, width=1)
        draw.text((xx, py0 + ph + 6), str(int(xv)), fill=SUB, font=f_tick, anchor="ma")
    # 轴线
    draw.line([px0, py0, px0, py0 + ph], fill=AXIS, width=2)
    draw.line([px0, py0 + ph, px0 + pw, py0 + ph], fill=AXIS, width=2)
    # 折线 + 最优点
    best_pts = []
    for label, color, pts in series:
        sp = sorted(pts)
        xy = [(px0 + pw * x / xmax, py0 + ph - ph * (y - lo) / (hi - lo)) for x, y in sp]
        if len(xy) > 1:
            draw.line(xy, fill=color, width=3, joint="curve")
        if best == "min":
            bx, by = min(sp, key=lambda t: t[1])
        else:
            bx, by = max(sp, key=lambda t: t[1])
        bpx = px0 + pw * bx / xmax
        bpy = py0 + ph - ph * (by - lo) / (hi - lo)
        draw.ellipse([bpx - 6, bpy - 6, bpx + 6, bpy + 6], fill=color, outline=(255, 255, 255), width=2)
        best_pts.append((label, bx, by, bpx, bpy, color))
    # 标题 / 轴名
    draw.text((x0 + w / 2, y0 + 10), title, fill=TXT, font=font(16, True), anchor="ma")
    draw.text((x0 + 14, y0 + 10), ylabel, fill=SUB, font=font(12), anchor="la")
    draw.text((px0 + pw / 2, y0 + h - 14), "Training steps", fill=SUB, font=font(12), anchor="ma")
    # 图例（右上，框内左侧，避免超出右边界）
    lx, ly = px0 + 16, py0 + 12
    for label, color, _ in series:
        draw.line([lx, ly + 7, lx + 18, ly + 7], fill=color, width=3)
        draw.text((lx + 24, ly), label, fill=TXT, font=font(12), anchor="la")
        ly += 20
    return best_pts


def make_loss_image(out):
    act_tr, act_ev = parse_log(ACT_LOG)
    bc_tr, bc_ev = parse_log(BC_LOG)
    # 训练用 L1 重建损失（index 2）；eval 用 eval_loss
    act_tr_l1 = [(s, l1) for s, _, l1, _ in act_tr]
    bc_tr_l1 = [(s, l1) for s, _, l1, _ in bc_tr]
    W, H = 1040, 900
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 16), "ACT vs BC - Training Dynamics", fill=TXT, font=font(22, True), anchor="ma")
    d.text((W / 2, 46), "identical dataset, encoder, optimizer; identical normalization space",
           fill=SUB, font=font(13), anchor="ma")
    b1 = panel(d, 20, 72, W - 40, 380,
               [("ACT (chunk40)", C_ACT, act_tr_l1), ("BC", C_BC, bc_tr_l1)],
               "Training L1 reconstruction loss (normalized action space)",
               "L1 loss", 4000, "min")
    b2 = panel(d, 20, 466, W - 40, 380,
               [("ACT (chunk40)", C_ACT, act_ev), ("BC", C_BC, bc_ev)],
               "Validation loss (held-out episodes, normalized L1)",
               "eval loss", 4000, "min")
    # 最优值脚注（分两行，避免重叠）
    f = font(12)
    d.text((30, 850), "Best training L1 : " + "  |  ".join(
        f"{lb} {by:.3f} @ step {int(bx)}" for lb, bx, by, *_ in b1), fill=C_ACT, font=f, anchor="la")
    d.text((30, 872), "Best validation : " + "  |  ".join(
        f"{lb} {by:.4f} @ step {int(bx)}" for lb, bx, by, *_ in b2), fill=C_BC, font=f, anchor="la")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    print("saved", out)


def make_success_image(json_path, out):
    with open(json_path) as fh:
        data = json.load(fh)
    series = []
    for label, name in [("act", "ACT (chunk40)"), ("bc", "BC")]:
        pts = []
        for step, v in sorted(data["models"].get(label, {}).items(), key=lambda t: int(t[0])):
            pts.append((int(step), v["success_rate"] * 100))
        if pts:
            series.append((name, C_ACT if label == "act" else C_BC, pts))
    W, H = 1040, 620
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 16), "Success Rate vs Training Steps", fill=TXT, font=font(22, True), anchor="ma")
    d.text((W / 2, 46), "identical 50 fixed seeds / model; success = cube within 3 cm of target",
           fill=SUB, font=font(13), anchor="ma")
    best = panel(d, 20, 72, W - 40, 480, series,
                 "Task success rate on identical test set", "success rate (%)",
                 4000, "max", ymin=0, ymax=100, xticks=[1000, 2000, 3000, 4000])
    f = font(13)
    d.text((W / 2, 570), "Peak: " + "  |  ".join(
        f"{lb} {by:.1f}% @ step {int(bx)}" for lb, bx, by, *_ in best),
        fill=TXT, font=f, anchor="ma")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curve-json", default=None,
                    help="CHECKPOINT_CURVE.json 路径；给定则同时画成功率曲线")
    args = ap.parse_args()
    mdir = os.path.join(REPO, "results/metrics")
    make_loss_image(os.path.join(mdir, "LOSS_CURVE_COMPARISON.png"))
    if args.curve_json and os.path.isfile(args.curve_json):
        make_success_image(args.curve_json, os.path.join(mdir, "SUCCESS_CURVE.png"))


if __name__ == "__main__":
    main()
