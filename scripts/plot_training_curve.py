#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 LeRobot 训练日志解析 loss 并绘制训练曲线 PNG（仅依赖 Pillow）。

日志中的两个关键模式：
  step:N ... loss:X ... l1_loss:... kld_loss:...   （训练 loss，每 log_freq 步）
  step N: eval_loss=Z                              （验证 loss，每 eval_steps 步）

用法：
    python scripts/plot_training_curve.py \
        --log /data/wzh/robot/outputs/act_push_cube_full_train.log \
        --out results/metrics/training_curve.png
"""
import argparse
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

# 训练 loss 行：形如 "step:20 smpl:320 ... loss:1.568 ... l1_loss:0.569 kld_loss:0.999"
# 后段 tqdm 用紧凑格式 "step:4K smpl:62K ..." —— step 支持 K 后缀。
# l1/kld 为可选，容错不同日志格式。
RE_TRAIN = re.compile(
    r"step:(\d+[KkMm]?)\s+smpl:\S+\s+ep:\S+\s+epch:\S+\s+loss:([\d.eE+\-]+)"
    r"(?:.*?l1_loss:([\d.eE+\-]+))?(?:.*?kld_loss:([\d.eE+\-]+))?"
)
# 验证 loss 行：形如 "step 200: eval_loss=0.2812" 或 "step:4000: eval_loss=0.1630"
RE_EVAL = re.compile(r"step\s*:?\s*(\d+[KkMm]?):\s*eval_loss=([\d.eE+\-]+)")


def _to_int(token):
    """将 '20' / '4K' / '2M' 之类的紧凑计数转换为整数。"""
    token = token.upper()
    mult = 1
    if token.endswith("K"):
        mult, token = 1000, token[:-1]
    elif token.endswith("M"):
        mult, token = 1_000_000, token[:-1]
    return int(float(token) * mult)


def parse_log(path):
    train, evals = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = RE_TRAIN.search(line)
            if m:
                l1 = float(m.group(3)) if m.group(3) else 0.0
                kld = float(m.group(4)) if m.group(4) else 0.0
                train.append((_to_int(m.group(1)), float(m.group(2)), l1, kld))
            m = RE_EVAL.search(line)
            if m:
                evals.append((_to_int(m.group(1)), float(m.group(2))))
    # 去重：同一 step 保留最后一条（tqdm 会重复刷新）
    def dedup(seq):
        d = {}
        for row in seq:
            d[row[0]] = row
        return [d[k] for k in sorted(d)]
    return dedup(train), dedup(evals)


def get_font(size):
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="results/metrics/training_curve.png")
    ap.add_argument("--title", default="ACT Training Loss (push_cube, full run 4000 steps)")
    args = ap.parse_args()

    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    train, evals = parse_log(args.log)
    if not train and not evals:
        raise SystemExit(f"[plot] 未从日志解析到 loss 数据: {args.log}")

    W, H = 900, 520
    L, R, T, B = 70, 30, 70, 55
    pw, ph = W - L - R, H - T - B
    img = Image.new("RGB", (W, H), "#FFFFFF")
    d = ImageDraw.Draw(img)
    f_title = get_font(20)
    f_label = get_font(13)
    f_small = get_font(11)

    max_step = max([r[0] for r in train] + [r[0] for r in evals] + [1])
    all_loss = [r[1] for r in train] + [r[2] for r in train] + [r[3] for r in train] + \
               [r[1] for r in evals]
    y_max = max(all_loss) * 1.05 if all_loss else 1.0
    y_max = max(y_max, 0.1)

    def xpos(step):
        return L + int(pw * step / max_step)
    def ypos(v):
        return T + ph - int(ph * min(v, y_max) / y_max)

    # 标题
    d.text((L, 22), args.title, fill="#1A1A1A", font=f_title)
    d.text((L, 48), "Blue=total loss  Orange=L1(action MSE)  Green=eval loss",
           fill="#555555", font=f_small)

    # 网格 + y 轴刻度
    for i in range(6):
        v = y_max * i / 5
        y = ypos(v)
        d.line((L, y, L + pw, y), fill="#E5E5E5", width=1)
        d.text((L - 46, y - 7), f"{v:.2f}", fill="#555555", font=f_small)
    # x 轴刻度
    for i in range(6):
        s = max_step * i / 5
        x = xpos(s)
        d.line((x, T, x, T + ph), fill="#F0F0F0", width=1)
        d.text((x - 12, T + ph + 8), f"{int(s)}", fill="#555555", font=f_small)

    # 绘制折线
    def draw_series(series, idx, color, width=2):
        if len(series) < 2:
            return
        pts = [(xpos(r[0]), ypos(r[idx])) for r in series]
        d.line(pts, fill=color, width=width)

    draw_series(train, 1, "#1F6FEB", 2)   # total loss
    draw_series(train, 2, "#E38616", 1)   # l1 loss
    draw_series(evals, 1, "#1A7F37", 3)   # eval loss

    # 标注最终值（右对齐，避免超出图框被裁切）
    if evals:
        fs, fv = evals[-1]
        label = f"final eval_loss={fv:.4f} @step {fs}"
        bbox = d.textbbox((0, 0), label, font=f_small)
        tw = bbox[2] - bbox[0]
        d.text((L + pw - tw - 8, ypos(fv) - 22), label,
               fill="#1A7F37", font=f_small)

    # 图框与轴标题
    d.rectangle((L, T, L + pw, T + ph), outline="#BBBBBB", width=1)
    d.text((L + pw // 2 - 30, H - 22), "training step", fill="#333333", font=f_label)
    d.text((8, T + ph // 2 - 30), "loss", fill="#333333", font=f_label)

    img.save(out)
    print("[plot] train points=%d eval points=%d" % (len(train), len(evals)))
    print("[plot] saved", out)
    print("PLOT DONE")


if __name__ == "__main__":
    main()
