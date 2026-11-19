#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACT 训练启动脚本 —— embodied-policy-zoo 统一训练入口。

职责
----
1. 解析本仓库 configs/act_push_cube/ 下的训练配置（默认 act_full_train.yaml）。
2. 先把兄弟仓库 mujoco-studio-lab 的 envs/ 加入 sys.path，触发自定义
   ``push_cube`` 环境注册（LeRobot 0.7.0 的插件发现机制不会自动扫描本环境）。
3. 透传参数给 LeRobot 官方训练入口 ``lerobot.scripts.lerobot_train.main``。

关键参数说明
------------
--config        训练 YAML 路径（相对本仓库根或绝对路径）。
--steps         训练步数；覆盖 YAML。约 80 step/epoch（batch 16）。
--batch-size    批大小；覆盖 YAML。
--output-dir    输出目录；覆盖 YAML。注意 LeRobot 要求该目录**不存在**。
--device        cuda / cpu；覆盖 policy.device。
--force         若输出目录已存在则先删除（默认拒绝，防止误删历史 checkpoint）。

跨仓依赖
--------
数据集与自定义环境由兄弟仓库 mujoco-studio-lab 提供，路径通过环境变量
STUDIO_LAB_ROOT 指定（默认 /data/wzh/robot/mujoco-studio-lab）。

用法（在 lerobot conda 环境）
---------------------------
    python scripts/train.py
    python scripts/train.py --config configs/act_push_cube/act_full_train.yaml \
        --steps 4000 --batch-size 16 --device cuda
"""
import argparse
import os
import shutil
import sys

# ---- 跨仓路径解析：从脚本位置推导本仓库根，再定位兄弟仓库 ----
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

STUDIO_LAB = os.environ.get("STUDIO_LAB_ROOT", "/data/wzh/robot/mujoco-studio-lab")
if not os.path.isdir(STUDIO_LAB):
    raise SystemExit(
        f"[train] 找不到兄弟仓库 mujoco-studio-lab: {STUDIO_LAB}\n"
        "设置环境变量 STUDIO_LAB_ROOT 指向其绝对路径后重试。"
    )

# 关键：在导入 LeRobot 训练入口之前，先 import 自定义环境完成注册。
sys.path.insert(0, STUDIO_LAB)
sys.path.insert(0, os.path.join(STUDIO_LAB, "envs"))
import envs.lerobot_push_cube_env  # noqa: E402,F401


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/act_push_cube/act_full_train.yaml",
                    help="训练配置 YAML（相对本仓库根或绝对路径）")
    ap.add_argument("--steps", type=int, default=None, help="覆盖配置中的 steps")
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖配置中的 batch_size")
    ap.add_argument("--output-dir", default=None, help="覆盖配置中的 output_dir")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="覆盖 policy.device")
    ap.add_argument("--force", action="store_true",
                    help="输出目录已存在时先删除（默认拒绝）")
    return ap.parse_args()


def main():
    args = parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(REPO, args.config)
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"[train] 配置文件不存在: {cfg_path}")
    print("[train] config       =", cfg_path)
    print("[train] studio_root  =", STUDIO_LAB)

    # 读取 YAML 以解析 output_dir（仅用于存在性检查；真正的配置解析由 draccus 完成）。
    import yaml
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    out_dir = args.output_dir or cfg.get("output_dir")
    if out_dir:
        if os.path.exists(out_dir):
            if not args.force:
                raise SystemExit(
                    f"[train] 输出目录已存在: {out_dir}\n"
                    "LeRobot 要求 output_dir 不存在。请换路径或加 --force 删除后重试。"
                )
            print("[train] --force: 删除已存在的输出目录", out_dir)
            shutil.rmtree(out_dir)
        print("[train] output_dir   =", out_dir)

    # 构造透传给 LeRobot 官方入口的参数（draccus 解析 sys.argv）。
    # 环境变量确保渲染走离屏 EGL，无需显示器。
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ["STUDIO_LAB_ROOT"] = STUDIO_LAB

    overrides = [f"--config_path={cfg_path}"]
    if args.steps is not None:
        overrides.append(f"--steps={args.steps}")
        print("[train] steps        =", args.steps)
    if args.batch_size is not None:
        overrides.append(f"--batch_size={args.batch_size}")
        print("[train] batch_size   =", args.batch_size)
    if args.output_dir is not None:
        overrides.append(f"--output_dir={args.output_dir}")
    if args.device is not None:
        overrides.append(f"--policy.device={args.device}")
        print("[train] device       =", args.device)

    print("[train] launching LeRobot official trainer ...")
    sys.argv = ["lerobot_train"] + overrides
    from lerobot.scripts.lerobot_train import main
    main()


if __name__ == "__main__":
    main()
