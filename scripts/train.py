#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACT / BC 训练启动脚本 —— embodied-policy-zoo 统一训练入口。

职责
----
1. 解析本仓库 configs/ 下的训练配置（支持 ${STUDIO_LAB_ROOT} / ${OUTPUTS_DIR} 占位符）。
2. 把兄弟仓库 mujoco-studio-lab 的 envs/ 加入 sys.path，触发自定义 push_cube 环境注册。
3. 透传参数给 LeRobot 官方训练入口 ``lerobot.scripts.lerobot_train.main``。

关键参数
--------
--config      训练 YAML（相对仓库根或绝对路径）
--steps       覆盖 YAML 的 steps
--batch-size  覆盖 YAML 的 batch_size
--output-dir  覆盖 YAML 的 output_dir（LeRobot 要求目录不存在）
--device      cuda / cpu
--force       输出目录已存在时先删除

用法（lerobot 环境、仓库根目录）：

    python scripts/train.py --config configs/act_push_cube/ablation/chunk40.yaml \
        --steps 4000 --device cuda
    python scripts/train.py --config configs/act_push_cube/bc_baseline.yaml --steps 5 \
        --device cpu --output-dir /tmp/bc_smoke --force
"""
import argparse
import os
import shutil
import sys
import tempfile

# ---- 路径解析：由脚本位置推导仓库根，环境适配层统一处理跨仓依赖 ----
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from envs.paths import (  # noqa: E402
    STUDIO_LAB, resolve_config_text, require_studio_lab, setup_sys_path, renderer_env,
)

require_studio_lab()
setup_sys_path()

# 关键：先 import 自定义环境完成注册，再导入 LeRobot 训练入口。
# 直接导入模块名（而非 envs.xxx），与本仓其他脚本一致；sys.path 已由
# setup_sys_path() 把兄弟仓库的 envs/ 置于最前，避免本仓 envs/ 的命名空间冲突。
import lerobot_push_cube_env  # noqa: E402,F401


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/act_push_cube/act_full_train.yaml",
                    help="训练配置 YAML（相对仓库根或绝对路径）")
    ap.add_argument("--steps", type=int, default=None, help="覆盖配置中的 steps")
    ap.add_argument("--batch-size", type=int, default=None, help="覆盖配置中的 batch_size")
    ap.add_argument("--output-dir", default=None, help="覆盖配置中的 output_dir")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"],
                    help="覆盖 policy.device")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="覆盖 DataLoader worker 数（0 = 单进程，便于冒烟/调试）")
    ap.add_argument("--force", action="store_true",
                    help="输出目录已存在时先删除（默认拒绝，防止误删历史 checkpoint）")
    return ap.parse_args()


def main():
    args = parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(REPO, args.config)
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"[train] 配置文件不存在: {cfg_path}")

    # 展开配置占位符，写到原配置同目录（保持相对路径解析一致），训练后删除。
    with open(cfg_path, encoding="utf-8") as fh:
        raw = resolve_config_text(fh.read())
    cfg_dir = os.path.dirname(cfg_path)
    fd, materialized = tempfile.mkstemp(
        prefix=".materialized_", suffix=".yaml", dir=cfg_dir
    )
    os.close(fd)
    with open(materialized, "w", encoding="utf-8") as fh:
        fh.write(raw)

    try:
        import yaml
        cfg = yaml.safe_load(raw)
        out_dir = args.output_dir or cfg.get("output_dir")
        if out_dir:
            if os.path.exists(out_dir):
                if not args.force:
                    raise SystemExit(
                        f"[train] 输出目录已存在: {out_dir}\n"
                        "LeRobot 要求 output_dir 不存在。请换路径或加 --force 后重试。"
                    )
                print("[train] --force: 删除已存在的输出目录", out_dir)
                shutil.rmtree(out_dir)
            print("[train] output_dir   =", out_dir)

        for k, v in renderer_env().items():
            if k == "STUDIO_LAB_ROOT":
                os.environ[k] = v
            else:
                os.environ.setdefault(k, v)

        print("[train] config       =", cfg_path)
        print("[train] studio_root  =", STUDIO_LAB)

        overrides = [f"--config_path={materialized}"]
        if args.steps is not None:
            overrides.append(f"--steps={args.steps}")
            print("[train] steps        =", args.steps)
        if args.batch_size is not None:
            overrides.append(f"--batch_size={args.batch_size}")
            print("[train] batch_size   =", args.batch_size)
        if args.output_dir is not None:
            overrides.append(f"--output_dir={args.output_dir}")
        if args.num_workers is not None:
            overrides.append(f"--num_workers={args.num_workers}")
            print("[train] num_workers  =", args.num_workers)
        if args.device is not None:
            overrides.append(f"--policy.device={args.device}")
            print("[train] device       =", args.device)

        print("[train] launching LeRobot official trainer ...")
        sys.argv = ["lerobot_train"] + overrides
        from lerobot.scripts.lerobot_train import main
        main()
    finally:
        if os.path.exists(materialized):
            os.remove(materialized)


if __name__ == "__main__":
    main()
