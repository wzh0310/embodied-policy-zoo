#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Teacher-forcing 计算验证集动作 MSE —— 与训练时 held-out split 严格一致。

 MSE 口径
--------
* 验证集 = 合并数据集最后 ceil(63*0.2)=13 个 episode（index 50..62），即训练时
  LeRobot make_train_eval_datasets 切出的 eval 集，两个模型完全相同。
* 对每个验证帧：把 raw observation 送进该 checkpoint 自带的 pre processor，
  再走 policy.predict_action_chunk 得到归一化动作 chunk，取第 0 步（当前帧对应的
  那只动作），经 post processor 反归一化回 raw 动作，与该帧数据集里的 raw action
  求逐维平方误差。全验证集逐元素平均 = 验证集动作 MSE。
* 单位：raw 动作空间 (dx,dy,dz,gripper) 的均方误差；同时给出 cm 尺度下的
  dx/dy/dz 平移 MSE 与整体 MSE，便于写进表格。

 注意：ACT 的 eval_loss 含 KL 且是归一化空间的 L1，不是动作 MSE；这里专门算
 raw 空间 MSE，作为与 rollout 成功率互补的"动作拟合精度"指标。

 用法（lerobot 环境）
--------------------
    python scripts/eval_action_mse.py \
        --checkpoint act=/path/to/outputs/act_chunk40/checkpoints/004000 \
        --checkpoint bc=/path/to/outputs/bc_baseline/checkpoints/004000 \
        --out results/metrics/ACTION_MSE.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STUDIO_LAB = os.environ.get("STUDIO_LAB_ROOT", os.path.join(os.path.dirname(REPO), "mujoco-studio-lab"))
DATA_ROOT = os.path.join(STUDIO_LAB, "data", "push_cube_demos_merged")
for p in (REPO, STUDIO_LAB, os.path.join(STUDIO_LAB, "envs")):
    if p not in sys.path:
        sys.path.insert(0, p)

import lerobot_push_cube_env as lc  # noqa: E402,F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.envs import make_env_config  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402

EVAL_SPLIT = 0.2


def load_policy(ckpt_dir):
    pretrained = os.path.join(ckpt_dir, "pretrained_model")
    if not os.path.isdir(pretrained):
        pretrained = ckpt_dir
    cfg = ACTConfig.from_pretrained(pretrained)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = make_env_config("push_cube")
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, pretrained_path=pretrained)
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=pretrained)
    return policy, pre, post, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH")
    ap.add_argument("--out", default="results/metrics/ACTION_MSE.json")
    args = ap.parse_args()

    checkpoints = {}
    for item in args.checkpoint:
        label, path = item.split("=", 1)
        checkpoints[label] = path

    # 复现 eval split：最后 ceil(n*0.2) 个 episode
    probe = LeRobotDataset(repo_id="push_cube_expert", root=DATA_ROOT,
                           video_backend="pyav")
    n = probe.num_episodes
    n_eval = math.ceil(n * EVAL_SPLIT)
    eval_eps = list(range(n - n_eval, n))
    print(f"[mse] total_episodes={n} eval_episodes={eval_eps} ({len(eval_eps)})")

    ds = LeRobotDataset(repo_id="push_cube_expert", root=DATA_ROOT,
                        episodes=eval_eps, video_backend="pyav")
    print(f"[mse] eval frames = {len(ds)}")

    # 预抽所有 eval 帧的 raw observation / action（避免每模型重复解码视频）
    imgs, states, actions = [], [], []
    for i in range(len(ds)):
        row = ds[i]
        im = row["observation.images.main"]
        if isinstance(im, torch.Tensor):
            im = im.numpy()
        im = np.asarray(im)
        # 转 CHW float32 [0,1]
        if im.ndim == 3 and im.shape[-1] == 3:
            im = np.transpose(im, (2, 0, 1))
        imgs.append(im.astype(np.float32) / 255.0 if im.max() > 1.5 else im.astype(np.float32))
        states.append(np.asarray(row["observation.state"], dtype=np.float32))
        actions.append(np.asarray(row["action"], dtype=np.float32))
    imgs = np.stack(imgs)
    states = np.stack(states)
    actions = np.stack(actions)
    print(f"[mse] stacked images {imgs.shape} states {states.shape} actions {actions.shape}")

    results = {}
    for label, path in checkpoints.items():
        policy, pre, post, cfg = load_policy(path)
        device = next(policy.parameters()).device
        preds = []
        bs = 64
        with torch.inference_mode():
            for s in range(0, len(imgs), bs):
                canon = {
                    "observation.images.main": torch.from_numpy(
                        np.ascontiguousarray(imgs[s:s+bs])).to(device),
                    "observation.state": torch.from_numpy(
                        np.ascontiguousarray(states[s:s+bs])).to(device),
                }
                proc = pre(canon)
                chunk = policy.predict_action_chunk(proc)   # (B, chunk, 4) normalized
                first = chunk[:, 0, :]                       # 当前步对应的动作
                raw = post(first)                            # 反归一化 -> raw action
                if isinstance(raw, dict):
                    raw = raw["action"]
                preds.append(raw.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        err = preds - actions
        mse_all = float(np.mean(err ** 2))
        mse_trans = float(np.mean(err[:, :3] ** 2))          # dx/dy/dz
        mse_trans_cm = mse_trans * 1e4                        # m^2 -> cm^2
        per_dim = [float(v) for v in np.mean(err ** 2, axis=0)]
        results[label] = {
            "checkpoint": path,
            "n_frames": int(len(actions)),
            "eval_episodes": eval_eps,
            "action_mse_raw_all": mse_all,
            "action_mse_translation_m2": mse_trans,
            "action_mse_translation_cm2": mse_trans_cm,
            "action_rmse_translation_cm": float(math.sqrt(mse_trans_cm)),
            "per_dim_mse": per_dim,
        }
        print(f"[mse] {label:6s} MSE(all)={mse_all:.6f} "
              f"trans_RMSE={math.sqrt(mse_trans_cm):.3f}cm "
              f"per_dim={[round(v,6) for v in per_dim]}")
        del policy, pre, post

    out = {"metric": "validation action MSE (teacher-forcing, raw action space)",
           "eval_split": EVAL_SPLIT, "models": results}
    out_path = os.path.join(REPO, "results/metrics", args.out) \
        if not (os.path.isabs(args.out) or args.out.startswith("results/metrics")) \
        else (args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("[mse] wrote", out_path)
    print("MSE DONE")


if __name__ == "__main__":
    main()
