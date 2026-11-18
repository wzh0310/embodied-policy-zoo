#!/usr/bin/env bash
# =============================================================================
# 一键启动：自定义 push_cube 场景的 ACT 冒烟训练（LeRobot 0.7.0）
# =============================================================================
# 复用关系：
#   - 数据集 / 环境：来自 mujoco-studio-lab 仓库（本工作区的兄弟仓库）
#   - 训练配置：本仓库 configs/act_push_cube.yaml（由官方 ACT 模板适配）
#   - 训练入口：mujoco-studio-lab/scripts/train_act_push_cube.py（wrapper，
#               先注册 push_cube 环境再进入 lerobot 官方训练）
#
# 环境约定：
#   - 用 `lerobot` conda 环境（训练 / 数据处理都在此环境）
#   - 可视化渲染走离屏 EGL（无需显示器）
#   - 本容器 GPU 驱动(12.4)与 torch(cu132)不匹配 -> 强制 CPU 训练
#
# 用法：
#   bash scripts/train_act_push_cube.sh
#   STEPS=200 BATCH=8 bash scripts/train_act_push_cube.sh   # 覆盖规模
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZOO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STUDIO_LAB="${STUDIO_LAB_ROOT:-/data/wzh/robot/mujoco-studio-lab}"

# LeRobot conda 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lerobot

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TQDM_DISABLE=1
export STUDIO_LAB_ROOT="${STUDIO_LAB}"

# 参数（可用环境变量覆盖）
STEPS="${STEPS:-100}"
BATCH="${BATCH:-8}"
CONFIG="${CONFIG:-${STUDIO_LAB}/configs/act_push_cube.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/wzh/robot/outputs/act_push_cube_smoke}"
TRAINER="${STUDIO_LAB}/scripts/train_act_push_cube.py"

# lerobot 要求 output_dir 不存在（resume=false），先清理
rm -rf "${OUTPUT_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "[train-act-push-cube] env=lerobot config=${CONFIG}"
echo "[train-act-push-cube] steps=${STEPS} batch=${BATCH} output=${OUTPUT_DIR}"
echo "[train-act-push-cube] trainer=${TRAINER}"

cd "${STUDIO_LAB}"
python -u "${TRAINER}" \
  --config_path="${CONFIG}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH}" \
  2>&1 | tee "${OUTPUT_DIR}.log"

echo "[train-act-push-cube] DONE (exit=${PIPESTATUS[0]})"
# 产物：
#   ${OUTPUT_DIR}/checkpoints/<step>/pretrained_model/{config.json,model.safetensors,...}
#   评估视频：checkpoints/<step>/eval/videos/*.mp4（env_eval_freq 触发）
