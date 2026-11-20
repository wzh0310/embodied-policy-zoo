# 状态输入 PPO 训练与评估

> 目标：完成低维状态输入的 PPO 训练全流程脚本开发，先快速验证奖励函数与学习信号有效性。
> 全程 `mujoco` 环境，动作空间沿用末端笛卡尔增量（±0.02m），与模仿学习一致。

## 1. 交付脚本

| 脚本 | 作用 |
|---|---|
| `algos/ppo/train_state_ppo.py` | PPO(`MlpPolicy`) 训练，可一键启动 |
| `utils/evaluate.py` | 统一评估（成功率 / 平均回合步数 / 平均回合奖励） |

### 1.1 训练脚本要点

- 策略：`PPO("MlpPolicy")`，观测用 13 维 `state`，headless CPU 可跑。
- 默认超参数（面向桌面操作任务）：

| 参数 | 默认值 |
|---|---|
| 总训练步数 `total_timesteps` | 500,000 |
| 单批次步数 `n_steps` | 2048 |
| 学习率 `learning_rate` | 3e-4 |
| 熵系数 `ent_coef` | 0.01 |
| batch_size / n_epochs / gamma / gae_lambda / clip_range | 256 / 10 / 0.99 / 0.95 / 0.2 |

- 工程能力：定期 `CheckpointCallback` 保存、SB3 标准 logger 输出 CSV + TensorBoard
  （`tensorboard --logdir <out_dir>/logs`）、训练中每 `eval_freq` 步跑确定性回合监控
  成功率/奖励。所有超参数可用命令行覆盖，`--n-envs` 可选并行（state 模式 fork 安全）。
- 路径遵循仓库约定：输出默认 `../outputs/ppo_state`，可用 `OUTPUTS_DIR` 覆盖；
  环境依赖由 `envs/paths.py` 解析 `STUDIO_LAB_ROOT`。

### 1.2 评估脚本要点

固定回合数（默认 50）跑确定性策略，输出三指标；可作为函数 `rollout_metrics` 被训练
脚本复用，也可命令行独立评估已保存模型：

```bash
python utils/evaluate.py --model <final_model.zip> --episodes 50 --out <result.json>
```

## 2. 学习信号快速验证（3 万步短训）

为在投入 50 万步前先确认奖励函数与学习信号有效，跑了一次 30,000 步短训（单环境，
lr 3e-4，ent 0.01，其余默认），每 5000 步评估 20 回合：

| 训练步数 | 评估回合奖励 | 训练 rollout 成功率 |
|---|---|---|
| 5,000 | -1.63 | 0% |
| 10,000 | -1.78 | 0% |
| 15,000 | -1.71 | 1% |
| 20,000 | -1.75 | 1% |
| 25,000 | -1.87 | 4% |
| 30,000 | -1.76 | 4% |

**结论：学习信号有效。** 评估回合奖励从随机基线的 **-3.15** 提升到 **-1.48**
（50 回合确定性评估，见 `results/metrics/PPO_STATE_EVAL50.json`），提升约 53%；
训练 rollout 成功率从 0 上升到 4%，价值函数 `explained_variance` 稳定在 0.5~0.8。
奖励函数（距离缩减塑形 + 接近引导 + 时间惩罚 + 稀疏成功）提供了清晰的优化梯度。

### 评估方差说明

当前环境 `reset(seed=)` 在不同 seed 下返回相同的初始立方体位姿（MJCF keyframe 固定，
seed 未用于随机化立方体初始位置）。因此**确定性策略**的 50 回合评估结果完全确定
（std=0），而随机基线的方差来自动作采样。这不影响奖励/成功率的有效比较，但若要评估
策略鲁棒性，需后续为环境加入初始位姿随机化（超出本次脚本开发范围，且改动会影响与
已有模仿学习对比的公平性，故未改动）。

## 3. 一键复现

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mujoco
cd embodied-policy-zoo
# 完整训练（默认 50 万步）
python algos/ppo/train_state_ppo.py
# 快速验证（3 万步）
python algos/ppo/train_state_ppo.py --total-timesteps 30000 --eval-freq 5000
# 评估
python utils/evaluate.py --model <out_dir>/<name>/final_model.zip --episodes 50
```

## 4. 数据

- `results/metrics/PPO_STATE_SIGNAL_CHECK.json`：短训评估曲线与配置
- `results/metrics/PPO_STATE_EVAL50.json`：50 回合确定性评估结果
