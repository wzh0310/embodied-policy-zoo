# results —— 实验结果归档

所有指标均在容器内由统一评估脚本产出，原始 JSON 与图表均已入库。
**跨实验可比的前提**：固定种子协议 + 统一指标定义；不同实验的测试集在表格中显式标注。

## 目录

| 目录 | 内容 |
|---|---|
| `logs/` | 训练日志（基线 / 增强 / chunk20 / 调优） |
| `metrics/` | 量化指标 JSON、对比表格、训练与消融曲线图 |
| `videos/` | 录制中间片段（`.clips/`，已忽略）；成片见 `assets/` |

## 核心实验条件速查

| 实验 | 数据 | 训练配置 | 评估协议 | 结果 |
|---|---|---|---|---|
| 基线 | 40 条 | chunk20 / 4000 步 / batch16 | 3 种子 ×10 = 30 回合 | 30.0% |
| 性能调优 | 63 条合并 | chunk20 / batch32 / lr1e-5 / 余弦退火 / 4000 步 | 30 回合 | 30.0%；宽偏移鲁棒性 0%→10% |
| chunk 消融 | 63 条 | chunk 4/8/16/20/30/40/50 / 4000 步 | 30 回合 | 13/13/23/30/40/60/60% |
| 训练量探针 | 63 条 | chunk20 / 24000 步 | 30 回合 | 20%（过拟合） |
| BC 基线 | 63 条 | 无 CVAE + chunk1 / 4000 步 | **50 固定种子** | 42.0% |
| 最优 ACT | 63 条 | chunk40 / 4000 步 | **50 固定种子** | 50.0% |
| 数据量消融 | 20 / 40 / 63 条 | chunk40 / 4000 步 | 50 固定种子 | 24% / 34% / 50% |

> 注意：早期实验用 30 回合协议，BC 对比及之后的实验统一用 50 固定种子协议。
> 两组成功率不可直接混比；同一表格内的对比均使用相同协议。

## 关键指标文件

| 文件 | 内容 |
|---|---|
| `metrics/BC_VS_ACT_COMPARISON.json` | ACT vs BC 对齐评估（50 种子，成功率/步数/位置误差，含逐回合明细） |
| `metrics/ACTION_MSE.json` | 验证集动作 MSE（teacher-forcing，13 条 held-out episode / 488 帧） |
| `metrics/QUANTITATIVE_TABLE.md` | 四指标对比表（可直接用于简历/文档） |
| `metrics/CHECKPOINT_CURVE.json` | 8 个 checkpoint 的成功率（训练/成功率曲线原始数据） |
| `metrics/DATA_ABLATION.json` | 数据量消融（20/40/63）原始指标 |
| `metrics/CHUNK_ABLATION_SUMMARY.json` | chunk 全量消融汇总 |
| `metrics/chunk_ablation_*.json` | 各 chunk 配置评估明细（含复现组） |
| `metrics/ablation_results.json` | 14 组控制变量消融 |
| `metrics/failure_analysis.json` | 失败回合归因统计 |
| `metrics/tuning_*.json` / `baseline_wide_metrics.json` | 调优与宽偏移鲁棒性探针 |

## 图表

| 文件 | 内容 |
|---|---|
| `metrics/LOSS_CURVE_COMPARISON.png` | ACT vs BC 训练 L1 loss / 验证 loss 曲线 |
| `metrics/SUCCESS_CURVE.png` | 成功率随 checkpoint 步数变化（ACT 44→30→60→50%，BC 18→12→32→42%） |
| `metrics/DATA_ABLATION_CURVE.png` | 数据量-成功率折线（24→34→50%） |
| `metrics/training_curve.png` | 基线训练/验证 loss 曲线 |

演示视频与关键帧见 [`../assets/`](../assets/README.md)，实验分析见 [`../docs/`](../docs/)。
