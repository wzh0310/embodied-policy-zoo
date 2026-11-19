# metrics

| 文件 | 内容 |
|---|---|
| `baseline_metrics.json` | 基线完整评估（成功率 30%，30 回合） |
| `ablation_results.json` | 14 组消融实验的成功率 / 距离误差 / 完成步数 |
| `failure_analysis.json` | 21 个失败回合的三类归因统计 |
| `training_curve.png` | 训练 loss + 验证 loss 曲线（由 plot_training_curve.py 生成） |
| `BC_VS_ACT_COMPARISON.json` | BC 基线 vs 最优 ACT 对齐评估（50 固定种子，成功率/步数/位置误差） |
