# results

本目录存放交付的复现产物（均已入库）：

- `logs/` —— 训练日志（基线 + 增强实验各一份完整 4000 步日志）
- `metrics/` —— 量化指标 JSON、14 组消融结果、失败分析、训练曲线 PNG
- `videos/` —— 双视角演示、成功回合、专家演示 MP4 + 关键帧截图

所有指标由 `scripts/eval.py` 在固定种子（3×10=30 回合）下产出，曲线由 `scripts/plot_training_curve.py` 从日志生成。

本轮基础训练调优（数据分布扩展 + batch32 + 余弦退火）新增指标：

- `metrics/tuning_final_metrics.json` —— 调优模型标准评估（3×10 回合）：成功率 30%、平均步数 34.1、距离误差 0.145 m
- `metrics/tuning_wide_metrics.json` —— 调优模型宽偏移鲁棒性探针：成功率 10%
- `metrics/baseline_wide_metrics.json` —— 基线模型同款宽偏移探针：成功率 0%

调优训练日志见 `results/logs/tuning_train.log`（受 .gitignore 的 `*.log` 规则约束未入库，仅存于运行环境）；
完整分析见 `docs/PERFORMANCE_TUNING_REPORT.md`。
