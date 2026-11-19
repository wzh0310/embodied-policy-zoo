# results

本目录存放第 3 周交付的复现产物（均已入库）：

- `logs/` —— 训练日志（基线 + 增强实验各一份完整 4000 步日志）
- `metrics/` —— 量化指标 JSON、14 组消融结果、失败分析、训练曲线 PNG
- `videos/` —— 双视角演示、成功回合、专家演示 MP4 + 关键帧截图

所有指标由 `scripts/eval.py` 在固定种子（3×10=30 回合）下产出，曲线由 `scripts/plot_training_curve.py` 从日志生成。
