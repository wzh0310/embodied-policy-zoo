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

动作分块（Action Chunking）消融新增指标（同一测试集 3×10 回合，含动作平滑度）:

- `metrics/CHUNK_ABLATION_SUMMARY.json` —— 四组分块长度对比汇总（含最优判定）
- `metrics/chunk_ablation_chunk{4,8,16,20}.json` —— 各组 4000 步评估明细
- `metrics/chunk_ablation_chunk20_full200ep.json` —— 最优 chunk20 的 12000 步全量训练评估（36.7%）
- `scripts/eval_chunk_ablation.py` —— 带平滑度指标（mean_jerk / direction_flip）的评估入口
- `configs/act_push_cube/ablation/chunk{4,8,16}.yaml` + `chunk20_full200ep.yaml` —— 消融配置
- 完整分析见 `docs/CHUNK_ABLATION_REPORT.md`。结论: chunk20 最优; 全量训练 36.7%, 80% 未达成。

分块上界延伸实验（回答"增大 chunk / 增加步数能否更好"）新增：

- `metrics/chunk_ablation_chunk{30,40,50}.json` —— chunk 上界探针（4000 步）：30→40%、40→60%、50→60%
- `metrics/chunk_ablation_chunk{40,50}_repro.json` —— 全新种子组复现：chunk40 两种子组均 60%（36/60，稳固）
- `metrics/chunk_ablation_chunk20_24k.json` —— 训练量翻倍探针：24000 步过拟合，36.7%→20%
- 配置：`configs/act_push_cube/ablation/chunk{30,40,50}.yaml`、`chunk20_24k.yaml`
- 结论：**最优 chunk_size=40（接近回合上限），成功率 30%→60% 翻倍；增加步数有害（过拟合）**。
  新最优 checkpoint：outputs/act_chunk40/checkpoints/004000。详见 `docs/CHUNK_ABLATION_REPORT.md` 第六节。
