# metrics

| 文件 | 内容 |
|---|---|
| `baseline_metrics.json` | 基线完整评估（成功率 30%，30 回合） |
| `ablation_results.json` | 14 组消融实验的成功率 / 距离误差 / 完成步数 |
| `failure_analysis.json` | 21 个失败回合的三类归因统计 |
| `training_curve.png` | 训练 loss + 验证 loss 曲线（由 plot_training_curve.py 生成） |
| `BC_VS_ACT_COMPARISON.json` | BC 基线 vs 最优 ACT 对齐评估（50 固定种子，成功率/步数/位置误差） |
| `ACTION_MSE.json` | 验证集动作 MSE（teacher-forcing，raw 动作空间） |
| `QUANTITATIVE_TABLE.md` / `QUANTITATIVE_SUMMARY.json` | 四指标量化对比表 |
| `CHECKPOINT_CURVE.json` | 各 checkpoint 成功率（成功率随步数曲线原始数据） |
| `DATA_ABLATION.json` | 数据量消融（20/40/63）原始指标 |
| `LOSS_CURVE_COMPARISON.png` / `SUCCESS_CURVE.png` / `DATA_ABLATION_CURVE.png` | 三张对比曲线图 |
| `RANDOM_POLICY_BASELINE.json` | SB3 环境适配验证 + 随机策略 10 回合基线（平均奖励 -3.15 / 成功率 0% / 4 并行吞吐） |
| `SB3_PPO_SMOKE.json` | SB3 二次验证 + PPO(MLP) 100 步最小闭环（累计奖励 -6.1341，无报错） |
| `PPO_STATE_SIGNAL_CHECK.json` | 状态 PPO 3 万步短训评估曲线（奖励 -3.15→-1.48，rollout 成功率 0→4%） |
| `PPO_STATE_EVAL50.json` | 状态 PPO 50 回合确定性评估（成功率 0%，平均奖励 -1.4832） |
| `PPO_TRAIN_CURVE.png` / `PPO_TRAIN_CURVE_TUNED.png` | PPO 基线/调优训练曲线截图（奖励/成功率/价值损失） |
| `PPO_TUNING_SUMMARY.json` | 训练调优量化汇总（基线震荡、调优点、stochastic 成功率 3%） |
| `PPO_BASELINE_PROGRESS.csv` / `PPO_TUNED_PROGRESS.csv` | PPO 原始训练日志 |
| `VISION_PIPELINE_VERIFY.json` | 视觉管线 9 项验证全 PASS（观测兼容/两档渲染/缓存失效/随机种子/动作归一化/check_env/训练档 481.8fps） |
| `CNN_PPO_VISION_SMOKE.json` | SB3 CnnPolicy + rgb 观测闭环与 state PPO 无回归冒烟结果 |
| `VISION_PPO_10K_SMOKE.json` | 端到端视觉 PPO 1 万步验证（CnnPolicy+NatureCNN+4 帧堆叠，观测 (4,84,84)，损失正常、模型已保存） |
| `GPU_ACCELERATION_FIX.json` | GPU 可用性纠正与性能修复（torch cu132→cu124，视觉 PPO CPU 6fps→GPU 209fps，35 倍提速） |
