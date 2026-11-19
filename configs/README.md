# configs

## act_push_cube/ —— ACT 推立方体训练配置

| 配置 | 用途 |
|---|---|
| `act_full_train.yaml` | **全量正式训练**（GPU / 4000 步 / batch16 / 8:2 split / 每 5 epoch 存 checkpoint）。基线 30% 即由此配置产出。 |
| `act_push_cube.yaml` | 冒烟训练（小步数，验证全链路） |
| `act_baseline.yaml` | 10 epoch 基线（第二周里程碑） |
| `final_best_train.yaml` | 图像增强实验配置（完整 4000 步，成功率 13.3%，证明增强有害） |

## act_push_cube/ablations/ —— 消融实验配置

第 5–6 周 14 组控制变量实验的配置变体，按实验组命名：

- `ablation_chunk{8,10,12,16}.yaml` —— 动作 chunk 长度
- `ablation_lr{1e4,3e4,5e4}.yaml` —— 学习率
- `ablation_nas{1,4,10}.yaml` —— 闭环重规划频率（n_action_steps）
- `ablation_smooth.yaml` —— 时序集成平滑
- `ablation_augment.yaml` / `ablation_aug_lr1e4.yaml` —— 图像增强
- `ablation_base_s.yaml` —— 短训纯基线对照（1600 步）

实验结果见 `results/metrics/ablation_results.json`，完整分析见 `docs/TUNING_REPORT.md`。
