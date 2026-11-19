# configs —— 训练配置

## 路径占位符（无硬编码）

配置 YAML 中的路径使用占位符，训练入口 `scripts/train.py` 会在解析前自动展开：

| 占位符 | 展开为 | 覆盖方式 |
|---|---|---|
| `${STUDIO_LAB_ROOT}` | 兄弟仓库 `mujoco-studio-lab` 路径 | 环境变量 `STUDIO_LAB_ROOT` |
| `${OUTPUTS_DIR}` | 训练输出根目录 | 环境变量 `OUTPUTS_DIR` 或 `--output-dir` |

因此克隆到任意机器都无需修改配置文件。

## act_push_cube/ —— 主配置

| 配置 | 用途 |
|---|---|
| `act_full_train.yaml` | 全量训练（4000 步 / batch16），基线 30% 由此产出 |
| `act_push_cube.yaml` | 冒烟训练（小步数验证全链路） |
| `act_baseline.yaml` | 短周期基线 |
| `final_best_train.yaml` | 图像增强实验（完整训练下成功率 13.3%，证明增强有害） |
| `tuning_train.yaml` | 性能调优（数据扩展 + batch32 + 余弦退火） |
| `bc_baseline.yaml` | BC 基线（无 CVAE + chunk1，与最优 ACT 公平对比） |

## act_push_cube/ablation/ —— 分块与数据量消融

- `chunk{4,8,16,20,30,40,50}.yaml` —— 动作分块长度全量消融
- `chunk20_full200ep.yaml` / `chunk20_24k.yaml` —— 训练量探针（12000 / 24000 步）
- `data{20,40,63}.yaml` —— 数据量消融（演示条数 20 / 40 / 63）

## act_push_cube/ablations/ —— 控制变量消融

14 组早期控制变量实验配置，按实验组命名：

- `ablation_chunk{8,10,12,16}.yaml` —— 动作块长度
- `ablation_lr{1e4,3e4,5e4}.yaml` —— 学习率
- `ablation_nas{1,4,10}.yaml` —— 闭环重规划频率（n_action_steps）
- `ablation_smooth.yaml` —— 时序集成平滑
- `ablation_augment.yaml` / `ablation_aug_lr1e4.yaml` —— 图像增强
- `ablation_base_s.yaml` —— 短训纯基线对照

## 结果与报告

实验结果见 [`../results/README.md`](../results/README.md)，完整分析见
[`../docs/TUNING_REPORT.md`](../docs/TUNING_REPORT.md) 与
[`../docs/CHUNK_ABLATION_REPORT.md`](../docs/CHUNK_ABLATION_REPORT.md)。
