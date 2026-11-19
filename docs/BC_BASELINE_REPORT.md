# BC 基线训练：与最优 ACT 的公平对比

> 目标：用**完全相同的数据集、输入/输出、训练节奏与评估协议**训练行为克隆（BC）基线，
> 量化 ACT（CVAE + 动作分块）相对纯模仿学习的净增益。所有对比在同一进程、同一批
> 固定种子下完成，唯一变量是策略算法本身。

---

## 1. 为什么这样定义"BC 基线"

本仓库使用的 LeRobot 0.7.0 定制版**没有独立的 `bc` policy 目录**（policies 下为
act / diffusion / tdmpc / vqbet 等）。获得严格意义上"行为克隆"的原生方式，是把 ACT
的两个算法增量逐一关闭：

| 维度 | ACT（最优） | BC 基线 | 说明 |
|---|---|---|---|
| CVAE 变分目标 | `use_vae: true` | `use_vae: false` | 关闭隐变量建模，退化为确定性回归 |
| 动作分块 | `chunk_size=40` / `n_action_steps=40` | `chunk_size=1` / `n_action_steps=1` | 退化为逐帧单步动作预测 |
| 图像编码器 | ImageNet 预训练 ResNet-18 | **完全相同** | 唯一变量是算法，不是骨干 |
| 观测/动作空间 | state + image → 4 维动作 | **完全相同** | 无手工二次归一化 |
| 数据集 | 63 条合并演示（50 train / 13 eval） | **完全相同** | 同一 `push_cube_demos_merged` |
| 训练步数 / 批量 | 4000 / 32 | **完全相同** | — |
| 学习率 / 调度 | 1e-5 / 余弦退火+warmup100 | **完全相同** | — |
| 图像增强 | 关闭 | **关闭** | — |
| 随机种子 | 1000 | 1000 | — |

因此这是一个**控制变量实验**：两者共享骨干、数据、优化配置与评估集，差异仅来自
"CVAE + 动作分块"这一 ACT 核心机制。BC 基线是 LeRobot 原生的（policy.type 仍为 `act`，
仅关闭上述开关），加载走与 ACT 完全相同的 `ACTConfig.from_pretrained` 路径。

---

## 2. 对齐评估协议

`scripts/eval_bc_comparison.py` 在**同一进程、同一环境实例**内依次加载两个 checkpoint，
用完全相同的 **50 个固定随机种子**（`default_rng(4242)` 生成，每种子 1 回合 = 50 次
离屏 rollout / 模型）评估，消除环境、渲染与随机性差异。立方体初始偏移按种子确定性
注入（x ±0.030 / y ±0.035），与训练数据同分布。

三项统一指标：

1. **成功率**：立方体中心进入目标区域（到目标距离 < 0.03 m）的回合占比；
2. **平均完成步数**：成功回合的平均执行步数；
3. **最终位置误差**：回合结束时立方体中心到目标的平均距离（米）。

---

## 3. 结果

| 指标 | ACT（chunk40, ckpt 004000） | BC（ckpt 004000） | 差异 |
|---|---|---|---|
| **成功率（核心）** | **50.0%**（25/50） | 42.0%（21/50） | **ACT +8.0 pp** |
| 平均完成步数（成功回合） | 35.12 | 34.33 | 接近 |
| **平均位置误差** | **0.0555 m** | 0.1225 m | **ACT 降低 55%** |
| 平均位置误差（成功回合） | 0.0244 m | 0.0231 m | 接近 |
| 平均总奖励 | +5.03 | +4.13 | ACT 更高 |

逐种子拆分（同一 50 种子）：ACT 单独成功 13 回合、BC 单独成功 9 回合、两者均成功
12 回合 → ACT 净胜 4 回合。训练侧：BC 4000 步用时 5 分 19 秒，train loss 收敛到
约 0.09，eval_loss 0.2221，无训练异常。

---

## 4. 分析：ACT 的算法价值体现在哪里

1. **成功率净增益 +8 pp，且逐种子净胜**：在同一测试集、同一数据与训练预算下，ACT
   稳定优于纯 BC，说明 CVAE + 动作分块不是"调参红利"，而是可复现的算法收益。
2. **最大差异在位置误差（0.0555 m vs 0.1225 m）**：BC 的整回合平均距离误差是 ACT
   的约 2.2 倍。BC 逐帧重规划，每步独立预测导致方向抖动与误差累积，回合结束时立方体
   平均离目标更远——这与前序 chunk 消融中"大 chunk 把方向翻转率从 4.5% 降到 0"的
   机制结论一致，从基线侧再次印证。
3. **成功回合两者步数/误差接近**：一旦成功，两条轨迹的完成质量相当；分块的优势主要
   体现在"更高的成功率 + 失败时离目标更近（更易补救）"，而非把单次成功做得更快。
4. **诚实边界**：50 回合下 8 pp 的差距属于中等样本量结论（ACT-only 13 vs BC-only 9），
   不宜夸大为压倒性优势；但配合位置误差 55% 的显著下降，方向性结论稳健。两者均未达
   到实用化的高成功率，这与前序"当前数据+架构天花板"的判断一致——BC 基线的价值在于
   建立公平对照，而非替代 ACT。

---

## 5. 产出清单

| 文件 | 内容 |
|---|---|
| `configs/act_push_cube/bc_baseline.yaml` | BC 基线训练配置（与 ACT 严格对齐） |
| `scripts/eval_bc_comparison.py` | 对齐评估脚本（50 固定种子，双模型同进程） |
| `results/metrics/BC_VS_ACT_COMPARISON.json` | 完整对比指标 + 逐回合明细 |
| `outputs/bc_baseline/checkpoints/004000`（容器） | BC 基线模型 checkpoint |

复现命令（lerobot 环境，容器内）：

```bash
# 训练 BC 基线
python scripts/train.py --config configs/act_push_cube/bc_baseline.yaml

# 同一测试集对齐评估（BC vs ACT）
python scripts/eval_bc_comparison.py \
  --checkpoint act=/data/wzh/robot/outputs/act_chunk40/checkpoints/004000 \
  --checkpoint bc=/data/wzh/robot/outputs/bc_baseline/checkpoints/004000 \
  --out results/metrics/BC_VS_ACT_COMPARISON.json
```
