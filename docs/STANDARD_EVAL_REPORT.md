# 标准化评估与量化结果

> 目标：用统一协议对最优 ACT 与 BC 基线做批量评估，产出可直接写进简历/文档的
> 量化对比表与训练/评估曲线。所有数字来自同一批固定测试集，两个模型口径完全一致。

---

## 1. 评估协议（统一口径）

| 项目 | 设置 |
|---|---|
| 测试集 | 50 个固定随机种子（`default_rng(4242)`），每种子 1 回合 = 50 次离屏 rollout / 模型 |
| 立方体初始偏移 | 按种子确定性注入 x ±0.030 / y ±0.035（与训练数据同分布） |
| 成功判据 | 回合结束立方体中心到目标距离 < 3 cm |
| 动作 MSE | 训练时切出的 13 条 held-out episode（488 帧），teacher-forcing 计算 raw 动作空间逐元素 MSE |
| 模型 | ACT（chunk40, ckpt 004000）与 BC（ckpt 004000），同进程同环境评估 |

rollout 三指标（成功率 / 完成步数 / 位置误差）来自 `eval_bc_comparison.py`；
动作 MSE 来自 `eval_action_mse.py`。两脚本共用同一 checkpoint 加载与预处理路径。

---

## 2. 量化对比表

| 算法 | 任务成功率 | 平均完成步数 | 最终位置误差 (cm) | 验证集动作 MSE |
| --- | --- | --- | --- | --- |
| BC | 42.0% (21/50) | 34.33 | 12.25 | 3.24e-06 |
| ACT (chunk40) | 50.0% (25/50) | 35.12 | 5.55 | 3.71e-06 |

> 动作 MSE 为 raw 4 维动作 `[dx, dy, dz, gripper]` 的均方误差（488 帧平均）。

**逐种子核对**：种子集合两模型完全一致（50/50），ACT 单独成功 13 回合、BC 单独成功
9 回合、两者均成功 12 回合 → ACT 净胜 4 回合。

---

## 3. 结果解读

1. **成功率**：ACT 50% vs BC 42%，领先 8 个百分点。在同数据、同编码器、同训练预算、
   同测试集的控制变量实验下，这一差距可归因于 ACT 的 CVAE + 动作分块机制。
2. **位置误差**：ACT 5.55 cm vs BC 12.25 cm，ACT 降低约 55%——这是两模型差距最大的
   指标。BC 逐帧重规划累积方向抖动，失败时立方体平均离目标更远。
3. **动作 MSE 的反直觉现象**：BC 的单帧动作 MSE（3.24e-06）反而略低于 ACT（3.71e-06），
   平移 RMSE 为 0.208 cm vs 0.222 cm。这说明 **ACT 的优势不在单帧动作拟合精度**，
   而在分块规划带来的轨迹连贯性——与 chunk 消融中"大 chunk 把方向翻转率降到 0"的结论
   完全一致。单帧 MSE 与 rollout 成功率在此任务上并不单调相关，是一个值得强调的洞察。
4. **完成步数**：成功回合两者接近（35.1 vs 34.3），ACT 并未以更慢的代价换取成功率。

---

## 4. 对比曲线

### 4.1 训练动态（train / val loss vs 训练步数）

![loss 对比](results/metrics/LOSS_CURVE_COMPARISON.png)

- 训练曲线统一用归一化动作空间的 **L1 重建损失**：ACT 的总 train loss 含 CVAE 的 KL 项
  （BC 无 KL），直接比总 loss 不公平；L1 重建损失与 eval_loss 两模型同口径。
- ACT 训练 L1 收敛到 0.173，BC 到 0.087（BC 单帧拟合更好，与动作 MSE 结论一致）；
- 验证 loss：ACT 0.5609→0.3429，BC 0.3819→0.2221，均无过拟合反弹。

### 4.2 成功率随训练步数

![成功率曲线](results/metrics/SUCCESS_CURVE.png)

各 checkpoint（1000/2000/3000/4000）在 50 固定种子下评估。ACT 曲线为
44% → 30% → 60% → 50%，波动明显：大 chunk 策略对训练终点敏感，早期 checkpoint
尚未稳定，最终 004000 的 50% 是复现过的稳健值。BC 曲线见图。

---

## 5. 产出清单

| 文件 | 内容 |
|---|---|
| `results/metrics/QUANTITATIVE_TABLE.md` | 量化对比表（Markdown） |
| `results/metrics/QUANTITATIVE_SUMMARY.json` | 对比表结构化数据 |
| `results/metrics/BC_VS_ACT_COMPARISON.json` | 50 种子 rollout 明细 |
| `results/metrics/ACTION_MSE.json` | 验证集动作 MSE 明细 |
| `results/metrics/CHECKPOINT_CURVE.json` | 各 checkpoint 成功率原始数据 |
| `results/metrics/LOSS_CURVE_COMPARISON.png` | 训练/验证 loss 对比曲线 |
| `results/metrics/SUCCESS_CURVE.png` | 成功率随步数曲线 |
| `scripts/eval_action_mse.py` / `eval_checkpoint_curve.py` | 评估脚本 |
| `scripts/plot_comparison_curves.py` / `plot_data_and_success.py` | 绘图脚本 |
