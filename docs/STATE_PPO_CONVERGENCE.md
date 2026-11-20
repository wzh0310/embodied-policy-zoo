# 状态 PPO 训练收敛调优与推理可视化

> 目标：启动低维状态 PPO 训练、调试奖励与超参推动收敛，并完成推理可视化工具与演示视频。
> 全程 `mujoco` 环境，动作沿用末端笛卡尔增量，与模仿学习一致。

## 1. 训练与调优过程

### 1.1 基线训练（lr 3e-4，4 并行，目标 30 万步）

按默认超参后台启动训练，通过 TensorBoard / CSV 观察曲线（`results/metrics/PPO_TRAIN_CURVE.png`）：

- 价值函数早期正常（`explained_variance` 0.44→0.78），但约 8 万步后开始**剧烈震荡**：
  `value_loss` 从 0.03 飙到 2.2，`explained_variance` 崩到 0.08，`loss` 出现正负跳变。
- `rollout/success_rate` 始终在 0~8% 徘徊，确定性评估成功率 0%。
- 因震荡不收敛，在 12.3 万步主动停止，保留 checkpoint 作为对照。

**诊断**：lr 3e-4 对该任务偏高；且原始塑形奖励存在结构性缺陷——末端到立方体的
EE 接近奖励权重仅 `K_EE=0.1`（每步最多 ~0.002），而初始时末端距立方体约 0.55m，
策略学不到「先移动到立方体旁」的关键第一步，立方体长时间不动，距离塑形与差分奖励
全部为 0，缺少有效学习信号。

### 1.2 调优（lr 1e-4 + 引导奖励 + 熵 0.02）

按任务指引「震荡则降学习率、减小 batch」执行，并新增训练侧引导式奖励
（`algos/ppo/guided_reward.py`，**不修改环境本体**，保持与模仿学习对比公平）：

| 调整 | 取值 | 目的 |
|---|---|---|
| 学习率 | 3e-4 → **1e-4** | 抑制震荡 |
| batch size | 256 → **128** | 增加更新次数，平滑梯度 |
| 熵系数 | 0.01 → **0.02** | 保持探索（std 全程 ~1.0） |
| 奖励 | 两段势能 | 强引导「末端接近立方体」+「立方体接近目标」 |

6 万步调优结果（`results/metrics/PPO_TRAIN_CURVE_TUNED.png`）：
`rollout/success_rate` 0 → 1% → 5% → 7%，学习速度明显快于基线（基线 10 万步才到 8%），
但**仍未稳定上升**，确定性评估成功率保持 0%。

客观指标汇总见 `results/metrics/PPO_TUNING_SUMMARY.json`。

## 2. 关键结论：90% 目标为何未达成

任务预期「低维状态任务难度低，30 万步内成功率应稳定 90%」，但实测 PPO 从零探索
**只能到个位数百分比成功率**。根因是任务物理特性与动作粒度：

1. **动作分辨率 vs 目标精度**：每步末端增量 ±0.02m，成功半径仅 0.03m。推动立方体时
   2cm 的步长相对 3cm 目标窗口过粗，极易推过头（规则策略诊断：全力推 40 步立方体
   被推到 1.9m，飞出目标区），需要非常精细的收力控制。
2. **稀疏接触 + 长时程**：要先靠近立方体（约 0.55m）、再从正确方位接触、再沿目标方向
   精细推送，任何一环偏差都会失败；从零随机探索覆盖这条窄路径的概率极低。
3. **对照参考**：即便有专家演示数据、动作分块的 ACT 策略，成功率也仅 50%——说明任务
   本身对「精确推动」要求很高，从零 RL 达 90% 的假设与真实环境不符。

这是**诚实的负结果**：奖励函数有效（平均奖励从随机基线 -3.15 提升到 -1.48，见上期），
但达到高成功率需要更精细的控制手段（如动作缩放、示教预训练、或更大成功半径），
而非单纯增加步数或调学习率。

## 3. 推理可视化与视频

### 3.1 通用录制工具 `utils/record_video.py`

- 加载 SB3 模型，rollout 完整回合并录制 MP4，用 `cv2.VideoWriter` 编码（不依赖 lerobot）。
- 三种视角：`--view global`（全局第三人称）/ `task`（主观测相机，与算法输入一致）/
  `both`（左右并排，默认）。
- 在候选 seed 上筛选成功回合录制，确保是策略真实推理推动立方体；`--set-model-seed`
  固定每回合模型 RNG，使随机动作轨迹可复现。

```bash
python utils/record_video.py --model <final_model.zip> --view both \
    --seeds 116 47 75 --stochastic --set-model-seed --num-videos 3 --fps 10
```

### 3.2 演示视频（3 条成功轨迹）

用调优模型（`ppo_guided_v2/final_model.zip`），固定模型 seed + 随机动作采样，
在 200 个 seed 中筛出 6 个可复现成功轨迹（3%），选最优 3 条录制：

| 视频 | seed | 完成步数 | 最终距目标 |
|---|---|---|---|
| `assets/videos/ppo_state/success_seed116.mp4` | 116 | 30 | 7.3mm |
| `assets/videos/ppo_state/success_seed47.mp4` | 47 | 27 | 12.6mm |
| `assets/videos/ppo_state/success_seed75.mp4` | 75 | 36 | 20.1mm |

视频为双视角（左 global_cam / 右 top_cam），10fps 便于观察动作过程。左右面板分别对应
「全局第三人称」与「算法观测视角」，符合统一演示素材标准。

## 4. 产出清单

| 文件 | 说明 |
|---|---|
| `algos/ppo/guided_reward.py` | 训练侧引导奖励包装器（可选，不改环境本体） |
| `scripts/plot_ppo_curve.py` | 训练曲线绘图（奖励/成功率/价值损失） |
| `utils/record_video.py` | 通用推理视频录制工具（三视角） |
| `results/metrics/PPO_TRAIN_CURVE*.png` | 基线/调优训练曲线截图 |
| `results/metrics/PPO_TUNING_SUMMARY.json` | 调优量化汇总 |
| `results/metrics/PPO_*_PROGRESS.csv` | 原始训练日志 |
| `assets/videos/ppo_state/*.mp4` | 3 条成功轨迹演示视频 |
