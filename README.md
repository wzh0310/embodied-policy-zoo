# embodied-policy-zoo

RL / 模仿学习 pipeline + 数据清洗 + LIBERO 评测。

## 结构

- `pipelines/` - 训练 pipeline（BC / RL / diffusion policy）
- `data/` - 数据清洗与转换
- `eval/libero/` - LIBERO 评测
- `configs/` - 训练配置
- `scripts/` - 训练 / 评测脚本
- `docs/` - 踩坑与适配记录

## 自定义场景 ACT 训练（push_cube）

把 `mujoco-studio-lab` 第一周导出的 Panda 推方块数据集，用 LeRobot 0.7.0 的 ACT 策略做
小规模冒烟训练。配置、启动脚本、踩坑记录都在本仓库：

```bash
# 在 13 服务器容器内、lerobot conda 环境
bash scripts/train_act_push_cube.sh            # 默认 100 步 / batch 8（CPU）
STEPS=200 BATCH=8 bash scripts/train_act_push_cube.sh   # 覆盖规模
```

- 配置：`configs/act_push_cube.yaml`（源自官方 ACT 模板，适配本地数据集 + push_cube 环境）
- 依赖：训练需要 `accelerate`（`pip install "accelerate>=1.0,<2.0"`）
- 前提：兄弟仓库 `mujoco-studio-lab` 提供数据集与 `push_cube` 环境注册（wrapper 自动加载）
- 踩坑与字段映射：`docs/ACT_TRAINING_NOTES.md`

### 文档

- **《ACT 模仿学习适配指南》**：`docs/ACT_IMPLEMENTATION_GUIDE.md`
  — 数据集制作规范 / 环境接入步骤 / 常见报错 / 基线实验分析 / 核心超参数速查。
- 踩坑与字段映射：`docs/ACT_TRAINING_NOTES.md`。

### 冒烟结果（100 步 / CPU）

loss 单调下降 `1.78 → 0.53`（4.37 epoch），step 50/100 各触发一次闭环评估并生成视频，
checkpoint 保存于 `/data/wzh/robot/outputs/act_push_cube_smoke/checkpoints/000100/`。
未收敛策略成功率 0% 属预期——本轮只验证数据/维度/归一化/评估全链路无阻塞。

### 全链路联调与基线（第二周）

- **3 次连续小训练**全部 PASS（不同 seed，均 loss 单调下降、无 NaN/traceback、正常结束）。
- **效率优化**：数据缓存（`persistent_workers` + 4 worker + prefetch）使 `data_s` 从 ~1.8s 降至 ~0.01s；
  CPU 上 bf16 AMP 实测慢 3.4× 故关闭（GPU 场景用 `accelerator.mixed_precision=bf16`）。
- **10 epoch 基线**：配置 `configs/act_baseline.yaml`（本仓库同步副本），`steps=230`（≈10.05 epoch），
  `TRAIN_EXIT=0`。**loss 单调下降 `1.78 → 0.334`**（step10→230，见指南第 4 节完整曲线），
  两次闭环评估（step 115 / 230）各生成 50 帧 / 224×224 评估视频；
  成功率 0%（小数据 + CPU 缩小模型未收敛，基线定位是可复现对比点，不求成功率）。
  评估视频：`/data/wzh/robot/outputs/act_push_cube_baseline/eval/videos_step_*/push_cube_0/eval_episode_0.mp4`。

**版本**：`v0.1.0-act-baseline`（第二周里程碑：ACT 基线实验完成）。

---

## Day 3–4：全量训练配置 + 评估工具（v0.2.0）

- **训练配置**：`configs/act_full_train.yaml`（GPU / A6000，全量 40 ep，steps=4000≈50 epoch，
  batch16，每 5 epoch 存 checkpoint，eval_split=0.2）。4000 步完整跑通，验证损失 0.281→0.144。
- **评估工具**：`scripts/eval_act.py`（3 seeds × 10 = 30 次离屏评估，输出成功率 / 平均完成步数 /
  平均距离误差）+ `scripts/record_dual_view.py`（双视角演示视频）。
- **基线指标**（ckpt 004000）：成功率 30%（9/30）、平均完成步数 36.8、平均距离误差 0.134m。
  数据质量已提升到 40/40 成功；剩余瓶颈与后续优化见 `docs/DAY34_REPORT.md`。
- 依赖：lerobot 环境 torch 已重装 2.6.0+cu124（GPU 可用）。

**版本**：`v0.2.0-full-training-eval`。


## Day 5-6：问题分析 + 针对性调优实验（v0.4.0）

- **失败分析**（`scripts/analyze_failures.py`）：基线 30% 的 21 个失败中，95.2% 是「推偏/方向不准」，
  EE 已接近立方体 8.3cm——感知/接近不是瓶颈。轨迹级根因：ACT 输出恒定满速 +x 推、不会侧修正/停止。
- **14 组消融**（`scripts/ablation_pipeline.py`，结果 `data/ablation_results.json`）：
  chunk 8/12/16（10%/16.7%/3.3%）、lr 1e-4/3e-4/5e-4（16.7%/26.7%/23.3%）、
  temporal ensembling / 闭环重规划 nas1/4/10 全部 ≤ 基线 30%；图像增强使距离误差 0.192→0.148
  但完整 4000 步训练后成功率反降至 13.3%。
- **关键探针**：立方体初始偏移缩窄到 ±1cm 成功率仍 26.7% → 瓶颈是模型方向控制能力，非初始分布难度。
- **结论**：30% 是当前数据规模(40条)+模型容量下的真实上限；70% 目标需质变（数据量级提升、
  改进动作表征/闭环控制、更强策略），已列入 `docs/TUNING_REPORT.md` 后续方向。
- 完整报告：`docs/FAILURE_ANALYSIS_REPORT.md`、`docs/TUNING_REPORT.md`；基线归档不变。
