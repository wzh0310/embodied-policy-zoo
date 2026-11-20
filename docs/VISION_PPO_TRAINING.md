# 端到端视觉 PPO 训练脚本

基于 SB3 `CnnPolicy` + `NatureCNN` + 4 帧堆叠，对 `PushCubeEnv` 的 84×84 灰度观测做
端到端像素级 PPO 训练。脚本：`algos/ppo/train_vision_ppo.py`。

## 1. 算法与输入管线

- **观测**：`obs_mode='rgb'`，几何代理离屏渲染的 84×84 灰度图，CHW + float32 [0,1]
  （训练档 ~482fps，见 `docs/VISION_PIPELINE_REPORT.md`）。
- **帧堆叠**：`VecFrameStack(n_stack=4, channels_order='first')`，观测由 (1,84,84) 变为
  (4,84,84)，为策略提供运动时序信息（立方体与末端的移动方向）。
- **主干**：显式指定 `NatureCNN`（`features_dim=256`），首层卷积自动适配 4 输入通道。
- **输入契约**：`policy_kwargs={'normalize_images': False}`，因环境已做 CHW + [0,1]，
  关闭 SB3 自带 /255 与转置，避免重复预处理。
- **动作空间**：末端笛卡尔增量 ±0.02m，与模仿学习（ACT/BC）及状态 PPO 完全一致。

## 2. 超参数（视觉任务适配）

| 参数 | 值 | 说明 |
|---|---|---|
| 总训练步数 | 2,000,000 | 视觉样本效率低于状态输入，需更多步数 |
| 学习率 | 2e-4 | 略低于状态 PPO 的 3e-4 |
| n_steps / batch_size | 2048 / 256 | PPO 标准配置 |
| ent_coef / gamma / gae_lambda | 0.01 / 0.99 / 0.95 | 保持探索与稳定 |
| 帧堆叠 | 4 | 提供时序信息 |
| 主干 | NatureCNN (features_dim=256) | 轻量 CNN，适配 84×84 |

## 3. 小批量验证（1 万步）

在容器 CPU 上跑 1 万步短程验证（`--total-timesteps 10000`）：

- 训练流程**无报错**，观测形状由 (1,84,84) 正确堆叠为 (4,84,84)；
- 损失函数正常输出：`policy_gradient_loss`、`value_loss`、`loss` 均有值并随更新波动，
  `n_updates` 正常累加，优化器工作正常；
- 模型 `final_model.zip` 与训练配置 `train_config.json` 已保存；
- 详细数值见 `results/metrics/VISION_PPO_10K_SMOKE.json`，原始日志见输出目录
  `logs/progress.csv`。

## 4. 工程约束与性能说明

- **EGL 段错误规避**：构造 PPO 前不可手动 reset 出图（MuJoCo EGL 上下文与 torch 卷积
  初始化在软件 EGL 主机上冲突）；脚本采用"torch 先导入 + 渲染器懒初始化 + 先构造 PPO
  再由 learn 触发渲染"的顺序，已验证稳定。
- **训练吞吐（已修正）**：此前记录的"无 GPU、CPU 6-8 fps"是错误判断——排查时混淆了
  NVIDIA 设备（`/dev/nvidia*`）与 `/dev/dri`。容器实际具备 RTX A6000 GPU，问题只是
  `mujoco` 环境 torch 为 cu132 与驱动 12.4 不匹配。修复为 `torch 2.6.0+cu124` 后，
  `device=cuda` 实测 **209 fps**（1 万步 48 秒），相比 CPU（6 fps / 1667 秒）提速约
  **35 倍**。默认 200 万步配置约需 2.7 小时，已完全可行。详见 `docs/GPU_ACCELERATION_FIX.md`。
- **与模仿学习对比**：本脚本产出视觉策略权重，可在统一 50 回合评估协议下与 ACT/BC、
  状态 PPO 横向对比（后续实验）。动作空间一致保证对比公平。
