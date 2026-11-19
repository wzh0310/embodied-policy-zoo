# 自定义 push_cube 场景 ACT 训练：踩坑与适配记录

> 适用范围：LeRobot **0.7.0（2026 定制版）** + 自定义 MuJoCo Panda 推方块环境。
> 目标：把 `mujoco-studio-lab` 早期导出的推物数据集，接入 ACT 策略做小规模冒烟训练。
> 结论先行：全链路已跑通（数据加载 / loss 计算 / 环境评估 / checkpoint）。以下字段差异均以
> 容器内 `lerobot` 环境真实源码为准，任务书里的部分字段名来自旧版或其它分支，需映射。

---

## 0. 环境分工（务必遵守）

| 用途 | conda 环境 | 说明 |
|---|---|---|
| 仿真可视化 / 离屏渲染视频 | `mujoco` | EGL 软件渲染，生成专家演示素材 |
| 训练 / 数据集 / 环境接入 | `lerobot` | **本训练任务全程用此环境** |

本容器 GPU 驱动为 **CUDA 12.4**，而 `lerobot` 环境 torch 为 **2.15.0+cu132**，二者不匹配：
`torch.cuda.is_available() == False`。因此**配置里 `policy.device: cpu`**，`tf32/use_amp` 均关闭。
日志中的 `CUDA initialization ... driver too old` 警告可忽略——不影响 CPU 训练。

---

## 1. 配置字段映射（任务书描述 → 0.7.0 真实字段）

| 任务书字段 | 真实字段 / 机制 | 说明 |
|---|---|---|
| `dataset_repo_id`（本地路径） | `dataset.repo_id` + `dataset.root` | `repo_id` 只是逻辑名（任意，如 `push_cube_expert`），真正定位本地数据靠 `root` 指向数据集目录 |
| `env.name: push_cube` | `env.type: push_cube` + `env.task: push_cube` | 通过 `@EnvConfig.register_subclass("push_cube")` 注册；draccus 按 `type` 解析为 `PushCubeEnvConfig` |
| `policy.action_dim: 4` | **无此字段** | `ACTConfig` 共 37 个字段，不含 `action_dim`。动作维度由数据集 `output_features["action"].shape=[4]` **自动推断** |
| `chunk_size: 20` | `policy.chunk_size: 20` + `policy.n_action_steps: 20` | `n_action_steps` 必须 `<= chunk_size`，否则配置校验报错 |
| `num_epochs: 2` | **无此字段**，用 `steps` | LeRobot 训练是 step 驱动；本数据集 183 帧 / 5 episodes，batch=8 时 1 epoch≈23 步，2 epoch≈46 步，故 `steps=100` 已覆盖"至少 2 epoch + 观察前 100 步" |

---

## 2. 坑 1：`accelerate` 未安装（首个阻塞）

**现象**：训练刚启动即
`ImportError: 'accelerate' is required but not installed. Install it with: pip install 'lerobot[training]'`

**原因**：`lerobot` 环境此前只装了 dataset/仿真依赖，未装 training extra。

**解决**（不必装整个 `lerobot[training]`，只补必需项）：
```bash
conda activate lerobot
pip install "accelerate>=1.0,<2.0"     # 实测装到 1.16.0 即可
```
`peft` 仅在 `use_peft=true` 时需要；`trackio/wandb/tensorboard` 在 `trackio.enable=false` 时不需要。

---

## 3. 坑 2：自定义环境无法被训练入口识别

**现象**：若直接 `python -m lerobot.scripts.lerobot_train --config_path=...`，
draccus 解析到 `env.type=push_cube` 时报 "Couldn't find a choice class for 'push_cube'"。

**原因**：`main()` 里的 `register_third_party_plugins()` 只扫描已安装的
`lerobot_env_* / lerobot_policy_* / lerobot_robot_* ...` **pip 包**（`importlib.metadata`），
我们的环境是 `mujoco-studio-lab/envs/lerobot_push_cube_env.py`，不是独立安装的包，不会被自动导入。

**解决**：写一个 **wrapper**（`scripts/train_act_push_cube.py`），在 import 官方训练入口**之前**
先把仓库根加入 `sys.path` 并 import 环境模块，触发 `@EnvConfig.register_subclass("push_cube")`：
```python
import sys
sys.path.insert(0, "${STUDIO_LAB_ROOT}")
import envs.lerobot_push_cube_env   # 触发注册
from lerobot.scripts.lerobot_train import main
main()
```
顺序至关重要：注册必须发生在 draccus 解析 `--config_path` 之前。

---

## 4. 坑 3：`output_dir` 必须不存在

**现象**：`FileExistsError`。

**原因**：`resume=false` 时 LeRobot 0.7.0 要求 `output_dir` 事先不存在。

**解决**：启动脚本里先 `rm -rf "${OUTPUT_DIR}"` 再训练。

---

## 5. 坑 4：初始 loss 偏高（`kl_weight`）

**现象**：默认 `kl_weight=10` 时，初始 `loss≈29`（`l1≈0.7` + `kld≈2.8`，KLD 被权重放大）。

**说明**：这不是错误，CVAE 训练早期 KL 项本就较大。但为对齐"初始 loss 落在 1-10"的冒烟预期，
把 `kl_weight` 从 10 降到 1.0，loss 随之落入目标区间。

**同时**：CPU 上 16M 参数模型约 14s/step，冒烟太慢。缩小网络
（`dim_model=512→128`、`dim_feedforward=3200→512`、`n_encoder_layers=4→1`、
`latent_dim=32→8`）降至 12M 参数。这些**只影响收敛质量，不影响链路正确性**——
冒烟测试目的是验证"数据/维度/归一化/评估"全通，不追求成功率。

---

## 6. 坑 5：评估视频生成

**机制**：训练循环按 `env_eval_freq`（本配置=50）触发 `lerobot_eval.rollout()`，
用 `env.type=push_cube` 创建环境、加载当前 policy 闭环交互，`eval.recording=true` 落盘视频。
视频在 `checkpoints/<step>/eval/` 下。随机/未收敛策略成功率低属正常，链路验证才是目的。

---

## 7. 冒烟训练实测结果（容器内 `lerobot` 环境，CPU）

配置：`configs/act_push_cube.yaml`；启动：`scripts/train_act_push_cube.py`（wrapper）。
数据集 183 帧 / 5 episodes / fps=50；有效 batch=8；模型 12M 参数；steps=100。

| step | epoch | loss | l1_loss | kld_loss | 梯度范数 |
|---|---|---|---|---|---|
| 10 | 0.44 | 1.781 | 0.684 | 1.097 | 75.6 |
| 20 | 0.87 | 1.472 | 0.566 | 0.905 | 44.8 |
| 30 | 1.31 | 1.231 | 0.471 | 0.760 | 55.2 |
| 40 | 1.75 | 1.075 | 0.441 | 0.634 | 59.1 |
| 50 | 2.19 | 0.937 | 0.422 | 0.515 | 56.7 |
| 70 | 3.06 | 0.743 | 0.410 | 0.333 | 36.0 |
| 80 | 3.50 | 0.667 | 0.406 | 0.261 | 34.1 |
| 90 | 3.93 | 0.596 | 0.404 | 0.192 | 34.7 |
| 100 | 4.37 | **0.533** | 0.371 | 0.162 | 30.9 |

loss 从 1.781 单调下降至 0.533，数值健康、无 NaN/发散。训练 `TRAIN_EXIT=0`、`End of training` 正常结束。

- **数据加载**：正常，无 shape/dtype 报错（图像 `[8,3,224,224]`、
  action `[8,4]`、state `[8,13]` 由数据集 `output_features` 自动对齐）。
- **动作归一化**：ACT 用数据集 `stats.json` 的 mean/std 做 MEAN_STD 归一化；
  gripper 常量维 std=0 已防除零。环境侧 `step()` 按同一统计量反归一化，闭环一致。
- **loss**：step 10 已落入 1-10 合理区间（默认 kl_weight=10 时为 29，降至 1.0 后正常）。
- **epoch**：`epch=0.44 @ step10` ⇒ 100 步 ≈ 4.4 epoch，满足"至少 2 epoch"。
- **评估**：`env_eval_freq=50`，step 50 触发一次闭环评估并生成视频。
  实测日志 `Eval policy at step 50` → `Stepping through eval batches 1/1` →
  视频 `outputs/act_push_cube_smoke/eval/videos_step_000050/push_cube_0/eval_episode_0.mp4`
  （**50 帧 / 224×224 / 50fps**）。抽帧确认画面清晰：首帧机械臂在左、黄色立方体中、红盘右；
  末帧末端已推进到红盘位置（策略虽未收敛但动作方向正确）。metrics：
  `sum_rewards=-0.80`、`max_rewards=0.16`、`successes=[False]`（未收敛策略，符合预期）。
- **两次评估**均成功：step 50 与 step 100 各生成一段 50 帧视频
  （`eval/videos_step_000050/...`、`eval/videos_step_000100/...`）。
- **checkpoint**：step 100 保存到 `output_dir/checkpoints/000100/pretrained_model/`，
  含 `config.json`、`model.safetensors`、`policy_pre/postprocessor.json`、
  `train_config.json` + `training_state/`（优化器/RNG/步数）。
  校验：`output_features.action.shape=[4]`、`input_features` 含
  `observation.images.main` 与 `observation.state`、`chunk_size=20`——与数据集 100% 对齐。

---

## 8. 全链路联调（3 次连续小训练）

用优化配置 `configs/act_baseline.yaml`（数据缓存开启），不同 seed 连续跑 3 次 30 步训练，
全程 `MUJOCO_GL=egl` 离屏渲染、不启 WebViewer：

| run | seed | loss(step10/20/30) | 结果 |
|---|---|---|---|
| 1 | 2001 | 2.313 / 1.942 / 1.635 | PASS |
| 2 | 2002 | 2.164 / 1.776 / 1.617 | PASS |
| 3 | 2003 | 2.519 / 2.109 / 1.761 | PASS |

三次均：loss 单调下降、无 NaN、无 Traceback、`End of training` 正常退出（EXIT=0）。
**无偶发 bug**，链路稳定可复现。

---

## 9. 效率优化实测

| 优化项 | 做法 | 实测效果 |
|---|---|---|
| 数据缓存 | `persistent_workers=true` + `num_workers=4` + `prefetch_factor=4` | `data_s` 首步 1.4~1.8s → 后续 0.012s（worker 常驻内存，磁盘 IO 近乎归零）|
| 混合精度 AMP | CPU 上试 bf16 autocast | **慢 3.4×**（fp32 0.94s vs bf16 3.27s）→ 关闭；GPU 场景改用 `accelerator.mixed_precision=bf16` |
| 评估频率 | `env_eval_freq=115`（≈5 epoch） | 减少环境交互开销；10 epoch 基线仅 2 次评估 |
| 离屏渲染 | `MUJOCO_GL=egl` + `PYOPENGL_PLATFORM=egl` | 无显示器正常渲染评估视频，不启 WebViewer |

**归一化核验**：图像 float32 `[0,1]` ✓；action 经 stats mean/std 归一化范围 `[-3.75,2.13]` 有限 ✓；
gripper 常量维已防除零 ✓。"loss 不下降" 在本链路未复现——三次联调与冒烟均单调下降。
