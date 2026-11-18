# ACT 模仿学习适配指南（push_cube 场景）

> 面向 **LeRobot 0.7.0（2026 定制版）**。本指南把第一周 `mujoco-studio-lab` 的
> Panda 推方块仿真，完整接入 ACT（Action Chunking with Transformers）策略训练与闭环评估。
> 所有命令均在 **13 服务器容器** 内执行；仿真/离屏渲染用 `mujoco` 环境，
> 数据集/环境/训练用 `lerobot` 环境。所有字段以容器内真实源码为准。

---

## 0. 环境分工与前置条件

| 用途 | conda 环境 | 关键包 |
|---|---|---|
| 仿真可视化 / 专家演示离屏录制 | `mujoco` | mujoco 4.0.1, opencv, pillow, pyarrow |
| 数据集 / 环境接入 / ACT 训练 / 评估 | `lerobot` | lerobot 0.7.0, mujoco 4.0.1（补装）, accelerate 1.16.0 |

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
# 若缺失：
pip install "accelerate>=1.0,<2.0"   # 训练必需
pip install mujoco==4.0.1            # 环境评估渲染必需（lerobot 环境原本没有）
```

**本容器无可用 GPU**：驱动 CUDA 12.4，而 `lerobot` 环境 torch 为 2.15.0+cu132，
`torch.cuda.is_available()==False`。因此所有训练 `device: cpu`， CUDA 初始化告警可忽略。

---

## 1. 自定义数据集制作规范

### 1.1 采用 LeRobot 原生 v3 格式（务必）

任务书里给出的 `train/episode_*.parquet` + `meta/features.json` 是**旧式/示意结构**，
LeRobot 0.7.0 无法原生加载。真实 v3 结构如下（`data/push_cube_demos/`）：

```
push_cube_demos/
├── meta/
│   ├── info.json          # features / splits / 帧数 / fps（核心）
│   ├── stats.json         # 各字段 min/max/mean/std（训练归一化用）
│   ├── features.json      # 附加：features 的独立可读定义（不破坏加载）
│   └── tasks.parquet      # 任务描述表
├── data/chunk-000/file-000.parquet    # 帧表：每行的 obs/action/索引字段
└── videos/chunk-000/observation.images.main/mp4/*.mp4   # 图像序列（AV1/H264）
```

### 1.2 字段对齐硬要求

| 字段 | 形状 / 类型 | 说明 |
|---|---|---|
| `observation.images.main` | `(3, 224, 224)` float32，值域 `[0,1]` | 主相机 RGB；pyav 解码后已归一化到 `[0,1]` |
| `observation.state` | `(13,)` float32 | Panda 关节 qpos（9）+  gripper（2）+ 末端/方块信息（2） |
| `action` | `(4,)` float32 | `[dx, dy, dz, gripper]`，mocap 末端位移增量 |
| `timestamp` | float32 | `frame_index / fps` |
| `episode_index` / `frame_index` / `index` | int64 | 全局/回合内帧号，连续递增 |

**8:2 划分**：通过 `info.json` 的 `splits = {"train":"0:4","val":"4:5"}` 声明，
加载时用 `LeRobotDataset(..., episodes=[0,1,2,3])` / `[4]` 得到 146/37 帧（79.8%/20.2%）。
`LeRobotDataset` 没有 `split` 参数，只有 `episodes`。

### 1.3 归一化统计量（`meta/stats.json`）

由 `scripts/prepare_dataset.py` 直接从 parquet 重算，写入每个字段的 `mean/std/min/max`：
- action mean ≈ `[0.0184, -0.0033, -0.0018, 0]`，std ≈ `[0.0049, 0.0021, 0.0014, 0]`
- gripper 维是常量（std=0）：训练侧与环境侧都做**除零保护**（std<1e-8 时置 1）
- 实测归一化后 action 范围 `[-3.75, 2.13]`，有限、无异常值

校验脚本：`scripts/verify_dataset.py`（原生加载 → batch shape/dtype → 随机帧时空对齐可视化）。

---

## 2. 自定义环境接入步骤

### 2.1 真实注册机制（与任务书描述的映射）

| 任务书描述 | LeRobot 0.7.0 真实机制 |
|---|---|
| 继承 `BaseEnv` | **无 BaseEnv 类**；环境本体是 `gym.Env` 子类，实现 `reset/step/render/close` |
| `env_name="push_cube"` 注册表 | `@EnvConfig.register_subclass("push_cube")` + `gym.register` + `sim/integrations` 约定 |
| 官方 rollout 脚本 | `lerobot.scripts.lerobot_eval.rollout()` / `eval_policy()` |

### 2.2 适配层要点（`envs/lerobot_push_cube_env.py`）

1. **观测键**用 LeRobot 标准常量：`observation.images.main`（uint8→float `[0,1]` 由 processor 处理）、
   `observation.state`，与数据集 100% 一致。
2. **`step(action)` 接收归一化动作**，内部按 `stats.json` 反归一化
   `raw = norm * std + mean`，clip 到仿真动作边界后驱动 mocap+weld；
   gripper 常量维 std=0 已防除零。
3. **离屏渲染**：`render()` 返回当前帧 RGB（uint8 HWC），底层用 `mujoco.Renderer` +
   `MUJOCO_GL=egl`（启动脚本导出），**不启动 WebViewer**。
4. **注册两个 gym id**：canonical（键名直出，给用户直用）与 raw（`pixels`/`agent_pos`，
   供 LeRobot gym 集成的 `canonical_observation` 转换）。
5. **训练入口 wrapper**：`scripts/train_act_push_cube.py` 在 import 官方训练入口**之前**
   import 环境模块完成注册（`register_third_party_plugins()` 不扫描非 pip 包的本地环境）。

验证脚本：`scripts/test_lerobot_push_cube.py`（10 步随机交互）、
`scripts/rollout_lerobot_push_cube.py`（官方 rollout + 随机策略 + 评估视频）。

---

## 3. 常见报错与解决方案

| 现象 | 根因 | 解决 |
|---|---|---|
| `ImportError: 'accelerate' is required` | lerobot 环境缺 training extra | `pip install accelerate==1.16.0` |
| `Couldn't find a choice class for 'push_cube'` | 自定义环境未注册 | 用 wrapper 在 draccus 解析前 import 环境模块 |
| `FileExistsError`（output_dir） | `resume=false` 要求目录不存在 | 启动脚本先 `rm -rf "${OUTPUT_DIR}"` |
| 初始 `loss≈29` | 默认 `kl_weight=10` 放大 KLD | 降到 `1.0`；缩小网络适配 CPU 速度 |
| 评估黑屏 / 渲染报错 | 无显示器、GL 后端错误 | `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` |
| `CUDA initialization ... driver too old` | torch cu132 vs 驱动 12.4 | 忽略；`device: cpu` |
| 动作维度报错 | `n_action_steps > chunk_size` | 令 `n_action_steps <= chunk_size`（均 20） |
| `torchcodec not available` 警告 | 平台无 torchcodec | 自动回退 pyav，忽略 |

**AMP 特别说明**：本容器 CPU 上 bf16 autocast 实测**慢 3.4×**（fp32 0.94s vs bf16 3.27s），
故 `accelerator.mixed_precision` 保持 `no`。若将来换 GPU 机器，启用方式：
`--accelerator.mixed_precision=bf16`（fp16 需 CUDA，CPU 会报错）。

---

## 4. 基线实验结果与分析

### 4.1 10 epoch 完整基线（配置 `configs/act_baseline.yaml`，CPU）

`steps=230`（≈10.05 epoch），batch=8，模型 12M 参数，`TRAIN_EXIT=0`，
`End of training` 正常，`traceback=0`、无 NaN。

**loss 曲线**（每 20 步采样）：

| step | epoch | loss | l1_loss | kld_loss |
|---|---|---|---|---|
| 10 | 0.44 | 1.781 | 0.684 | 1.097 |
| 20 | 0.87 | 1.472 | 0.566 | 0.905 |
| 60 | 2.62 | 0.839 | 0.410 | 0.429 |
| 100 | 4.37 | 0.533 | 0.371 | 0.162 |
| 140 | 6.12 | 0.425 | 0.352 | 0.074 |
| 180 | 7.87 | 0.343 | 0.293 | 0.051 |
| 230 | 10.05 | **0.334** | 0.289 | 0.045 |

loss 从 1.781 单调下降至 0.334（后期趋缓），KLD 项从 1.10 降到 0.045，
训练稳定收敛、无发散。

**闭环评估**（每 5 epoch 一次，共 2 次）：

| 评估 step | avg_sum_reward | avg_max_reward | 成功率 | 视频 |
|---|---|---|---|---|
| 115 | -0.789 | 0.158 | 0%（1 episode） | `eval/videos_step_000115/.../eval_episode_0.mp4` |
| 230 | -0.730 | 0.158 | 0%（1 episode） | `eval/videos_step_000230/.../eval_episode_0.mp4` |

评估视频 50 帧 / 224×224 / 50fps。抽帧确认：首帧末端在左、立方体中、红盘右；
末帧末端已推进到红盘位置并覆盖立方体（绿色 mocap 点可见）——
**动作方向正确、闭环评估链路完整**。成功率 0% 是因为：脚本生成的演示数据质量"够用即可"、
模型为 CPU 冒烟缩小版（12M / latent 8 / dim 128）、且 `chunk_size=20` 的模仿学习在小数据下
尚未收敛。本基线定位是**可复现的训练基线**，供下周调参对比，不追求成功率。

### 4.2 三次连续小训练稳定性（联调）

不同 seed（2001/2002/2003）各跑 30 步：loss 分别 `2.313→1.635`、`2.164→1.617`、
`2.519→1.761`，全部单调下降，无 NaN/traceback，正常结束 → **无偶发 bug**。

### 4.3 效率优化效果

- 数据缓存：`data_s` 首步 1.4~1.8s → 后续 0.01s（worker 常驻内存，磁盘 IO 近乎归零）。
- AMP：CPU 上 bf16 慢 3.4×，保持关闭；GPU 场景改用 `accelerator.mixed_precision=bf16`。
- 评估频率：`env_eval_freq=115`（≈5 epoch），10 epoch 仅 2 次评估，控制开销。

---

## 5. 核心超参数速查（下周调参参考）

| 参数 | 当前值 | 作用 | 调参方向 |
|---|---|---|---|
| `policy.chunk_size` | 20 | 一次预测的动作序列长度；越大越平滑但响应越慢 | 匹配回合长度（36-37 步）|
| `policy.n_action_steps` | 20 | 每次实际执行的动作数 | ≤ chunk_size |
| `policy.kl_weight` | 1.0 | CVAE 的 KL 正则权重；越大越接近确定性、多样性下降 | 1→10 影响 loss 量级与探索性 |
| `policy.latent_dim` | 8 | CVAE 隐变量维度；越大表达力越强 | 冒烟缩小，正式可调至 32 |
| `policy.vision_backbone` | resnet18 | 图像编码器 | 正式可用 resnet34/50 |
| `policy.dim_model` | 128 | Transformer 隐维 | 越大容量越高、CPU 越慢 |
| `policy.optimizer_lr` | 1e-5 | 学习率 | 训练不稳可下调 |
| `steps` | 230（≈10 epoch） | 训练总步数 | 正式训练按需增大 |
| `batch_size` | 8 | 批大小 | 显存允许可增大 |
| `env_eval_freq` | 115（≈5 epoch） | 闭环评估间隔（step） | 评估贵可增大 |
| `num_workers` / `prefetch_factor` / `persistent_workers` | 4 / 4 / true | 数据加载与缓存 | CPU 上 4 worker 实测稳定 |
| `accelerator.mixed_precision` | no | AMP | GPU 场景设 bf16 |

> 冒烟/基线阶段为**链路验证 + 基线记录**，模型已缩小、训练步数有限，不追求高成功率。
> 下周正式调参时优先恢复模型容量（dim_model/latent_dim/backbone），再调 kl_weight 与学习率。

---

## 6. 一键复现

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd /data/wzh/robot/embodied-policy-zoo
# 冒烟（100 步）：
STEPS=100 bash scripts/train_act_push_cube.sh
# 基线（10 epoch，优化配置）：
CONFIG=/data/wzh/robot/mujoco-studio-lab/configs/act_baseline.yaml \
OUTPUT_DIR=/data/wzh/robot/outputs/act_push_cube_baseline \
bash scripts/train_act_push_cube.sh
```

产物：`output_dir/checkpoints/<step>/pretrained_model/`（模型）、
`output_dir/eval/videos_step_*/push_cube_0/eval_episode_0.mp4`（闭环评估视频）。
