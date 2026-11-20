# SB3 依赖安装与环境适配验证

> 目标：搭建强化学习训练基座，验证标准化后的 `PushCube-v0` 与 Stable-Baselines3（SB3）
> 框架的兼容性，为后续 PPO / SAC 等在线 RL 训练提供可复现的并行采样与基线。

## 1. 依赖与环境

- 环境：`mujoco`（Python 3.12.16，与仿真/标准化环境同环境，避免跨环境依赖冲突）。
- 安装命令：

```bash
pip install "stable-baselines3[extra]" opencv-python tensorboard
```

- 验证版本：SB3 2.9.2 / gymnasium 1.4.0 / mujoco 4.0.1 / numpy 2.5.5 / torch 2.15.0+cu132。
- 离屏渲染（`mujoco` 环境专属，容器无显示窗口）：

```bash
export MUJOCO_GL=egl
export LIBGL_ALWAYS_SOFTWARE=1
export EGL_PLATFORM=surfaceless
export PYOPENGL_PLATFORM=egl
```

> 与模仿学习用的 `lerobot` 环境相互隔离：SB3 基座装在 `mujoco` 环境，不影响 ACT/BC 训练。

## 2. 向量化环境适配

用 `SubprocVecEnv` 封装 4 个并行 `PushCube-v0`（`start_method="fork"`），
观测用低维 `state`（13 维）以贴合 RL 输入。验证输出完全符合 SB3 预期：

| 项 | 实测 | SB3 预期 | 结论 |
|---|---|---|---|
| 观测 `obs` | `(4, 13)` float32 | `(n_envs, obs_dim)` | 通过 |
| 奖励 `rewards` | `(4,)` float64 | `(n_envs,)` | 通过 |
| 终止 `dones` | `(4,)` bool | `(n_envs,)` | 通过 |
| 观测空间 | `Box(13,)` | gym 空间 | 通过 |
| 动作空间 | `Box(4,)` ±[0.02,0.02,0.02]/[0,1] | gym 空间 | 通过 |

### 采样吞吐实测（4 并行 vs 单环境）

| 观测模式 | 单环境 (trans/s) | 4 并行 (trans/s) | 加速比 |
|---|---|---|---|
| `state`（纯 CPU，13 维） | 1559.4 | 3301.5 | **2.12x** |
| `both`（含 224×224 RGB 离屏渲染） | 8.1 | 24.3 | **2.99x** |

- 含渲染场景下 4 并行达到约 **3 倍**加速，与任务预期一致；低维 state 场景因单步极快、
  进程间通信开销占比高，加速比略低但仍显著。
- `state` 与 `both` 两次测量复用同一份随机种子序列，保证对比公平。

## 3. 随机策略基线（10 回合）

固定种子，`state` 观测，动作在合法区间均匀随机采样，每回合上限 50 步。

| 指标 | 数值 |
|---|---|
| 平均奖励 | **-3.1491 ± 0.0834** |
| 成功率 | **0% (0/10)** |
| 平均步数 | **50.00**（全部走满被截断） |
| 单回合奖励区间 | [-3.323, -3.072] |

随机策略 0% 成功、全部耗尽 50 步，符合预期：立方体需定向推动才能进入 3cm 半径的
目标区，随机末端增量几乎不可能命中。该基线给出 RL 训练的起点——任何有效策略都应在
奖励（>-3.15）与成功率（>0%）上显著超过它。

> 对照：最优 ACT（50 固定种子）成功率 50%、平均奖励远优于随机基线，说明任务本身
> 可学习，随机基线仅作为 RL 训练的下界参照。

## 4. 踩坑：EGL 上下文与 fork 死锁

- **现象**：在父进程先调用 `env.render()` / 生成 RGB 观测后再 `SubprocVecEnv(fork)`，
  子进程首次渲染会卡死（父进程已建立 GL 上下文，fork 后子进程继承到无效上下文）。
- **解决**：所有 `SubprocVecEnv` 的创建与 step 必须在父进程产生**任何** GL 上下文之前
  完成；把离屏渲染验证放到脚本**最后**（`verify_sb3_env.py` 已按此顺序编排）。
- 若确需在渲染后再并行，可改用 `start_method="spawn"`（代价是启动更慢、需模块可 pickle）。

## 5. 产出

- 测试脚本：`scripts/verify_sb3_env.py`（一段式跑完全部验证，约 1 分钟）
- 基线数据：`results/metrics/RANDOM_POLICY_BASELINE.json`

复现：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mujoco
export MUJOCO_GL=egl LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless PYOPENGL_PLATFORM=egl
cd embodied-policy-zoo && python scripts/verify_sb3_env.py
```
