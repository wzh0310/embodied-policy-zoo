# 环境标准化与 RL 最小闭环验证

> 目标：完成 `PushCubeEnv` 的 Gymnasium API 合规改造并跑通 SB3「环境—算法」最小闭环，
> 全程仅用 `mujoco` conda 环境，动作空间沿用末端笛卡尔增量，底层运动完全复用 MuJoCo，
> 不涉及任何控制算法开发。

## 1. 环境合规性改造（Gymnasium API）

标准化环境来自兄弟仓库 `mujoco-studio-lab`，以 `pip install -e .` 开发模式安装为
`pushcube-envs` 包，在本仓库直接 `import pushcube_envs` 引用，不重复搬运代码。

### 1.1 标准方法齐备

| 方法 | 实现 | 说明 |
|---|---|---|
| `seed(seed)` | 显式 setter | 统一随机种子；同时支持标准的 `reset(seed=...)` |
| `render()` | `rgb_array` / `human` 双模式 | 离屏 RGB 帧；human 模式 headless 下安全降级为 no-op |
| `close()` | 释放离屏渲染器 | 置空 `_renderer`，无资源泄漏 |
| `reset()` / `step()` | 标准 5 元组 | 返回 `(obs, reward, terminated, truncated, info)` |

### 1.2 空间定义（`gymnasium.spaces`）

- **动作空间**：`Box(4,)` float32，`[dx, dy, dz, gripper]`；前三维修正增量范围
  **±0.02m**，`gripper∈[0,1]`。与模仿学习**完全一致**，保证对比公平。
- **观测空间**：按 `obs_mode` 切换——`state` 为 `Box(13,)` float32（7 关节角 + 立方体 3 +
  目标 3），`image` 为 `Box(224,224,3)` uint8，`both` 为二者组成的 `spaces.Dict`。

### 1.3 `gymnasium.utils.env_checker.check_env()` 校验

| 观测模式 | 结果 | 备注 |
|---|---|---|
| `state`（RL 输入） | **PASS** | 无 error |
| `both`（Dict 观测） | **PASS** | 无 error |

唯一警告：Gymnasium 1.4 提示官方已弃用独立 `seed()` 方法、建议用 `reset(seed=...)`。
这是向后兼容提示，不影响合规——本环境两种播种方式均支持。

## 2. SB3 集成与最小闭环

### 2.1 环境注册

`import pushcube_envs` 即把 `PushCube-v0` 注册进 Gymnasium 全局表，可直接
`gym.make("PushCube-v0")`。SB3 自带 `env_checker.check_env()` 二次验证 **PASS**。

### 2.2 PPO(MLP) 最小闭环

- 策略：`PPO("MlpPolicy")`，观测用低维 `state`（13 维），未引入视觉编码器。
- 交互：100 步 `predict → step`，打印观测/动作/奖励，确认格式与流程无报错。

| 项 | 数值 |
|---|---|
| 交互步数 | 100 |
| 累计奖励 | -6.1341 |
| 动作样例（step 3） | `[+0.0200, -0.0200, -0.0200, +1.0000]` |
| 观测样例（step 1） | `[0.038, 0.022, 0.034]`（关节角前 3 维） |
| 报错 | 无 |

动作落在合法区间内（前三维修正被 clip 到 ±0.02），格式正确；此步为随机初始化策略的
短程探索，奖励为负属正常。两个无害警告：

- SB3 建议动作空间归一化为 `[-1,1]`——本任务刻意保留 ±0.02m 物理增量以对齐模仿学习，
  不做改写。
- 当前镜像 torch（cu132 编译）检测到宿主驱动偏旧而回退 CPU——本闭环只用 MLP/CPU，
  训练正常，不影响验证。

## 3. 产出

- `scripts/verify_sb3_env.py`：向量化适配 + 随机基线（见 `SB3_SETUP_REPORT.md`）
- `scripts/verify_sb3_ppo_closedloop.py`：SB3 二次验证 + PPO 100 步闭环
- `results/metrics/RANDOM_POLICY_BASELINE.json`
- `results/metrics/SB3_PPO_SMOKE.json`

复现：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate mujoco
cd embodied-policy-zoo
python scripts/verify_sb3_ppo_closedloop.py
```
