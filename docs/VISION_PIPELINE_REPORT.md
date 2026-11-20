# 视觉观测管线工程化与环境解耦

本阶段完成两件事：（1）给 `PushCubeEnv` 增加工业级离屏视觉观测能力，解决训练时图像
采集效率问题；（2）修复环境重置/随机化/动作归一化三类问题，并统一 `embodied-policy-zoo`
目录结构，使环境与算法完全解耦。所有改动保持对既有模仿学习（ACT/BC）与状态 PPO 流程
的向后兼容。

## 1. 离屏渲染：两档预设 + 几何代理加速

### 1.1 为什么需要几何代理
容器装配 RTX A6000（`/dev/nvidia*` 设备已直通，详见 `docs/GPU_ACCELERATION_FIX.md`），
但 MuJoCo 离屏渲染当前仍走 mesa 软件光栅化（已装 NVIDIA EGL 库，切换 GPU 渲染未成功）。实测完整 Panda 场景（90 geom / 67 mesh）单帧 140ms ≈ **7fps**，与
分辨率无关（84 与 224 耗时相同），瓶颈在 Panda 高面数 mesh 的固定光栅化开销，关阴影也
仅 11fps。因此 200fps 目标在完整场景的纯软件渲染下不可达。

采用工业界标准做法——**训练/演示双路径渲染**（物理仿真始终用完整场景，仅渲染图像分流）：

| 预设 | 场景 | 输出 | 用途 | 实测帧率 |
|---|---|---|---|---|
| `train` | 几何代理（cube/target/桌面/EE 代理，0 mesh） | 84×84 灰度 CHW float32 [0,1] | CNN 策略训练 | **~482 fps** |
| `demo` | 完整 Panda 场景 | 224×224 RGB CHW float32 [0,1] | 展示 / 与 VLA 对齐 | ~8 fps |

代理场景的 cube 位姿与 hand 位置每帧从真实 `MjData` 同步，保证图像与任务几何对齐
（验证：移动真实 cube 后代理解图像随之变化）。训练档帧率 482fps，远超 200fps 目标，
不再成为训练瓶颈。

### 1.2 接口
- 独立工具模块 `mujoco-studio-lab/envs/offscreen_render.py` 的 `OffscreenRenderer`：
  `get_rgb_image(preset='train'|'demo', camera=..., chw=True, normalize=True)`，支持
  HWC uint8（`normalize=False, chw=False`）与自定义分辨率。
- 环境新增 `obs_mode='rgb'`：`gym.make("PushCube-v0", obs_mode="rgb")` 直接得到 CHW
  归一化观测；`env.get_rgb_image(preset)` 供录制/调试直接取图。
- 从 `mujoco-studio-lab` 版本 `0.0.6` 起随 `pushcube-envs` 包发布，算法仓通过 `pip
  install -e` 引用，不搬运代码。

### 1.3 与 SB3 CNN 策略对齐
环境已做 HWC→CHW + [0,1] 归一化，故 CnnPolicy 必须设 `policy_kwargs=
{'normalize_images': False}`，关闭 SB3 自带的 /255 与转置，避免重复预处理。训练脚本
`algos/ppo/train_rgb_ppo.py` 已固化该配置并冒烟通过（learn + 60 步 rollout 正常）。

## 2. 三类问题修复

1. **重置图像缓存**：旧实现 `reset()` 不刷新已渲染的观测帧，跨回合可能读到上一回合图像。
   现在 `reset()` 与每步 `step()` 均将 `_rgb_cache` 置空，强制按当前状态重渲染。验证：
   同环境连续两次不同 seed 的 `reset()`，首帧图像不同。
2. **随机初始状态一致性**：旧 `reset(seed=)` 用固定 keyframe，seed 不改变立方体初始位姿
   （这解释了此前确定性评估方差为 0）。新增 `randomize_start` 开关（默认关，保持既有评估
   可比性），开启后用 Gymnasium 的 `np_random` 对 cube 初始 xy 施加 ±`start_xy_range`
   扰动。验证：同 seed 三次 reset 起点完全一致，不同 seed 起点不同——可复现且可控。
3. **动作归一化统一**：明确唯一契约——动作空间 `Box([-0.02]*3+[0], [0.02]*3+[1])`，
   策略输出 [-1,1] 经 `lo + (u+1)/2*(hi-lo)` 映射到物理增量；训练与推理共用同一
   `action_space`，`step()` 内只 clip 一次，不做二次缩放。往返无损验证通过。

## 3. 工程踩坑

- **段错误**：若在构造 `PPO("CnnPolicy")` 前手动 `reset()` 出图（建立 MuJoCo EGL 上下文），
  随后 torch 卷积初始化会段错误（软件 EGL 主机）。解决：`OffscreenRenderer` 懒初始化 +
  训练脚本"先构造 PPO、后由 learn 触发渲染"的顺序约束。纯 torch CNN 与 GL 互不冲突，
  问题特定于 SB3 构造路径。
- **EGL + fork 死锁**：父进程先渲染再 `SubprocVecEnv` fork 会继承 GL 上下文而死锁；视觉
  训练默认单环境，规避该问题（历史记录见 `docs/SB3_SETUP_REPORT.md`）。
- **CUDA 驱动告警（已修复）**：此前 torch 2.15+cu132 遇驱动 12.4 不匹配；现已将
  `mujoco` 环境 torch 修为 2.6.0+cu124，CUDA 可用，视觉 PPO 可用 `device=cuda` 训练
  达 209 fps（详见 `docs/GPU_ACCELERATION_FIX.md`）。

## 4. 目录结构与解耦

统一为 `algos/`（算法实现：ppo、后续 dt/iql）、`configs/`（超参配置）、`utils/`（评估/
录屏/日志）、`scripts/`（一键入口与验证）四层，环境与算法分仓：
`mujoco-studio-lab` 提供 `pushcube-envs` 包（含环境与离屏渲染工具），本仓仅 `import`
引用。新增算法只需在 `algos/` 加目录、`configs/` 加配置，不改环境代码。

## 5. 验证与产物
- 标准化验证 `scripts/verify_vision_pipeline.py`：9 项检查全 PASS，结果
  `results/metrics/VISION_PIPELINE_VERIFY.json`（训练档 481.8 fps）。
- CNN PPO 闭环 `results/metrics/CNN_PPO_VISION_SMOKE.json`：视觉与 state 策略均正常。
