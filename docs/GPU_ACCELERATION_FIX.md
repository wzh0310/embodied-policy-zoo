# GPU 加速可用性纠正与视觉 PPO 性能修复

本记录纠正此前"容器无 GPU 可用"的错误判断，并给出修复后的实测性能。

## 1. 问题复盘：一次错误的硬件判断

此前曾判定"容器无 GPU 直通，视觉 PPO 只能在 CPU 上运行（6-8 fps）"。该判断**错误**，根因是排查时检查了 `/dev/dri`（AMD/Intel 的 DRM 接口），而 NVIDIA GPU 走的是 `/dev/nvidia*` 设备节点，两者被混淆。

实测确认容器具备完整 GPU 能力：

| 项 | 实测值 |
|---|---|
| GPU | NVIDIA RTX A6000，49140 MiB 显存 |
| 驱动 | 550.107.02（CUDA 12.4） |
| 设备节点 | `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 均存在 |
| `nvidia-smi` | 正常返回 GPU 状态 |

`/dev/dri` 缺失只说明没有 DRM 直通，与 NVIDIA GPU 能否使用无关。

## 2. torch CUDA 不可用的真正原因

GPU 在，但 `mujoco` 环境的 torch 无法调用：

```
torch 2.15.0+cu132  →  cuda_available = False
UserWarning: The NVIDIA driver on your system is too old (found version 12040)
```

根因：**torch 是用 CUDA 13.2 编译的（cu132），而驱动只支持到 CUDA 12.4**。这是编译版本与驱动版本不匹配，不是硬件问题。`lerobot` 环境此前已将 torch 修为 `2.6.0+cu124` 并验证 CUDA 可用，证明环境本身完全支持 GPU。

## 3. 修复方案

将 `mujoco` 环境的 torch 对齐到已验证的 cu124 配置：

- `torch 2.6.0+cu124` + `torchvision 0.21.0+cu124`
- 配套 nvidia cu12 运行时库（cublas / cudnn / cufft / curand / cusolver / cusparse / cusparselt / nccl / nvjitlink / nvtx / cuda_runtime 等）
- 关键补漏：torch `libtorch_cuda.so` 通过 RPATH 依赖顶层 `site-packages/cusparselt/lib/libcusparseLt.so.0`，需一并就位

修复后：

```
torch 2.6.0+cu124  →  cuda_available = True  →  NVIDIA RTX A6000
```

## 4. 修复前后性能对比

### 4.1 CNN 前反向微基准（batch 256，NatureCNN 类卷积）

| 设备 | 单次更新耗时 | 吞吐 |
|---|---|---|
| CPU | 590 ms | 1.7 upd/s |
| GPU (CUDA) | 2.3 ms | **432 upd/s** |

约 **250 倍**加速。

### 4.2 端到端视觉 PPO 训练（1 万步，CnnPolicy + NatureCNN + 4 帧堆叠）

| 配置 | 总耗时 | 吞吐 | 结果 |
|---|---|---|---|
| `device=cpu` | 1667 s（27.8 分钟） | 6 fps | 1 万步完成 |
| `device=cuda` | **48 s** | **209 fps** | 1 万步完成，模型保存 |

约 **35 倍**端到端提速。损失下降正常（loss -0.045、policy_gradient_loss -0.009、value_loss 1.36→0.12、explained_variance 转正）。

**结论**：默认 200 万步视觉 PPO 配置在修复后约需 2.7 小时，已完全可行；此前"数周、不现实"的结论作废。

## 5. MuJoCo 离屏渲染的现状（次要因素，如实记录）

- 已安装 NVIDIA EGL 库（`libnvidia-gl-550`、`libnvidia-egl-wayland1`，EGL vendor 出现 `10_nvidia.json`）
- 但 MuJoCo 离屏渲染仍走 mesa llvmpipe 软件光栅化（`GL_RENDERER=llvmpipe`），尝试 `EGL_PLATFORM=device/nvidia/surfaceless`、清除 `LIBGL_ALWAYS_SOFTWARE`、`NVIDIA_DRIVER_CAPABILITIES=all` 等组合均未切换成功
- **影响评估**：渲染不是视觉 PPO 的瓶颈——训练档几何代理渲染已达 ~482 fps，演示档完整场景 ~8 fps 仅供展示。CNN 前反向（真正的瓶颈）已通过 torch CUDA 解决，故 GPU 渲染未切换不影响训练吞吐目标
- 视觉 PPO 在 `device=cuda` 下 209 fps 的完整闭环已验证通过，说明当前渲染配置与 GPU 训练兼容

## 6. 待办与建议

- 若要进一步让 MuJoCo 走 GPU 渲染（用于演示档提速），需排查容器 EGL device 平台与 NVIDIA vendoring（可能需重建容器时挂载 graphics capability 或使用 NVIDIA 官方 CUDA 基础镜像）
- 训练侧结论不受影响：视觉 PPO 已具备 GPU 训练能力，可按 200 万步默认配置启动
