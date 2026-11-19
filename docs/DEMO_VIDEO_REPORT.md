# 项目演示视频

## 成片

[`results/videos/demo_short.mp4`](results/videos/demo_short.mp4) —— 约 13.8 秒，
896×224 双视角，25 fps，H.264。

| 属性 | 值 |
|---|---|
| 时长 | 13.76 秒 |
| 分辨率 | 896 × 224（左右各 448 × 224） |
| 帧数 / 帧率 | 344 帧 / 25 fps |
| 内容 | 4 个真实成功回合，每回合约 3.4 秒（原始 33-36 帧放慢 5×，逐帧策略推理） |

## 视角说明

- **左（第三人称）**：`global_cam`，Panda 机械臂完整推物过程的场景全景。
- **右（第一人称）**：`top_cam`，喂给策略的 RGB 观测——立方体（黄）与目标圆盘（红）。

两个视角同步来自同一真实 rollout，展示"模型看到的输入 → 输出的动作 → 场景中的效果"。

## 片段来源

| 片段 | 种子 | 完成步数 | 最终距离 |
|---|---|---|---|
| 1 | 47988 | 36 | 0.0253 m |
| 2 | 85728 | 35 | 0.0222 m |
| 3 | 24500 | 33 | 0.0249 m |
| 4 | 18766 | 33 | 0.0092 m |

均来自最优 ACT checkpoint（`act_chunk40/checkpoints/004000`），从评估同分布的 50 个
种子中挑出的成功回合，非开环回放。

## 复现

```bash
# 1. 批量录制成功片段
python scripts/record_demo_clips.py \
  --checkpoint /data/wzh/robot/outputs/act_chunk40/checkpoints/004000 \
  --num-clips 4 --out-dir results/videos/.clips
# 2. 合成为约 15 秒短视频
python scripts/make_demo_video.py
```

首帧预览：[`results/videos/demo_short_preview.png`](results/videos/demo_short_preview.png)
