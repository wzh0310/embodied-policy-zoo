# assets —— 演示素材

本目录归档项目演示视频与结果截图，均为真实推理产出（非开环回放）。视频总量约 1.3 MB，
体积小，直接入库，无需网盘外链。

## videos/

| 文件 | 内容 | 实验条件 |
|---|---|---|
| `demo_short.mp4` | **主演示**：4 个成功回合合成，13.8 秒，双视角（左第三人称全局 / 右第一人称观测），896×224 | 最优 ACT `chunk40` checkpoint `004000`；4 个成功种子最终距离 0.9–2.5 cm |
| `dual_view.mp4` | 单回合双视角逐帧推理，约 3.4 秒 | 最优 ACT checkpoint，成功回合 |
| `success_rollout.mp4` | 策略单视角成功 rollout，约 4.0 秒 | 最优 ACT checkpoint |
| `expert_demo.mp4` | 专家推动轨迹回放 | 专家演示数据 |

## figures/

| 文件 | 内容 |
|---|---|
| `demo_short_preview.png` | 演示视频首帧（双视角预览，README 内嵌） |
| `keyframe_scene.png` | 场景概览关键帧 |
| `keyframe_goal.png` | 目标达成关键帧 |

## 复现方式

```bash
# 录制若干成功回合片段（中间产物落在 results/videos/.clips/）
python scripts/run.py record_clips --checkpoint ../outputs/act_chunk40/checkpoints/004000 --num-clips 4
# 合成为约 15 秒短视频（输出到本目录 videos/）
python scripts/make_demo_video.py
```
