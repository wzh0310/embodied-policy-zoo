# videos —— 录制中间产物

成片与关键帧已归档到 [`../../assets/`](../../assets/README.md)：

- `demo_short.mp4`（主演示短视频）、`dual_view.mp4`、`success_rollout.mp4`、`expert_demo.mp4`
- `demo_short_preview.png`、`keyframe_scene.png`、`keyframe_goal.png`

本目录仅保留 `.clips/`（批量录制的单回合片段，属于中间产物，已在 `.gitignore` 中忽略）。
重新生成成片：先 `python scripts/record_demo_clips.py`，再 `python scripts/make_demo_video.py`。
