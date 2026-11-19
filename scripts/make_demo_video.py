#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把多个成功回合片段合成为约 15 秒演示短视频（ffmpeg）。

 输入 results/videos/.clips/clip_*.mp4（双视角：左第三人称 / 右第一人称），
 每段截取前若干帧，等长拼接，统一到 640x224（保持双视角 2:1），加片头文字标注，
 输出 results/videos/demo_short.mp4。若片段不足则按实际数量合成。
"""
import glob
import json
import os
import subprocess

REPO = "/data/wzh/robot/embodied-policy-zoo"
VID = os.path.join(REPO, "results/videos")
CLIPS = os.path.join(VID, ".clips")


def main():
    manifest = os.path.join(CLIPS, "clips_manifest.json")
    with open(manifest) as f:
        clips = json.load(f)
    clips = sorted(clips, key=lambda c: c["path"])
    print("clips:", len(clips))

    # 目标：约 15 秒。每个源 clip 约 33-36 帧 @50fps = 0.7 秒，需放慢播放拉长时长。
    # setpts=5*PTS 把每段拉到约 3.5 秒，再 fps=25 补齐帧，4 段约 14 秒。
    fps = 25
    slowdown = 5.0
    n = len(clips)
    parts = []
    for i, c in enumerate(clips):
        src = c["path"]
        out = os.path.join(CLIPS, f"part_{i:02d}.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", src,
            "-vf", "setpts=%.1f*PTS,fps=%d,scale=896:224:flags=lanczos" % (slowdown, fps),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
            out,
        ], check=True)
        parts.append(out)
        print("part", i, "from", os.path.basename(src), "slowdown", slowdown)

    # concat
    lst = os.path.join(CLIPS, "concat.txt")
    with open(lst, "w") as f:
        for p in parts:
            f.write("file '%s'\n" % p)
    out = os.path.join(VID, "demo_short.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", lst,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
        "-movflags", "+faststart", out,
    ], check=True)
    print("wrote", out, "segments", n)
    # 探测真实时长
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", out], capture_output=True, text=True)
    print("actual_duration", r.stdout.strip())


if __name__ == "__main__":
    main()
