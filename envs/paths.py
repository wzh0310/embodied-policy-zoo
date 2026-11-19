#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一路径解析层 —— 消除脚本与配置里的绝对路径硬编码。

设计原则
--------
* 本仓库根 REPO 由 __file__ 推导，克隆到任意位置都能工作。
* 兄弟仓库 mujoco-studio-lab 默认与仓库同级（REPO.parent/mujoco-studio-lab），
  可用环境变量 STUDIO_LAB_ROOT 覆盖。
* 训练输出默认写到 REPO.parent/outputs，可用环境变量 OUTPUTS_DIR 覆盖。
* 配置模板用 ${STUDIO_LAB_ROOT} / ${OUTPUTS_DIR} 占位符，由 resolve_config_text
  在训练入口展开，因此 YAML 中不含任何机器相关的绝对路径。
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

STUDIO_LAB = Path(
    os.environ.get("STUDIO_LAB_ROOT", str(REPO.parent / "mujoco-studio-lab"))
).resolve()

OUTPUTS = Path(
    os.environ.get("OUTPUTS_DIR", str(REPO.parent / "outputs"))
).resolve()

PLACEHOLDERS = {
    "${STUDIO_LAB_ROOT}": str(STUDIO_LAB),
    "${OUTPUTS_DIR}": str(OUTPUTS),
}


def resolve_config_text(text: str) -> str:
    """把配置模板中的占位符替换为当前机器的实际绝对路径。"""
    for token, value in PLACEHOLDERS.items():
        text = text.replace(token, value)
    return text


def require_studio_lab() -> Path:
    if not STUDIO_LAB.is_dir():
        raise SystemExit(
            f"[envs] 找不到兄弟仓库 mujoco-studio-lab: {STUDIO_LAB}\n"
            "请设置环境变量 STUDIO_LAB_ROOT 指向其绝对路径后重试。"
        )
    return STUDIO_LAB


def setup_sys_path():
    """把本仓库、脚本目录、兄弟仓库环境层加入 sys.path，完成环境注册。"""
    import sys
    require_studio_lab()
    for p in (str(REPO), str(REPO / "scripts"), str(STUDIO_LAB),
              str(STUDIO_LAB / "envs")):
        if p not in sys.path:
            sys.path.insert(0, p)


def renderer_env():
    """返回无显示器离屏渲染所需的环境变量字典。"""
    return {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "EGL_PLATFORM": "surfaceless",
        "TQDM_DISABLE": "1",
        "STUDIO_LAB_ROOT": str(STUDIO_LAB),
    }
