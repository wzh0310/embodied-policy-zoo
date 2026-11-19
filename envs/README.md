# envs —— 环境适配层

本目录是**策略仓库与仿真仓库之间的适配层**。本仓库只做算法，不拥有仿真资产；
PushCube 的 MJCF 场景、Gym 环境、LeRobot 环境注册和专家数据都由兄弟仓库
[`mujoco-studio-lab`](https://github.com/wzh0310/mujoco-studio-lab) 提供。

## 文件

| 文件 | 职责 |
|---|---|
| `paths.py` | 统一路径解析：由 `__file__` 推导本仓库根，定位兄弟仓库与输出目录，展开配置占位符 |

## 路径约定（无硬编码）

| 项 | 默认值 | 覆盖方式 |
|---|---|---|
| 本仓库根 | 由 `__file__` 自动推导 | 无需配置 |
| 兄弟仓库 | 与仓库同级的 `../mujoco-studio-lab` | 环境变量 `STUDIO_LAB_ROOT` |
| 训练输出 | 仓库同级的 `../outputs` | 环境变量 `OUTPUTS_DIR` 或 `--output-dir` |

配置 YAML 中使用占位符 `${STUDIO_LAB_ROOT}` / `${OUTPUTS_DIR}`，
训练入口 `scripts/train.py` 会在解析前把它们替换为当前机器的实际路径。
因此克隆到任意机器都无需改动配置文件。

## 环境注册机制

LeRobot 0.7.0 定制版不会自动扫描插件目录，必须在导入训练入口**之前**导入
自定义环境模块完成注册。这一逻辑集中在 `scripts/train.py` / `eval.py` 中，
通过 `paths.setup_sys_path()` 统一完成，避免每个脚本各写一遍。
