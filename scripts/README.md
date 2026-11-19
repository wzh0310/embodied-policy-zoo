# scripts

## 主入口（日常使用）

这三个脚本是标准化统一入口，README「快速上手」即基于它们：

| 脚本 | 职责 | 命令示例 |
|---|---|---|
| `train.py` | 训练启动（解析配置 → 注册环境 → 进 LeRobot 官方训练） | `python scripts/train.py --config configs/act_push_cube/act_full_train.yaml` |
| `eval.py` | 固定种子量化评估（3×10=30 回合，输出指标 JSON） | `python scripts/eval.py --checkpoint <ckpt>/004000` |
| `record_video.py` | 双视角演示视频录制（全局 + 任务相机） | `python scripts/record_video.py --checkpoint <ckpt>/004000` |

三者均通过环境变量 `STUDIO_LAB_ROOT` 定位兄弟仓库 `mujoco-studio-lab`（数据集 + 自定义环境），默认 `/data/wzh/robot/mujoco-studio-lab`。

## 辅助工具

| 脚本 | 职责 |
|---|---|
| `plot_training_curve.py` | 从训练日志解析 loss 生成曲线 PNG（仅依赖 Pillow） |
| `preprocessing.py` | 归一化规则固化（图像 →[-1,1]，动作 min-max →[-1,1]） |

## 研究脚本（复现第 5–6 周实验）

以下脚本用于复现已完成的消融与失败分析实验，**非日常主链路**；保留以支撑 `docs/TUNING_REPORT.md` 与 `docs/FAILURE_ANALYSIS_REPORT.md` 的可复现性：

| 脚本 | 职责 |
|---|---|
| `ablation_pipeline.py` | 14 组消融实验编排（生成配置变体 → 训练 → 评估 → 汇总） |
| `analyze_failures.py` | 失败轨迹逐帧分析与三类失败归因 |
| `collect_demos_v3.py` | 针对性演示数据生成（扩大 y 偏移覆盖） |

## 历史脚本（保留兼容）

`eval_act.py`、`record_dual_view.py`、`train_act_push_cube.sh` 是规范化之前的第一版脚本，功能已被上面三个主入口取代。历史文档与消融脚本中仍引用它们，故保留不删。**新实验请使用主入口。**
