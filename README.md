# embodied-policy-zoo

RL / 模仿学习 pipeline + 数据清洗 + LIBERO 评测。

## 结构

- `pipelines/` - 训练 pipeline（BC / RL / diffusion policy）
- `data/` - 数据清洗与转换
- `eval/libero/` - LIBERO 评测
- `configs/` - 训练配置
- `scripts/` - 训练 / 评测脚本
- `docs/` - 踩坑与适配记录

## 自定义场景 ACT 训练（push_cube）

把 `mujoco-studio-lab` 第一周导出的 Panda 推方块数据集，用 LeRobot 0.7.0 的 ACT 策略做
小规模冒烟训练。配置、启动脚本、踩坑记录都在本仓库：

```bash
# 在 13 服务器容器内、lerobot conda 环境
bash scripts/train_act_push_cube.sh            # 默认 100 步 / batch 8（CPU）
STEPS=200 BATCH=8 bash scripts/train_act_push_cube.sh   # 覆盖规模
```

- 配置：`configs/act_push_cube.yaml`（源自官方 ACT 模板，适配本地数据集 + push_cube 环境）
- 依赖：训练需要 `accelerate`（`pip install "accelerate>=1.0,<2.0"`）
- 前提：兄弟仓库 `mujoco-studio-lab` 提供数据集与 `push_cube` 环境注册（wrapper 自动加载）
- 踩坑与字段映射：`docs/ACT_TRAINING_NOTES.md`

### 冒烟结果（100 步 / CPU）

loss 单调下降 `1.78 → 0.53`（4.37 epoch），step 50/100 各触发一次闭环评估并生成视频，
checkpoint 保存于 `/data/wzh/robot/outputs/act_push_cube_smoke/checkpoints/000100/`。
未收敛策略成功率 0% 属预期——本轮只验证数据/维度/归一化/评估全链路无阻塞。
