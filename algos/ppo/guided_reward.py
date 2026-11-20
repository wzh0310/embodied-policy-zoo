#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练侧引导式奖励包装器（Gymnasium Wrapper，仅用于在线 RL 训练）。

不修改环境本体，保持与模仿学习对比的公平性。

原始奖励的问题：EE-to-cube 引导权重仅 0.1（每步最多 ~0.002），而末端初始距
立方体约 0.55m —— 策略学不到"先移动到立方体旁"这关键第一步，立方体始终不动，
距离势能与差分塑形全部为 0，导致奖励长期卡在 -0.5/step 无学习信号。

本包装器用两段式势能奖励明确引导：
    阶段一：末端(EE)接近立方体      —— 强势能，学会靠近；
    阶段二：立方体接近目标圆盘      —— 强势能 + 到达分级奖励；
外加较大的成功奖励与掉落惩罚。势能形式比差分塑形更稳定，不依赖上一步状态。
"""
import numpy as np
import gymnasium as gym

TARGET_XY = np.array([0.82, -0.12])

# 归一化尺度（用于把距离映射到 0~1 的势能）
EE_DIST0 = 0.55   # 末端到立方体的初始距离
CUBE_DIST0 = 0.20  # 立方体到目标的初始距离


class GuidedRewardWrapper(gym.Wrapper):
    def __init__(self, env, ee_weight=8.0, cube_weight=15.0,
                 success_bonus=100.0, fall_penalty=-10.0, time_penalty=0.01):
        super().__init__(env)
        self.ee_weight = ee_weight
        self.cube_weight = cube_weight
        self.success_bonus = success_bonus
        self.fall_penalty = fall_penalty
        self.time_penalty = time_penalty

    def _potentials(self):
        u = self.env.unwrapped
        cube = u.data.xpos[u._cube_body]
        ee = u.data.xpos[u._hand_body]
        cube_d = float(np.linalg.norm(cube[:2] - TARGET_XY))
        ee_d = float(np.linalg.norm(ee[:2] - cube[:2]))
        return cube_d, ee_d

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cube_d, ee_d = self._potentials()
        success = bool(info.get("success", False))
        fell = bool(info.get("fell_off", False))

        # 两段势能：靠近立方体 + 推动立方体靠近目标
        r_ee = self.ee_weight * (EE_DIST0 - min(ee_d, EE_DIST0)) / EE_DIST0
        r_cube = self.cube_weight * (CUBE_DIST0 - min(cube_d, CUBE_DIST0)) / CUBE_DIST0
        guided = r_ee + r_cube - self.time_penalty
        # 到达分级奖励：越接近目标额外奖励（引导精细收力）
        if cube_d < 0.10:
            guided += 5.0
        if cube_d < 0.05:
            guided += 10.0
        if success:
            guided += self.success_bonus
        if fell:
            guided += self.fall_penalty

        info["guided_reward"] = float(guided)
        info["raw_reward"] = float(reward)
        info["ee_dist"] = ee_d
        info["cube_dist"] = cube_d
        return obs, float(guided), terminated, truncated, info
