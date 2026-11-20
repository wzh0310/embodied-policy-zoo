#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练侧视觉观测增强（Gymnasium ObservationWrapper）。

对 obs_mode='rgb' 的 CHW float32 [0,1] 灰度图做两类轻量增强，缓解视觉 PPO 过拟合：
  - 随机平移裁剪：四边 edge padding 后随机 crop 回原尺寸，模拟视角微抖；
  - 亮度/对比抖动：乘性亮度 + 加性噪声，clip 回 [0,1]。

只在训练环境启用；评估/推理不包裹此 wrapper，保证评估确定性。
随机性取自 gym 的 self.np_random，随 reset(seed=) 可复现。
"""
import numpy as np
import gymnasium as gym


class VisionAugmentationWrapper(gym.ObservationWrapper):
    def __init__(self, env, crop_pad=4, brightness=(0.85, 1.15),
                 noise_std=0.03, enable_crop=True, enable_color=True):
        super().__init__(env)
        self.crop_pad = int(crop_pad)
        self.brightness = brightness
        self.noise_std = float(noise_std)
        self.enable_crop = enable_crop
        self.enable_color = enable_color

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        out = self.env.reset(**kwargs)
        if seed is not None:
            # 让增强随机流也随 seed 复现
            try:
                self._np_random, _ = gym.utils.seeding.np_random(seed)
            except Exception:
                pass
        return self.observation(out[0]), out[1]

    def observation(self, obs):
        img = np.asarray(obs, dtype=np.float32)
        if img.ndim != 3:
            return obs
        c, h, w = img.shape
        if self.enable_crop and self.crop_pad > 0:
            p = self.crop_pad
            padded = np.pad(img, ((0, 0), (p, p), (p, p)), mode="edge")
            top = int(self.np_random.integers(0, 2 * p + 1))
            left = int(self.np_random.integers(0, 2 * p + 1))
            img = padded[:, top:top + h, left:left + w]
        if self.enable_color:
            b = float(self.np_random.uniform(*self.brightness))
            img = img * b + self.np_random.normal(0.0, self.noise_std, size=img.shape)
            img = np.clip(img, 0.0, 1.0)
        return img.astype(np.float32)
