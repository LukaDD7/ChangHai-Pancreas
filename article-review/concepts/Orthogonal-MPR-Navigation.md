---
title: "Orthogonal MPR Navigation (3D 正交巡航)"
created: 2026-04-17
updated: 2026-04-17
tags: [concept, MPR, navigation, 3D, centerline]
sources: [PancreasMDT, PathAgent]
mocs: (Interdisciplinary Index)
---

## Definition
沿 3D 血管中心线 (Centerline)，以固定步长（如 1-2mm）推进，在每个采样点计算切向量并生成与之**严格垂直的 2D 横截面 (Orthogonal Plane)**，用于后续的形态分析和包绕角计算。

## Theory/Mechanism
1. CT 原始轴状面 (Axial) 是水平切片，血管斜着穿过时截面变形为椭圆 → 测量不准。
2. 使用 `extract_centerline.py` (PAN-VIQ) 提取血管骨架线。
3. 在中心线上每个点计算切向量 (Tangent vector)。
4. 生成与切向量垂直的 2D 平面 (via SimpleITK ResampleImageFilter + Euler3DTransform)。
5. 在正交切面上，血管始终呈圆形靶心 → 包绕角可精确测量。

## 与 PathAgent Navigator 的类比
| 维度 | PathAgent | PancreasMDT |
| :--- | :--- | :--- |
| 探索空间 | 2D WSI (XY) | 3D CT volume (沿中心线) |
| 导航方式 | 多尺度放大 | 正交切面步进 |
| 停留判断 | 发现可疑 ROI | 偏心率异常 / 脂肪间隙模糊 |

## Related Concepts
- [[\../concepts/Inverse-Geometric-Reasoning]]
- [[\../concepts/Teardrop-Sign]]

