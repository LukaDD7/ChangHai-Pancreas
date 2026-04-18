---
title: "Teardrop Sign (泪滴征)"
created: 2026-04-17
updated: 2026-04-17
tags: [concept, radiology, vascular-invasion, pancreatic-cancer]
sources: [PancreasMDT]
mocs: (Interdisciplinary Index)
---

## Definition
在 CT 增强扫描的血管正交切面上，当肿瘤局部侵犯血管壁时，血管截面由正圆变为一端尖锐的泪滴形状。这是放射科医生判断 SMA 受侵犯的实用影像学征象。

## 检测方法 (PancreasMDT)
1. 使用 `cv2.findContours` 提取血管轮廓
2. 使用 `cv2.fitEllipse` 拟合椭圆 → 计算偏心率
3. 使用 `cv2.convexHull` + `cv2.convexityDefects` → 检测局部尖锐凹陷
4. 圆度 (Circularity) = 4πA/P² → 偏离 1.0 越多越显著

## Clinical Significance
- 泪滴征的出现强烈提示 SMA 已被肿瘤包裹/侵犯
- 即使在等密度肿瘤无法被分割时，血管形变依然可被检测
- 这是 [[\../concepts/Inverse-Geometric-Reasoning]] 的核心依据

## Related Concepts
- [[\../concepts/Inverse-Geometric-Reasoning]]
- [[\../concepts/Orthogonal-MPR-Navigation]]

