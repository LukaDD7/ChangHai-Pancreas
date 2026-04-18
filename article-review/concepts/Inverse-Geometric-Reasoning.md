---
title: "Inverse Geometric Reasoning (逆向几何推理)"
created: 2026-04-17
updated: 2026-04-17
tags: [concept, pancreatic-cancer, vascular-invasion, geometry]
sources: [PancreasMDT]
mocs: (Interdisciplinary Index)
---

## Definition
一种规避等密度肿瘤直接分割难题的推理策略：不分割肿瘤边界，而是通过测量被侵犯血管的**形态变化**（偏心率增高、泪滴征出现）来反推肿瘤的存在和侵犯程度。

## Theory/Mechanism
1. 增强 CT 下，血管 (SMA/CA) 被造影剂充盈，对比度极高，分割可靠。
2. 在血管中心线的正交切面上，正常血管呈**正圆**。
3. 当肿瘤侵犯时，血管截面受力变形:
   - 偏心率 (Eccentricity) 增大: 圆 → 椭圆
   - 泪滴征 (Teardrop sign): 局部尖锐凹陷
   - 凸包缺陷 (Convexity Defects): 边缘出现不规则凹陷
4. 上述形态特征可用 OpenCV 几何算子精确量化 (`cv2.fitEllipse`, `cv2.convexityDefects`)。

## 为什么是创新
- AI 学界路径依赖于"端到端像素级分割"，忽略了**工程化几何特征**直接提取。
- 目前所有系统 (PAN-VIQ, LungNoduleAgent, PathAgent) 都假设分割成功。
- **PancreasMDT 是首个处理"分割失败"场景的系统**，这是一个确认的蓝海。

## Related Concepts
- [[\../concepts/Orthogonal-MPR-Navigation]]
- [[\../concepts/Teardrop-Sign]]

