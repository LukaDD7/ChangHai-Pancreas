---
title: "Method: Skeletonization Fallback (Zhang-Suen)"
created: 2026-04-17
type: method
mocs: (Interdisciplinary Index)
project: Pancreatic_Cancer
---

# Method: Skeletonization Fallback (Zhang-Suen)

## Overview
在 `scikit-image >= 0.25` 中，原有的 `skeletonize_3d` 函数已被移除。为了保持 PancreasMDT 在现代 Python 环境下的稳定性，系统实现了基于 **Zhang-Suen** 算法的逐层 (per-slice) Fallback 方案。

## Core Components
### 1. Zhang-Suen Algorithm
**Formulation:**
一种用于细化（Thinning）二值图像的迭代算法。在每个迭代周期内，根据 8-邻域像素分布决定当前像素是否保留。

**Key Characteristics:**
- **优点**: 保证拓扑连通性，对于细长的管状结构（如血管）非常有效。
- **缺点**: 2D 算法直接逐层应用可能因 Z 轴不连续性导致 3D 骨架在纵向上不够平滑。

## Implementation Details
### Tools/Frameworks
- **scikit-image**: 用于基础图像处理。
- **networkx**: 用于将提取的骨架像素点转化为可拓扑排序的图结构。

### Critical Considerations
- **Environment Matrix Warning**: 在 `Ubuntu_GPU_Server` 等现代服务器上安装最新版依赖时，必须检测 `scikit-image` 版本。
- **Fallback Trigger**: 
  ```python
  try:
      from skimage.morphology import skeletonize_3d
  except ImportError:
      # Use Zhang-Suen per-slice fallback
  ```

## Related Methods
- [[\../methods/Vessel-Centerline-Extraction]]
- [[\../methods/MPR-Resampling]]

