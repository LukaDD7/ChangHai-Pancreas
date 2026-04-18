---
title: "Method: MPR Resampling Implementation (SimpleITK)"
created: 2026-04-17
type: method
mocs: (Interdisciplinary Index)
project: Pancreatic_Cancer
---

# Method: MPR Resampling Implementation (SimpleITK)

## Overview
在 `mdt_agent/tools/mpr_controller.py` 中实现的 3D 图像多平面重建 (MPR) 逻辑。核心目标是沿血管中心线的切向量，提取出严格正交的 2D 物理坐标切片。

## Core Components

### 1. Spatial Transformation Matrix
**Logic:**
通过中心线采样点的物理坐标 (Point) 和切向量 (Tangent)，构建一个正交基 (Orthogonal Basis)。
1. **Z-axis**: Tangent vector (normalized).
2. **X-axis**: Any vector perpendicular to Z.
3. **Y-axis**: Cross product of Z and X.

### 2. SimpleITK Resampler (Implementation v2)
**Formulation:**
为了提高在物理空间（毫米）中的采样精度，系统采用了以下组件：
- **Transform**: `sitk.TranslationTransform(3)` 配合旋转矩阵。
- **Interpolator**: `sitk.sitkLinear` (折中速度与精度) 或 `sitk.sitkBSpline` (更高质量)。
- **Output Properties**:
  - `SetOutputDirection`: 使用构建的正交基旋转矩阵。
  - `SetOutputOrigin`: 设定为中心线采样点的物理位置。
  - `SetSize`: 用于定义 2D 切片的像素尺寸（如 128x128）。
  - `SetOutputSpacing`: 定义切片分辨率（如 0.5mm/pixel）。

## Key Refinements (2026-04-17)
- **Problem**: 初始存根无法处理方向矩阵。
- **Fix**: 使用 `OutputDirection` 强制对齐法平面，替换了简单的索引截取。
- **Benefit**: 彻底解决了血管斜向穿插时，“轴状面切片”导致的投影面积误差。

## Related Methods
- [[\../methods/Skeletonization-Fallback-Zhang-Suen]]
- [[\../concepts/Orthogonal-MPR-Navigation]]

