# Sprint 1 — 几何管线底层模块搭建

**日期**: 2026-04-17

---

## 工程结构

```
mdt_agent/
├── __init__.py
└── tools/
    ├── __init__.py
    ├── centerline_tool.py      # 3D骨架提取 (VesselCenterlineExtractor)
    ├── geometry_detector.py    # 2D几何分析 (VascularShapeEvaluator)
    └── mpr_controller.py       # MPR切面重采样 (extract_orthogonal_slice)
```

---

## 逐模块实现

### centerline_tool.py — VesselCenterlineExtractor

**问题**: PAN-VIQ 使用的 `skeletonize_3d` 在 scikit-image ≥0.25 中已移除，所有 conda 环境均为 0.25.2。

**实现**:
- 自动检测是否可用 `skeletonize_3d`（scikit-image < 0.25），不可用则降级为 per-slice zhang 方案
- per-slice zhang：对每个 z-slice 分别做 2D skeletonize，再 stack 回 3D
- 图排序：26-neighborhood networkx graph → Dijkstra 找最远点对 → 最短路径 = 有序中心线

**测试**:
```python
ext = VesselCenterlineExtractor()
coords = ext.extract(sma_data)  # AbdomenAtlas SMA mask
# SMA centerline: 59 points (skeleton backend: slice_by_slice)
# First 3: [(268, 176, 156), (268, 177, 156), (268, 178, 156)]
# Last 3:  [(255, 215, 120), (255, 215, 119), (255, 215, 118)]
```

---

### geometry_detector.py — VascularShapeEvaluator

**问题**: opencv-python 未安装在任何 conda 环境；AbdomenAtlas mask 是 0/1 二值而非 0/255。

**实现**:
- `calculate_eccentricity(mask)`: cv2.findContours → cv2.fitEllipse → sqrt(1-(b/a)²)
- `detect_teardrop_sign(mask)`: circularity < 0.7 AND convexityDefects > 5% → 泪滴检测
- `circularity(mask)`: 4πA/P²
- 自动兼容 binary (0/1) 和 uint8 (0/255) mask 格式

**测试** (pytest inline):
```
Perfect circle:   eccentricity=0.0000, circularity=0.8786  ✅
Elongated ellipse: eccentricity=0.9269, circularity=0.6430  ✅
Teardrop:         circularity=0.4938, is_teardrop=True, defects=14  ✅
```

真实切片测试:
```
Aorta  z=260: eccentricity=0.989, circularity=0.193
SMA    z=262: eccentricity=0.969, circularity=0.360
```

---

### mpr_controller.py — extract_orthogonal_slice

**问题**: 初版 Euler3DTransform + SetMatrix 旋转导致 SimpleITK 空值（direction/origin 语义混淆）。

**实现** (最终方案):
1. Gram-Schmidt 构建 (T, B, N) 正交标架（tangent/binormal/normal）
2. OutputDirection = [n|b|t]（列向量），编码切面朝向物理空间
3. TranslationTransform offset = 输入图像中 cruise point 的物理坐标
4. OutputOrigin = [0,0,0]，OutputSpacing = 输入 x/y/z spacing
5. SimpleITK size = (W, H, 1) — 单层 3D volume 即 2D 切面

**测试** (真实数据):
```python
slice_out = extract_orthogonal_slice(ct_nii, coords[mid], tangent, (128, 128))
# shape: (128, 128), HU = [-974, 315]  ✅
```

---

## Git 提交记录

| 提交 | 内容 |
|------|------|
| `0149710` | feat: 初始 mdt_agent 框架 + 3 个工具模块初版 |
| `8f72e5c` | fix: geometry_detector 支持 binary (0/1) 和 uint8 (0/255) mask 两种格式 |
| `a7f6ab8` | fix: mpr_controller 重写为 TranslationTransform + OutputDirection，可工作 |

---

## 环境配置变更

- ChangHai Python: 3.11
- 额外安装: `opencv-python 4.13.0`, `SimpleITK 2.5.3`
- 运行路径: `/home/luzhenyang/anaconda3/envs/ChangHai/bin/python`

---

## 几何管线流程

```
血管 mask (nii.gz)
  → VesselCenterlineExtractor.extract()  → 有序 3D 中心线点列表
  → 取相邻两点算 tangent（单位向量）
  → extract_orthogonal_slice(CT, point, tangent) → 2D MPR 切面
  → VascularShapeEvaluator.evaluate() → eccentricity / circularity / teardrop sign
  → 几何异常 → 触发 Agent 认知失调报告
```

---

## 已知限制

- `slice_by_slice` skeleton 后端仅生成 3 个 SMA 中心线点（per-slice 骨架连续性不足），skeletonize_3d 后端可生成 59 个（需 scikit-image < 0.25）
- 如需更密集采样点，需升级 scikit-image 版本或使用其他骨架化库
