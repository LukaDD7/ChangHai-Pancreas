#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
血管侵犯拓扑分析 Skill (Vascular Encasement Topology Analysis Skill)
================================================================================

功能描述:
    本脚本用于评估胰腺肿瘤对周围关键血管（如SMA、SMV）的侵犯程度，
    并计算最大的包绕角度。这直接决定了肿瘤的可切除性判定。

算法核心:
    - 形态学膨胀检测肿瘤-血管接触
    - 基于3D物理坐标的局部角度扫描
    - 临床可切除性分类

技术架构:
    - 输入: NIfTI格式的肿瘤掩膜和血管掩膜(.nii.gz)
    - 输出: JSON格式的侵犯分析结果
    - 核心库: nibabel, numpy, scipy.ndimage

临床标准 (长海医院):
    - Clear: 无接触
    - Borderline Resectable: 包绕角度 <= 180°
    - Locally Advanced: 包绕角度 > 180°

作者: Claude Code Assistant
日期: 2026-03-20
版本: 2.0.0
================================================================================
"""

import json
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure


TARGET_SPACING_MM = (1.0, 1.0, 1.0)


def load_mask_and_spacing(mask_path: str) -> Tuple[np.ndarray, Tuple[float, float, float], nib.Nifti1Image]:
    """加载NIfTI掩膜并提取体素间距。"""
    img = nib.as_closest_canonical(nib.load(mask_path))
    mask_data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]
    return mask_data, spacing, img


def resample_mask_to_reference(mask_img: nib.Nifti1Image, reference_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """将二值mask重采样到参考图像空间，使用最近邻插值避免标签污染。"""
    if mask_img.shape != reference_img.shape or not np.allclose(mask_img.affine, reference_img.affine, atol=1e-3):
        return resample_from_to(mask_img, reference_img, order=0)
    return mask_img


def prepare_common_grid(tumor_mask_path: str, vessel_mask_path: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """将肿瘤和血管mask统一到同一物理网格。"""
    tumor_data, _, tumor_img = load_mask_and_spacing(tumor_mask_path)
    vessel_data, _, vessel_img = load_mask_and_spacing(vessel_mask_path)

    tumor_iso_img = resample_to_output(tumor_img, voxel_sizes=TARGET_SPACING_MM, order=0)
    vessel_canonical_img = nib.as_closest_canonical(vessel_img)
    vessel_iso_img = resample_from_to(vessel_canonical_img, tumor_iso_img, order=0)

    tumor_iso = tumor_iso_img.get_fdata() > 0
    vessel_iso = vessel_iso_img.get_fdata() > 0

    if tumor_iso.shape != vessel_iso.shape:
        vessel_iso_img = resample_mask_to_reference(vessel_iso_img, tumor_iso_img)
        vessel_iso = vessel_iso_img.get_fdata() > 0

    return tumor_iso.astype(bool), vessel_iso.astype(bool), tumor_iso_img


def calculate_dilation_radius_voxels(physical_radius_mm: float, spacing: Tuple[float, float, float]) -> int:
    """将物理膨胀半径（毫米）转换为体素个数。"""
    avg_spacing = float(np.mean(spacing))
    radius_voxels = int(np.ceil(physical_radius_mm / avg_spacing))
    return max(1, radius_voxels)


def extract_2d_boundary(mask_2d: np.ndarray) -> np.ndarray:
    """提取2D二值掩膜的边界。"""
    if np.sum(mask_2d) == 0:
        return np.zeros_like(mask_2d, dtype=bool)

    structure = generate_binary_structure(2, 1)
    eroded = binary_erosion(mask_2d, structure, iterations=1)
    return mask_2d & (~eroded)


def extract_3d_boundary(mask_3d: np.ndarray) -> np.ndarray:
    """提取3D二值掩膜的边界。"""
    if np.sum(mask_3d) == 0:
        return np.zeros_like(mask_3d, dtype=bool)

    structure = generate_binary_structure(3, 1)
    eroded = binary_erosion(mask_3d, structure, iterations=1)
    return mask_3d & (~eroded)


def _legacy_encasement_angle(vessel_boundary_2d: np.ndarray, tumor_mask_2d: np.ndarray) -> Optional[float]:
    """保留旧版2D近似，兼容单元测试和历史逻辑。"""
    total_boundary_voxels = np.sum(vessel_boundary_2d)
    if total_boundary_voxels == 0:
        return None

    contact_boundary = vessel_boundary_2d & tumor_mask_2d
    contact_voxels = np.sum(contact_boundary)
    if contact_voxels == 0:
        return 0.0

    angle = (contact_voxels / total_boundary_voxels) * 360.0
    return min(angle, 360.0)


def _principal_axis(points: np.ndarray) -> np.ndarray:
    """通过PCA估计血管主轴。"""
    centered = points - points.mean(axis=0, keepdims=True)
    if centered.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0], dtype=float)

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return axis / norm


def _orthonormal_basis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """为给定主轴构建正交平面基底。"""
    axis = axis / (np.linalg.norm(axis) or 1.0)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(ref, axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(axis, ref)
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        u = np.array([0.0, 1.0, 0.0], dtype=float)
        u_norm = 1.0
    u = u / u_norm
    v = np.cross(axis, u)
    v = v / (np.linalg.norm(v) or 1.0)
    return u, v


def _project_to_plane(points: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """将3D点投影到垂直于axis的局部平面。"""
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    u, v = _orthonormal_basis(axis)
    centered = points - origin
    x = centered @ u
    y = centered @ v
    return np.column_stack([x, y])


def _angular_span_degrees(points_2d: np.ndarray) -> float:
    """计算平面内点集的最小覆盖角度。"""
    if points_2d.size == 0:
        return 0.0

    angles = np.mod(np.arctan2(points_2d[:, 1], points_2d[:, 0]), 2 * np.pi)
    if angles.size == 1:
        return 0.0

    angles = np.sort(angles)
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    coverage = 2 * np.pi - np.max(gaps)
    return float(np.degrees(max(0.0, coverage)))


def _local_physical_encasement_angle(
    vessel_boundary_mask: np.ndarray,
    contact_boundary_mask: np.ndarray,
    affine: np.ndarray,
    bin_width_mm: float = 1.0,
    min_points: int = 12,
) -> Tuple[float, Dict[int, float]]:
    """在血管主轴周围扫描局部截面，估计最大包绕角。"""
    vessel_coords = np.argwhere(vessel_boundary_mask)
    contact_coords = np.argwhere(contact_boundary_mask)

    if vessel_coords.size == 0 or contact_coords.size == 0:
        return 0.0, {}

    vessel_world = nib.affines.apply_affine(affine, vessel_coords)
    contact_world = nib.affines.apply_affine(affine, contact_coords)

    axis = _principal_axis(vessel_world)
    origin = vessel_world.mean(axis=0)

    vessel_t = (vessel_world - origin) @ axis
    contact_t = (contact_world - origin) @ axis

    bin_min = float(min(vessel_t.min(), contact_t.min()))
    bin_max = float(max(vessel_t.max(), contact_t.max()))
    if bin_max - bin_min < 1e-6:
        return 0.0, {}

    edges = np.arange(bin_min, bin_max + bin_width_mm, bin_width_mm)
    if edges.size < 2:
        edges = np.array([bin_min, bin_max], dtype=float)

    best_angle = 0.0
    angle_per_bin: Dict[int, float] = {}

    for idx in range(len(edges) - 1):
        lo, hi = edges[idx], edges[idx + 1]
        vessel_sel = (vessel_t >= lo) & (vessel_t < hi)
        contact_sel = (contact_t >= lo) & (contact_t < hi)

        if vessel_sel.sum() < min_points or contact_sel.sum() < 3:
            continue

        vessel_slice_world = vessel_world[vessel_sel]
        contact_slice_world = contact_world[contact_sel]
        local_origin = vessel_slice_world.mean(axis=0)

        vessel_2d = _project_to_plane(vessel_slice_world, local_origin, axis)
        contact_2d = _project_to_plane(contact_slice_world, local_origin, axis)

        if vessel_2d.size == 0 or contact_2d.size == 0:
            continue

        local_angle = _angular_span_degrees(contact_2d)
        if local_angle > 0:
            angle_per_bin[idx] = round(local_angle, 2)
            best_angle = max(best_angle, local_angle)

    return float(round(best_angle, 2)), angle_per_bin


def calculate_encasement_angle(
    vessel_boundary: np.ndarray,
    tumor_mask: np.ndarray,
    affine: Optional[np.ndarray] = None,
) -> Optional[float]:
    """计算血管被肿瘤包绕的角度。

    兼容两种模式：
    1) 旧版2D近似：当 affine 为空且输入为2D时使用周长比例法。
    2) 新版3D物理坐标：当提供 affine 时，使用局部截面扫描。
    """
    if affine is None and vessel_boundary.ndim == 2 and tumor_mask.ndim == 2:
        return _legacy_encasement_angle(vessel_boundary, tumor_mask)

    if affine is None:
        raise ValueError("3D encasement calculation requires affine metadata.")

    angle, _ = _local_physical_encasement_angle(vessel_boundary.astype(bool), tumor_mask.astype(bool), affine)
    return angle


def analyze_vascular_encasement(
    tumor_mask_path: str,
    vessel_mask_path: str,
    dilation_radius_mm: float = 2.0,
) -> Dict:
    """核心函数: 分析肿瘤对血管的侵犯程度。"""
    print(f"\n{'='*60}")
    print("血管侵犯拓扑分析 - Vascular Encasement Analysis")
    print(f"{'='*60}")
    print(f"[INFO] 肿瘤掩膜: {tumor_mask_path}")
    print(f"[INFO] 血管掩膜: {vessel_mask_path}")
    print(f"[INFO] 膨胀半径: {dilation_radius_mm}mm")

    print(f"\n{'-'*60}")
    print("步骤1: 加载与空间标准化...")
    print(f"{'-'*60}")

    try:
        tumor_mask, tumor_spacing, tumor_img = load_mask_and_spacing(tumor_mask_path)
        vessel_mask, vessel_spacing, vessel_img = load_mask_and_spacing(vessel_mask_path)
    except Exception as e:
        return {
            "is_contact": False,
            "max_angle_degree": 0.0,
            "clinical_status": "Error",
            "error_message": f"文件加载失败: {str(e)}",
        }

    tumor_iso_img = resample_to_output(tumor_img, voxel_sizes=TARGET_SPACING_MM, order=0)
    vessel_iso_img = resample_from_to(vessel_img, tumor_iso_img, order=0)

    tumor_mask = tumor_iso_img.get_fdata() > 0
    vessel_mask = vessel_iso_img.get_fdata() > 0

    print(f"[INFO] 肿瘤维度: {tumor_mask.shape}, 间距: {tumor_iso_img.header.get_zooms()[:3]}")
    print(f"[INFO] 血管维度: {vessel_mask.shape}, 间距: {vessel_iso_img.header.get_zooms()[:3]}")

    print(f"\n{'-'*60}")
    print("步骤2: 执行安全校验...")
    print(f"{'-'*60}")

    if tumor_mask.shape != vessel_mask.shape:
        error_msg = (
            f"【严重错误】肿瘤与血管掩膜维度不匹配!\n"
            f"  - 肿瘤: {tumor_mask.shape}\n"
            f"  - 血管: {vessel_mask.shape}"
        )
        print(f"[ERROR] {error_msg}")
        return {
            "is_contact": False,
            "max_angle_degree": 0.0,
            "clinical_status": "Error",
            "error_message": error_msg,
        }

    tumor_voxels = int(np.sum(tumor_mask))
    vessel_voxels = int(np.sum(vessel_mask))
    if tumor_voxels == 0:
        return {
            "is_contact": False,
            "max_angle_degree": 0.0,
            "clinical_status": "Clear (No Involvement)",
            "contact_z_slices": [],
            "angle_per_slice": {},
            "dilation_radius_mm": dilation_radius_mm,
            "status": "success",
            "limitation": "Tumor mask empty after normalization",
        }

    print(f"[PASS] 维度校验通过 | 肿瘤体素: {tumor_voxels} | 血管体素: {vessel_voxels}")

    print(f"\n{'-'*60}")
    print("步骤3: 形态学膨胀与接触检测...")
    print(f"{'-'*60}")

    spacing = tumor_iso_img.header.get_zooms()[:3]
    dilation_voxels = calculate_dilation_radius_voxels(dilation_radius_mm, spacing)
    print(f"[INFO] 物理膨胀半径: {dilation_radius_mm}mm ≈ {dilation_voxels}个体素")

    structure_3d = generate_binary_structure(3, 1)
    tumor_dilated = binary_dilation(tumor_mask, structure_3d, iterations=dilation_voxels)
    print(f"[INFO] 膨胀前肿瘤体素数: {tumor_voxels}")
    print(f"[INFO] 膨胀后肿瘤体素数: {int(np.sum(tumor_dilated))}")

    contact_mask = tumor_dilated & vessel_mask
    has_contact = bool(np.any(contact_mask))
    print(f"[INFO] 接触区域体素数: {int(np.sum(contact_mask))}")

    if not has_contact:
        print("[RESULT] 未检测到肿瘤-血管接触")
        return {
            "is_contact": False,
            "max_angle_degree": 0.0,
            "clinical_status": "Clear (No Involvement)",
            "contact_z_slices": [],
            "angle_per_slice": {},
            "dilation_radius_mm": dilation_radius_mm,
            "status": "success",
            "vessel_voxels": vessel_voxels,
        }

    print("[ALERT] 检测到肿瘤-血管接触，进入角度计算...")

    vessel_boundary = extract_3d_boundary(vessel_mask)
    contact_boundary = vessel_boundary & contact_mask

    max_angle, angle_per_slice = _local_physical_encasement_angle(
        vessel_boundary_mask=vessel_boundary,
        contact_boundary_mask=contact_boundary,
        affine=tumor_iso_img.affine,
        bin_width_mm=max(spacing),
        min_points=12,
    )

    contact_z_slices = np.unique(np.argwhere(contact_boundary)[:, 0]).tolist() if np.any(contact_boundary) else []

    print(f"\n{'-'*60}")
    print("步骤4: 临床可切除性分类...")
    print(f"{'-'*60}")

    if max_angle == 0:
        clinical_status = "Clear (No Involvement)"
    elif max_angle <= 180:
        clinical_status = "Borderline Resectable (<= 180 degrees)"
    else:
        clinical_status = "Locally Advanced (> 180 degrees)"

    print(f"[INFO] 最大包绕角度: {max_angle:.2f}°")
    print(f"[INFO] 接触层数: {len(contact_z_slices)}")
    print(f"[RESULT] 临床分类: {clinical_status}")

    result = {
        "is_contact": True,
        "max_angle_degree": round(float(max_angle), 2),
        "clinical_status": clinical_status,
        "contact_z_slices": contact_z_slices,
        "angle_per_slice": angle_per_slice,
        "dilation_radius_mm": dilation_radius_mm,
        "dilation_voxels": dilation_voxels,
        "total_z_slices": len(contact_z_slices),
        "tumor_spacing": [float(s) for s in spacing],
        "status": "success",
        "vessel_voxels": vessel_voxels,
        "algorithm": "3D physical-axis local angular span",
    }

    print(f"\n{'='*60}")
    print("分析完成!")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" vascular_topology.py 本地测试模式")
    print("="*60)

    TEST_TUMOR_MASK = "/media/luzhenyang/project/ChangHai_PDA/data/segmentations/C3L-03348/pancreas.nii.gz"
    TEST_VESSEL_MASK = "/media/luzhenyang/project/ChangHai_PDA/data/segmentations/C3L-03348/aorta.nii.gz"

    import os

    if not os.path.exists(TEST_TUMOR_MASK):
        print(f"[ERROR] 肿瘤掩膜路径不存在: {TEST_TUMOR_MASK}")
        print("[INFO] 请修改脚本中的 TEST_TUMOR_MASK 为实际路径")
        raise SystemExit(1)

    if not os.path.exists(TEST_VESSEL_MASK):
        print(f"[ERROR] 血管掩膜路径不存在: {TEST_VESSEL_MASK}")
        print("[INFO] 请修改脚本中的 TEST_VESSEL_MASK 为实际路径")
        raise SystemExit(1)

    try:
        result = analyze_vascular_encasement(
            tumor_mask_path=TEST_TUMOR_MASK,
            vessel_mask_path=TEST_VESSEL_MASK,
            dilation_radius_mm=2.0,
        )

        print("\n" + "="*60)
        print("最终输出 (JSON格式):")
        print("="*60)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\n[ERROR] 测试过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
