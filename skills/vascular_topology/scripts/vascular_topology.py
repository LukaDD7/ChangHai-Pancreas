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
    - 逐层切片周长比例法计算包绕角度
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
版本: 1.0.0
================================================================================
"""

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from typing import Dict, List, Tuple, Optional
import json


def load_mask_and_spacing(mask_path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    加载NIfTI掩膜并提取体素间距

    参数:
        mask_path (str): 掩膜文件路径

    返回:
        Tuple: (mask_data, spacing)
            - mask_data: 3D numpy数组
            - spacing: (dx, dy, dz) 体素间距（毫米）
    """
    img = nib.load(mask_path)
    mask_data = img.get_fdata()
    header = img.header
    spacing = header.get_zooms()[:3]  # 获取前3个维度

    return mask_data, spacing


def calculate_dilation_radius_voxels(physical_radius_mm: float, spacing: Tuple[float, float, float]) -> int:
    """
    将物理膨胀半径（毫米）转换为体素个数

    由于各向异性，我们取平均 spacing 进行估算

    参数:
        physical_radius_mm: 物理半径（毫米）
        spacing: 体素间距 (dx, dy, dz)

    返回:
        int: 膨胀半径对应的体素个数
    """
    avg_spacing = np.mean(spacing)
    radius_voxels = int(np.ceil(physical_radius_mm / avg_spacing))
    return max(1, radius_voxels)  # 至少膨胀1个体素


def extract_2d_boundary(mask_2d: np.ndarray) -> np.ndarray:
    """
    提取2D二值掩膜的边界（边缘）体素

    使用形态学腐蚀找出内部，然后用原掩膜减去内部得到边界

    参数:
        mask_2d: 2D二值数组

    返回:
        边界体素的布尔数组
    """
    if np.sum(mask_2d) == 0:
        return np.zeros_like(mask_2d, dtype=bool)

    # 生成2D四连通结构元素
    structure = generate_binary_structure(2, 1)

    # 轻微腐蚀得到内部
    eroded = binary_erosion(mask_2d, structure, iterations=1)

    # 边界 = 原掩膜 - 腐蚀后的内部
    boundary = mask_2d & (~eroded)

    return boundary


def calculate_encasement_angle(
    vessel_boundary: np.ndarray,
    tumor_mask: np.ndarray
) -> Optional[float]:
    """
    使用周长比例法计算血管被肿瘤包绕的角度

    核心思想: 角度 = (接触边界长度 / 血管总周长) * 360°

    参数:
        vessel_boundary: 血管边界体素的布尔数组
        tumor_mask: 肿瘤掩膜（已膨胀）的布尔数组

    返回:
        float: 包绕角度（度），如果无法计算则返回None
    """
    # 统计血管边界体素总数
    total_boundary_voxels = np.sum(vessel_boundary)

    # 安全检查: 防止除以零
    if total_boundary_voxels == 0:
        return None

    # 找出与肿瘤接触的血管边界体素
    contact_boundary = vessel_boundary & tumor_mask
    contact_voxels = np.sum(contact_boundary)

    if contact_voxels == 0:
        return 0.0

    # 周长比例法计算角度
    # 假设边界体素均匀分布，比例直接对应角度比例
    angle = (contact_voxels / total_boundary_voxels) * 360.0

    # 角度上限为360度
    return min(angle, 360.0)


def analyze_vascular_encasement(
    tumor_mask_path: str,
    vessel_mask_path: str,
    dilation_radius_mm: float = 2.0
) -> Dict:
    """
    核心函数: 分析肿瘤对血管的侵犯程度

    计算流程:
    1. 加载掩膜并校验维度一致性
    2. 对肿瘤进行形态学膨胀（检测接触）
    3. 检测膨胀后肿瘤与血管的交集
    4. 逐层计算包绕角度
    5. 临床分类

    参数:
        tumor_mask_path: 肿瘤掩膜路径
        vessel_mask_path: 血管掩膜路径
        dilation_radius_mm: 膨胀半径（毫米），默认2mm

    返回:
        Dict: {
            "is_contact": bool,
            "max_angle_degree": float,
            "clinical_status": str,
            "contact_z_slices": List[int],
            "angle_per_slice": Dict[int, float],
            "vessel_info": Dict
        }
    """
    print(f"\n{'='*60}")
    print("血管侵犯拓扑分析 - Vascular Encasement Analysis")
    print(f"{'='*60}")
    print(f"[INFO] 肿瘤掩膜: {tumor_mask_path}")
    print(f"[INFO] 血管掩膜: {vessel_mask_path}")
    print(f"[INFO] 膨胀半径: {dilation_radius_mm}mm")

    # ======================================================================
    # 步骤1: 数据加载
    # ======================================================================
    print(f"\n{'-'*60}")
    print("步骤1: 加载掩膜数据...")
    print(f"{'-'*60}")

    try:
        tumor_data, tumor_spacing = load_mask_and_spacing(tumor_mask_path)
        vessel_data, vessel_spacing = load_mask_and_spacing(vessel_mask_path)
    except Exception as e:
        return {
            "is_contact": False,
            "max_angle_degree": 0.0,
            "clinical_status": "Error",
            "error_message": f"文件加载失败: {str(e)}"
        }

    print(f"[INFO] 肿瘤维度: {tumor_data.shape}, 间距: {tumor_spacing}")
    print(f"[INFO] 血管维度: {vessel_data.shape}, 间距: {vessel_spacing}")

    # ======================================================================
    # 步骤2: 安全校验
    # ======================================================================
    print(f"\n{'-'*60}")
    print("步骤2: 执行安全校验...")
    print(f"{'-'*60}")

    if tumor_data.shape != vessel_data.shape:
        error_msg = (
            f"【严重错误】肿瘤与血管掩膜维度不匹配!\n"
            f"  - 肿瘤: {tumor_data.shape}\n"
            f"  - 血管: {vessel_data.shape}"
        )
        print(f"[ERROR] {error_msg}")
        return {
            "is_contact": False,
            "max_angle_degree": 0.0,
            "clinical_status": "Error",
            "error_message": error_msg
        }

    # 转换为布尔数组
    tumor_mask = tumor_data > 0
    vessel_mask = vessel_data > 0

    print(f"[PASS] 维度校验通过")

    # ======================================================================
    # 步骤3: 形态学膨胀与接触检测
    # ======================================================================
    print(f"\n{'-'*60}")
    print("步骤3: 形态学膨胀与接触检测...")
    print(f"{'-'*60}")

    # 计算膨胀半径（体素个数）
    dilation_voxels = calculate_dilation_radius_voxels(dilation_radius_mm, tumor_spacing)
    print(f"[INFO] 物理膨胀半径: {dilation_radius_mm}mm ≈ {dilation_voxels}个体素")

    # 生成3D结构元素（球状）
    structure_3d = generate_binary_structure(3, 1)

    # 对肿瘤进行形态学膨胀
    tumor_dilated = binary_dilation(tumor_mask, structure_3d, iterations=dilation_voxels)
    print(f"[INFO] 膨胀前肿瘤体素数: {np.sum(tumor_mask)}")
    print(f"[INFO] 膨胀后肿瘤体素数: {np.sum(tumor_dilated)}")

    # 检测接触: 膨胀后肿瘤与血管的交集
    contact_mask = tumor_dilated & vessel_mask
    has_contact = np.any(contact_mask)

    print(f"[INFO] 接触区域体素数: {np.sum(contact_mask)}")

    if not has_contact:
        print("[RESULT] 未检测到肿瘤-血管接触")
        return {
            "is_contact": False,
            "max_angle_degree": 0.0,
            "clinical_status": "Clear (No Involvement)",
            "contact_z_slices": [],
            "angle_per_slice": {},
            "dilation_radius_mm": dilation_radius_mm
        }

    print("[ALERT] 检测到肿瘤-血管接触，进入角度计算...")

    # ======================================================================
    # 步骤4: 逐层角度计算（核心算法）
    # ======================================================================
    print(f"\n{'-'*60}")
    print("步骤4: 逐层计算包绕角度...")
    print(f"{'-'*60}")

    nz, ny, nx = tumor_mask.shape
    max_angle = 0.0
    contact_z_slices = []
    angle_per_slice = {}

    # 遍历Z轴每一层
    for z in range(nz):
        # 提取当前切片的肿瘤和血管
        tumor_slice = tumor_dilated[z, :, :]
        vessel_slice = vessel_mask[z, :, :]

        # 跳过不包含血管的切片
        if not np.any(vessel_slice):
            continue

        # 提取血管边界
        vessel_boundary = extract_2d_boundary(vessel_slice)
        boundary_voxels = np.sum(vessel_boundary)

        if boundary_voxels == 0:
            continue

        # 计算该层的包绕角度
        angle = calculate_encasement_angle(vessel_boundary, tumor_slice)

        if angle is not None and angle > 0:
            contact_z_slices.append(z)
            angle_per_slice[z] = round(angle, 2)
            max_angle = max(max_angle, angle)

            print(f"  [Slice {z}] 边界体素: {boundary_voxels:4d}, 包绕角度: {angle:.2f}°")

    # ======================================================================
    # 步骤5: 临床分类
    # ======================================================================
    print(f"\n{'-'*60}")
    print("步骤5: 临床可切除性分类...")
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

    # ======================================================================
    # 步骤6: 结果封装
    # ======================================================================
    result = {
        "is_contact": True,
        "max_angle_degree": round(max_angle, 2),
        "clinical_status": clinical_status,
        "contact_z_slices": contact_z_slices,
        "angle_per_slice": angle_per_slice,
        "dilation_radius_mm": dilation_radius_mm,
        "dilation_voxels": dilation_voxels,
        "total_z_slices": len(contact_z_slices),
        "tumor_spacing": [float(s) for s in tumor_spacing],
        "status": "success"
    }

    print(f"\n{'='*60}")
    print("分析完成!")
    print(f"{'='*60}")

    return result


# =============================================================================
# 测试入口 (Test Entrypoint)
# =============================================================================

if __name__ == "__main__":
    """
    本地调试和测试用例

    使用方法:
        1. 激活conda环境: conda activate totalseg
        2. 运行脚本: python vascular_topology.py
        3. 修改下方的路径为您的实际数据路径
    """

    print("\n" + "="*60)
    print(" vascular_topology.py 本地测试模式")
    print("="*60)

    # -------------------------------------------------------------------------
    # 测试配置: 请根据实际情况修改以下路径
    # -------------------------------------------------------------------------

    # 测试用例: 胰腺肿瘤与SMA/SMV的侵犯分析
    # 注意：需要准备真实的肿瘤掩膜和血管掩膜

    TEST_TUMOR_MASK = "/media/luzhenyang/project/ChangHai_PDA/data/segmentations/C3L-03348/pancreas.nii.gz"
    TEST_VESSEL_MASK = "/media/luzhenyang/project/ChangHai_PDA/data/segmentations/C3L-03348/aorta.nii.gz"

    # -------------------------------------------------------------------------
    # 执行测试
    # -------------------------------------------------------------------------

    import os

    if not os.path.exists(TEST_TUMOR_MASK):
        print(f"[ERROR] 肿瘤掩膜路径不存在: {TEST_TUMOR_MASK}")
        print("[INFO] 请修改脚本中的 TEST_TUMOR_MASK 为实际路径")
        exit(1)

    if not os.path.exists(TEST_VESSEL_MASK):
        print(f"[ERROR] 血管掩膜路径不存在: {TEST_VESSEL_MASK}")
        print("[INFO] 请修改脚本中的 TEST_VESSEL_MASK 为实际路径")
        exit(1)

    # 执行分析
    try:
        result = analyze_vascular_encasement(
            tumor_mask_path=TEST_TUMOR_MASK,
            vessel_mask_path=TEST_VESSEL_MASK,
            dilation_radius_mm=2.0  # 2mm膨胀半径
        )

        # 打印格式化后的JSON结果
        print("\n" + "="*60)
        print("最终输出 (JSON格式):")
        print("="*60)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\n[ERROR] 测试过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
