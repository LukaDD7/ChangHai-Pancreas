#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
医疗影像物理量化 Skill (Medical Imaging Physical Quantification Skill)
================================================================================

功能描述:
    本脚本用于从3D CT影像和对应的分割掩膜(Mask)中提取精确的物理参数，
    包括器官/病灶的体积（毫升）和CT密度值（Hounsfield Unit）。

技术架构:
    - 输入: NIfTI格式的CT影像(.nii.gz)和分割掩膜(.nii.gz)
    - 输出: JSON格式的量化指标（体积、平均HU、标准差）
    - 核心库: nibabel (医学影像I/O), numpy (数值计算)

设计原则:
    1. 安全性优先: 严格的维度校验和边界条件处理
    2. 物理精确性: 基于体素间距的物理体积计算
    3. 临床可用性: 输出单位符合临床标准（ml, HU）

作者: Claude Code Assistant
日期: 2026-03-20
版本: 1.0.0
================================================================================
"""

import nibabel as nib
import numpy as np
from typing import Dict, Optional, Tuple
import json
from scipy.ndimage import binary_erosion


def load_nifti_image(file_path: str) -> Tuple[np.ndarray, nib.Nifti1Header]:
    """
    加载NIfTI格式影像文件

    参数:
        file_path (str): NIfTI文件的完整路径（支持.nii或.nii.gz）

    返回:
        Tuple[np.ndarray, nib.Nifti1Header]: 包含两个元素的元组
            - data: 3D/4D numpy数组，存储影像体素值
            - header: NIfTI文件头信息，包含空间参数（spacing、origin等）

    异常:
        FileNotFoundError: 文件不存在时抛出
        nibabel.filebasedimages.ImageFileError: 文件格式错误时抛出
    """
    # 使用nibabel加载NIfTI文件
    img = nib.load(file_path)

    # 提取影像数据为numpy数组
    data = img.get_fdata()

    # 提取文件头信息（包含体素间距等关键参数）
    header = img.header

    return data, header


def validate_image_compatibility(
    ct_data: np.ndarray,
    mask_data: np.ndarray,
    ct_header: nib.Nifti1Header,
    mask_header: nib.Nifti1Header
) -> None:
    """
    安全校验: 验证CT影像与分割掩膜的兼容性

    这是整个量化流程的关键安全闸门。医学影像分析中，由于重采样(resampling)、
    配准(registration)或导出错误，CT和Mask的维度可能不一致，这将导致
    错误的布尔索引和完全错误的量化结果。

    参数:
        ct_data (np.ndarray): CT影像的numpy数组
        mask_data (np.ndarray): 分割掩膜的numpy数组
        ct_header (nib.Nifti1Header): CT影像的文件头
        mask_header (nib.Nifti1Header): Mask影像的文件头

    异常:
        ValueError: 当维度不匹配时抛出，包含详细的错误信息
    """
    # -------------------------------------------------------------------------
    # 校验1: 空间维度一致性检查 (Spatial Dimension Compatibility)
    # -------------------------------------------------------------------------
    if ct_data.shape != mask_data.shape:
        error_msg = (
            f"【严重错误】CT影像与分割掩膜的空间维度不匹配!\n"
            f"  - CT 维度: {ct_data.shape}\n"
            f"  - Mask 维度: {mask_data.shape}\n"
            f"  - 可能原因: 重采样参数不一致、配准失败或数据导出错误\n"
            f"  - 建议: 请使用相同的空间参数重新生成Mask，或对Mask进行重采样"
        )
        raise ValueError(error_msg)

    # -------------------------------------------------------------------------
    # 校验2: 体素间距一致性检查 (Voxel Spacing Consistency)
    # -------------------------------------------------------------------------
    # 获取体素间距（单位：毫米）
    ct_spacing = ct_header.get_zooms()
    mask_spacing = mask_header.get_zooms()

    # 允许微小的浮点误差（1e-6），但体素间距应该基本一致
    if not np.allclose(ct_spacing, mask_spacing, atol=1e-6):
        error_msg = (
            f"【警告】CT与Mask的体素间距存在差异!\n"
            f"  - CT Spacing: {ct_spacing}\n"
            f"  - Mask Spacing: {mask_spacing}\n"
            f"  - 影响: 体积计算将基于CT影像的spacing，可能存在误差"
        )
        # 对于spacing不匹配，我们选择警告而非报错，
        # 因为在某些配准后处理中这可能是有意的
        print(f"[WARN] {error_msg}")


def calculate_voxel_volume(header: nib.Nifti1Header) -> float:
    """
    计算单个体素的物理体积

    在医学影像中，体素(voxel)是3D空间中的最小单位。不同扫描仪和重建参数
    会产生不同大小的体素。物理体积计算必须基于真实的体素间距，而非像素计数。

    参数:
        header (nib.Nifti1Header): NIfTI文件头，包含pixdim字段存储间距信息

    返回:
        float: 单个体素的物理体积（单位：立方毫米 mm³）

    计算公式:
        V_voxel = dx × dy × dz
        其中 dx, dy, dz 分别是x、y、z三个方向的体素间距（单位：毫米）
    """
    # -------------------------------------------------------------------------
    # 提取体素间距参数 (Voxel Spacing Parameters)
    # -------------------------------------------------------------------------
    # nibabel的get_zooms()方法从header中提取pixdim字段
    # 返回元组 (dx, dy, dz)，单位：毫米 (mm)
    spacing = header.get_zooms()

    # 确保我们至少获取3个维度（某些NIfTI可能有4维时间维度）
    if len(spacing) < 3:
        raise ValueError(
            f"【错误】无法从影像头信息中提取3D体素间距!\n"
            f"  - 获取到的spacing: {spacing}\n"
            f"  - 期望: (dx, dy, dz) 三个数值"
        )

    dx, dy, dz = spacing[0], spacing[1], spacing[2]

    # -------------------------------------------------------------------------
    # 体素体积计算 (Voxel Volume Calculation)
    # -------------------------------------------------------------------------
    voxel_volume_mm3 = dx * dy * dz

    print(f"[INFO] 体素间距参数: dx={dx:.4f}mm, dy={dy:.4f}mm, dz={dz:.4f}mm")
    print(f"[INFO] 单个体素体积: {voxel_volume_mm3:.4f} mm³")

    return voxel_volume_mm3


def extract_masked_hu_values(
    ct_data: np.ndarray,
    mask_data: np.ndarray,
    apply_erosion: bool = False,
    erosion_iterations: int = 1
) -> Optional[np.ndarray]:
    """
    使用布尔索引提取Mask区域内的CT值（HU）

    这是医学影像分析的核心操作。通过将Mask转换为布尔掩膜，
    我们可以精确提取感兴趣区域(ROI)内的所有体素值。

    可选的形态学腐蚀功能：
        当Mask边缘正好卡在器官和周围组织的交界处时（如胰腺与脂肪），
        边缘体素可能存在"部分容积效应"——即一个体素内混合了两种组织，
        导致HU值被拉低或拉高。通过形态学腐蚀（向内收缩1个像素），
        可以剥离边界噪声，获得更纯粹的器官内部HU值。

    参数:
        ct_data (np.ndarray): CT影像数据（单位：Hounsfield Unit）
        mask_data (np.ndarray): 分割掩膜（前景>0，背景=0）
        apply_erosion (bool): 是否应用形态学腐蚀，默认False
        erosion_iterations (int): 腐蚀迭代次数，默认1（向内收缩1个像素）

    返回:
        Optional[np.ndarray]: 一维数组，包含所有ROI内的CT值
            - 如果Mask为空（无前景体素），返回None
            - 如果腐蚀后Mask为空，返回None并打印警告
            - 否则返回提取的HU值数组

    技术细节:
        - 使用 mask_data > 0 生成布尔掩膜
        - 可选：使用scipy.ndimage.binary_erosion进行形态学腐蚀
        - 使用布尔索引 ct_data[mask_bool] 提取对应位置的值
        - 结果展平为一维数组便于统计计算
    """
    # -------------------------------------------------------------------------
    # 生成布尔掩膜 (Boolean Mask Generation)
    # -------------------------------------------------------------------------
    # 假设Mask中任何大于0的值都代表前景（某些分割软件可能使用不同值表示不同类别）
    mask_bool = mask_data > 0

    # 统计前景体素数量
    voxel_count = np.sum(mask_bool)
    print(f"[INFO] Mask中前景体素数量: {voxel_count}")

    # -------------------------------------------------------------------------
    # 边界条件处理: 空Mask检测 (Empty Mask Handling)
    # -------------------------------------------------------------------------
    if voxel_count == 0:
        print("[WARN] 检测到空Mask（无前景体素），可能原因：")
        print("  1. 分割目标在当前切片/体素中不存在")
        print("  2. 阈值设置错误导致所有体素被归类为背景")
        print("  3. 掩膜文件损坏或读取错误")
        return None

    # -------------------------------------------------------------------------
    # 可选：形态学腐蚀 (Optional Morphological Erosion)
    # -------------------------------------------------------------------------
    # 目的：消除边界部分容积效应，获得更纯粹的器官内部HU值
    if apply_erosion:
        print(f"[INFO] 应用形态学腐蚀: iterations={erosion_iterations}")
        mask_bool = binary_erosion(mask_bool, iterations=erosion_iterations)
        eroded_count = np.sum(mask_bool)
        print(f"[INFO] 腐蚀后前景体素数量: {eroded_count} (减少了 {voxel_count - eroded_count} 个)")

        if eroded_count == 0:
            print("[WARN] 腐蚀后Mask为空，请检查腐蚀参数是否过于激进")
            return None

    # -------------------------------------------------------------------------
    # 布尔索引提取HU值 (Boolean Indexing for HU Extraction)
    # -------------------------------------------------------------------------
    # 使用布尔数组作为索引，提取CT影像中对应位置的值
    # 结果是一个一维数组，包含所有ROI内的HU值
    hu_values = ct_data[mask_bool]

    return hu_values


def calculate_3d_metrics(
    ct_path: str,
    mask_path: str,
    apply_erosion: bool = False,
    erosion_iterations: int = 1
) -> Dict:
    """
    核心函数: 计算3D医学影像的物理体积和CT密度指标

    这是本Skill的主入口函数，整合了整个量化流程：
    1. 数据加载 -> 2. 安全校验 -> 3. 体积计算 -> 4. HU统计 -> 5. 结果封装

    参数:
        ct_path (str): CT影像的NIfTI文件路径（.nii或.nii.gz）
        mask_path (str): 分割掩膜的NIfTI文件路径（.nii或.nii.gz）
        apply_erosion (bool): 是否应用形态学腐蚀以消除边界部分容积效应，默认False
        erosion_iterations (int): 腐蚀迭代次数，默认1（向内收缩1个像素）

    返回:
        Dict: JSON格式的量化结果字典，包含以下字段：
            - volume_ml (float): 物理体积，单位毫升(ml)，保留2位小数
            - mean_hu (float): 平均CT值(HU)，保留2位小数
            - std_hu (float): CT值标准差(HU)，保留2位小数
            - voxel_count (int): 前景体素数量
            - voxel_volume_mm3 (float): 单个体素体积(mm³)
            - status (str): 处理状态，"success"或"error"
            - error_message (str, optional): 错误时的详细描述
            - erosion_applied (bool): 是否应用了形态学腐蚀

    示例:
        >>> result = calculate_3d_metrics(
        ...     ct_path="patient_001_ct.nii.gz",
        ...     mask_path="patient_001_liver_mask.nii.gz"
        ... )
        >>> print(result)
        {
            'volume_ml': 1256.78,
            'mean_hu': 56.32,
            'std_hu': 12.45,
            'voxel_count': 1345678,
            'voxel_volume_mm3': 0.9339,
            'status': 'success'
        }
    """
    # -------------------------------------------------------------------------
    # 步骤1: 数据加载 (Data Loading)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("开始执行 3D 医学影像物理量化分析")
    print(f"{'='*60}")
    print(f"[INFO] CT影像路径: {ct_path}")
    print(f"[INFO] Mask路径: {mask_path}")

    try:
        # 加载CT影像
        ct_data, ct_header = load_nifti_image(ct_path)
        print(f"[INFO] CT影像加载成功，维度: {ct_data.shape}")

        # 加载分割掩膜
        mask_data, mask_header = load_nifti_image(mask_path)
        print(f"[INFO] Mask加载成功，维度: {mask_data.shape}")

    except FileNotFoundError as e:
        return {
            "volume_ml": 0.0,
            "mean_hu": 0.0,
            "std_hu": 0.0,
            "status": "error",
            "error_message": f"文件未找到: {str(e)}"
        }
    except Exception as e:
        return {
            "volume_ml": 0.0,
            "mean_hu": 0.0,
            "std_hu": 0.0,
            "status": "error",
            "error_message": f"文件加载失败: {str(e)}"
        }

    # -------------------------------------------------------------------------
    # 步骤2: 安全校验 (Safety Validation)
    # -------------------------------------------------------------------------
    print(f"\n{'-'*60}")
    print("步骤2: 执行安全校验...")
    print(f"{'-'*60}")

    try:
        validate_image_compatibility(ct_data, mask_data, ct_header, mask_header)
        print("[PASS] 所有安全校验通过")
    except ValueError as e:
        return {
            "volume_ml": 0.0,
            "mean_hu": 0.0,
            "std_hu": 0.0,
            "status": "error",
            "error_message": str(e)
        }

    # -------------------------------------------------------------------------
    # 步骤3: 体积计算 (Volume Calculation)
    # -------------------------------------------------------------------------
    print(f"\n{'-'*60}")
    print("步骤3: 计算物理体积...")
    print(f"{'-'*60}")

    # 计算单个体素体积
    voxel_volume_mm3 = calculate_voxel_volume(ct_header)

    # 统计前景体素数量 (N)
    mask_bool = mask_data > 0
    voxel_count = int(np.sum(mask_bool))
    print(f"[INFO] 前景体素总数 N = {voxel_count}")

    # -------------------------------------------------------------------------
    # 物理体积计算公式:
    # Volume (ml) = (N × V_voxel) / 1000
    # 其中:
    #   - N: 前景体素数量
    #   - V_voxel: 单个体素体积 (mm³)
    #   - 除以1000: 将 mm³ 转换为 ml (1 ml = 1000 mm³)
    # -------------------------------------------------------------------------
    total_volume_mm3 = voxel_count * voxel_volume_mm3
    volume_ml = total_volume_mm3 / 1000.0

    print(f"[INFO] 总体积: {total_volume_mm3:.2f} mm³ = {volume_ml:.2f} ml")

    # -------------------------------------------------------------------------
    # 步骤4: CT值量化 (HU Value Quantification)
    # -------------------------------------------------------------------------
    print(f"\n{'-'*60}")
    print("步骤4: 计算CT密度值(HU)...")
    print(f"{'-'*60}")

    hu_values = extract_masked_hu_values(
        ct_data,
        mask_data,
        apply_erosion=apply_erosion,
        erosion_iterations=erosion_iterations
    )

    if hu_values is None:
        # 空Mask情况
        return {
            "volume_ml": 0.0,
            "mean_hu": 0.0,
            "std_hu": 0.0,
            "voxel_count": 0,
            "voxel_volume_mm3": voxel_volume_mm3,
            "erosion_applied": apply_erosion,
            "status": "success",
            "warning": "Mask为空，未检测到前景体素"
        }

    # -------------------------------------------------------------------------
    # 统计学计算: 均值和标准差
    # -------------------------------------------------------------------------
    mean_hu = float(np.mean(hu_values))
    std_hu = float(np.std(hu_values))

    # 额外的统计信息（可选）
    min_hu = float(np.min(hu_values))
    max_hu = float(np.max(hu_values))
    median_hu = float(np.median(hu_values))

    print(f"[INFO] HU统计指标:")
    print(f"  - 均值 (Mean): {mean_hu:.2f} HU")
    print(f"  - 标准差 (Std): {std_hu:.2f} HU")
    print(f"  - 中位数 (Median): {median_hu:.2f} HU")
    print(f"  - 最小值 (Min): {min_hu:.2f} HU")
    print(f"  - 最大值 (Max): {max_hu:.2f} HU")

    # -------------------------------------------------------------------------
    # 步骤5: 结果封装 (Result Packaging)
    # -------------------------------------------------------------------------
    # 注意: 使用float()和int()将numpy类型转换为Python原生类型，确保JSON可序列化
    result = {
        "volume_ml": float(round(volume_ml, 2)),
        "mean_hu": float(round(mean_hu, 2)),
        "std_hu": float(round(std_hu, 2)),
        "voxel_count": int(voxel_count),
        "voxel_volume_mm3": float(round(voxel_volume_mm3, 4)),
        "erosion_applied": apply_erosion,
        "status": "success"
    }

    # 可选：添加详细统计信息
    result["detail"] = {
        "median_hu": float(round(median_hu, 2)),
        "min_hu": float(round(min_hu, 2)),
        "max_hu": float(round(max_hu, 2)),
        "total_volume_mm3": float(round(total_volume_mm3, 2))
    }

    print(f"\n{'='*60}")
    print("量化分析完成!")
    print(f"{'='*60}")
    print(f"[RESULT] 体积: {result['volume_ml']} ml")
    print(f"[RESULT] 平均HU: {result['mean_hu']} ± {result['std_hu']}")

    return result


# =============================================================================
# 测试入口 (Test Entrypoint)
# =============================================================================

if __name__ == "__main__":
    """
    本地调试和测试用例

    使用方法:
        1. 激活conda环境: conda activate totalseg
        2. 运行脚本: python quantify_metrics.py
        3. 修改下方的路径为您的实际数据路径
    """

    print("\n" + "="*60)
    print(" quantify_metrics.py 本地测试模式")
    print("="*60)

    # -------------------------------------------------------------------------
    # 测试配置: 请根据实际情况修改以下路径
    # -------------------------------------------------------------------------

    # 测试用例1: 使用CPTAC-PDA患者C3L-03348的数据（肝脏分割）
    TEST_CT_PATH = "/media/luzhenyang/project/ChangHai_PDA/data/nifti_output/C3L-03348_CT.nii.gz"
    TEST_MASK_PATH = "/media/luzhenyang/project/ChangHai_PDA/data/segmentations/C3L-03348/liver.nii.gz"

    # 测试用例2: 脾脏分割（取消注释以测试）
    # TEST_MASK_PATH = "/media/luzhenyang/project/ChangHai_PDA/data/segmentations/C3L-03348/spleen.nii.gz"

    # 测试用例3: 左肾分割（取消注释以测试）
    # TEST_MASK_PATH = "/media/luzhenyang/project/ChangHai_PDA/data/segmentations/C3L-03348/kidney_left.nii.gz"

    # -------------------------------------------------------------------------
    # 执行测试
    # -------------------------------------------------------------------------

    # 检查路径是否存在
    import os
    if not os.path.exists(TEST_CT_PATH):
        print(f"[ERROR] CT影像路径不存在: {TEST_CT_PATH}")
        print("[INFO] 请修改脚本中的 TEST_CT_PATH 为实际路径")
        exit(1)

    if not os.path.exists(TEST_MASK_PATH):
        print(f"[ERROR] Mask路径不存在: {TEST_MASK_PATH}")
        print("[INFO] 请修改脚本中的 TEST_MASK_PATH 为实际路径")
        exit(1)

    # 执行量化分析
    try:
        # 默认模式：不应用腐蚀
        result = calculate_3d_metrics(TEST_CT_PATH, TEST_MASK_PATH)

        # 可选：启用形态学腐蚀（消除边界部分容积效应）
        # result = calculate_3d_metrics(
        #     TEST_CT_PATH,
        #     TEST_MASK_PATH,
        #     apply_erosion=True,
        #     erosion_iterations=1
        # )

        # 打印格式化后的JSON结果
        print("\n" + "="*60)
        print("最终输出 (JSON格式):")
        print("="*60)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\n[ERROR] 测试过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
