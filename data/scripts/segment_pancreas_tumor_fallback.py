#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
备选方案：使用 TotalSegmentator 的胰腺分割 + 形态学方法模拟肿瘤

当 nnU-Net MSD Task07 模型不可用时，这个脚本作为备选方案：
1. 使用 TotalSegmentator 的胰腺分割
2. 在胰腺内部使用阈值/形态学方法识别可疑区域
3. 输出二值化的肿瘤 Mask

注意：这不是真正的肿瘤分割，仅用于演示工作流。
真实肿瘤分割需要：
- 医生手动标注
- 或训练专门的肿瘤分割模型（如 nnU-Net）
"""

import os
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from typing import Tuple


def simulate_tumor_from_pancreas(
    ct_path: str,
    pancreas_mask_path: str,
    output_path: str,
    hu_threshold_low: int = 30,
    hu_threshold_high: int = 70
) -> str:
    """
    基于胰腺 Mask 和 CT 密度值模拟肿瘤分割

    这是一个简化的启发式方法：
    - 在胰腺区域内
    - 寻找 HU 值在 30-70 之间的区域（胰腺实质密度）
    - 寻找 HU 值异常（<0 或 >100）的区域（可能为肿瘤/囊变/钙化）

    参数:
        ct_path: CT 影像路径
        pancreas_mask_path: 胰腺掩膜路径
        output_path: 输出肿瘤掩膜路径
        hu_threshold_low: HU 值下限
        hu_threshold_high: HU 值上限

    返回:
        输出文件路径
    """
    print(f"\n{'='*60}")
    print("备选方案：基于启发式的肿瘤模拟分割")
    print(f"{'='*60}")
    print("[WARNING] 这不是真正的肿瘤分割！")
    print("真实肿瘤分割需要：")
    print("  1. 医生手动标注")
    print("  2. 或 MSD Task07 等预训练模型")
    print("  3. 或自行训练的 nnU-Net 模型")
    print()

    # 加载数据
    ct_img = nib.load(ct_path)
    ct_data = ct_img.get_fdata()
    pancreas_img = nib.load(pancreas_mask_path)
    pancreas_mask = pancreas_img.get_fdata() > 0

    print(f"[INFO] CT 维度: {ct_data.shape}")
    print(f"[INFO] 胰腺体素数: {np.sum(pancreas_mask)}")

    # 在胰腺区域内寻找异常密度区域
    # 正常胰腺 HU: 40-60
    # 肿瘤可能表现为：低密度（<0，囊变）或高密度（>80，实性肿瘤）

    # 方法：寻找胰腺内的密度异常区域
    tumor_candidates = np.zeros_like(ct_data, dtype=bool)

    # 条件1：胰腺区域内 HU < 20 或 HU > 80 的区域（异常密度）
    low_density = (ct_data < 20) & pancreas_mask
    high_density = (ct_data > 80) & pancreas_mask

    # 合并候选区域
    tumor_candidates = low_density | high_density

    # 形态学清理
    # 去除小噪点
    tumor_mask = binary_erosion(tumor_candidates, iterations=1)
    tumor_mask = binary_dilation(tumor_mask, iterations=1)

    # 如果没有找到候选区域，创建一个占位提示
    if np.sum(tumor_mask) == 0:
        print("[WARNING] 未找到可疑肿瘤区域")
        print("[INFO] 创建占位 Mask（全 0）")
        print("[INFO] 请使用真实肿瘤标注或训练模型")

    # 保存结果
    tumor_img = nib.Nifti1Image(tumor_mask.astype(np.uint8), ct_img.affine, ct_img.header)
    nib.save(tumor_img, output_path)

    print(f"\n[RESULT] 模拟肿瘤 Mask 已保存: {output_path}")
    print(f"[INFO] 肿瘤体素数: {np.sum(tumor_mask)}")
    print(f"[INFO] 占胰腺比例: {np.sum(tumor_mask)/np.sum(pancreas_mask)*100:.2f}%")

    return output_path


def main():
    """主函数"""
    print("="*60)
    print("胰腺肿瘤分割 - 备选方案 (启发式方法)")
    print("="*60)

    # 输入路径
    CT_PATH = "/media/luzhenyang/project/ChangHai_PDA/data/nifti_output/C3L-03348_CT.nii.gz"
    PANCREAS_MASK = "/media/luzhenyang/project/ChangHai_PDA/data/segmentations/C3L-03348/pancreas.nii.gz"
    OUTPUT_PATH = "/media/luzhenyang/project/ChangHai_PDA/data/nnunet_tumor_output/simulated_tumor_mask.nii.gz"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 检查输入
    if not os.path.exists(CT_PATH):
        print(f"[ERROR] CT 文件不存在: {CT_PATH}")
        sys.exit(1)

    if not os.path.exists(PANCREAS_MASK):
        print(f"[ERROR] 胰腺掩膜不存在: {PANCREAS_MASK}")
        sys.exit(1)

    # 运行模拟分割
    simulate_tumor_from_pancreas(
        ct_path=CT_PATH,
        pancreas_mask_path=PANCREAS_MASK,
        output_path=OUTPUT_PATH
    )

    print("\n" + "="*60)
    print("说明:")
    print("="*60)
    print("此脚本使用启发式方法模拟肿瘤分割，仅供演示工作流使用。")
    print("真实医疗应用需要：")
    print("  1. 放射科医生手动勾画肿瘤边界")
    print("  2. 或使用 MSD Task07 等医学分割竞赛的预训练模型")
    print("  3. 或自行收集数据训练 nnU-Net v2 模型")


if __name__ == "__main__":
    main()
