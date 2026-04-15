#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提取胰腺 Master Slice - 用于 LLaVA-Med 视觉分析
"""

import nibabel as nib
import numpy as np
from PIL import Image
import os

def extract_pancreas_master_slice(ct_path, pancreas_mask_path, output_png, window_center=40, window_width=400):
    """
    提取胰腺最大面积的切片

    Args:
        ct_path: CT NIfTI 文件路径
        pancreas_mask_path: 胰腺 Mask 路径
        output_png: 输出 PNG 路径
        window_center: 窗位 (默认 40 HU)
        window_width: 窗宽 (默认 400 HU)
    """
    print("="*70)
    print("提取胰腺 Master Slice")
    print("="*70)
    print(f"CT: {ct_path}")
    print(f"Mask: {pancreas_mask_path}")
    print(f"输出: {output_png}")

    # 1. 加载数据
    print("\n加载数据...")
    ct_img = nib.load(ct_path)
    ct = ct_img.get_fdata()
    mask_img = nib.load(pancreas_mask_path)
    mask = mask_img.get_fdata()

    print(f"CT 维度: {ct.shape}")
    print(f"Mask 维度: {mask.shape}")

    # 2. 统计 Z 轴各层胰腺面积
    print("\n计算各层胰腺面积...")
    z_areas = np.sum(mask > 0, axis=(0, 1))
    max_z_idx = np.argmax(z_areas)
    max_area = z_areas[max_z_idx]

    print(f"胰腺层数: {np.sum(z_areas > 0)}")
    print(f"最大面积层: Z={max_z_idx}, 面积={max_area} 体素")

    # 显示面积分布
    non_zero_z = np.where(z_areas > 0)[0]
    if len(non_zero_z) > 0:
        print(f"胰腺覆盖 Z 范围: [{non_zero_z[0]}, {non_zero_z[-1]}]")

    # 3. 提取切片并应用腹部窗 (W:400, L:40)
    print(f"\n提取切片 Z={max_z_idx}...")
    slice_2d = ct[:, :, max_z_idx].copy()

    # 应用窗宽窗位
    lower = window_center - window_width // 2
    upper = window_center + window_width // 2
    print(f"窗宽窗位: [{lower}, {upper}] HU")

    slice_2d = np.clip(slice_2d, lower, upper)
    slice_2d = ((slice_2d - lower) / (upper - lower) * 255).astype(np.uint8)

    # 4. 旋转纠正 (NIfTI 到标准图像坐标)
    img = Image.fromarray(np.rot90(slice_2d, k=1))

    # 保存
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    img.save(output_png, quality=95)

    print(f"\n✅ Master slice 已保存: {output_png}")
    print(f"   尺寸: {img.size}")
    print(f"   Z 位置: {max_z_idx}")

    return max_z_idx, img.size

if __name__ == "__main__":
    # 处理 CL-03356
    base_dir = "/media/luzhenyang/project/ChangHai_PDA/data"

    ct_path = f"{base_dir}/processed/nifti/CL-03356/CL-03356_CT.nii.gz"
    pancreas_mask_path = f"{base_dir}/processed/segmentations/CL-03356/pancreas.nii.gz"
    output_png = f"{base_dir}/results/images/CL-03356_master_slice.png"

    if not os.path.exists(ct_path):
        print(f"❌ CT 文件不存在: {ct_path}")
        exit(1)

    if not os.path.exists(pancreas_mask_path):
        print(f"❌ 胰腺 Mask 不存在: {pancreas_mask_path}")
        exit(1)

    extract_pancreas_master_slice(ct_path, pancreas_mask_path, output_png)
