#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DICOM 转换为 NIfTI 脚本
"""

import pydicom
import nibabel as nib
import numpy as np
import os
import sys
from glob import glob

def dicom_to_nifti(dicom_dir, output_path):
    """将 DICOM 目录转换为 NIfTI 文件"""
    print(f"输入: {dicom_dir}")
    print(f"输出: {output_path}")

    # 读取所有 DICOM 文件
    dcm_files = sorted(glob(os.path.join(dicom_dir, "*.dcm")))
    print(f"找到 {len(dcm_files)} 个 DICOM 文件")

    if len(dcm_files) == 0:
        raise ValueError("未找到 DICOM 文件")

    # 读取第一个文件获取元数据
    first = pydicom.dcmread(dcm_files[0])
    print(f"患者ID: {first.PatientID}")
    print(f"模态: {first.Modality}")

    # 读取所有切片
    print("读取所有切片...")
    slices = []
    for f in dcm_files:
        dcm = pydicom.dcmread(f)
        slices.append(dcm)

    # 按 Z 轴位置排序
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # 构建 3D 数组
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape, dtype=np.float32)

    for i, s in enumerate(slices):
        img2d = s.pixel_array.astype(np.float32)
        if hasattr(s, 'RescaleSlope') and hasattr(s, 'RescaleIntercept'):
            img2d = img2d * s.RescaleSlope + s.RescaleIntercept
        img3d[:, :, i] = img2d

    # 构建仿射矩阵
    pixel_spacing = first.PixelSpacing
    slice_thickness = first.SliceThickness
    affine = np.eye(4)
    affine[0, 0] = pixel_spacing[0]
    affine[1, 1] = pixel_spacing[1]
    affine[2, 2] = slice_thickness
    affine[:3, 3] = first.ImagePositionPatient

    # 创建 NIfTI
    nifti_img = nib.Nifti1Image(img3d.astype(np.int16), affine)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nifti_img, output_path)

    print(f"✅ 转换完成!")
    print(f"输出维度: {img3d.shape}")
    print(f"数值范围: [{img3d.min():.1f}, {img3d.max():.1f}] HU")
    print(f"间距: {pixel_spacing} x {slice_thickness} mm")

    return output_path

if __name__ == "__main__":
    dicom_dir = "/media/luzhenyang/project/ChangHai_PDA/data/raw/dicom/dicom_data/CPTAC-PDA/CL-03356/6.000000-Axial-31651"
    output_path = "/media/luzhenyang/project/ChangHai_PDA/data/processed/nifti/CL-03356/CL-03356_CT.nii.gz"

    dicom_to_nifti(dicom_dir, output_path)
