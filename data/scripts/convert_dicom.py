#!/usr/bin/env python
"""
将DICOM转换为NIfTI格式
使用SimpleITK
"""

import SimpleITK as sitk
import os
from pathlib import Path

def dicom_to_nifti(dicom_dir, output_path):
    """
    将DICOM文件夹转换为NIfTI文件
    """
    print(f"读取DICOM目录: {dicom_dir}")

    # 读取DICOM序列
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))

    if len(dicom_names) == 0:
        raise ValueError(f"在 {dicom_dir} 中没有找到DICOM文件")

    print(f"找到 {len(dicom_names)} 个DICOM文件")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    print(f"图像尺寸: {image.GetSize()}")
    print(f"图像间距: {image.GetSpacing()}")

    # 保存为NIfTI
    print(f"保存到: {output_path}")
    sitk.WriteImage(image, str(output_path))

    print("转换完成!")
    return output_path

if __name__ == "__main__":
    # 患者C3L-03348的CT数据
    dicom_dir = "/media/luzhenyang/project/ChangHai_PDA/data/dicom_data/CPTAC-PDA/C3L-03348/05-16-2004-NA-CT ABDOMEN - AUGMENTAB-19086/7.000000-Axial VenousPhase-71265"
    output_dir = "/media/luzhenyang/project/ChangHai_PDA/data/nifti_output"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "C3L-03348_CT.nii.gz")

    dicom_to_nifti(dicom_dir, output_path)
