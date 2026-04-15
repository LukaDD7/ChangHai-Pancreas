#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nnU-Net v1 胰腺肿瘤分割 - 针对 CL-03356
"""

import os
import sys
import numpy as np
import nibabel as nib
import subprocess

# 设置路径
BASE_DIR = "/media/luzhenyang/project/ChangHai_PDA/data"
INPUT_NIFTI = f"{BASE_DIR}/processed/nifti/CL-03356/CL-03356_CT.nii.gz"
OUTPUT_DIR = f"{BASE_DIR}/processed/segmentations/nnunet_tumor_output_CL-03356"

# nnU-Net 环境变量
os.environ['nnUNet_raw_data_base'] = f"{BASE_DIR}/models/nnunet/nnunet_v1_workspace/raw"
os.environ['nnUNet_preprocessed'] = f"{BASE_DIR}/models/nnunet/nnunet_v1_workspace/preprocessed"
os.environ['RESULTS_FOLDER'] = f"{BASE_DIR}/models/nnunet/nnunet_v1_workspace/results"
os.environ['TMPDIR'] = "/media/luzhenyang/project/ChangHai_PDA/tmp"

print("="*70)
print("nnU-Net v1 胰腺肿瘤分割")
print("="*70)
print(f"患者: CL-03356")
print(f"输入: {INPUT_NIFTI}")
print(f"输出: {OUTPUT_DIR}")

# 检查输入文件
if not os.path.exists(INPUT_NIFTI):
    print(f"❌ 输入文件不存在: {INPUT_NIFTI}")
    sys.exit(1)

# 创建 nnU-Net 输入目录结构
input_dir = f"{os.environ['nnUNet_raw_data_base']}/nnUNet_raw_data/Task007_Pancreas/imagesTs"
os.makedirs(input_dir, exist_ok=True)

# 复制并重命名为 nnU-Net 格式 (需要 _0000.nii.gz 后缀)
nnunet_input = f"{input_dir}/CL-03356_0000.nii.gz"
print(f"\n准备输入文件: {nnunet_input}")

import shutil
shutil.copy(INPUT_NIFTI, nnunet_input)
print("✅ 输入文件准备完成")

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n运行 nnU-Net 推理...")
print("(这可能需要 5-10 分钟)")

# 运行 nnU-Net
result = subprocess.run([
    "nnUNet_predict",
    "-i", f"{os.environ['nnUNet_raw_data_base']}/nnUNet_raw_data/Task007_Pancreas/imagesTs",
    "-o", OUTPUT_DIR,
    "-t", "Task007_Pancreas",
    "-m", "3d_fullres",
    "--num_threads_preprocessing", "4",
    "--num_threads_nifti_save", "4"
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print(f"❌ nnU-Net 运行失败")
    print(result.stderr)
    sys.exit(1)

print("\n✅ nnU-Net 推理完成!")

# 提取肿瘤标签
raw_output = f"{OUTPUT_DIR}/CL-03356.nii.gz"
if os.path.exists(raw_output):
    print("\n提取肿瘤 Mask...")

    img = nib.load(raw_output)
    data = img.get_fdata()

    print(f"  原始输出标签: {np.unique(data)}")

    # 提取 Label=2 (肿瘤)
    tumor_mask = (data == 2).astype(np.uint8)
    tumor_voxels = np.sum(tumor_mask)

    # 保存二值化肿瘤 Mask
    tumor_img = nib.Nifti1Image(tumor_mask, img.affine, img.header)
    tumor_path = f"{OUTPUT_DIR}/true_tumor_mask.nii.gz"
    nib.save(tumor_img, tumor_path)

    print(f"  ✅ 肿瘤体素数: {tumor_voxels}")
    print(f"  ✅ 肿瘤 Mask 已保存: {tumor_path}")

    # 计算体积
    spacing = img.header.get_zooms()
    voxel_volume = np.prod(spacing) / 1000  # 转换为 ml
    tumor_volume = tumor_voxels * voxel_volume
    print(f"  ✅ 估计肿瘤体积: {tumor_volume:.2f} ml")

print("\n" + "="*70)
print("分割完成!")
print("="*70)
print(f"输出目录: {OUTPUT_DIR}")
