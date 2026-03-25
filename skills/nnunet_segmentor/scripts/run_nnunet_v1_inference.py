#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nnU-Net v1 胰腺肿瘤分割推理
"""

import os
import sys
import subprocess
import shutil
import numpy as np
import nibabel as nib

# 设置环境变量
os.environ["nnUNet_raw_data_base"] = "/media/luzhenyang/project/ChangHai_PDA/data/nnunet_v1_workspace/raw"
os.environ["nnUNet_preprocessed"] = "/media/luzhenyang/project/ChangHai_PDA/data/nnunet_v1_workspace/preprocessed"
os.environ["RESULTS_FOLDER"] = "/media/luzhenyang/project/ChangHai_PDA/data/nnunet_v1_workspace/results"

print("="*60)
print("nnU-Net v1 胰腺肿瘤分割")
print("="*60)

# 配置
CT_PATH = "/media/luzhenyang/project/ChangHai_PDA/data/nifti_output/C3L-03348_CT.nii.gz"
OUTPUT_DIR = "/media/luzhenyang/project/ChangHai_PDA/data/nnunet_tumor_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 步骤1: 准备输入数据
print("\n[步骤1] 准备输入数据...")
task_dir = "/media/luzhenyang/project/ChangHai_PDA/data/nnunet_v1_workspace/raw/nnUNet_raw_data/Task007_Pancreas"
images_ts_dir = f"{task_dir}/imagesTs"
os.makedirs(images_ts_dir, exist_ok=True)

# 复制并重命名CT文件
target_path = f"{images_ts_dir}/C3L_03348_0000.nii.gz"
shutil.copy2(CT_PATH, target_path)
print(f"  输入文件: {target_path}")

# 步骤2: 运行推理
print("\n[步骤2] 运行 nnU-Net v1 推理...")
output_dir = f"{OUTPUT_DIR}/nnunet_raw_output"
os.makedirs(output_dir, exist_ok=True)

cmd = [
    "nnUNet_predict",
    "-i", images_ts_dir,
    "-o", output_dir,
    "-t", "Task007_Pancreas",
    "-m", "3d_fullres"
]

print(f"  命令: {' '.join(cmd)}")
print("  推理中，约需5-15分钟...")

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"[ERROR] 推理失败:")
    print(f"stderr: {result.stderr}")
    sys.exit(1)

print("  [SUCCESS] 推理完成")

# 步骤3: 提取肿瘤标签
print("\n[步骤3] 提取肿瘤标签 (Label=2)...")
nnunet_output = f"{output_dir}/C3L_03348.nii.gz"

if not os.path.exists(nnunet_output):
    print(f"[ERROR] 输出文件不存在: {nnunet_output}")
    print(f"  检查目录内容: {os.listdir(output_dir)}")
    sys.exit(1)

# 加载并提取肿瘤
img = nib.load(nnunet_output)
data = img.get_fdata()

print(f"  原始Mask唯一值: {np.unique(data)}")
for label in np.unique(data):
    count = np.sum(data == label)
    name = {0: "背景", 1: "胰腺", 2: "肿瘤"}.get(int(label), "未知")
    print(f"    Label {int(label)} ({name}): {count} 体素")

# 提取肿瘤 (Label=2)
tumor_mask = (data == 2).astype(np.uint8)
print(f"\n  肿瘤体素数: {np.sum(tumor_mask)}")

# 保存
tumor_output = f"{OUTPUT_DIR}/true_tumor_mask.nii.gz"
tumor_img = nib.Nifti1Image(tumor_mask, img.affine, img.header)
nib.save(tumor_img, tumor_output)

print(f"\n[SUCCESS] 肿瘤Mask已保存: {tumor_output}")

# 同时保存胰腺+肿瘤
full_mask = data.astype(np.uint8)
full_output = f"{OUTPUT_DIR}/pancreas_and_tumor_mask.nii.gz"
full_img = nib.Nifti1Image(full_mask, img.affine, img.header)
nib.save(full_img, full_output)

print(f"[SUCCESS] 胰腺+肿瘤Mask已保存: {full_output}")

print("\n" + "="*60)
print("分割完成!")
print("="*60)
print(f"肿瘤Mask: {tumor_output}")
print(f"完整Mask: {full_output}")
