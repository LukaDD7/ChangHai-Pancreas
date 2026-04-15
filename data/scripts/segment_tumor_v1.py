#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nnU-Net v1 胰腺肿瘤分割 - 完整工作流
"""

import os
import sys
import subprocess
import zipfile
import requests
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm


# =============================================================================
# 配置路径 (在数据盘上，避免 /tmp 空间不足)
# =============================================================================
BASE_DIR = "/media/luzhenyang/project/ChangHai_PDA/data/nnunet_v1_workspace"
NNUNET_RAW = f"{BASE_DIR}/raw"
NNUNET_PREPROCESSED = f"{BASE_DIR}/preprocessed"
RESULTS_FOLDER = f"{BASE_DIR}/results"

def setup_environment():
    """设置 nnU-Net v1 环境变量"""
    os.makedirs(NNUNET_RAW, exist_ok=True)
    os.makedirs(NNUNET_PREPROCESSED, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    os.environ["nnUNet_raw_data_base"] = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
    os.environ["RESULTS_FOLDER"] = RESULTS_FOLDER

    print(f"[INFO] nnU-Net v1 环境已设置:")
    print(f"  nnUNet_raw_data_base: {NNUNET_RAW}")
    print(f"  nnUNet_preprocessed: {NNUNET_PREPROCESSED}")
    print(f"  RESULTS_FOLDER: {RESULTS_FOLDER}")


def download_model():
    """下载 Task007 Pancreas 模型权重"""
    print("\n" + "="*60)
    print("下载 MSD Task007 Pancreas 模型")
    print("="*60)

    zip_path = f"{BASE_DIR}/Task007_Pancreas.zip"

    # 检查是否已下载
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000000000:  # > 1GB
        print(f"[INFO] 模型文件已存在: {zip_path}")
    else:
        # 从 Zenodo 下载
        url = "https://zenodo.org/record/4003545/files/Task007_Pancreas.zip"
        print(f"[INFO] 从 {url} 下载...")
        print("[INFO] 文件大小约 2.7GB，可能需要 10-20 分钟...")

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"[SUCCESS] 下载完成: {zip_path}")

    # 解压
    extract_dir = f"{RESULTS_FOLDER}/nnUNet/3d_fullres/Task007_Pancreas"
    if os.path.exists(extract_dir):
        print(f"[INFO] 模型已解压: {extract_dir}")
        return extract_dir

    print("[INFO] 解压模型文件...")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"[SUCCESS] 模型已解压到: {extract_dir}")
    return extract_dir


def prepare_input(ct_path: str) -> str:
    """准备输入数据"""
    print("\n" + "="*60)
    print("准备输入数据")
    print("="*60)

    # 创建 Task007 测试目录结构
    task_dir = f"{NNUNET_RAW}/nnUNet_raw_data/Task007_Pancreas"
    images_ts_dir = f"{task_dir}/imagesTs"
    os.makedirs(images_ts_dir, exist_ok=True)

    # 复制并重命名 CT 文件 (添加 _0000 模态后缀)
    target_path = f"{images_ts_dir}/C3L_03348_0000.nii.gz"

    import shutil
    shutil.copy2(ct_path, target_path)

    print(f"[INFO] 输入文件已准备: {target_path}")
    return images_ts_dir


def run_inference(input_dir: str, output_dir: str):
    """运行 nnU-Net v1 推理"""
    print("\n" + "="*60)
    print("运行 nnU-Net v1 推理")
    print("="*60)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "nnUNet_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-t", "Task007_Pancreas",
        "-m", "3d_fullres"
    ]

    print(f"[INFO] 执行命令: {' '.join(cmd)}")
    print("[INFO] 推理中，可能需要 5-15 分钟...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] 推理失败:")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError("nnU-Net 推理失败")

    print("[SUCCESS] 推理完成")


def extract_tumor_mask(nnunet_output_path: str, output_path: str):
    """提取肿瘤标签 (Label=2)"""
    print("\n" + "="*60)
    print("提取肿瘤标签")
    print("="*60)

    # 加载 nnU-Net 输出
    img = nib.load(nnunet_output_path)
    data = img.get_fdata()

    print(f"[INFO] 原始 Mask 唯一值: {np.unique(data)}")

    # 统计各标签
    for label in np.unique(data):
        count = np.sum(data == label)
        name = {0: "Background", 1: "Pancreas", 2: "Tumor"}.get(int(label), "Unknown")
        print(f"  Label {int(label)} ({name}): {count} voxels")

    # 提取肿瘤 (Label=2)
    tumor_mask = (data == 2).astype(np.uint8)

    print(f"\n[INFO] 肿瘤体素数: {np.sum(tumor_mask)}")

    # 保存
    tumor_img = nib.Nifti1Image(tumor_mask, img.affine, img.header)
    nib.save(tumor_img, output_path)

    print(f"[SUCCESS] 肿瘤 Mask 已保存: {output_path}")
    return output_path


def main():
    """主函数"""
    print("="*70)
    print("nnU-Net v1 胰腺肿瘤分割")
    print("="*70)

    # 配置
    CT_PATH = "/media/luzhenyang/project/ChangHai_PDA/data/nifti_output/C3L-03348_CT.nii.gz"
    OUTPUT_DIR = "/media/luzhenyang/project/ChangHai_PDA/data/nnunet_tumor_output"

    if not os.path.exists(CT_PATH):
        print(f"[ERROR] CT 文件不存在: {CT_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 步骤 1: 设置环境
    setup_environment()

    # 步骤 2: 下载模型
    model_dir = download_model()

    # 步骤 3: 准备输入
    input_dir = prepare_input(CT_PATH)

    # 步骤 4: 运行推理
    nnunet_output_dir = f"{OUTPUT_DIR}/nnunet_raw_output"
    run_inference(input_dir, nnunet_output_dir)

    # 步骤 5: 提取肿瘤
    nnunet_output_file = f"{nnunet_output_dir}/C3L_03348.nii.gz"
    tumor_output = f"{OUTPUT_DIR}/true_tumor_mask.nii.gz"

    if os.path.exists(nnunet_output_file):
        extract_tumor_mask(nnunet_output_file, tumor_output)

        print("\n" + "="*70)
        print("分割完成!")
        print("="*70)
        print(f"[RESULT] 肿瘤 Mask: {tumor_output}")
    else:
        print(f"[ERROR] 未找到输出文件: {nnunet_output_file}")
        print(f"[INFO] 检查目录内容: {nnunet_output_dir}")
        if os.path.exists(nnunet_output_dir):
            print(f"  文件列表: {os.listdir(nnunet_output_dir)}")


if __name__ == "__main__":
    main()
