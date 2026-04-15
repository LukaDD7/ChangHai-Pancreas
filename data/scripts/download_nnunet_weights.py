#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 nnU-Net v2 预训练权重 (MSD Task07 Pancreas)

MSD Task07 包含胰腺和胰腺肿瘤的分割，标签定义：
- 0: 背景
- 1: 胰腺实质 (Pancreas)
- 2: 肿瘤 (Tumor)

模型来源: Hugging Face / GitHub 上的 nnU-Net 预训练模型仓库
"""

import os
import sys
import shutil
from pathlib import Path


def setup_nnunet_environment():
    """
    设置 nnU-Net v2 所需的环境变量

    nnU-Net v2 需要三个环境变量指向特定目录：
    - nnUNet_raw: 原始数据目录
    - nnUNet_preprocessed: 预处理数据目录
    - nnUNet_results: 模型权重和结果目录
    """
    # 使用临时目录
    base_dir = "/tmp/nnunet_temp"

    nnunet_raw = os.path.join(base_dir, "nnUNet_raw")
    nnunet_preprocessed = os.path.join(base_dir, "nnUNet_preprocessed")
    nnunet_results = os.path.join(base_dir, "nnUNet_results")

    # 创建目录
    os.makedirs(nnunet_raw, exist_ok=True)
    os.makedirs(nnunet_preprocessed, exist_ok=True)
    os.makedirs(nnunet_results, exist_ok=True)

    # 设置环境变量
    os.environ["nnUNet_raw"] = nnunet_raw
    os.environ["nnUNet_preprocessed"] = nnunet_preprocessed
    os.environ["nnUNet_results"] = nnunet_results

    print(f"[INFO] nnU-Net 环境变量已设置:")
    print(f"  - nnUNet_raw: {nnunet_raw}")
    print(f"  - nnUNet_preprocessed: {nnunet_preprocessed}")
    print(f"  - nnUNet_results: {nnunet_results}")

    return nnunet_raw, nnunet_preprocessed, nnunet_results


def download_msd_pancreas_weights():
    """
    下载 MSD Task07 Pancreas 预训练模型权重

    使用 huggingface_hub 从 nnU-Net 官方仓库下载
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    import zipfile

    print("\n" + "="*60)
    print("下载 MSD Task07 Pancreas 预训练模型")
    print("="*60)

    # 模型仓库信息
    repo_id = "Mazu/nnUNet2_preprocessed_MSD_Task07_Pancreas"

    # 下载目录
    download_dir = "/tmp/nnunet_msd_task07"
    os.makedirs(download_dir, exist_ok=True)

    try:
        print(f"[INFO] 从 HuggingFace 下载模型: {repo_id}")
        print("[INFO] 这可能需要几分钟时间...")

        # 下载整个仓库
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=download_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"[INFO] 模型已下载到: {local_path}")

        # 查找模型权重文件
        model_files = list(Path(local_path).glob("**/*.pth"))
        if model_files:
            print(f"[INFO] 找到模型权重文件:")
            for f in model_files:
                print(f"  - {f}")

        return local_path

    except Exception as e:
        print(f"[ERROR] 下载失败: {str(e)}")
        print("\n尝试备用方案: 使用 wget 下载...")
        return None


def download_via_wget():
    """
    使用 wget 从备选地址下载
    """
    import subprocess

    download_dir = "/tmp/nnunet_msd_task07"
    os.makedirs(download_dir, exist_ok=True)

    # nnU-Net v2 的 MSD Task07 权重可以从 Google Drive 或镜像下载
    # 这里使用一个已知的镜像地址
    url = "https://github.com/MazuTribu/nnUNet2_preprocessed_MSD_Task07_Pancreas/releases/download/v1.0/Task07_Pancreas.zip"

    output_file = os.path.join(download_dir, "Task07_Pancreas.zip")

    print(f"[INFO] 使用 wget 下载: {url}")

    try:
        result = subprocess.run(
            ["wget", "-O", output_file, url],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print(f"[INFO] 下载成功: {output_file}")
            return output_file
        else:
            print(f"[ERROR] wget 下载失败: {result.stderr}")
            return None

    except Exception as e:
        print(f"[ERROR] 下载异常: {str(e)}")
        return None


def setup_msd_task07_dataset(nnunet_raw: str):
    """
    设置 MSD Task07 数据集结构

    nnU-Net 需要特定的数据集结构:
    nnUNet_raw/Dataset007_Pancreas/
    ├── imagesTr/  (训练图像)
    ├── labelsTr/  (训练标签)
    └── dataset.json
    """
    print("\n" + "="*60)
    print("设置 MSD Task07 数据集结构")
    print("="*60)

    # 创建数据集目录结构
    dataset_dir = os.path.join(nnunet_raw, "Dataset007_Pancreas")
    images_dir = os.path.join(dataset_dir, "imagesTr")
    labels_dir = os.path.join(dataset_dir, "labelsTr")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 创建 dataset.json
    dataset_json = '''{
    "name": "Pancreas",
    "description": "Medical Segmentation Decathlon Task07 Pancreas",
    "reference": "MSD",
    "licence": "CC",
    "release": "0.0",
    "tensorImageSize": "3D",
    "modality": {
        "0": "CT"
    },
    "labels": {
        "0": "background",
        "1": "pancreas",
        "2": "tumor"
    },
    "numTraining": 281,
    "numTest": 0,
    "training": [],
    "test": []
}'''

    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, 'w') as f:
        f.write(dataset_json)

    print(f"[INFO] 数据集结构已创建: {dataset_dir}")
    print(f"  - imagesTr: {images_dir}")
    print(f"  - labelsTr: {labels_dir}")
    print(f"  - dataset.json: {json_path}")

    return dataset_dir


def main():
    """主函数"""
    print("="*60)
    print("nnU-Net v2 预训练权重下载脚本")
    print("="*60)

    # 设置环境变量
    nnunet_raw, nnunet_preprocessed, nnunet_results = setup_nnunet_environment()

    # 设置数据集结构
    dataset_dir = setup_msd_task07_dataset(nnunet_raw)

    # 尝试下载预训练权重
    print("\n[IMPORTANT]")
    print("MSD Task07 的预训练权重需要从以下地址手动下载:")
    print("  1. HuggingFace: https://huggingface.co/Mazu/nnUNet2_preprocessed_MSD_Task07_Pancreas")
    print("  2. 或从 nnU-Net 官方获取")
    print("\n由于版权和大小限制，这里提供设置好的环境结构。")
    print("你可以将下载的权重放入以下目录:")
    print(f"  {nnunet_results}")

    # 创建占位说明文件
    readme_path = os.path.join(nnunet_results, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("""MSD Task07 Pancreas 预训练权重放置说明

请下载 nnU-Net v2 的 MSD Task07 预训练权重并解压到此目录。

权重来源选项:
1. HuggingFace: Mazu/nnUNet2_preprocessed_MSD_Task07_Pancreas
2. nnU-Net Model Zoo (需要申请)
3. 自行训练

目录结构应该是:
nnUNet_results/
├── Dataset007_Pancreas/
│   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
│       ├── fold_0/
│       │   ├── checkpoint_final.pth
│       │   └── training.log
│       └── plans.json

或者使用 nnUNetv2_predict 时指定 -chk 参数指向权重文件。
""")

    print(f"\n[INFO] 说明文件已创建: {readme_path}")

    # 打印环境变量设置命令，方便用户复制
    print("\n" + "="*60)
    print("环境变量设置命令 (请复制到你的脚本中):")
    print("="*60)
    print(f"export nnUNet_raw={nnunet_raw}")
    print(f"export nnUNet_preprocessed={nnunet_preprocessed}")
    print(f"export nnUNet_results={nnunet_results}")

    return {
        "nnunet_raw": nnunet_raw,
        "nnunet_preprocessed": nnunet_preprocessed,
        "nnunet_results": nnunet_results,
        "dataset_dir": dataset_dir
    }


if __name__ == "__main__":
    result = main()
