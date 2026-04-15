#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
胰腺肿瘤分割 Skill (nnU-Net v2 Based Pancreas Tumor Segmentation)
================================================================================

功能描述:
    基于 nnU-Net v2 框架的 MSD Task07 Pancreas 预训练模型，
    从 CT 影像中自动分割胰腺导管腺癌（Tumor）。

技术特点:
    - 使用 MSD Task07 预训练模型（包含胰腺和肿瘤联合分割）
    - 自动提取肿瘤标签（Label=2），生成纯净的二值化肿瘤 Mask
    - 支持批量推理和后处理

模型输出标签定义:
    - 0: 背景 (Background)
    - 1: 胰腺实质 (Pancreas)
    - 2: 肿瘤 (Tumor) ← 我们提取的目标

作者: Claude Code Assistant
日期: 2026-03-21
版本: 1.0.0
================================================================================
"""

import os
import sys
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Optional
import tempfile
import shutil


# =============================================================================
# 环境配置
# =============================================================================

def setup_nnunet_environment() -> Tuple[str, str, str]:
    """
    设置 nnU-Net v2 运行所需的环境变量

    nnU-Net v2 严格依赖三个环境变量：
    - nnUNet_raw: 原始数据目录
    - nnUNet_preprocessed: 预处理缓存目录
    - nnUNet_results: 模型权重和输出目录

    返回:
        Tuple: (nnUNet_raw, nnUNet_preprocessed, nnUNet_results)
    """
    # 使用项目目录下的持久化路径，而非临时目录
    base_dir = "/media/luzhenyang/project/ChangHai_PDA/data/nnunet_workspace"

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

    print(f"[INFO] nnU-Net v2 环境变量已设置:")
    print(f"  - nnUNet_raw: {nnunet_raw}")
    print(f"  - nnUNet_preprocessed: {nnunet_preprocessed}")
    print(f"  - nnUNet_results: {nnunet_results}")

    return nnunet_raw, nnunet_preprocessed, nnunet_results


# =============================================================================
# 模型权重管理
# =============================================================================

def download_msd_task07_model(nnunet_results: str) -> Optional[str]:
    """
    下载 MSD Task07 Pancreas 预训练模型权重

    使用 huggingface-cli 从公开仓库下载

    参数:
        nnunet_results: nnU-Net results 目录

    返回:
        模型权重目录路径，如果下载失败返回 None
    """
    print(f"\n{'='*60}")
    print("下载 MSD Task07 Pancreas 预训练模型")
    print(f"{'='*60}")

    # 模型仓库
    repo_id = "Mazu/nnUNet2_preprocessed_MSD_Task07_Pancreas"

    # 目标目录
    model_dir = os.path.join(nnunet_results, "Dataset007_Pancreas")
    os.makedirs(model_dir, exist_ok=True)

    try:
        import huggingface_hub

        print(f"[INFO] 从 HuggingFace 下载: {repo_id}")
        print("[INFO] 这可能需要 5-10 分钟，请耐心等待...")

        # 使用 snapshot_download 下载整个仓库
        local_path = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"[SUCCESS] 模型已下载到: {local_path}")
        return local_path

    except Exception as e:
        print(f"[ERROR] 下载失败: {str(e)}")
        print("\n[备选方案]")
        print("请手动下载模型权重并放置到以下目录:")
        print(f"  {model_dir}")
        print("\n下载地址:")
        print(f"  https://huggingface.co/{repo_id}")
        return None


# =============================================================================
# 数据准备
# =============================================================================

def prepare_input_data(ct_path: str, nnunet_raw: str) -> str:
    """
    准备 nnU-Net 推理所需的输入数据

    nnU-Net 要求特定的文件命名格式:
    - 图像: CASE_0000.nii.gz (0000 表示模态，CT 是 0000)

    参数:
        ct_path: 输入 CT 文件路径
        nnunet_raw: nnU-Net raw 目录

    返回:
        准备好的输入文件路径
    """
    # 创建临时数据集目录
    dataset_dir = os.path.join(nnunet_raw, "Dataset007_Pancreas")
    images_ts_dir = os.path.join(dataset_dir, "imagesTs")
    os.makedirs(images_ts_dir, exist_ok=True)

    # 复制并重命名 CT 文件（添加 0000 模态后缀）
    case_id = "TEMP_CASE"
    target_filename = f"{case_id}_0000.nii.gz"
    target_path = os.path.join(images_ts_dir, target_filename)

    # 复制文件
    shutil.copy2(ct_path, target_path)
    print(f"[INFO] 输入数据已准备: {target_path}")

    return target_path, case_id


# =============================================================================
# 核心推理函数
# =============================================================================

def run_nnunet_inference(
    input_path: str,
    output_dir: str,
    model_path: Optional[str] = None
) -> str:
    """
    运行 nnU-Net v2 推理

    使用 nnUNetv2_predict 命令行工具进行推理

    参数:
        input_path: 输入图像路径
        output_dir: 输出目录
        model_path: 自定义模型路径（可选）

    返回:
        输出分割文件路径
    """
    print(f"\n{'='*60}")
    print("运行 nnU-Net v2 推理")
    print(f"{'='*60}")

    # 检查 nnUNetv2_predict 是否可用
    try:
        result = subprocess.run(
            ["nnUNetv2_predict", "--help"],
            capture_output=True,
            text=True
        )
        print("[INFO] nnUNetv2_predict 命令可用")
    except FileNotFoundError:
        print("[ERROR] nnUNetv2_predict 命令未找到")
        print("请确保 nnunetv2 已正确安装")
        raise

    # 构建命令
    # 注意：nnUNetv2_predict 的完整参数
    cmd = [
        "nnUNetv2_predict",
        "-i", os.path.dirname(input_path),  # 输入目录
        "-o", output_dir,                    # 输出目录
        "-d", "Dataset007_Pancreas",         # 数据集 ID
        "-c", "3d_fullres",                  # 配置（3D 全分辨率）
        "-f", "0",                           # 使用第 0 fold
        "--disable_tta"                      # 禁用测试时增强（更快）
    ]

    # 如果提供了自定义模型路径
    if model_path:
        cmd.extend(["-chk", os.path.join(model_path, "checkpoint_final.pth")])

    print(f"[INFO] 执行命令: {' '.join(cmd)}")
    print("[INFO] 推理进行中，可能需要几分钟...")

    # 执行推理
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"[ERROR] 推理失败:")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        raise RuntimeError(f"nnU-Net 推理失败: {result.stderr}")

    print(f"[SUCCESS] 推理完成")

    # 返回输出文件路径
    case_id = os.path.basename(input_path).replace("_0000.nii.gz", "")
    output_file = os.path.join(output_dir, f"{case_id}.nii.gz")

    return output_file


# =============================================================================
# 肿瘤标签提取与后处理
# =============================================================================

def extract_tumor_label(
    raw_mask_path: str,
    output_path: str,
    preserve_pancreas: bool = False
) -> str:
    """
    从 nnU-Net 原始输出中提取肿瘤标签

    MSD Task07 标签定义：
    - 0: 背景
    - 1: 胰腺实质
    - 2: 肿瘤

    提取逻辑：
    - 将所有值为 2 的体素设为 1（肿瘤）
    - 将所有值为 0 或 1 的体素设为 0（背景和胰腺）

    参数:
        raw_mask_path: nnU-Net 原始输出 Mask 路径
        output_path: 输出肿瘤 Mask 路径
        preserve_pancreas: 是否保留胰腺标签（设为 1，肿瘤设为 2）

    返回:
        输出文件路径
    """
    print(f"\n{'='*60}")
    print("提取肿瘤标签")
    print(f"{'='*60}")

    # 加载原始 Mask
    img = nib.load(raw_mask_path)
    mask_data = img.get_fdata()
    header = img.header
    affine = img.affine

    print(f"[INFO] 原始 Mask 维度: {mask_data.shape}")
    print(f"[INFO] 唯一值: {np.unique(mask_data)}")

    # 统计各标签数量
    for label in np.unique(mask_data):
        count = np.sum(mask_data == label)
        label_name = {0: "Background", 1: "Pancreas", 2: "Tumor"}.get(int(label), "Unknown")
        print(f"  - Label {int(label)} ({label_name}): {count} voxels ({count/np.prod(mask_data.shape)*100:.2f}%)")

    # 创建肿瘤专用 Mask
    tumor_mask = np.zeros_like(mask_data, dtype=np.uint8)

    if preserve_pancreas:
        # 保留胰腺（1）和肿瘤（2）
        tumor_mask[mask_data == 1] = 1  # 胰腺
        tumor_mask[mask_data == 2] = 2  # 肿瘤
        print("[INFO] 模式: 保留胰腺和肿瘤标签")
    else:
        # 仅保留肿瘤（二值化）
        tumor_mask[mask_data == 2] = 1
        print("[INFO] 模式: 仅保留肿瘤（二值化）")

    # 保存结果
    tumor_img = nib.Nifti1Image(tumor_mask, affine, header)
    nib.save(tumor_img, output_path)

    print(f"[SUCCESS] 肿瘤 Mask 已保存: {output_path}")
    print(f"[INFO] 肿瘤体素数: {np.sum(tumor_mask > 0)}")

    return output_path


# =============================================================================
# 主函数: 完整的肿瘤分割流程
# =============================================================================

def segment_pancreas_tumor(
    ct_path: str,
    output_dir: str,
    model_path: Optional[str] = None,
    download_model: bool = True
) -> dict:
    """
    完整的胰腺肿瘤分割流程

    这是主入口函数，执行以下步骤：
    1. 设置 nnU-Net 环境变量
    2. 下载/加载预训练模型（如果需要）
    3. 准备输入数据
    4. 运行 nnU-Net 推理
    5. 提取肿瘤标签并保存

    参数:
        ct_path: 输入 CT 影像路径（NIfTI 格式）
        output_dir: 输出目录
        model_path: 自定义模型路径（可选）
        download_model: 是否自动下载模型

    返回:
        dict: 包含所有输出路径的结果字典
    """
    print(f"\n{'='*70}")
    print("胰腺肿瘤分割 - Pancreas Tumor Segmentation")
    print(f"{'='*70}")
    print(f"[INFO] 输入 CT: {ct_path}")
    print(f"[INFO] 输出目录: {output_dir}")

    # 验证输入
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"输入 CT 文件不存在: {ct_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ======================================================================
    # 步骤 1: 设置环境变量
    # ======================================================================
    nnunet_raw, nnunet_preprocessed, nnunet_results = setup_nnunet_environment()

    # ======================================================================
    # 步骤 2: 下载/检查模型
    # ======================================================================
    if download_model and model_path is None:
        model_path = download_msd_task07_model(nnunet_results)

    # ======================================================================
    # 步骤 3: 准备输入数据
    # ======================================================================
    input_path, case_id = prepare_input_data(ct_path, nnunet_raw)

    # ======================================================================
    # 步骤 4: 运行推理
    # ======================================================================
    nnunet_output_dir = os.path.join(output_dir, "nnunet_raw_output")
    os.makedirs(nnunet_output_dir, exist_ok=True)

    try:
        raw_output_path = run_nnunet_inference(
            input_path=input_path,
            output_dir=nnunet_output_dir,
            model_path=model_path
        )
    except Exception as e:
        print(f"[ERROR] 推理失败: {str(e)}")
        print("\n[备选方案]")
        print("请确保已下载 MSD Task07 预训练模型")
        raise

    # ======================================================================
    # 步骤 5: 提取肿瘤标签
    # ======================================================================
    tumor_mask_path = os.path.join(output_dir, "true_tumor_mask.nii.gz")
    tumor_mask_path = extract_tumor_label(
        raw_mask_path=raw_output_path,
        output_path=tumor_mask_path,
        preserve_pancreas=False  # 仅保留肿瘤
    )

    # 同时保存胰腺+肿瘤的完整分割
    full_mask_path = os.path.join(output_dir, "pancreas_and_tumor_mask.nii.gz")
    extract_tumor_label(
        raw_mask_path=raw_output_path,
        output_path=full_mask_path,
        preserve_pancreas=True
    )

    # ======================================================================
    # 结果封装
    # ======================================================================
    result = {
        "status": "success",
        "input_ct": ct_path,
        "output_dir": output_dir,
        "raw_nnunet_output": raw_output_path,
        "tumor_mask": tumor_mask_path,
        "pancreas_and_tumor_mask": full_mask_path,
        "case_id": case_id
    }

    print(f"\n{'='*70}")
    print("分割完成!")
    print(f"{'='*70}")
    print(f"[RESULT] 肿瘤 Mask: {tumor_mask_path}")
    print(f"[RESULT] 胰腺+肿瘤 Mask: {full_mask_path}")

    return result


# =============================================================================
# 测试入口
# =============================================================================

if __name__ == "__main__":
    """
    测试用例：使用 CPTAC-PDA 患者 C3L-03348 的 CT 进行肿瘤分割
    """
    print("="*70)
    print(" segment_pancreas_tumor.py 测试模式")
    print("="*70)

    # 测试配置
    TEST_CT_PATH = "/media/luzhenyang/project/ChangHai_PDA/data/nifti_output/C3L-03348_CT.nii.gz"
    OUTPUT_DIR = "/media/luzhenyang/project/ChangHai_PDA/data/nnunet_tumor_output"

    # 检查输入文件
    if not os.path.exists(TEST_CT_PATH):
        print(f"[ERROR] 测试 CT 文件不存在: {TEST_CT_PATH}")
        print("[INFO] 请修改脚本中的 TEST_CT_PATH 为实际路径")
        sys.exit(1)

    # 运行肿瘤分割
    try:
        result = segment_pancreas_tumor(
            ct_path=TEST_CT_PATH,
            output_dir=OUTPUT_DIR,
            download_model=True
        )

        print("\n" + "="*70)
        print("最终输出路径:")
        print("="*70)
        for key, value in result.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
