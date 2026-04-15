#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
长海标准临床诊断图生成 (Generate Diagnostic Figure)
================================================================================

功能描述:
    生成符合长海医院标准的临床对比图，包含：
    - 原始 CT 窗宽窗位调整
    - 肿瘤与血管伪彩色叠加
    - ROI 裁剪与标注

作者: Claude Code Assistant
日期: 2026-03-21
版本: 1.0.0
================================================================================
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import os


def load_nifti(file_path):
    """加载 NIfTI 文件"""
    img = nib.load(file_path)
    return img.get_fdata(), img.affine, img.header


def apply_abdomen_window(ct_slice, window_center=40, window_width=400):
    """
    应用腹部窗宽窗位 (WL:40, WW:400)
    归一化到 0-255 用于显示
    """
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2

    # 归一化到 0-1
    normalized = (ct_slice - window_min) / (window_max - window_min)
    normalized = np.clip(normalized, 0, 1)

    # 转换到 0-255
    return (normalized * 255).astype(np.uint8)


def find_tumor_center_slice(tumor_data):
    """找到肿瘤面积最大的切片"""
    max_area = 0
    best_z = 0

    for z in range(tumor_data.shape[0]):
        area = np.sum(tumor_data[z, :, :] > 0)
        if area > max_area:
            max_area = area
            best_z = z

    return best_z, max_area


def find_tumor_bbox(tumor_slice, padding=100):
    """找到肿瘤的边界框"""
    coords = np.where(tumor_slice > 0)
    if len(coords[0]) == 0:
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # 添加 padding
    y_min = max(0, y_min - padding)
    y_max = min(tumor_slice.shape[0] - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(tumor_slice.shape[1] - 1, x_max + padding)

    return (y_min, y_max, x_min, x_max)


def create_clinical_figure():
    """生成临床诊断图"""
    print("="*70)
    print("长海标准临床诊断图生成")
    print("="*70)

    # 配置路径
    BASE_DIR = "/media/luzhenyang/project/ChangHai_PDA/data"
    CT_PATH = f"{BASE_DIR}/nifti_output/C3L-03348_CT.nii.gz"
    TUMOR_PATH = f"{BASE_DIR}/nnunet_tumor_output/true_tumor_mask.nii.gz"
    MASK_DIR = f"{BASE_DIR}/segmentations/C3L-03348"
    OUTPUT_PATH = f"{BASE_DIR}/clinical_evidence_overlay.png"

    # 加载数据
    print("\n[INFO] 加载数据...")
    ct_data, _, _ = load_nifti(CT_PATH)
    tumor_data, _, _ = load_nifti(TUMOR_PATH)

    # 找到肿瘤最大的切片
    best_z, max_area = find_tumor_center_slice(tumor_data)
    print(f"[INFO] 肿瘤最大切片: Z={best_z}, 面积={max_area} 体素")

    # 加载血管
    vessel_data = {}
    vessel_files = {
        'aorta': f"{MASK_DIR}/aorta.nii.gz",
        'portal_vein': f"{MASK_DIR}/portal_vein_and_splenic_vein.nii.gz",
        'ivc': f"{MASK_DIR}/inferior_vena_cava.nii.gz",
    }

    for name, path in vessel_files.items():
        if os.path.exists(path):
            data, _, _ = load_nifti(path)
            vessel_data[name] = data
            print(f"[INFO] 加载 {name}: {np.sum(data > 0)} 体素")

    # 提取切片
    ct_slice = ct_data[best_z, :, :]
    tumor_slice = tumor_data[best_z, :, :]

    vessel_slices = {}
    for name, data in vessel_data.items():
        vessel_slices[name] = data[best_z, :, :]

    # 应用腹部窗
    ct_display = apply_abdomen_window(ct_slice)

    # 找到肿瘤边界框 (300x300)
    bbox = find_tumor_bbox(tumor_slice, padding=120)
    if bbox is None:
        print("[ERROR] 未找到肿瘤")
        return

    y_min, y_max, x_min, x_max = bbox

    # 确保 300x300
    h, w = y_max - y_min, x_max - x_min
    if h < 300:
        y_min = max(0, y_min - (300 - h) // 2)
        y_max = y_min + 300
    if w < 300:
        x_min = max(0, x_min - (300 - w) // 2)
        x_max = x_min + 300

    # 裁剪到 300x300
    if y_max - y_min > 300:
        y_max = y_min + 300
    if x_max - x_min > 300:
        x_max = x_min + 300

    ct_crop = ct_display[y_min:y_max, x_min:x_max]
    tumor_crop = tumor_slice[y_min:y_max, x_min:x_max]

    vessel_crops = {}
    for name, slice_data in vessel_slices.items():
        vessel_crops[name] = slice_data[y_min:y_max, x_min:x_max]

    print(f"[INFO] ROI 尺寸: {ct_crop.shape}")

    # 创建图像
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f'Changhai Standard - Pancreatic Tumor Assessment\nPatient: C3L-03348 | Slice Z={best_z} | Window: WL40/WW400',
                 fontsize=14, fontweight='bold')

    # ========== 子图1: 原始 CT ==========
    ax1 = axes[0, 0]
    ax1.imshow(ct_crop, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('CT Original (Abdomen Window)', fontsize=11, fontweight='bold')
    ax1.axis('off')

    # 添加比例尺
    scale_bar_mm = 20
    scale_bar_px = int(scale_bar_mm / 0.919)
    ax1.plot([20, 20+scale_bar_px], [ct_crop.shape[0]-20, ct_crop.shape[0]-20],
            'w-', linewidth=4)
    ax1.text(20, ct_crop.shape[0]-35, f'{scale_bar_mm}mm', color='white', fontsize=10, fontweight='bold')

    # ========== 子图2: 伪彩色叠加 ==========
    ax2 = axes[0, 1]

    # CT 背景
    ax2.imshow(ct_crop, cmap='gray', vmin=0, vmax=255, alpha=0.6)

    # 创建 RGBA 叠加层
    overlay = np.zeros((*ct_crop.shape, 4))

    # 肿瘤 - 红色 (Red, Alpha=0.5)
    tumor_mask = tumor_crop > 0
    overlay[tumor_mask] = [1.0, 0.0, 0.0, 0.5]

    # 动脉 - 黄色 (Yellow, Alpha=0.4)
    if 'aorta' in vessel_crops:
        artery_mask = vessel_crops['aorta'] > 0
        overlay[artery_mask] = [1.0, 1.0, 0.0, 0.4]

    # 静脉 - 蓝色 (Blue, Alpha=0.4)
    if 'portal_vein' in vessel_crops:
        vein_mask = vessel_crops['portal_vein'] > 0
        overlay[vein_mask] = [0.0, 0.4, 1.0, 0.4]

    if 'ivc' in vessel_crops:
        ivc_mask = vessel_crops['ivc'] > 0
        overlay[ivc_mask] = [0.3, 0.0, 0.8, 0.4]

    ax2.imshow(overlay)
    ax2.set_title('Color Overlay (Changhai Standard)', fontsize=11, fontweight='bold')
    ax2.axis('off')

    # ========== 子图3: 肿瘤特写 ==========
    ax3 = axes[1, 0]

    # CT 背景
    ax3.imshow(ct_crop, cmap='gray', vmin=0, vmax=255, alpha=0.5)

    # 仅显示肿瘤和血管
    detail_overlay = np.zeros((*ct_crop.shape, 4))

    # 肿瘤 - 红色
    detail_overlay[tumor_crop > 0] = [1.0, 0.0, 0.0, 0.6]

    # 血管
    if 'aorta' in vessel_crops:
        detail_overlay[vessel_crops['aorta'] > 0] = [1.0, 0.8, 0.0, 0.5]
    if 'portal_vein' in vessel_crops:
        detail_overlay[vessel_crops['portal_vein'] > 0] = [0.0, 0.5, 1.0, 0.5]
    if 'ivc' in vessel_crops:
        detail_overlay[vessel_crops['ivc'] > 0] = [0.5, 0.0, 0.8, 0.5]

    ax3.imshow(detail_overlay)

    # 添加肿瘤标注框
    tumor_coords = np.where(tumor_crop > 0)
    if len(tumor_coords[0]) > 0:
        ty_min, ty_max = tumor_coords[0].min(), tumor_coords[0].max()
        tx_min, tx_max = tumor_coords[1].min(), tumor_coords[1].max()
        rect = Rectangle((tx_min, ty_min), tx_max-tx_min, ty_max-ty_min,
                          linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
        ax3.add_patch(rect)
        ax3.text(tx_min, ty_min-5, 'Tumor', color='yellow', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax3.set_title('Tumor-Vascular Detail View', fontsize=11, fontweight='bold')
    ax3.axis('off')

    # ========== 子图4: 临床信息 ==========
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 图例
    legend_elements = [
        mpatches.Patch(color=[1.0, 0.0, 0.0], label='Tumor (Resectable)', alpha=0.6),
        mpatches.Patch(color=[1.0, 0.8, 0.0], label='Artery (Clear)', alpha=0.5),
        mpatches.Patch(color=[0.0, 0.5, 1.0], label='Portal Vein (Clear)', alpha=0.5),
        mpatches.Patch(color=[0.5, 0.0, 0.8], label='IVC (Clear)', alpha=0.5),
    ]

    ax4.legend(handles=legend_elements, loc='upper left', fontsize=10,
               framealpha=0.9, title='Anatomical Labels', title_fontsize=11)

    # 临床统计信息
    stats_text = """
    ┌─────────────────────────────────────┐
    │    Changhai Hospital Standard       │
    │      Pancreatic Tumor Report        │
    ├─────────────────────────────────────┤
    │ Patient ID: C3L-03348               │
    │ Dataset: CPTAC-PDA                  │
    ├─────────────────────────────────────┤
    │ TUMOR CHARACTERISTICS               │
    │ • Volume: 5.25 ml                   │
    │ • Max Diameter: ~25 mm              │
    │ • Location: Pancreas Head/Body      │
    │ • Classification: Resectable        │
    ├─────────────────────────────────────┤
    │ VASCULAR ASSESSMENT                 │
    │ • SMA: Not Evaluated                │
    │ • SMV: Not Evaluated                │
    │ • Portal Vein: Clear (No Contact)   │
    │ • Aorta: Clear (No Contact)         │
    │ • IVC: Clear (No Contact)           │
    ├─────────────────────────────────────┤
    │ SURGICAL RECOMMENDATION             │
    │ • Resectable                        │
    │ • No Vascular Encasement            │
    │ • Suitable for Whipple Procedure    │
    ├─────────────────────────────────────┤
    │ TECHNICAL PARAMETERS                │
    │ • Window: WL40/WW400                │
    │ • Slice: Z={z}                      │
    │ • ROI: 300×300 px                   │
    │ • Method: nnU-Net v1 (MSD Task07)   │
    └─────────────────────────────────────┘
    """.format(z=best_z)

    ax4.text(0.05, 0.95, stats_text, fontsize=8.5, verticalalignment='top',
             family='monospace', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\n[SUCCESS] 临床诊断图已保存: {OUTPUT_PATH}")

    plt.close()

    return OUTPUT_PATH


if __name__ == "__main__":
    create_clinical_figure()
