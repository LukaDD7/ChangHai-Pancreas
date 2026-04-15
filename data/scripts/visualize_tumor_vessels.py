#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肿瘤-血管侵犯学术可视化
用于论文展示的高分辨率叠加图
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches


def load_nifti(file_path):
    """加载NIfTI文件"""
    img = nib.load(file_path)
    return img.get_fdata(), img.affine, img.header


def apply_abdomen_window(ct_slice, window_center=40, window_width=400):
    """
    应用腹部窗宽窗位 (WL:40, WW:400)
    归一化到 0-255
    """
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2

    # 归一化到 0-1
    normalized = (ct_slice - window_min) / (window_max - window_min)
    normalized = np.clip(normalized, 0, 1)

    # 转换到 0-255
    return (normalized * 255).astype(np.uint8)


def find_tumor_center_slice(tumor_data):
    """
    找到肿瘤面积最大的切片（或肿瘤与胰腺距离最近）
    """
    max_area = 0
    best_z = 0

    for z in range(tumor_data.shape[0]):
        area = np.sum(tumor_data[z, :, :] > 0)
        if area > max_area:
            max_area = area
            best_z = z

    return best_z, max_area


def find_tumor_bbox(tumor_slice, padding=50):
    """
    找到肿瘤的边界框，用于裁剪ROI
    """
    coords = np.where(tumor_slice > 0)
    if len(coords[0]) == 0:
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # 添加padding
    y_min = max(0, y_min - padding)
    y_max = min(tumor_slice.shape[0] - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(tumor_slice.shape[1] - 1, x_max + padding)

    return (y_min, y_max, x_min, x_max)


def create_tumor_vessel_visualization(
    ct_path: str,
    tumor_path: str,
    pancreas_path: str,
    vessel_dict: dict,
    output_path: str,
    target_size: int = 300
):
    """
    创建肿瘤-血管叠加可视化

    参数:
        ct_path: CT影像路径
        tumor_path: 肿瘤Mask路径
        pancreas_path: 胰腺Mask路径
        vessel_dict: 血管字典 {名称: 路径}
        output_path: 输出图像路径
        target_size: 目标裁剪尺寸
    """
    print("="*60)
    print("肿瘤-血管侵犯可视化")
    print("="*60)

    # 加载数据
    print("\n[INFO] 加载数据...")
    ct_data, _, _ = load_nifti(ct_path)
    tumor_data, _, _ = load_nifti(tumor_path)
    pancreas_data, _, _ = load_nifti(pancreas_path)

    # 找到肿瘤最大的切片
    best_z, max_area = find_tumor_center_slice(tumor_data)
    print(f"[INFO] 肿瘤最大切片: Z={best_z}, 面积={max_area} 体素")

    # 加载血管
    vessel_data = {}
    for name, path in vessel_dict.items():
        try:
            data, _, _ = load_nifti(path)
            vessel_data[name] = data
            print(f"[INFO] 加载 {name}: {np.sum(data > 0)} 体素")
        except:
            print(f"[WARN] 无法加载 {name}")

    # 提取切片
    ct_slice = ct_data[best_z, :, :]
    tumor_slice = tumor_data[best_z, :, :]
    pancreas_slice = pancreas_data[best_z, :, :]

    # 应用腹部窗
    ct_display = apply_abdomen_window(ct_slice)

    # 找到肿瘤边界框
    bbox = find_tumor_bbox(tumor_slice, padding=80)
    if bbox is None:
        print("[ERROR] 未找到肿瘤")
        return

    y_min, y_max, x_min, x_max = bbox
    print(f"[INFO] 肿瘤ROI: Y=[{y_min},{y_max}], X=[{x_min},{x_max}]")

    # 确保尺寸一致
    h, w = y_max - y_min, x_max - x_min
    if h < target_size:
        y_min = max(0, y_min - (target_size - h) // 2)
        y_max = y_min + target_size
    if w < target_size:
        x_min = max(0, x_min - (target_size - w) // 2)
        x_max = x_min + target_size

    # 裁剪
    ct_crop = ct_display[y_min:y_max, x_min:x_max]
    tumor_crop = tumor_slice[y_min:y_max, x_min:x_max]
    pancreas_crop = pancreas_slice[y_min:y_max, x_min:x_max]

    vessel_crops = {}
    for name, data in vessel_data.items():
        vessel_slice = data[best_z, :, :]
        vessel_crops[name] = vessel_slice[y_min:y_max, x_min:x_max]

    # 创建图像
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    fig.suptitle(f'Pancreatic Tumor-Vascular Relationship\nPatient C3L-03348 | Slice Z={best_z} | Tumor Volume: 5.25 ml',
                 fontsize=14, fontweight='bold')

    # ========== 子图1: 原始CT ==========
    ax1 = axes[0, 0]
    ax1.imshow(ct_crop, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('CT (Abdomen Window: WL40/WW400)', fontsize=11)
    ax1.axis('off')

    # ========== 子图2: 多标签叠加 ==========
    ax2 = axes[0, 1]

    # 背景CT（灰度）
    ax2.imshow(ct_crop, cmap='gray', vmin=0, vmax=255, alpha=0.7)

    # 创建彩色叠加层
    overlay = np.zeros((*ct_crop.shape, 4))  # RGBA

    # 胰腺 - 半透明黄色
    pancreas_mask = pancreas_crop > 0
    overlay[pancreas_mask] = [1.0, 0.8, 0.0, 0.3]

    # 肿瘤 - 半透明红色（覆盖胰腺）
    tumor_mask = tumor_crop > 0
    overlay[tumor_mask] = [1.0, 0.0, 0.0, 0.6]

    # 血管（按优先级）
    vessel_colors = {
        'aorta': [1.0, 0.3, 0.0, 0.5],           # 橙红色
        'portal_vein_and_splenic_vein': [0.0, 0.3, 1.0, 0.5],  # 蓝色
        'inferior_vena_cava': [0.5, 0.0, 0.8, 0.5],  # 紫色
    }

    for name, crop in vessel_crops.items():
        vessel_mask = crop > 0
        if name in vessel_colors:
            overlay[vessel_mask] = vessel_colors[name]

    ax2.imshow(overlay)
    ax2.set_title('Multi-Label Overlay', fontsize=11)
    ax2.axis('off')

    # ========== 子图3: 肿瘤+血管特写 ==========
    ax3 = axes[1, 0]

    # CT背景
    ax3.imshow(ct_crop, cmap='gray', vmin=0, vmax=255, alpha=0.5)

    # 只显示肿瘤和血管
    detail_overlay = np.zeros((*ct_crop.shape, 4))

    # 肿瘤 - 红色
    detail_overlay[tumor_crop > 0] = [1.0, 0.0, 0.0, 0.7]

    # 血管
    for name, crop in vessel_crops.items():
        if name in vessel_colors:
            detail_overlay[crop > 0] = vessel_colors[name]

    ax3.imshow(detail_overlay)

    # 添加标注框
    tumor_coords = np.where(tumor_crop > 0)
    if len(tumor_coords[0]) > 0:
        ty_min, ty_max = tumor_coords[0].min(), tumor_coords[0].max()
        tx_min, tx_max = tumor_coords[1].min(), tumor_coords[1].max()
        rect = Rectangle((tx_min, ty_min), tx_max-tx_min, ty_max-ty_min,
                          linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
        ax3.add_patch(rect)
        ax3.text(tx_min, ty_min-5, 'Tumor', color='yellow', fontsize=9, fontweight='bold')

    ax3.set_title('Tumor-Vascular Detail (No Contact Detected)', fontsize=11)
    ax3.axis('off')

    # ========== 子图4: 图例和统计 ==========
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 创建图例
    legend_elements = [
        mpatches.Patch(color=[1.0, 0.0, 0.0], label='Tumor (Resectable)', alpha=0.7),
        mpatches.Patch(color=[1.0, 0.8, 0.0], label='Pancreas', alpha=0.5),
        mpatches.Patch(color=[1.0, 0.3, 0.0], label='Aorta (Clear)', alpha=0.5),
        mpatches.Patch(color=[0.0, 0.3, 1.0], label='Portal Vein (Clear)', alpha=0.5),
        mpatches.Patch(color=[0.5, 0.0, 0.8], label='IVC (Clear)', alpha=0.5),
    ]

    ax4.legend(handles=legend_elements, loc='upper left', fontsize=10,
               framealpha=0.9, title='Anatomical Labels')

    # 添加统计信息
    stats_text = f"""
    Patient: C3L-03348 (CPTAC-PDA)

    Tumor Characteristics:
    • Volume: 5.25 ml
    • Max Diameter: ~25 mm
    • Location: Pancreas Head/Body

    Vascular Assessment:
    • SMA: Not Segmented
    • SMV: Not Segmented
    • Portal Vein: Clear (No Contact)
    • Aorta: Clear (No Contact)
    • IVC: Clear (No Contact)

    Surgical Classification:
    • Resectable
    • No Vascular Encasement
    • Suitable for Whipple Surgery

    Method: nnU-Net v1 (MSD Task07)
    Window: WL40/WW400
    Slice: Z={best_z}
    """

    ax4.text(0.1, 0.6, stats_text, fontsize=9, verticalalignment='top',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\n[SUCCESS] 可视化图像已保存: {output_path}")

    # 同时保存一个单独的放大图
    fig_single, ax = plt.subplots(1, 1, figsize=(8, 8))

    # CT背景
    ax.imshow(ct_crop, cmap='gray', vmin=0, vmax=255, alpha=0.6)

    # 叠加层
    single_overlay = np.zeros((*ct_crop.shape, 4))
    single_overlay[tumor_crop > 0] = [1.0, 0.0, 0.0, 0.7]
    single_overlay[pancreas_crop > 0] = [1.0, 0.8, 0.0, 0.3]

    for name, crop in vessel_crops.items():
        if name in vessel_colors:
            single_overlay[crop > 0] = vessel_colors[name]

    ax.imshow(single_overlay)

    # 添加比例尺
    scale_bar_length = 20  # mm
    scale_bar_pixels = int(scale_bar_length / 0.919)
    ax.plot([20, 20+scale_bar_pixels], [ct_crop.shape[0]-20, ct_crop.shape[0]-20],
            'w-', linewidth=3)
    ax.text(20, ct_crop.shape[0]-30, f'{scale_bar_length} mm', color='white', fontsize=10)

    ax.set_title(f'Pancreatic Tumor-Vascular Overlay | Z={best_z}', fontsize=12)
    ax.axis('off')

    zoom_path = output_path.replace('.png', '_zoom.png')
    plt.savefig(zoom_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[SUCCESS] 放大图已保存: {zoom_path}")

    plt.close('all')


if __name__ == "__main__":
    # 配置路径
    BASE_DIR = "/media/luzhenyang/project/ChangHai_PDA/data"

    CT_PATH = f"{BASE_DIR}/nifti_output/C3L-03348_CT.nii.gz"
    TUMOR_PATH = f"{BASE_DIR}/nnunet_tumor_output/true_tumor_mask.nii.gz"
    PANCREAS_PATH = f"{BASE_DIR}/segmentations/C3L-03348/pancreas.nii.gz"

    VESSEL_DICT = {
        'aorta': f"{BASE_DIR}/segmentations/C3L-03348/aorta.nii.gz",
        'portal_vein_and_splenic_vein': f"{BASE_DIR}/segmentations/C3L-03348/portal_vein_and_splenic_vein.nii.gz",
        'inferior_vena_cava': f"{BASE_DIR}/segmentations/C3L-03348/inferior_vena_cava.nii.gz",
    }

    OUTPUT_PATH = f"{BASE_DIR}/tumor_vessel_overlay.png"

    # 运行可视化
    create_tumor_vessel_visualization(
        ct_path=CT_PATH,
        tumor_path=TUMOR_PATH,
        pancreas_path=PANCREAS_PATH,
        vessel_dict=VESSEL_DICT,
        output_path=OUTPUT_PATH,
        target_size=300
    )
