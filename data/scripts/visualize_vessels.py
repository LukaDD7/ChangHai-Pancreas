#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
血管分割可视化脚本
展示 CT、胰腺和关键血管的空间关系
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os


def load_nifti(file_path):
    """加载NIfTI文件"""
    img = nib.load(file_path)
    return img.get_fdata(), img.header.get_zooms()


def create_overlay_visualization(
    ct_path: str,
    pancreas_mask_path: str,
    vessel_dict: dict,
    output_dir: str,
    num_slices: int = 9
):
    """
    创建血管分割可视化图像

    参数:
        ct_path: CT影像路径
        pancreas_mask_path: 胰腺掩膜路径
        vessel_dict: 血管字典 {名称: 掩膜路径}
        output_dir: 输出目录
        num_slices: 展示的切片数量
    """
    # 加载数据
    ct_data, ct_spacing = load_nifti(ct_path)
    pancreas_mask, _ = load_nifti(pancreas_mask_path)

    # 加载所有血管掩膜
    vessel_masks = {}
    vessel_colors = {
        'aorta': 'red',
        'portal_vein_and_splenic_vein': 'blue',
        'inferior_vena_cava': 'purple',
        'superior_mesenteric_artery': 'orange',
        'celiac_trunk': 'yellow',
        'spleen': 'green'
    }

    for name, path in vessel_dict.items():
        if os.path.exists(path):
            mask, _ = load_nifti(path)
            vessel_masks[name] = mask

    # 找到胰腺的中心位置，选择最佳切片
    pancreas_coords = np.where(pancreas_mask > 0)
    if len(pancreas_coords[0]) == 0:
        print("[ERROR] 胰腺掩膜为空")
        return

    z_min, z_max = pancreas_coords[0].min(), pancreas_coords[0].max()
    z_center = (z_min + z_max) // 2

    # 选择切片范围
    z_start = max(0, z_center - num_slices // 2)
    z_end = min(ct_data.shape[0], z_start + num_slices)
    slice_indices = np.linspace(z_start, z_end - 1, num_slices, dtype=int)

    # 创建图像
    fig, axes = plt.subplots(3, num_slices // 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, z in enumerate(slice_indices):
        ax = axes[idx]

        # 显示CT底层 (调整窗宽窗位到腹部软组织窗)
        ct_slice = ct_data[z, :, :]
        # 腹部窗: 窗位40, 窗宽400 (范围: -160 到 240)
        ct_display = np.clip((ct_slice + 160) / 400, 0, 1)

        ax.imshow(ct_display, cmap='gray', aspect='equal')

        # 叠加胰腺 (半透明黄色)
        pancreas_slice = pancreas_mask[z, :, :]
        pancreas_overlay = np.zeros((*pancreas_slice.shape, 4))
        pancreas_overlay[pancreas_slice > 0] = [1, 0.8, 0, 0.4]  # 黄色半透明
        ax.imshow(pancreas_overlay)

        # 叠加血管
        for vessel_name, vessel_mask in vessel_masks.items():
            vessel_slice = vessel_mask[z, :, :]
            if np.any(vessel_slice > 0):
                color = vessel_colors.get(vessel_name, 'cyan')
                vessel_overlay = np.zeros((*vessel_slice.shape, 4))

                # 设置颜色
                color_map = {
                    'red': [1, 0, 0, 0.6],
                    'blue': [0, 0, 1, 0.6],
                    'purple': [0.5, 0, 0.5, 0.6],
                    'orange': [1, 0.5, 0, 0.6],
                    'yellow': [1, 1, 0, 0.6],
                    'green': [0, 1, 0, 0.6],
                    'cyan': [0, 1, 1, 0.6]
                }

                vessel_overlay[vessel_slice > 0] = color_map.get(color, [0, 1, 1, 0.6])
                ax.imshow(vessel_overlay)

        # 添加切片信息
        ax.set_title(f'Slice {z}', fontsize=10)
        ax.axis('off')

    # 添加图例
    legend_elements = [Patch(facecolor='yellow', edgecolor='black', label='Pancreas', alpha=0.5)]

    for vessel_name, color_name in vessel_colors.items():
        if vessel_name in vessel_masks:
            color_map = {
                'red': [1, 0, 0],
                'blue': [0, 0, 1],
                'purple': [0.5, 0, 0.5],
                'orange': [1, 0.5, 0],
                'yellow': [1, 1, 0],
                'green': [0, 1, 0],
                'cyan': [0, 1, 1]
            }
            color_rgb = color_map.get(color_name, [0, 1, 1])
            legend_elements.append(Patch(
                facecolor=color_rgb,
                edgecolor='black',
                label=vessel_name.replace('_', ' ').title(),
                alpha=0.6
            ))

    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

    plt.suptitle('Vascular Assessment - Pancreas and Key Vessels\nPatient: C3L-03348', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # 保存图像
    output_path = os.path.join(output_dir, 'vascular_segmentation_overview.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"[INFO] 可视化图像已保存: {output_path}")

    plt.close()

    # 创建3D投影图
    create_3d_projection(ct_data, pancreas_mask, vessel_masks, vessel_colors, output_dir)


def create_3d_projection(
    ct_data: np.ndarray,
    pancreas_mask: np.ndarray,
    vessel_masks: dict,
    vessel_colors: dict,
    output_dir: str
):
    """创建3D最大强度投影图"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # 轴状面投影 (Axial - Z轴)
    pancreas_mip = np.max(pancreas_mask, axis=0)
    ax = axes[0, 0]
    ax.imshow(pancreas_mip > 0, cmap='YlOrRd', alpha=0.3)

    for vessel_name, vessel_mask in vessel_masks.items():
        vessel_mip = np.max(vessel_mask, axis=0)
        if np.any(vessel_mip > 0):
            color = vessel_colors.get(vessel_name, 'cyan')
            cmap_dict = {
                'red': 'Reds', 'blue': 'Blues', 'purple': 'Purples',
                'orange': 'Oranges', 'yellow': 'YlOrRd', 'green': 'Greens'
            }
            cmap = cmap_dict.get(color, 'Blues')
            ax.imshow(vessel_mip > 0, cmap=cmap, alpha=0.6)

    ax.set_title('Axial View (Top-Down)', fontsize=12)
    ax.axis('off')

    # 冠状面投影 (Coronal - Y轴)
    pancreas_mip = np.max(pancreas_mask, axis=1)
    ax = axes[0, 1]
    ax.imshow(pancreas_mip > 0, cmap='YlOrRd', alpha=0.3)

    for vessel_name, vessel_mask in vessel_masks.items():
        vessel_mip = np.max(vessel_mask, axis=1)
        if np.any(vessel_mip > 0):
            color = vessel_colors.get(vessel_name, 'cyan')
            cmap_dict = {
                'red': 'Reds', 'blue': 'Blues', 'purple': 'Purples',
                'orange': 'Oranges', 'yellow': 'YlOrRd', 'green': 'Greens'
            }
            cmap = cmap_dict.get(color, 'Blues')
            ax.imshow(vessel_mip > 0, cmap=cmap, alpha=0.6)

    ax.set_title('Coronal View (Front-Back)', fontsize=12)
    ax.axis('off')

    # 矢状面投影 (Sagittal - X轴)
    pancreas_mip = np.max(pancreas_mask, axis=2)
    ax = axes[1, 0]
    ax.imshow(pancreas_mip > 0, cmap='YlOrRd', alpha=0.3)

    for vessel_name, vessel_mask in vessel_masks.items():
        vessel_mip = np.max(vessel_mask, axis=2)
        if np.any(vessel_mip > 0):
            color = vessel_colors.get(vessel_name, 'cyan')
            cmap_dict = {
                'red': 'Reds', 'blue': 'Blues', 'purple': 'Purples',
                'orange': 'Oranges', 'yellow': 'YlOrRd', 'green': 'Greens'
            }
            cmap = cmap_dict.get(color, 'Blues')
            ax.imshow(vessel_mip > 0, cmap=cmap, alpha=0.6)

    ax.set_title('Sagittal View (Left-Right)', fontsize=12)
    ax.axis('off')

    # 3D 示意图
    ax = axes[1, 1]
    ax.text(0.5, 0.8, '3D Relationship', fontsize=14, ha='center', transform=ax.transAxes)

    # 列出检测到的血管
    y_pos = 0.6
    for vessel_name in vessel_masks.keys():
        color_name = vessel_colors.get(vessel_name, 'white')
        ax.text(0.5, y_pos, f"● {vessel_name.replace('_', ' ').title()}",
                fontsize=10, ha='center', transform=ax.transAxes,
                color=color_name)
        y_pos -= 0.08

    ax.text(0.5, 0.3, f"● Pancreas (Target Organ)",
            fontsize=10, ha='center', transform=ax.transAxes,
            color='yellow')

    ax.text(0.5, 0.15, f"Total Vessels: {len(vessel_masks)}",
            fontsize=10, ha='center', transform=ax.transAxes,
            color='white')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor('black')

    plt.suptitle('3D Maximum Intensity Projection\nPatient: C3L-03348', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'vascular_3d_projection.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"[INFO] 3D投影图已保存: {output_path}")

    plt.close()


if __name__ == "__main__":
    # 配置路径
    BASE_DIR = "/media/luzhenyang/project/ChangHai_PDA/data"
    CT_PATH = f"{BASE_DIR}/nifti_output/C3L-03348_CT.nii.gz"
    MASK_DIR = f"{BASE_DIR}/segmentations/C3L-03348"
    OUTPUT_DIR = f"{BASE_DIR}/visualizations"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 定义要显示的血管
    VESSEL_DICT = {
        'aorta': f"{MASK_DIR}/aorta.nii.gz",
        'portal_vein_and_splenic_vein': f"{MASK_DIR}/portal_vein_and_splenic_vein.nii.gz",
        'inferior_vena_cava': f"{MASK_DIR}/inferior_vena_cava.nii.gz",
        'spleen': f"{MASK_DIR}/spleen.nii.gz",
        'liver': f"{MASK_DIR}/liver.nii.gz",
    }

    # 执行可视化
    create_overlay_visualization(
        ct_path=CT_PATH,
        pancreas_mask_path=f"{MASK_DIR}/pancreas.nii.gz",
        vessel_dict=VESSEL_DICT,
        output_dir=OUTPUT_DIR,
        num_slices=9
    )

    print("\n[INFO] 可视化完成!")
