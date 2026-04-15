#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用真实肿瘤 Mask 进行血管侵犯评估
"""

import sys
sys.path.insert(0, '/media/luzhenyang/project/ChangHai_PDA/data')

from panel_vascular_assessment import run_full_vascular_panel
import json

# 真实肿瘤 Mask（来自 nnU-Net v1）
TUMOR_MASK_PATH = "/media/luzhenyang/project/ChangHai_PDA/data/nnunet_tumor_output/true_tumor_mask.nii.gz"
MASK_DIR = "/media/luzhenyang/project/ChangHai_PDA/data/segmentations/C3L-03348"

print("="*70)
print("真实肿瘤血管侵犯评估")
print("="*70)
print(f"肿瘤 Mask: {TUMOR_MASK_PATH}")
print(f"血管目录: {MASK_DIR}")

# 运行评估
result = run_full_vascular_panel(
    tumor_mask_path=TUMOR_MASK_PATH,
    mask_dir=MASK_DIR,
    dilation_radius_mm=2.0
)

# 打印 JSON 结果
print("\n" + "="*70)
print("结构化评估结果 (JSON)")
print("="*70)
print(json.dumps(result, indent=2, ensure_ascii=False))

# 保存结果
import os
output_json = "/media/luzhenyang/project/ChangHai_PDA/data/vascular_panel_true_tumor.json"
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\n[INFO] 结果已保存: {output_json}")
