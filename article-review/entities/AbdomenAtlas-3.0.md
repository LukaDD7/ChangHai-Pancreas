---
title: "AbdomenAtlas 3.0"
created: 2026-04-17
updated: 2026-04-17
tags: [entity, dataset, CT, abdomen, segmentation]
sources: [AbdomenAtlas]
mocs: (Interdisciplinary Index)
---

## Definition
大规模腹部 CT 多器官+血管标注数据集，由 JHU MrGiovanni 团队开发。首个公开提供逐体素胰周血管标注的数据集。

## Properties

### 标注覆盖 (Claude Code 实审验证 2026-04-17)
- **celiac_trunk** (label=44): ✅ 存在 (578 voxels/例)
- **superior_mesenteric_artery** (label=54): ✅ 存在 (1019 voxels/例)
- **veins** (label=55): ✅ SMV + Portal Vein + Splenic Vein 合并 (16734 voxels/例)
- **portal_vein_and_splenic_vein** (label=47): ✅ 存在 (3833 voxels/例)
- **CHA** (Common Hepatic Artery): ❌ 无独立标注
- **pancreas** (label=6): 仅 21 voxels (被子区覆盖)
- **pancreas_head/body/tail** (label=7/8/9): ✅ 完整

### 数据格式
- 每例包含 `ct.nii.gz` + `segmentations/*.nii.gz` (每类独立 mask)
- `combine_labels.py` (RadGPT) 可合并为 nnU-Net v2 兼容的单文件多类 label map
- Shape 示例: (511, 404, 339), Spacing: (0.822×0.822×2.5mm)

### 下载
- HuggingFace: https://huggingface.co/AbdomenAtlas
- 每块 232 例
- HF-Mirror (hf-mirror.com) 加速: 25-30 MB/s (300x vs 直连)

## Related Entities
- [[\../entities/PanTS]]
- [[\../entities/nnU-Net]]

## Used By
- [Pancreatic_Cancer Architecture] — P0 级训练数据源

