---
title: "PanTS Dataset"
created: 2026-04-17
updated: 2026-04-17
tags: [entity, dataset, CT, pancreas, segmentation]
sources: [PanTS]
mocs: (Interdisciplinary Index)
---

## Definition
大规模胰腺肿瘤与血管分割基准数据集 (NeurIPS 2024)，由 JHU MrGiovanni 团队开发。

## Properties (Claude Code 实审验证 2026-04-17)
- **28 类标注**: 每类独立 `.nii.gz` 文件
- **SMA** (label=26): ✅
- **celiac_artery** (label=5): ✅
- **veins** (label=27): 合并类 (无独立 PV/SMV/CHA)
- **pancreas** (label=17), head(19), body(18), tail(20), duct(21), lesion(28)
- 训练集: 9,000 例; 测试集: 901 例; 总量: 36,390 CT
- 下载: HuggingFace ~300GB; 每包 1000 例
- nnU-Net 不兼容: 需转换脚本

## Related Entities
- [[\../entities/AbdomenAtlas-3.0]]

## Used By
- [Pancreatic_Cancer Architecture] — OOD 验证/补充训练数据

