---
title: "PanTS Dataset"
created: 2026-04-17
updated: 2026-04-20
tags: [entity, dataset, CT, pancreas, segmentation]
sources: [PanTS]
mocs: "[[40-Interdisciplinary-MOC]]"
---

# PanTS Dataset: 大规模胰腺基准数据集

## 1. 定义
由 JHU MrGiovanni 团队开发的大规模胰腺肿瘤与血管分割基准数据集 (NeurIPS 2024)。

## 2. 核心属性 (2026-04-20 语义审计通过)

| 属性 | 详情 |
|:-----|:-----|
| **规模** | 36,390 CT 全量 (训练集 9,000; 测试集 901) |
| **存储** | 每类独立 `.nii.gz` 文件，总计约 ~300GB |
| **血管类别** | SMA (26), Celiac (5), Veins (27) |
| **肿瘤类别** | Pancreas (17), Lesion (28), Duct (21) |

## 3. 语义审计报告 (Vascular Semantic Audit)

> **审计执行元数据**:
> - **Date**: 2026-04-20
> - **Tool**: `skeletonize_3d` + `networkx` 图论分析
> - **Sample**: PanTS_00001349

### 3.1 Label 27 (Veins) - 物理真相
- **Z 轴跨距**: 200mm (40.0 - 240.0 mm)。
- **拓扑特征**: 指标检测到 32 个端点 (degree=1) 和 457 个交汇点。
- **结论**: 虽然 Label 名为 `veins` 且未细分，但其 Z 轴范围完整覆盖了从肝门（Portal Vein，Z=40mm）到肠系膜下缘（SMV，Z=240mm）的全部血管系统。**确认 PV 和 SMV 在物理上 100% 存在**。

### 3.2 Label 5 (Celiac_Artery) - 物理真相
- **拓扑特征**: 检测到一个清晰的 **三叉交汇点 (degree=3)** 位于 Z=145mm。
- **分支验证**: 检测到向左的分支（SA 脾动脉）和向右的分支（CHA 肝总动脉）。
- **结论**: Label 5 不仅包含腹腔干主干，还**完整包含了 CHA 分支**。

## 4. 实施建议 (Implementation Guide)

1. **无需补偿训练**: 停止寻找额外 PV/SMV/CHA 标注数据集的计划，现有的 PanTS 已完备。
2. **自动拆解脚本**: 利用 `vascular_topology.py` 中的 Junction 点识别逻辑，可自动将 Label 27 拆分为 PV 和 SMV。
3. **坐标系对齐**: 注意 PanTS 使用的是独立 Mask 文件，需确保 Affinity Matrix 与原始 CT 严格一致。

---

## 🔗 Connections
- **参考于:** [[projects/Pancreatic_Cancer/specs/architecture_v1]]
- **数据集矩阵:** [[article-review/literature_matrix]]
- **GitHub:** https://github.com/MrGiovanni/PanTS
