---
title: "Source: RadGPT (ICCV 2025)"
id: "RadGPT-ICCV2025"
author: "Bassi et al."
year: 2025
venue: "ICCV"
project: "Pancreatic_Cancer"
type: "paper"
tags: ["agent", "abdominal-ct", "report-generation", "AbdomenAtlas-3.0"]
---

# RadGPT: 结构化影像语料库构建 Agent (Structural Imaging Corpus Construction Agent)

## 0. 论文自述核心故事 (The Story)
RadGPT 是首个实现 **3D 空间、Mask 与叙述性报告自动配对** 的结构化 Agent。它解决了公共影像领域“缺乏高质量图文配对数据”的瓶颈，通过 Segment-Extract-Transform 三阶段 Pipeline，为大规模放射大模型的训练提供了标准化的语义土壤。

## Evidence Map
- **Evidence_ID: RadGPT_Arch_01**: The framework uses a three-stage pipeline: **Segment → Extract → Transform**. [p.4]
- **Evidence_ID: RadGPT_Vessel_01**: Automated resectability is assessed via **Algorithm 3**, specifically measuring contact angles with **SMA, SMV, Celiac Axis, and Portal Vein**. [p.7]
- **Evidence_ID: RadGPT_Isodense_01**: Sensitivity for **isodense tumors** reached **82%** by leveraging mass-effect cues. [p.10]
- **Evidence_ID: RadGPT_Gap_01**: Authors identify the primary gap as the **scarcity of paired 3D masks and narrative clinical reports** in public domains. [p.2]

## Connection to PancreasMDT
- **Convergence**: Both use hierarchical reasoning (Layer 1/2/3).
- **Divergence**: RadGPT is a forward pipeline; PancreasMDT implements **Cognitive Dissonance Monitoring** to audit segmentation failures (especially in isodense cases where RadGPT still has an 18% miss rate).

