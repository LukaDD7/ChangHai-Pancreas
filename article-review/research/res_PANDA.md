---
title: "Research: PANDA"
created: 2026-04-20
updated: 2026-04-20
paper: "Large-scale pancreatic cancer detection via non-contrast CT and deep learning"
authors: Cao K, Lu J, et al.
institution: SJTU-Changhai Hospital / Alibaba DAMO Academy
journal: Nature Medicine
doi: 10.1038/s41591-023-02640-w
tags: [pancreatic-cancer, non-contrast-ct, deep-learning, texture-features, memory-network]
---

# PANDA: 非增强 CT 下的等密度胰腺癌检测标杆 — 微观纹理感知的革新

![PANDA Architecture](../_assets/PANDA_arch.png)

## 0. 论文自述核心故事 (The Story)
PANDA 证明了非增强 CT (NCCT) 并不是等密度胰腺癌的“死胡同”。通过大规模多中心数据训练和**跨模态监督策略**，它成功捕捉到了人眼无法察觉的微观纹理变化，将非增强 CT 的筛查准确率提升到了准专家级别，为大规模人群筛查提供了物理上可行的 AI 方案。

## 1. 核心技术架构 (Technical Architecture)

### 1.1 三阶段处理框架
- **Stage 1: 器官定位 (Localization)**: 实现胰腺的初筛与边界锁定。
- **Stage 2: 多任务学习 (Multi-task Learning)**: 
    - 同时执行 **Segmentation** 与 **Classification**。
    - **核心逻辑**: 利用 **NCCT-CECT 配准数据**，将增强 CT 上的精准病灶 Mask 映射回非增强图像，强制模型在 NCCT 上学习“本应可见”的病理特征。
- **Stage 3: 双路径记忆 Transformer (Dual-path Memory Transformer)**:
    - 引入 **200 个可学习的 Memory Tokens**，作为不同疾病（PDAC, CP, NET 等）的“纹理原型”。
    - 通过 Attention 机制强化图像中的细微灰度波动。

## 2. 诊断标准与疾病全病谱
PANDA 针对以下 8 类胰腺状况进行了稳健检测：
1. Normal (正常)
2. PDAC (导管腺癌)
3. pNET (神内)
4. IPMN (囊性)
5. SCN (浆液性)
6. MCN (黏液性)
7. SPN (实假)
8. CP (慢性胰腺炎)

## 3. 对 PancreasMDT 的启发
- **微观纹理的可行性**: 证明了等密度病灶在特征空间（Latent Space）中是有迹可循的。
- **标签引导思路**: “借用增强期 Mask 训练非增强模型”的策略可直接用于我们 `Geometry_Detector` 的冷启动。
- **防御逻辑补位**: PANDA 的“纹理抗性”是我们 **Innovation 3 (流形对齐)** 的重要学术支撑。

---

## 🔗 Connections
- **迁移到:** [PancreasMDT Innovation 3] (流形对齐与等密度特征提取)
- **DOI:** https://doi.org/10.1038/s41591-023-02640-w
