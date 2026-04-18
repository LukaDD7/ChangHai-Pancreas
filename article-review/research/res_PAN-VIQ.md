---
title: "Research: PAN-VIQ"
created: 2026-04-17
updated: 2026-04-18
paper: "A clinically validated 3D deep learning approach for quantifying vascular invasion in pancreatic cancer"
authors: Zhang Y, Zhang H, Yang Y, et al.
institution: SJTU-Ruijin Hospital / UIH
journal: npj Digital Medicine (IF ~15)
doi: 10.1038/s41746-025-02260-3
type: method_paper
domain: [Engineering, Interdisciplinary]
tags: [pancreatic-cancer, vascular-invasion, nnU-Net, 3D-geometry, centerline, PDAC]
mocs: (Interdisciplinary Index)
---

# PAN-VIQ: Pancreatic Vascular Invasion Quantifier — DFR Phase 1 骨折级解构

![PAN-VIQ Architecture](../_assets/PAN-VIQ_arch.png)

---

## 1. 研究链路图谱 (Research Chain)

### 1.1 观察 (Observation)

- **核心问题**: 胰腺导管腺癌 (PDAC) 术前血管侵犯评估的**主观性**与**不可复现性**。
- **现状量化**:
  - 放射科医生对"包绕角 (encasement angle)" 的判读 **Kappa = 0.28–0.55** (来源: Al-Hawary et al., Radiology 2014; Hong et al., Radiology 2018)
  - 血管侵犯概率从 <180° 时的 ~40% 跃升至 ≥180° 时的 >80% (来源: Ferrone et al., Ann. Surg. 2015; Michelakos et al., Ann. Surg. 2019)
  - 仅 **15–20%** PDAC 患者在确诊时有手术切除机会 (来源: Oettle et al., JAMA 2013; Conroy et al., NEJM 2018)
- **痛点本质**: 医生在 2D 轴位切片上**目测**角度 → 主观、不可重复、无法标准化跨机构比较

### 1.2 假设 (Hypotheses)

| # | 假设 | 论文中对应实验验证 |
|---|------|-------------------|
| H1 | 基于 nnU-Net 的双期 CT 分割可以同时准确勾画 PDAC 肿瘤 + 5 条血管 | Table S2: DSC 0.853 (内部), 0.789 (外部); SMA DSC 0.871, Sensitivity 0.983 |
| H2 | 从分割 Mask 可自动计算 3D 连续包绕角，精度优于人工 2D 目测 | Fig. 3: 手工 121° vs 半自动 154.4° vs PAN-VIQ 172.4° (同一病例) |
| H3 | PAN-VIQ 性能可匹敌高年资放射科医生，且优于低年资 | Table 2: CHA 准确率 PAN-VIQ 90.22% vs 低年资 56.86%; 与高年资非劣效 (Δ<5%, p>0.05) |

### 1.3 临床 Staging 关键概念

> **Portal Venous Phase (门脉期)** 是增强 CT 扫描中注射碘造影剂后 **60–70 秒** 的采集窗口。此时造影剂充盈门静脉系统 (PV, SMV)，使**静脉**显影最佳。与之对应的 **Arterial Phase (动脉期)** 在注射后 **25–35 秒** 采集，动脉 (CA, CHA, SMA) 显影最佳。PAN-VIQ 对这两期分别训练独立的分割模型。

---

## 2. 技术方案深度解构 (Technical Architecture)

### 2.0 总览 Pipeline

```
双期 CE-CT (DICOM)
    │
    ├─ 动脉期 ──→ nnU-Net Model_A ──→ {PDAC_mask, CA_mask, CHA_mask, SMA_mask}
    │
    └─ 门脉期 ──→ nnU-Net Model_V ──→ {PDAC_mask, SMV_mask, PV_mask}
                        │
                        ▼
              ┌─────────────────────┐
              │ Post-Processing     │
              │ · 体积阈值过滤       │
              │   (art<415 voxels,  │
              │    ven<450 voxels)  │
              │ · 连通域过滤         │
              └────────┬────────────┘
                       ▼
              ┌─────────────────────┐
              │ Centerline          │
              │ Extraction          │
              │ · skeletonize_3d    │
              │ · networkx 图排序   │
              └────────┬────────────┘
                       ▼
              ┌─────────────────────┐
              │ Angle Quantification│
              │ · 沿中心线逐点:     │
              │   1. 计算切向量      │
              │   2. 生成正交法平面  │
              │   3. 重采样 tumor/   │
              │      vessel mask    │
              │   4. 余弦定理算角度  │
              │ · 取最大角 = 包绕角  │
              └────────┬────────────┘
                       ▼
              ┌─────────────────────┐
              │ Occlusion Detection │
              │ · 球形邻域中断检测   │
              │ · 与肿瘤 mask 重叠  │
              │   判定 → 真闭塞 360°│
              └────────┬────────────┘
                       ▼
              输出: 每条血管的连续角度值
              + 三类分类: Class 0 (无接触)
                         Class 1 (≤180°)
                         Class 2 (>180° / 闭塞)
```

### 2.1 模块 1: 分割 (Segmentation Module)

| 维度 | 详情 |
|:-----|:-----|
| **骨干网络** | nnU-Net v2 (自适应配置) (来源: Isensee et al., Nat. Methods 2021) |
| **框架** | PyTorch 1.13.1, CUDA 11.6 |
| **硬件** | NVIDIA A100-PCIE-40GB, 128GB RAM, Intel i7-14700K |
| **OS** | Ubuntu 22.04.5 LTS (Jammy, kernel 6.8.0-59-generic) |
| **输入** | 双期 CE-CT (DICOM → NIfTI), 动脉期/门脉期分别处理 |
| **输出** | 逐体素 Mask: 动脉期 4 类 (PDAC+CA+CHA+SMA), 门脉期 3 类 (PDAC+SMV+PV) |
| **损失函数** | Dice + Cross-Entropy (combined) |
| **学习率** | Poly decay, 起始 lr = 1e-3 |
| **Batch Size** | 2 |
| **验证策略** | 5-fold stratified CV (按血管侵犯类型分层) |

#### 三阶段渐进训练 (Three-Phase Progressive Training)

| 阶段 | 数据规模 | 标注来源 | 策略 | 改进效果 |
|:-----|:---------|:---------|:-----|:---------|
| **Phase 1** | 380 例 | 高年资放射科医生手工标注 | 全监督 (Fully Supervised) | 初始基线; 存在肿瘤勾画不完整、血管断裂 |
| **Phase 2** | +500 例 (共 880) | Phase 1 模型推理 → 放射科医生审核修正 | 半监督 (Semi-Supervised) | 边界歧义改善; 减少假阳性 |
| **Phase 3** | +320 例 (共 1200) | Phase 2 模型推理 → 专家修正 | 专家引导微调 | 解剖连贯性显著提升; 血管分叉处连续性修复 |

> ⚠️ **未披露细节**: 半监督训练的具体伪标签筛选策略 (置信度阈值? Self-Training? Mean Teacher?) 论文未说明。

### 2.2 模块 2: 后处理 (Post-Processing)

| 操作 | 参数 | 目的 |
|:-----|:-----|:-----|
| 体积阈值过滤 | 动脉期 < 415 voxels 剔除; 门脉期 < 450 voxels 剔除 | 消除噪声/假阳性小 Mask |
| 连通域过滤 | `volume_filter.py` (131行), 保留最大连通域 | 确保分割结果解剖学合理 |

> ⚠️ **注意**: 415/450 阈值为"经验确定 (empirically determined)"，未给出系统搜索过程。

### 2.3 模块 3: 中心线提取 (Centerline Extraction)

| 维度 | 详情 |
|:-----|:-----|
| **对应代码** | `extract_centerline.py` (87行) |
| **输入** | 单条血管的 3D binary Mask (NIfTI) |
| **算法** | `skimage.morphology.skeletonize_3d` → 得到骨架体素 → 构建 `networkx.Graph` → 找最长路径 → 有序中心线坐标列表 |
| **输出** | `List[(x, y, z)]` — 沿血管走行的有序 3D 坐标点序列 |
| **关键依赖** | scikit-image, networkx |

### 2.4 模块 4: 3D 包绕角量化 (Angle Quantification)

| 维度 | 详情 |
|:-----|:-----|
| **对应代码** | `angle_quantify.py` (266行) |
| **输入** | 中心线坐标 + 肿瘤 Mask + 血管 Mask |
| **核心算法** | |

**逐中心线点 O 的计算步骤:**

1. **切向量计算**: 取 O 点前后邻域点, 拟合切线方向 $\vec{t}$
2. **法平面生成**: 过点 O, 法向量为 $\vec{t}$ 的 2D 平面 CDEF
3. **Mask 重采样**: 将 tumor_mask 和 vessel_mask **重采样**到法平面上, 得到 2D 截面
4. **交界点提取**: 找到 tumor 和 vessel 截面的边界交点 b, c
5. **质心定位**: tumor 截面的质心 a
6. **角度计算**: ∠bac = arccos[(|ab|² + |ac|² - |bc|²) / (2·|ab|·|ac|)] **(余弦定理)**
7. **汇聚**: 取所有中心线点上的 **最大角** 作为该血管的包绕角

| **输出** | 连续角度值 (0°–360°) + 三类分类 |

### 2.5 模块 5: 闭塞检测 (Occlusion Detection)

| 维度 | 详情 |
|:-----|:-----|
| **对应代码** | `detect_occlusion.py` (216行) |
| **输入** | 血管 Mask + 肿瘤 Mask |
| **算法** | 在血管中心线上检测 Mask 中断点 → 球形邻域 (半径 ⚠️ 未披露) 内检查是否存在肿瘤 Mask 重叠 → 重叠 = 真闭塞 (360°); 不重叠 = 分割伪影 (标记待审) |
| **输出** | Boolean: 是否闭塞; 闭塞位置坐标 |

---

## 3. 数据集构成与临床参与 (Datasets & Clinical Grounding)

### 3.1 数据集

| 数据集 | 规模 | 机构 | 公开/私有 | 时间范围 | 用途 |
|:-------|:-----|:-----|:---------|:---------|:-----|
| 瑞金医院 (内部) | 1,759 例 | SJTU Ruijin Hospital, 上海 | ❌ 私有 | 2018.01–2023.12 | 训练+内部验证 |
| 外部验证 1 | 30 例 | 唐山人民医院 | ❌ 私有 | 2021.01–2022.12 | 外部验证 |
| 外部验证 2 | 85 例 | 无锡二院 | ❌ 私有 | 2022.04–2023.04 | 外部验证 |
| 外部验证 3 | 54 例 | 重庆医大附一院 | ❌ 私有 | 2021.12–2022.11 | 外部验证 |
| 前瞻验证 | 202 例 | SJTU Ruijin Hospital | ❌ 私有 | 2024.01–2024.12 | 前瞻 (Prospective) 验证 |
| **总计** | **2,130 例** | 4 家医院 | - | - | - |

**数据分布**: 79.7% 无接触, 14.7% ≤180°, 5.6% >180°/闭塞 — 高度类别不平衡

### 3.2 临床参与分析

| 环节 | 是否有临床医生参与 | 具体形式 | 评价 |
|:-----|:-------------------|:---------|:-----|
| 数据标注 | ✅ | 高年资放射科医生手工勾画 + 迭代审核修正 | 金标准级 |
| 分割质量评估 | ✅ | 2 位放射科医生对 400 例做 5 分 Likert 量表评分 (ICC 评估) | Cohen's Kappa ≥ 0.689 |
| 前瞻性验证 | ✅ | 202 例连续入组; 模型预测与盲法高/低年资放射科医生评估对比 | 非劣效设计 (Δ<5%) |
| 与医生直接对比 | ✅ | 1 位 20 年经验高年资 + 2 位 6-7 年低年资 | 模型全面超越低年资, 与高年资非劣效 |
| 参考标准 | ✅ | 术中发现 (intraoperative findings) 作为金标准 | 限制: 仅含手术患者 |

### 3.3 核心性能指标

**内部验证 (1,759 例)**:

| 血管 | Class 1 准确率 | Class 2 准确率 | Class 3 准确率 |
|:-----|:--------------|:--------------|:--------------|
| CA | 97.54% | — | 74.29% |
| CHA | 95.87% | — | — |
| SMA | 94.38% | — | — |
| SMV | 93.30% | — | — |
| PV | 90.82% | — | — |

**前瞻验证 (202 例) — vs 放射科医生**:

| 血管 | PAN-VIQ | 低年资 | 高年资 |
|:-----|:--------|:-------|:-------|
| CA | 87.78% | — | ~同等 |
| CHA | **90.22%** | **56.86%** (+33.36%) | ~同等 |
| SMA | 87.43% | — | ~同等 |
| SMV | **88.55%** | **63.44%** (+25.11%) | ~同等 |
| PV | 87.36% | — | ~同等 |

---

## 4. 代码与可复现性 (Reproducibility)

| 维度 | 结论 |
|:-----|:-----|
| **GitHub** | [IMIT-PMCL/PDAC](https://github.com/IMIT-PMCL/PDAC) |
| **模型权重** | ❌ **不可用** — GitHub Releases 为空, README 未提供下载链接 (实审: 2026-04-17) |
| **核心代码** | ✅ 后处理管线 4 个 Python 文件完整可用 (extract_centerline.py, angle_quantify.py, detect_occlusion.py, volume_filter.py) |
| **社区活跃度** | ⚠️ 极低 — 3 Stars, 0 Forks, 0 Issues (截至 2026-04) |
| **依赖链** | nnU-Net v2, PyTorch 1.13.1, SimpleITK, scikit-image, networkx, opencv-python |
| **复现难度** | **高** — 权重不可用意味着无法直接使用分割模块; 但几何后处理代码独立于分割模型, 可直接复用 |

---

## 5. 未披露细节清单 (Undisclosed Details Inventory)

| # | 未披露内容 | 影响程度 | 为什么重要 |
|:--|:----------|:---------|:----------|
| 1 | 半监督训练的具体伪标签策略 (Self-Training / Mean Teacher / 置信度阈值?) | 🟡 中 | 直接影响 Phase 2→3 训练效果的复现 |
| 2 | 体积阈值 415/450 的搜索过程和敏感性分析 | 🟡 中 | 不同数据集可能需要不同阈值 |
| 3 | 闭塞检测的球形邻域半径 | 🟡 中 | 影响真闭塞 vs 分割伪影的区分能力 |
| 4 | 中心线提取时 networkx 图的边权重/排序策略 | 🟢 低 | 影响中心线平滑性, 但大概率是标准最长路径 |
| 5 | nnU-Net 自动配置的 patch size, 具体网络深度 | 🟢 低 | nnU-Net 自配置, 理论上可复现 |
| 6 | 血管变异 (15.86% 病例有血管变异) 的处理策略 — 是否有专门的分支? | 🟡 中 | 论文仅说"性能无显著差异", 但未说明是否有特殊处理 |
| 7 | 新辅助治疗 (NAT) 后病例的 Ground Truth 确定方式 | 🔴 高 | NAT 改变血管周围解剖, 术中发现可能不反映原始侵犯状态 |

---

## 6. 对 PancreasMDT 项目的迁移价值 (Transferable Insights)

### ✅ 可直接复用

| 模块 | 复用方式 |
|:-----|:---------|
| `angle_quantify.py` | 作为 `Geometry_Detector` 的核心引擎, 接收任何分割模型输出的 Mask |
| `extract_centerline.py` | 作为 `Centerline_Extractor` 工具, 输入 vessel Mask → 输出有序 3D 坐标 |
| `detect_occlusion.py` | 作为闭塞判定逻辑 |
| 双期独立处理策略 | 动脉/静脉分别训练分割模型是合理的工程选择 |

### 🔧 需要改造

| 方面 | 改造需求 |
|:-----|:---------|
| 分割模型 | PAN-VIQ 权重不可用, 需基于 PanTS/AbdomenAtlas 3.0 自训 nnU-Net |
| 法平面采样 | 当前实现可能对弯曲血管段 (如 SMA 弯曲处) 产生采样偏差, 需加入曲率自适应 |
| 闭塞检测 | 球形半径需参数化, 最好可配置 |

### ❌ 明确不适用

| 方面 | 原因 |
|:-----|:-----|
| 纯几何管线的"零降级"设计 | PAN-VIQ 分割失败时整条管线崩溃, **这正是我们 Volume-Topology Agent 的创新空间** — 引入 Agent 推理降级策略 |
| 仅输出角度值 | 临床需要的是 R/BR/LA 分类, 需要结合 NCCN 指南做决策映射 |

---

## 🔗 Connections

- **复用于:** [PancreasMDT Architecture] — `Centerline_Extractor`, `Geometry_Detector`
- **GitHub:** https://github.com/IMIT-PMCL/PDAC
- **DOI:** https://doi.org/10.1038/s41746-025-02260-3
- **PDF:** https://www.nature.com/articles/s41746-025-02260-3.pdf
- **Supplementary:** https://static-content.springer.com/esm/art%3A10.1038%2Fs41746-025-02260-3/MediaObjects/41746_2025_2260_MOESM1_ESM.pdf
