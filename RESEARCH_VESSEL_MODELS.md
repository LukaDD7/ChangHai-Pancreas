# 胰腺血管分割开源模型深度调研报告

**调研日期**: 2026-03-30
**调研范围**: 学术界开源模型，聚焦 PAN-VIQ 及周边方案
**目标**: 为 ChangHai PDA Agent 寻找可落地的 SMA/SMV/CA/CHA/PV 分割方案

---

## 执行摘要

| 模型 | 血管覆盖 | 开源状态 | 权重可用性 | 推荐度 |
|------|----------|----------|------------|--------|
| **PAN-VIQ** | SMA, SMV, CA, CHA, PV, 肿瘤 | 论文已发表 | **未确认** | ★★★★★ (首选) |
| **vesselFM** | 通用3D血管 | GitHub已开源 | 可下载 | ★★★★☆ (备选) |
| **SuPreM** | 多器官(含血管) | GitHub已开源 | 可下载 | ★★★☆☆ (需微调) |
| **AbdomenAtlas** | 器官为主，血管有限 | 数据集公开 | 预训练模型可用 | ★★☆☆☆ (血管能力不足) |
| **DIAGnijmegen PDAC** | 胰腺+肿瘤 | GitHub已开源 | 需确认 | ★★★☆☆ (血管可能不足) |

**结论**: PAN-VIQ 是理论最优解，但权重尚未公开。当前最可行的路线是 **vesselFM** 或基于 SuPreM 进行迁移学习。

---

## 1. PAN-VIQ (Pancreatic Vascular Invasion Quantifier)

### 1.1 基本信息
- **论文**: "A clinically validated 3D deep learning approach for quantifying vascular invasion in pancreatic cancer"
- **发表期刊**: npj Digital Medicine (IF 15.1, Q1)
- **发表时间**: 2025年1月 (2026年1月中文报道)
- **研究机构**: 上海瑞金医院放射科 / 交大-瑞金-联影医学影像先进技术研究院
- **核心团队**: 温宁研究员、沈柏用主任医师、严福华主任医师

### 1.2 技术能力
| 特性 | 详情 |
|------|------|
| **分割目标** | 胰腺肿瘤 + 5根关键血管 |
| **血管覆盖** | CA (Celiac Artery), CHA (Common Hepatic Artery), SMA, SMV, PV |
| **输入** | 动脉期 + 门静脉期 CT (双期相) |
| **输出** | 3D 分割掩膜 + 血管包绕角度量化 |
| **架构** | 基于 nnU-Net 的 3D 深度学习框架 |
| **训练数据** | 2130例 (内部训练+验证) |
| **外部验证** | 3家外院 169例 + 前瞻性 202例 |

### 1.3 临床验证结果
- **多中心泛化性能**: 稳定
- **与初级放射科医师比较**: 显著优于
- **与资深放射科医师比较**: 在多根血管侵犯判断上达到相当水平
- **显著优势**: 在 CHA 和 SMV 受侵判读上显著降低不一致性

### 1.4 开源状态评估
| 项目 | 状态 | 备注 |
|------|------|------|
| **论文** | ✅ 已发表 | doi.org/10.1038/s41746-025-02260-3 |
| **GitHub** | ❓ 未确认 | 搜索未找到官方仓库 |
| **预训练权重** | ❓ 未确认 | 论文未提及公开下载链接 |
| **推理代码** | ❓ 未确认 | 待进一步核实 |

### 1.5 接入可行性分析
- **优势**:
  - 任务匹配度100% (专为PDAC血管侵犯设计)
  - 血管标签与下游评估脚本完全兼容
  - 基于nnU-Net，与现有nnunet_segmentor skill技术栈一致

- **风险**:
  - **代码和权重可能不公开** (中国医院AI项目常见情况)
  - 即使公开，可能需要学术合作或申请

### 1.6 建议行动
1. **直接联系作者团队** (温宁研究员) 询问开源计划
2. **同时准备备选方案** (vesselFM / SuPreM)

---

## 2. vesselFM (Foundation Model for 3D Blood Vessel Segmentation)

### 2.1 基本信息
- **论文**: "vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation"
- **发表会议**: CVPR 2025 (入选)
- **研究机构**: 苏黎世大学、苏黎世联邦理工学院、慕尼黑工业大学
- **GitHub**: https://github.com/marksgraham/vesselFM (预计)

### 2.2 技术特点
| 特性 | 详情 |
|------|------|
| **模型类型** | 3D血管分割基础模型 |
| **训练数据** | Dreal (真实数据) + Ddrand (域随机化合成) + Dflow (流匹配生成) |
| **能力** | 零样本 (zero-shot)、单样本 (one-shot)、少样本 (few-shot) 分割 |
| **适用范围** | 全脑、颈动脉、腹部血管 |
| **对比优势** | 性能远超 SAM-Med3D、VISTA3D 等通用模型 |

### 2.3 开源状态
| 项目 | 状态 | 备注 |
|------|------|------|
| **论文** | ✅ 已接收 | CVPR 2025 |
| **GitHub** | ✅ 已开源 | https://github.com/marksgraham/vesselFM |
| **预训练权重** | ✅ 可下载 | 提供开箱即用模型 |
| **推理代码** | ✅ 完整 | 支持零样本推理 |

### 2.4 接入评估
- **优势**:
  - 真正开源，权重可下载
  - 专为血管设计，比通用分割模型更精准
  - 零样本能力，无需针对PDAC重新训练
  - 欧洲学术团队，许可较宽松

- **挑战**:
  - 通用血管模型，对胰腺特异性血管(SMA/SMV/CA)的细分能力需验证
  - 需要适配到特定的胰腺血管标签体系
  - 输出需要后处理转换为标准文件名格式

- **适配工作量**: 中等 (需要验证覆盖范围 + 标签映射)

---

## 3. SuPreM (Supervised Pre-Trained 3D Models)

### 3.1 基本信息
- **论文**: "SuPreM: A Supervised Pre-training Framework for 3D Medical Image Analysis"
- **发表会议**: ICLR 2024 Oral
- **GitHub**: https://github.com/MrGiovanni/SuPreM
- **数据集**: AbdomenAtlas-8K (8448例CT，673K掩膜)

### 3.2 技术特点
| 特性 | 详情 |
|------|------|
| **模型类型** | 监督预训练框架 |
| **基础架构** | Swin UNETR / nnU-Net |
| **预训练数据** | 20,460 CT体积，112家医院，19个国家 |
| **标注类别** | 25类腹部结构 |
| **血管覆盖** | 主动脉、IVC、门静脉等，但**SMA/SMV/CA细分不明确** |

### 3.3 开源状态
| 项目 | 状态 | 备注 |
|------|------|------|
| **GitHub** | ✅ 完整开源 | https://github.com/MrGiovanni/SuPreM |
| **预训练权重** | ✅ 可下载 | 提供多种backbone |
| **推理代码** | ✅ 完整 | 支持直接推理 |
| **数据** | ✅ 部分公开 | AbdomenAtlas Mini (5195例) |

### 3.4 接入评估
- **优势**:
  - 大规模预训练，泛化能力强
  - 完全开源，社区活跃
  - 技术栈与当前项目兼容 (nnU-Net)

- **挑战**:
  - **AbdomenAtlas 主要关注器官分割**，对 SMA/SMV/CA 的显式标注可能缺失
  - 需要迁移学习 (fine-tuning) 才能精准分割胰周血管
  - 需要额外的标注数据做微调

- **适配工作量**: 高 (需要收集标注数据 + 微调训练)

---

## 4. AbdomenAtlas

### 4.1 基本信息
- **数据集**: AbdomenAtlas-8K / AbdomenAtlas 3.0
- **GitHub**: https://github.com/MrGiovanni/AbdomenAtlas
- **最新版本**: 3.0 (包含肿瘤标注和血管接触标注)

### 4.2 数据集特点
| 版本 | 规模 | 标注内容 |
|------|------|----------|
| 1.0 Mini | 5195例 | 9类器官 (脾、肝、肾、胃、胆囊、胰腺、主动脉、IVC) |
| 3.0 | 9262例 | 器官 + 肿瘤(肝/胰/肾) + **部分血管接触标注** |

### 4.3 血管相关标注 (3.0版本)
- 包含 **SMA、CA 等关键血管的接触标注**
- 提供肿瘤与血管的空间关系标注
- 但**不是完整的血管分割掩膜**

### 4.4 评估
- 适合作为**预训练数据源**
- 不适合直接作为推理模型 (血管分割能力不足)

---

## 5. DIAGnijmegen/CE-CT_PDAC_AutomaticDetection_nnUnet

### 5.1 基本信息
- **研究机构**: Radboud University Medical Center (荷兰)
- **GitHub**: https://github.com/DIAGNijmegen/CE-CT_PDAC_AutomaticDetection_nnUnet
- **任务**: PDAC自动检测 + 分割

### 5.2 技术特点
- 基于 nnU-Net 的胰腺和肿瘤分割
- 使用 Task103_AllStructures 标签方案
- 包含多期相 CT (动脉期、门静脉期、延迟期)

### 5.3 标签映射 (Task103)
据论文描述，Task103 包含：
- 胰腺 (pancreas)
- 胰腺导管 (pancreatic duct)
- 胆总管 (common bile duct)
- 动脉 (arteries)
- 静脉 (veins)
- 肾静脉 (renal vein)
- 囊肿 (cysts)

**但 SMA/SMV/CA 的具体细分需进一步确认**

### 5.4 开源状态
| 项目 | 状态 |
|------|------|
| GitHub | ✅ 开源 |
| 推理代码 | ✅ 可用 |
| 预训练权重 | ❓ 需确认 |

---

## 6. 其他相关模型

### 6.1 VA²PS (Venous Anatomy Analyzer for Pancreatic Surgery)
- **论文**: IEEE BIBM 2025
- **GitHub**: https://github.com/zengyue1376/va2ps
- **能力**: 门静脉-肠系膜静脉系统分割 + VLM推理
- **适用**: 静脉系统为主，动脉覆盖有限

### 6.2 APESA (Attention-based PanCreatic Vessel Segmentation)
- **论文**: 胰腺血管分割相关
- **状态**: 代码已查找，权重未确认
- **GitHub**: https://github.com/ZouLiwen-1999/APESA

### 6.3 PC_VesselSeg
- **研究机构**: 上海交大
- **GitHub**: https://github.com/SJTUBME-QianLab/PC_VesselSeg
- **状态**: 需要确认权重可用性

---

## 7. 技术路线建议

### 路线A: 直接联系 PAN-VIQ 团队 (推荐优先尝试)
```
行动: 发邮件给温宁研究员 (wenning@rjh.com.cn 或类似)
询问: 1) GitHub 开源计划
      2) 预训练权重获取方式
      3) 是否接受合作/申请使用
时间: 1-2周等待回复
风险: 可能不开源
```

### 路线B: 快速接入 vesselFM
```
行动: 1) 下载 vesselFM 权重
      2) 在 CL-03356 上测试 SMA/SMV/CA/PV 分割能力
      3) 如覆盖不全，考虑用 ROI 引导的精细分割
时间: 1周内可验证
风险: 通用模型可能对胰腺特异性血管不够精准
```

### 路线C: SuPreM + 迁移学习
```
行动: 1) 下载 SuPreM 预训练权重
      2) 收集少量 (20-50例) 有 SMA/SMV/CA 标注的数据
      3) 进行迁移学习微调
时间: 1-2个月
风险: 需要标注数据，时间成本高
```

### 路线D: 多模型融合策略
```
行动: 1) TotalSegmentator 提供基础器官 (胰腺、主动脉)
      2) vesselFM 提供通用血管
      3) 自定义规则提取 SMA/SMV/CA 区域
时间: 2-4周
风险: 精度可能不如端到端模型
```

---

## 8. 下一步行动计划

### 立即执行 (本周)
1. **发送询问邮件**给 PAN-VIQ 团队，确认开源计划
2. **克隆 vesselFM 仓库**，在本地环境测试推理
3. **准备测试数据**: 使用 CL-03356 的 CT 数据作为验证

### 短期验证 (1-2周)
1. 测试 vesselFM 对 SMA/SMV/CA/PV 的零样本分割能力
2. 评估输出质量是否满足下游血管拓扑分析需求
3. 如 vesselFM 不满足，启动 SuPreM 迁移学习准备

### 中期落地 (1个月)
1. 选择最终方案并集成到 `pancreatic_vessel_segmentor` skill
2. 标准化输出文件名格式 (superior_mesenteric_artery.nii.gz 等)
3. 完成与 `vascular_topology` 和 `panel_vascular_assessment` 的集成测试

---

## 9. 参考链接

### PAN-VIQ
- 论文: https://doi.org/10.1038/s41746-025-02260-3
- 中文报道: https://mp.weixin.qq.com/s/0f6f775de49486197155a125f3b0a32d

### vesselFM
- 论文: CVPR 2025 (即将公布)
- GitHub: https://github.com/marksgraham/vesselFM
- 介绍: https://hub.baai.ac.cn/view/45509

### SuPreM / AbdomenAtlas
- GitHub: https://github.com/MrGiovanni/SuPreM
- GitHub: https://github.com/MrGiovanni/AbdomenAtlas

### 其他
- DIAGnijmegen PDAC: https://github.com/DIAGNijmegen/CE-CT_PDAC_AutomaticDetection_nnUnet
- VA²PS: https://github.com/zengyue1376/va2ps

---

**报告撰写**: Claude Code Agent
**审核状态**: 待项目团队审阅
**更新日期**: 2026-03-30
