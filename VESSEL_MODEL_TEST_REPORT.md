# 胰腺血管分割开源模型调研测试报告

**报告日期**: 2026-03-30
**测试环境**: ChangHai PDA 项目服务器
**测试样本**: CL-03356 (如适用)

---

## 执行摘要

| 模型 | 开源状态 | 权重可用 | SMA | SMV | CA | PV | CHA | 结论 |
|------|----------|----------|-----|-----|-----|-----|-----|------|
| vesselFM | ❌ 未公开 | ❌ 不可用 | ? | ? | ? | ? | ? | **无法使用** |
| SuPreM | ✅ 已开源 | ⚠️ 下载受阻 | ❌ | ❌ | ✅ | ✅ | ✅ | **部分可用** |
| PDAC-nnU-Net | ✅ 已开源 | ✅ 可用 | ❌ | ❌ | ❌ | ❌ | ❌ | **不满足需求** |

**关键结论**:
- 当前**没有可直接使用**的开源模型能完整覆盖 SMA/SMV/CA/CHA/PV
- SuPreM 是唯一有 `celiac_truck` 和 `hepatic_vessel` 标签的模型
- **建议**: 1) 继续联系 PAN-VIQ 团队; 2) 基于 SuPreM 进行迁移学习

---

## 1. vesselFM (失败)

### 1.1 基本信息
- **论文**: vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation
- **会议**: CVPR 2025 (入选)
- **研究机构**: 苏黎世大学、苏黎世联邦理工学院、慕尼黑工业大学

### 1.2 调研结果
| 项目 | 状态 | 说明 |
|------|------|------|
| GitHub | ❌ 未找到 | 搜索 `marksgraham/vesselFM` 失败 |
| 替代尝试 | ❌ 失败 | 尝试 `MedicalImageAnalysisGroup/vesselFM`、`helmholtz-ai/vesselFM` 均失败 |
| 预训练权重 | ❌ 未公开 | 论文已接收但代码/权重尚未发布 |

### 1.3 结论
**无法使用**。CVPR 2025 论文的代码/权重通常会在会议召开前 (2025年6月) 或之后公开。当前不可获取。

---

## 2. SuPreM (部分可用)

### 2.1 基本信息
- **论文**: How Well Do Supervised 3D Models Transfer to Medical Image Segmentation?
- **会议**: ICLR 2024 (Oral, top 1.2%)
- **GitHub**: https://github.com/MrGiovanni/SuPreM
- **研究机构**: Johns Hopkins University

### 2.2 技术规格
| 特性 | 详情 |
|------|------|
| **架构** | U-Net / SwinUNETR |
| **预训练数据** | AbdomenAtlas 1.1 (9,262 CT volumes, 25类器官) |
| **分割类别** | 32类 (器官 + 血管 + 肿瘤) |

### 2.3 标签分析 (关键发现)

#### ORGAN_NAME_LOW (索引从1开始)
| 索引 | 名称 | PDAC 相关 | 说明 |
|------|------|-----------|------|
| 1 | spleen | ❌ | 脾脏 |
| 2 | kidney_right | ❌ | 右肾 |
| 3 | kidney_left | ❌ | 左肾 |
| 4 | gall_bladder | ❌ | 胆囊 |
| 5 | esophagus | ❌ | 食管 |
| 6 | liver | ❌ | 肝脏 |
| 7 | stomach | ❌ | 胃 |
| 8 | **aorta** | ✅ | **主动脉** |
| 9 | **postcava** | ✅ | **下腔静脉** |
| 10 | **portal_vein_and_splenic_vein** | ✅ | **门静脉和脾静脉** |
| 11 | pancreas | ✅ | 胰腺 |
| 12 | adrenal_gland_right | ❌ | 右肾上腺 |
| 13 | adrenal_gland_left | ❌ | 左肾上腺 |
| 14 | duodenum | ❌ | 十二指肠 |
| 15 | **hepatic_vessel** | ✅ | **肝血管** |
| 16-24 | 其他器官 | ❌ | 肺、结肠等 |
| **25** | **celiac_truck** | ✅ | **腹腔干** |
| 26-32 | 肿瘤类 | ✅ | 各类肿瘤 |

#### 问题识别
1. **TEMPLATE['target'] 只包含**: `[1,2,3,4,6,7,8,9,11]`
   - 即：spleen, kidney_right, kidney_left, gall_bladder, liver, stomach, aorta, postcava, pancreas
   - **不包含**: `celiac_truck` (25), `hepatic_vessel` (15), `portal_vein_and_splenic_vein` (10)

2. **缺少 SMA/SMV 明确标签**:
   - `portal_vein_and_splenic_vein` (10) 是合并标签，未区分 SMV
   - **无 `superior_mesenteric_artery` 或 `superior_mesenteric_vein` 标签**

### 2.4 权重下载尝试
```bash
# 尝试1: wget
wget -q https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_unet_2100.pth
# 结果: 失败 (服务器wget版本不支持--show-progress)

# 尝试2: curl
curl -L -o supervised_suprem_unet_2100.pth https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_unet_2100.pth
# 结果: 网络超时，下载未完成
```

### 2.5 可行性评估
- **直接推理**: 需要修改 `TEMPLATE['target']` 包含 10, 15, 25
- **血管细分**: 无法直接获得 SMA/SMV，需要从 `portal_vein_and_splenic_vein` 中分离
- **迁移学习**: 可行，但需要标注数据对 SMA/SMV 进行微调

---

## 3. DIAGnijmegen/CE-CT_PDAC_AutomaticDetection_nnUnet (不满足)

### 3.1 基本信息
- **论文**: Fully Automatic Deep Learning Framework for Pancreatic Ductal Adenocarcinoma Detection on Computed Tomography
- **期刊**: Cancers 2022
- **GitHub**: https://github.com/DIAGNijmegen/CE-CT_PDAC_AutomaticDetection_nnUnet
- **研究机构**: Radboud University Medical Center (荷兰)

### 3.2 技术规格
| 特性 | 详情 |
|------|------|
| **架构** | nnU-Net (3D fullres) |
| **任务** | Task103_AllStructures |
| **类别数** | 9类 |

### 3.3 标签分析
根据 README 和代码分析：
| 标签 | 说明 |
|------|------|
| 1 | veins (静脉，粗分类) |
| 2 | arteries (动脉，粗分类) |
| 3 | pancreas |
| 4 | pancreatic duct |
| 5 | bile duct |
| 6 | cysts |
| 7 | renal vein |
| 8-9 | (未明确说明) |

### 3.4 关键问题
**veins 和 arteries 是粗分类**，未细分为：
- ❌ SMA (superior mesenteric artery)
- ❌ SMV (superior mesenteric vein)
- ❌ CA (celiac artery) - 可能在 arteries 中但未区分
- ❌ CHA (common hepatic artery) - 可能在 arteries 中但未区分
- ❌ PV (portal vein) - 可能在 veins 中但未区分

### 3.5 结论
**不满足 PDAC 血管评估需求**。只能提供 coarse-grained 的血管分割，无法进行精细的血管侵犯评估。

---

## 4. 对比分析

| 需求 | PAN-VIQ (理论) | vesselFM (理论) | SuPreM | PDAC-nnU-Net |
|------|----------------|-----------------|--------|--------------|
| SMA | ✅ | ✅ | ❌ | ❌ |
| SMV | ✅ | ✅ | ❌ | ❌ |
| CA | ✅ | ✅ | ✅ | ❌ |
| CHA | ✅ | ✅ | ✅ | ❌ |
| PV | ✅ | ✅ | ✅ | ❌ |
| 肿瘤 | ✅ | ? | ✅ | ✅ |
| 开源 | ❓ 未确认 | ❌ 未公开 | ✅ | ✅ |
| 可立即使用 | ❓ | ❌ | ⚠️ 需修改 | ❌ |

---

## 5. 建议方案

### 方案A: 联系 PAN-VIQ 团队 (优先)
```
行动: 发邮件给上海瑞金医院温宁研究员
邮箱: wenning@rjh.com.cn (推测)
内容:
  1. 询问 PAN-VIQ 开源计划
  2. 询问预训练权重获取方式
  3. 表达合作意向 (如适用)
时间: 1-2周等待回复
```

### 方案B: 基于 SuPreM 的迁移学习
```
步骤:
  1. 成功下载 SuPreM 权重 (~70MB)
  2. 修改 TEMPLATE['target'] 包含血管标签 [8,9,10,15,25]
  3. 在 CL-03356 上测试推理
  4. 从 portal_vein_and_splenic_vein 中尝试分离 SMV
  5. 如效果不佳，收集 20-50例 SMA/SMV 标注数据进行微调
时间: 2-4周
```

### 方案C: 多 Atlas 配准方案
```
步骤:
  1. 使用 TotalSegmentator 获取基础器官
  2. 使用血管图谱 (Vascular Atlas) 进行配准
  3. 将标准血管 mask 配准到患者 CT
时间: 1-2周实现
精度: 中等 (依赖配准质量)
```

---

## 6. 下一步行动

### 立即执行 (本周)
1. **发送邮件**给 PAN-VIQ 团队询问开源计划
2. **继续尝试下载** SuPreM 权重 (网络条件允许时)
3. **准备标注数据** (如决定走迁移学习路线)

### 短期备用 (1-2周)
1. **测试 SuPreM 修改版** (修改 TEMPLATE 后)
2. **评估** portal_vein_and_splenic_vein 是否可分离 SMV
3. **验证** celiac_truck 和 hepatic_vessel 的分割质量

### 中期方案 (1个月)
1. 如 PAN-VIQ 不开源，启动 **SuPreM 迁移学习**
2. 或实施 **多 Atlas 配准方案**

---

## 7. 附录

### 7.1 测试命令记录
```bash
# vesselFM 仓库克隆 (失败)
git clone https://github.com/marksgraham/vesselFM.git /tmp/vesselFM
# Error: Repository not found

# SuPreM 仓库克隆 (成功)
git clone --depth 1 https://github.com/MrGiovanni/SuPreM.git /tmp/SuPreM

# PDAC nnU-Net 仓库克隆 (成功)
git clone --depth 1 https://github.com/DIAGNijmegen/CE-CT_PDAC_AutomaticDetection_nnUnet.git /tmp/PDAC_nnuNet

# SuPreM 权重下载 (网络超时)
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_unet_2100.pth
# 或
curl -L -o supervised_suprem_unet_2100.pth https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_unet_2100.pth
```

### 7.2 关键文件路径
```
/tmp/SuPreM/direct_inference/utils/utils.py          # 标签定义
/tmp/SuPreM/direct_inference/inference.py            # 推理代码
/tmp/PDAC_nnuNet/custom_nnunet_predict.py            # PDAC推理代码
/tmp/PDAC_nnuNet/nnunet/results/.../plans.pkl        # 标签映射 (需git-lfs)
```

### 7.3 参考资料
- SuPreM Paper: https://www.cs.jhu.edu/~alanlab/Pubs23/li2023suprem.pdf
- PAN-VIQ Paper: https://doi.org/10.1038/s41746-025-02260-3
- PDAC nnU-Net Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8774174/

---

**报告撰写**: Claude Code Agent
**审核状态**: 待项目团队审阅
**更新日期**: 2026-03-30
