# Pan-Agent 胰腺多疾病智能诊断系统技术路线图

**版本**: v1.0
**日期**: 2026-03-30
**目标**: 构建从CT输入到精准分型的端到端AI诊断系统

---

## 一、系统整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Pan-Agent 智能诊断系统                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input Layer                                                                │
│  ├── DICOM/NIfTI CT (动脉期+静脉期)                                          │
│  ├── 患者病历 (症状/实验室/病史)                                              │
│  └── 既往影像 (随访对比)                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Perception Layer (器官-病灶-血管感知)                                         │
│  ├── 胰腺分割 (TotalSegmentator/SuPreM)                                      │
│  ├── 病灶检测与分割 (nnU-Net/MedSAM)                                         │
│  ├── 血管分割 (SMA/SMV/CA/CHA/PV - 待解决)                                   │
│  └── 多期相配准 (动脉期/静脉期/延迟期)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Cognition Layer (疾病认知与分型)                                             │
│  ├── 实性 vs 囊性分类 (影像组学+VLM)                                         │
│  ├── 疾病特异性分型 (PDAC/pNET/SCN/MCN/IPMN/SPN/AP/CP)                        │
│  ├── 严重程度评估 (NCCN/WHO/Atlanta分级)                                     │
│  └── 恶性风险预测 (多模态融合模型)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Decision Layer (临床决策支持)                                               │
│  ├── 可切除性评估 (血管侵犯角度计算)                                          │
│  ├── 治疗建议生成 (指南匹配)                                                 │
│  ├── 随访计划制定 (时间间隔/检查项目)                                         │
│  └── 预后预测模型 (生存分析)                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Output Layer                                                                │
│  ├── 结构化报告 (符合指南标准)                                               │
│  ├── 三维可视化 (肿瘤-血管关系)                                              │
│  └── MDT讨论摘要 (关键决策点)                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、疾病分型技术路线详解

### 2.1 第一层：实性 vs 囊性鉴别

#### 2.1.1 临床指南标准

| 特征 | 实性肿瘤 | 囊性肿瘤 |
|------|----------|----------|
| **CT密度** | >20 HU (软组织密度) | <20 HU (液体密度) |
| **强化模式** | 动脉期/静脉期明显强化 | 囊壁/分隔强化，囊液不强化 |
| **形态** | 实性占位，边界不规则多见 | 圆形/类圆形，边界清晰 |
| **钙化** | 少见，散在 | SCN中央瘢痕钙化特征性 |
| **胰管** | 截断/包绕 | 扩张/沟通 |

#### 2.1.2 技术实现方案

```python
# 实性-囊性分类模块
class SolidCysticClassifier:
    """
    基于影像组学+深度学习的实囊分类器
    """

    def __init__(self):
        self.ct_threshold = 20  # HU阈值
        self.enhancement_threshold = 20  # 强化阈值

    def extract_features(self, roi_mask, ct_arterial, ct_venous):
        """
        提取关键影像组学特征
        """
        features = {
            # 1. 密度特征
            'mean_hu': np.mean(ct_arterial[roi_mask]),
            'std_hu': np.std(ct_arterial[roi_mask]),
            'min_hu': np.min(ct_arterial[roi_mask]),

            # 2. 强化特征
            'enhancement': np.mean(ct_venous[roi_mask]) - np.mean(ct_arterial[roi_mask]),

            # 3. 纹理特征 (GLCM/GLRLM)
            'contrast': self.compute_glcm_contrast(ct_arterial, roi_mask),
            'homogeneity': self.compute_glcm_homogeneity(ct_arterial, roi_mask),

            # 4. 形态特征
            'volume': np.sum(roi_mask) * voxel_volume,
            'sphericity': self.compute_sphericity(roi_mask),
            'surface_area': self.compute_surface_area(roi_mask),
        }
        return features

    def classify(self, features):
        """
        基于规则的初级分类 + ML精细分类
        """
        # 规则1: CT值<20 HU → 囊性
        if features['mean_hu'] < 20 and features['enhancement'] < 10:
            return 'Cystic', 0.95

        # 规则2: 明显强化(>40 HU) → 实性
        if features['enhancement'] > 40:
            return 'Solid', 0.90

        # 模糊情况: 使用XGBoost分类器
        return self.ml_classifier.predict(features)
```

#### 2.1.3 关键注意事项

| 陷阱 | 解决方案 | 技术细节 |
|------|----------|----------|
| **等密度PDAC** | 窄窗增强(W:150) | 窄窗提高20 HU范围→34灰度级 |
| **囊性变实性肿瘤** | 囊实比计算 | 囊性成分>50%单独标注 |
| **出血性囊肿** | 多期相分析 | 出血灶动脉期高密度 |
| **假性囊肿** | 病史整合 | 胰腺炎病史+无强化壁结节 |

---

### 2.2 第二层：实性肿瘤分型 (PDAC vs pNET)

#### 2.2.1 临床指南鉴别标准 (NCCN 2025 + WHO 2024)

| 鉴别点 | PDAC | pNET |
|--------|------|------|
| **流行病学** | 老年(>60岁)，男=女 | 任何年龄，女性略多 |
| **症状** | 腹痛、黄疸、消瘦 | 激素综合征(功能型)/无症状(无功能型) |
| **CT强化** | 乏血供(动脉期低强化) | 富血供(动脉期明显强化) |
| **边界** | 浸润性，边界模糊 | 膨胀性，边界清晰 |
| **胰管** | 截断征(+)，上游扩张 | 推压为主，较少截断 |
| **血管侵犯** | 早期包绕血管 | 推压为主，晚期侵犯 |
| **钙化** | 罕见 | 可见(长期病变) |
| **CA19-9** | 升高(>37 U/mL) | 正常或轻度升高 |
| **CgA/Syn** | 阴性 | 阳性(病理确诊) |

#### 2.2.2 强化模式量化分析

```python
class EnhancementAnalyzer:
    """
    多期相强化曲线分析
    """

    def analyze_curve(self, roi_mask, ct_pre, ct_arterial, ct_venous, ct_delayed):
        """
        生成时间-密度曲线(TDC)
        """
        # 计算各期相平均CT值
        hu_pre = np.mean(ct_pre[roi_mask])
        hu_arterial = np.mean(ct_arterial[roi_mask])
        hu_venous = np.mean(ct_venous[roi_mask])
        hu_delayed = np.mean(ct_delayed[roi_mask])

        # 关键指标
        arterial_enhancement = hu_arterial - hu_pre
        venous_enhancement = hu_venous - hu_pre

        # 强化模式分类
        if arterial_enhancement > 50 and arterial_enhancement > venous_enhancement:
            pattern = 'Arterial_hyperenhancement'  # pNET典型
            washout = arterial_enhancement - (hu_delayed - hu_pre)

        elif venous_enhancement > arterial_enhancement and arterial_enhancement < 30:
            pattern = 'Delayed_enhancement'  # PDAC典型
            washout = None

        else:
            pattern = 'Iso/Hypoenhancement'  # 需进一步评估

        return {
            'pattern': pattern,
            'arterial_enhancement': arterial_enhancement,
            'venous_enhancement': venous_enhancement,
            'washout': washout,
            'curve_type': self.classify_tdc_type([hu_pre, hu_arterial, hu_venous, hu_delayed])
        }
```

#### 2.2.3 技术实现架构

```
实性肿瘤鉴别诊断流程
│
├─ 1. 强化特征分析
│   ├─ 动脉期强化值 >50 HU? → pNET可能性高
│   └─ 静脉期为主强化? → PDAC可能性高
│
├─ 2. 形态学特征
│   ├─ 边界清晰+圆形 → pNET
│   └─ 浸润性+不规则 → PDAC
│
├─ 3. 胰管关系
│   ├─ 双管征/胰管截断 → PDAC
│   └─ 胰管推压扩张 → pNET
│
├─ 4. 血管关系 (关键)
│   ├─ SMA/SMV/CA侵犯 → PDAC典型
│   └─ 血管推压无侵犯 → pNET典型
│
└─ 5. 临床信息整合
    ├─ 激素症状? → 功能性pNET
    ├─ CA19-9>1000? → PDAC
    └─ 年龄<40+女性? → SPN可能
```

#### 2.2.4 VLM (视觉语言模型) 提示设计

```python
pdac_vs_pnet_prompt = """
You are a radiologist specializing in pancreatic imaging.
Analyze this CT scan and differentiate between PDAC and pNET.

Key features to evaluate:
1. Enhancement pattern: Arterial phase enhancement compared to pancreatic parenchyma
2. Vessel relationship: Encasement vs displacement of SMA/SMV/PV
3. Ductal involvement: Double duct sign, abrupt cutoff vs displacement
4. Morphology: Infiltrative irregular vs well-circumscribed round

Provide your assessment in this format:
- Primary diagnosis: [PDAC/pNET/Indeterminate]
- Confidence: [0-100%]
- Key supporting features: [list]
- Differential considerations: [list]
- Recommended next step: [Biochemistry/Pathology/Follow-up]
"""
```

---

### 2.3 第三层：囊性肿瘤四分型

#### 2.3.1 临床指南鉴别要点

| 特征 | SCN | MCN | IPMN | SPN |
|------|-----|-----|------|-----|
| **好发人群** | 老年女性 | 中年女性 | 老年男性 | 青年女性 |
| **位置** | 任何部位 | 体尾部 | 头体部 | 体尾部 |
| **形态** | 分叶状多囊 | 类圆形厚壁 | 杵状指/葡萄串 | 大囊实性 |
| **囊数** | >6小囊 | <6囊 | 单/多房 | 单/多房 |
| **壁结节** | 无 | 可有 | 可有(恶性征) | 可有 |
| **钙化** | 中央瘢痕 | 周边 | 无 | 无 |
| **胰管关系** | 压迫 | 压迫 | **相通** | 压迫 |
| **强化** | 无/轻度 | 壁强化 | 壁结节强化 | 实性部分强化 |

#### 2.3.2 IPMN关键鉴别点

```python
class IPMNDiagnoser:
    """
    IPMN分型与恶性风险评估
    """

    def classify_type(self, ct_pancreas, duct_mask):
        """
        IPMN分型诊断
        """
        # 1. 测量主胰管直径
        main_duct_diameter = self.measure_duct_diameter(duct_mask)

        # 2. 检测囊性病变与胰管关系
        cyst_connected = self.check_duct_communication(
            ct_pancreas, cyst_mask, duct_mask
        )

        # 3. 分型判断
        if main_duct_diameter >= 10:  # mm
            if cyst_connected:
                return 'Mixed_type_IPMN', 'HIGH_RISK'
            else:
                return 'Main_duct_IPMN', 'HIGH_RISK'
        else:
            if cyst_connected:
                return 'Branch_duct_IPMN', self.bd_risk_assessment()
            else:
                return 'Not_IPMN', 'N/A'

    def bd_risk_assessment(self, features):
        """
        BD-IPMN恶性高危因素评估 (中国指南2022)
        """
        high_risk_stigmata = {
            'enhancing_mural_nodule_≥5mm': features.get('mural_nodule', 0) >= 5,
            'main_duct_dilation_≥10mm': features.get('main_duct', 0) >= 10,
            'obstructive_jaundice': features.get('jaundice', False),
        }

        worrisome_features = {
            'cyst_diameter_>3cm': features.get('cyst_size', 0) > 30,
            'thickened_enhancing_walls': features.get('wall_thickness', 0) > 2,
            'main_duct_5-9mm': 5 <= features.get('main_duct', 0) < 10,
            'lymphadenopathy': features.get('lymph_node', False),
        }

        # 决策树
        if any(high_risk_stigmata.values()):
            return 'RESECTION_INDICATED'
        elif any(worrisome_features.values()):
            return 'ENHANCED_SURVEILLANCE'
        else:
            return 'ROUTINE_SURVEILLANCE'
```

#### 2.3.3 囊性肿瘤AI分型流程

```
囊性肿瘤鉴别诊断树
│
├─ 1. 患者 demographics
│   ├─ 年龄>50+女性 → SCN可能性
│   ├─ 年龄20-40+女性 → SPN可能性
│   └─ 年龄>60+男性 → IPMN可能性
│
├─ 2. 形态学分析
│   ├─ >6小囊+分叶状 → SCN
│   ├─ 单/少囊+厚壁+体尾 → MCN
│   ├─ 杵状指/葡萄串+胰管通 → IPMN
│   └─ 大囊实性+青年女性 → SPN
│
├─ 3. 关键征象检测
│   ├─ 中央瘢痕钙化 → SCN特征
│   ├─ 壁结节≥5mm+强化 → MCN/IPMN恶性
│   ├─ 主胰管≥10mm → IPMN主胰管型
│   └─ 实性成分延迟强化 → SPN
│
└─ 4. 恶性风险评估
    ├─ SCN: 极低风险 → 保守随访
    ├─ MCN: 5-20%恶变 → >4cm手术
    ├─ IPMN: 按高危征象分层管理
    └─ SPN: 低度恶性 → 全部手术
```

---

### 2.4 第四层：急性胰腺炎严重程度评估

#### 2.4.1 Atlanta 2024 标准量化

```python
class AcutePancreatitisGrader:
    """
    急性胰腺炎CT严重度指数 (CTSI) 自动评分
    """

    def compute_ctsi(self, ct_image, pancreas_mask):
        """
        计算改良CT严重度指数 (MCTSI)
        """
        # 1. Balthazar分级 (0-4分)
        balthazar_score = self.grade_balthazar(ct_image, pancreas_mask)

        # 2. 坏死评分 (0-4分)
        necrosis_mask = self.segment_necrosis(ct_image, pancreas_mask)
        necrosis_percentage = np.sum(necrosis_mask) / np.sum(pancreas_mask)

        if necrosis_percentage == 0:
            necrosis_score = 0
        elif necrosis_percentage < 0.3:
            necrosis_score = 2
        elif necrosis_percentage < 0.5:
            necrosis_score = 4
        else:
            necrosis_score = 6

        # 3. 器官衰竭评估 (Marshall评分)
        organ_failure = self.assess_organ_failure(clinical_data)

        # 4. 并发症检测
        complications = {
            'apfc': self.detect_apfc(ct_image),  # 急性胰周液体积聚
            'anc': self.detect_anc(ct_image),    # 急性坏死性积聚
            'walled_off_necrosis': self.detect_won(ct_image),
            'pseudoaneurysm': self.detect_pseudoaneurysm(ct_image),
        }

        # 总分
        mctsi = balthazar_score + necrosis_score

        # 严重度分级
        if mctsi <= 3 and not organ_failure['persistent']:
            grade = 'MAP'  # 轻症
        elif mctsi <= 6 and not organ_failure['persistent']:
            grade = 'MSAP'  # 中度重症
        else:
            grade = 'SAP'  # 重症

        if organ_failure['persistent'] and complications['infected_necrosis']:
            grade = 'CAP'  # 危重症

        return {
            'mctsi': mctsi,
            'balthazar': balthazar_score,
            'necrosis_score': necrosis_score,
            'necrosis_percentage': necrosis_percentage,
            'grade': grade,
            'complications': complications,
        }

    def grade_balthazar(self, ct_image, pancreas_mask):
        """
        Balthazar CT分级
        """
        # A: 正常胰腺
        if self.is_normal_pancreas(ct_image, pancreas_mask):
            return 0

        # B: 胰腺局限性肿大
        elif self.is_focal_enlargement(ct_image, pancreas_mask):
            return 1

        # C: 胰周脂肪炎症
        elif self.has_peripancreatic_inflammation(ct_image, pancreas_mask):
            return 2

        # D: 单发液体积聚
        elif self.count_fluid_collections(ct_image) == 1:
            return 3

        # E: 多发液体积聚/坏死
        else:
            return 4
```

#### 2.4.2 胰腺坏死自动分割

```python
def segment_pancreatic_necrosis(ct_venous, pancreas_mask):
    """
    胰腺坏死自动分割
    坏死定义：静脉期CT无强化区域
    """
    # 1. 在胰腺区域内检测低强化区域
    pancreas_region = ct_venous * pancreas_mask

    # 2. 计算正常胰腺实质CT值参考
    normal_hu = np.percentile(pancreas_region[pancreas_region > 0], 75)

    # 3. 坏死阈值：低于正常强化50%以上
    necrosis_threshold = normal_hu * 0.5

    # 4. 二值化+形态学操作
    necrosis_mask = (pancreas_region < necrosis_threshold) & (pancreas_region > 0)
    necrosis_mask = morphological_cleanup(necrosis_mask)

    return necrosis_mask
```

---

### 2.5 第五层：慢性胰腺炎严重程度评估

#### 2.5.1 IAP/APA 2024 影像评分

```python
class ChronicPancreatitisScorer:
    """
    慢性胰腺炎影像严重程度评分 (CT/MRI)
    """

    def __init__(self):
        self.cambridge_criteria = {
            'normal': 0,
            'equivocal': 1,
            'mild': 2,
            'moderate': 3,
            'severe': 4
        }

    def score_calcification(self, ct_image, pancreas_mask):
        """
        钙化严重程度评分
        """
        # 检测钙化点
        calc_mask = ct_image > 300  # HU > 300为钙化
        calc_points = detect_calcifications(calc_mask)

        n_calc = len(calc_points)

        # 分级
        if n_calc == 0:
            return 0, 'None'
        elif n_calc < 7:
            return 1, 'Mild'  # 轻度
        elif n_calc < 50:
            return 2, 'Moderate'  # 中度
        else:
            return 3, 'Severe'  # 重度

    def score_duct_changes(self, mrcp_or_ct, duct_mask):
        """
        胰管改变评分
        """
        # 测量主胰管直径
        main_duct_diam = self.measure_main_duct_diameter(duct_mask)

        # 检测狭窄/扩张
        strictures = self.detect_strictures(duct_mask)

        # 评分
        score = 0
        if main_duct_diam > 3:  # mm
            score += 1
        if len(strictures) > 0:
            score += 2  # 狭窄是金标准
        if main_duct_diam > 6:
            score += 1

        return score, {
            'main_duct_diameter': main_duct_diam,
            'strictures': strictures,
            'beading': self.detect_beading(duct_mask)  # 串珠样改变
        }

    def compute_total_score(self, ct_image, mri=None):
        """
        综合评分 (总分0-10)
        """
        calc_score, calc_grade = self.score_calcification(ct_image, pancreas_mask)
        duct_score, duct_info = self.score_duct_changes(mri or ct_image, duct_mask)
        atrophy_score = self.score_atrophy(ct_image, pancreas_mask)

        total = calc_score + duct_score + atrophy_score

        # 严重程度分级
        if total <= 2:
            severity = 'Mild'
        elif total <= 5:
            severity = 'Moderate'
        else:
            severity = 'Severe'

        return {
            'total_score': total,
            'severity': severity,
            'calcification': {'score': calc_score, 'grade': calc_grade},
            'duct_changes': {'score': duct_score, 'details': duct_info},
            'atrophy': atrophy_score,
        }
```

---

## 三、多模态融合诊断

### 3.1 临床信息整合

```python
class MultimodalFusion:
    """
    影像+临床信息融合诊断
    """

    def fuse_data(self, image_features, clinical_data):
        """
        多模态特征融合
        """
        # 1. 影像特征 (CNN输出)
        img_vector = self.image_encoder(image_features)

        # 2. 临床特征结构化
        clinical_vector = self.encode_clinical({
            'age': clinical_data['age'],
            'gender': clinical_data['gender'],
            'symptoms': clinical_data['symptoms'],  # 腹痛/黄疸/消瘦
            'ca19_9': clinical_data['ca19_9'],
            'cea': clinical_data['cea'],
            'history': clinical_data['history'],  # 胰腺炎/糖尿病/家族史
        })

        # 3. 实验室检查
        lab_vector = self.encode_labs({
            'amylase': clinical_data['amylase'],
            'lipase': clinical_data['lipase'],
            'bilirubin': clinical_data['bilirubin'],
            'albumin': clinical_data['albumin'],
        })

        # 4. 特征融合 (Concatenate + Attention)
        fused = self.fusion_layer([img_vector, clinical_vector, lab_vector])

        # 5. 多任务输出
        return {
            'disease_type': self.classifier_disease(fused),  # 疾病分型
            'malignant_prob': self.classifier_malignancy(fused),  # 恶性概率
            'severity_grade': self.classifier_severity(fused),  # 严重度
            'resectability': self.classifier_resectability(fused),  # 可切除性
        }
```

### 3.2 不确定性量化

```python
def monte_carlo_dropout_inference(model, input_data, n_samples=50):
    """
    使用MC Dropout估计模型不确定性
    """
    predictions = []
    model.train()  # 保持dropout开启

    for _ in range(n_samples):
        pred = model(input_data)
        predictions.append(pred)

    # 计算均值和方差
    mean_pred = np.mean(predictions, axis=0)
    uncertainty = np.var(predictions, axis=0)

    return mean_pred, uncertainty

# 决策规则
if uncertainty > threshold:
    recommendation = "ESCALATE_TO_RADIOLOGIST"
else:
    recommendation = "AI_DIAGNOSIS_CONFIDENT"
```

---

## 四、系统验证与质控

### 4.1 诊断准确性验证

| 验证项目 | 方法 | 标准 |
|----------|------|------|
| **实性-囊性分类** | 测试集(AUC) | AUC > 0.95 |
| **PDAC vs pNET** | 独立测试集 | 准确率 > 90% |
| **囊性肿瘤四分型** | 多中心验证 | 准确率 > 85% |
| **血管侵犯评估** | 与金标准对比 | 敏感性 > 90% |
| **CTSI评分** | 专家一致性 | ICC > 0.85 |

### 4.2 安全性检查清单

```python
SAFETY_CHECKLIST = {
    'pre_inference': [
        'CT_phase_verified',  # 确认期相(动脉/静脉)
        'image_quality_check',  # 图像质量检查
        'contrast_enhancement_confirmed',  # 确认增强
    ],
    'post_inference': [
        'confidence_threshold_met',  # 置信度>0.8
        'uncertainty_acceptable',  # 不确定性<0.15
        'clinical_plausibility_check',  # 临床合理性检查
        'contradiction_detection',  # 矛盾结果检测
    ],
    'report_generation': [
        'citations_verified',  # 引用验证
        'confidence_disclosed',  # 置信度披露
        'limitations_stated',  # 局限性说明
        'radiologist_review_triggered',  # 触发专家审核
    ]
}
```

---

## 五、实施路线图

### Phase 1: 基础能力 (1-2月)
- [ ] 胰腺自动分割 (TotalSegmentator集成)
- [ ] 实性-囊性二分类
- [ ] 基础PDAC/pNET鉴别

### Phase 2: 血管分析 (2-3月) - **关键阻塞点**
- [ ] SMA/SMV/CA/CHA/PV分割模型获取/训练
- [ ] 血管侵犯角度自动计算
- [ ] 可切除性AI评估

### Phase 3: 囊性肿瘤 (1-2月)
- [ ] 囊性肿瘤四分型模型
- [ ] IPMN恶性风险评估
- [ ] 随访计划自动生成

### Phase 4: 胰腺炎 (1-2月)
- [ ] 急性胰腺炎CTSI自动评分
- [ ] 慢性胰腺炎严重程度评分
- [ ] 坏死自动分割

### Phase 5: 整合与验证 (2-3月)
- [ ] 多疾病统一诊断平台
- [ ] 多中心临床验证
- [ ] 监管认证准备

---

**文档版本**: v1.0
**最后更新**: 2026-03-30
**作者**: Claude Code Agent
**审核**: 待临床专家审核
