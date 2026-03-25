# ChangHai PDAC Agent 技术架构详解 (Mermaid图表)

## 一、系统整体架构流程图

```mermaid
flowchart TB
    subgraph Input["📥 输入层"]
        PatientID["患者ID: CL-03356"]
        DICOM["DICOM序列<br/>CT静脉期增强扫描"]
    end

    subgraph AgentCore["🧠 Agent核心决策层"]
        direction TB
        EnvAware["环境感知<br/>发现已有数据"]
        Decision1{"nnU-Net结果?"}
        NormalFlow["常规流程"]
        DeepDrillTrigger["🚨 触发Deep Drill"]
    end

    subgraph SkillsLayer["🔧 SKILL执行层"]
        direction TB

        subgraph DicomSkill["SKILL: DICOM处理器"]
            DicomConvert["DICOM → NIfTI转换<br/>SimpleITK读取DICOM序列<br/>Spacing/Origin/方向保留"]
        end

        subgraph TotalSegSkill["SKILL: TotalSegmentator分割"]
            TotalSeg["多器官分割<br/>104类解剖结构<br/>包括:胰腺/血管/器官"]
            PancreasMask["胰腺Mask提取<br/>Label: pancreas"]
            VesselMasks["血管Masks<br/>aorta/portal_vein/splenic_vein"]
        end

        subgraph NnunetSkill["SKILL: nnU-Net肿瘤分割"]
            Nnunet["3D U-Net推理<br/>MSD Task07预训练权重"]
            TumorMask["肿瘤Mask<br/>Label 2 = 肿瘤"]
            VolumeCalc["体积计算<br/>voxel_count × voxel_volume"]
        end

        subgraph TiledSkill["SKILL: 多窗位Tiled切片 ⭐"]
            direction TB
            MaxAreaSlice["最大面积切片定位<br/>argmax(胰腺mask面积)"]
            WindowTransform["HU值窗位变换 ⭐"]
            ThreeWindow["三窗位生成"]
            TiledConcat["水平拼接<br/>1536×512 PNG"]
        end

        subgraph VLMSkill["SKILL: VLM视觉分析 ⭐"]
            Base64Encode["Base64图像编码"]
            QwenVL["Qwen-VL-Plus API调用"]
            VLMReasoning["视觉推理<br/>不规则轮廓/低衰减/占位效应"]
            SuspicionScore["怀疑度评估<br/>LOW/MODERATE/HIGH"]
        end

        subgraph VascularSkill["SKILL: 血管拓扑评估 ⭐"]
            direction TB
            CenterlineExtract["血管中心线提取<br/>3D Skeletonization"]
            BifurcationDetect["分叉点检测<br/>SMA/SMV分支识别"]
            AngleCalc["包绕角度计算 ⭐<br/>肿瘤质心→血管切线平面"]
            Resectability["可切除性分级<br/>NCCN标准"]
        end
    end

    subgraph CognitiveLayer["🧩 认知失调检测层"]
        Compare["结果比较"]
        Dissonance{"存在冲突?"]
        EndogenousFN["ENDOGENOUS_FALSE_NEGATIVE<br/>等密度假阴性检测"]
    end

    subgraph Output["📤 输出层"]
        MDTReport["MDT报告<br/>引用验证审计"]
        Warning["⚠️ 需人工复核标记"]
    end

    %% 流程连接
    PatientID --> EnvAware
    DICOM --> DicomConvert

    EnvAware --> TotalSeg
    TotalSeg --> PancreasMask
    TotalSeg --> VesselMasks

    EnvAware --> Nnunet
    Nnunet --> TumorMask
    TumorMask --> VolumeCalc

    VolumeCalc --> Decision1
    Decision1 -->|Volume > 0| NormalFlow
    Decision1 -->|Volume = 0| DeepDrillTrigger

    DeepDrillTrigger --> MaxAreaSlice
    PancreasMask --> MaxAreaSlice

    MaxAreaSlice --> WindowTransform
    WindowTransform --> ThreeWindow
    ThreeWindow --> TiledConcat

    TiledConcat --> Base64Encode
    Base64Encode --> QwenVL
    QwenVL --> VLMReasoning
    VLMReasoning --> SuspicionScore

    TumorMask --> CenterlineExtract
    VesselMasks --> CenterlineExtract
    CenterlineExtract --> BifurcationDetect
    BifurcationDetect --> AngleCalc
    AngleCalc --> Resectability

    VolumeCalc --> Compare
    SuspicionScore --> Compare
    Compare --> Dissonance
    Dissonance -->|Yes| EndogenousFN

    NormalFlow --> MDTReport
    EndogenousFN --> Warning
    Warning --> MDTReport
    Resectability --> MDTReport

    style DeepDrillTrigger fill:#ff9999
    style EndogenousFN fill:#ffcc99
    style TiledSkill fill:#99ccff
    style VLMSkill fill:#99ff99
    style VascularSkill fill:#ffccff
```

---

## 二、SKILL: 多窗位Tiled切片 - 技术实现详解

```mermaid
flowchart LR
    subgraph InputTiled["输入"]
        CTNifti["CT NIfTI<br/>Shape: (512,512,194)<br/>Spacing: (1,1,1)mm"]
        PancreasNifti["胰腺Mask NIfTI<br/>Binary: 0/1"]
    end

    subgraph CoreAlgorithm["核心算法 ⭐"]
        direction TB

        subgraph Step1["步骤1: 最大面积切片定位"]
            SliceLoop["遍历所有Z轴切片<br/>for z in range(depth)"]
            AreaCalc["计算胰腺面积<br/>area = np.sum(mask[:,:,z])"]
            ArgMax["取最大面积索引<br/>z_max = argmax(area)"]
        end

        subgraph Step2["步骤2: HU值窗位变换 ⭐"]
            RawHU["原始HU值<br/>范围: [-1000, +1000]"]
            WindowFormula["窗位公式 ⭐<br/>windowed = (HU - C) / W × 255 + 128"]
            Clip["截断到[0,255]<br/>np.clip(windowed, 0, 255)"]
            UInt8["转换为uint8<br/>8位灰度图像"]
        end

        subgraph Step3["步骤3: 三窗位并行生成"]
            direction LR

            subgraph Standard["标准窗"]
                StdParam["W:400, C:40"]
                StdRange["HU范围: [-160, 240]<br/>跨度: 400 HU"]
                StdImg["512×512图像"]
            end

            subgraph Narrow["窄窗 ⭐"]
                NarrowParam["W:150, C:40"]
                NarrowRange["HU范围: [-35, 115]<br/>跨度: 150 HU"]
                NarrowImg["512×512图像"]
            end

            subgraph Soft["软组织窗"]
                SoftParam["W:250, C:50"]
                SoftRange["HU范围: [-75, 175]<br/>跨度: 250 HU"]
                SoftImg["512×512图像"]
            end
        end

        subgraph Step4["步骤4: Tiled拼接"]
            HConcat["水平拼接<br/>np.concatenate([std, narrow, soft], axis=1)"]
            OutputShape["输出形状<br/>(512, 1536, 3)"]
            SavePNG["保存为PNG<br/>无损压缩"]
        end
    end

    subgraph Physics["物理原理 ⭐"]
        direction TB
        NormalPancreas["正常胰腺<br/>HU: 40-60"]
        PDACTumor["PDAC肿瘤<br/>HU: 35-50<br/>(纤维化)"]
        Diff["差异: ~10 HU"]

        StandardContrast["标准窗对比度<br/>10/400 = 2.5%"]
        NarrowContrast["窄窗对比度 ⭐<br/>10/150 = 6.7%<br/>增强2.6倍"]
    end

    %% 连接
    CTNifti --> SliceLoop
    PancreasNifti --> SliceLoop
    SliceLoop --> AreaCalc --> ArgMax

    ArgMax --> RawHU
    RawHU --> WindowFormula --> Clip --> UInt8

    UInt8 --> StdParam
    UInt8 --> NarrowParam
    UInt8 --> SoftParam

    StdParam --> StdRange --> StdImg
    NarrowParam --> NarrowRange --> NarrowImg
    SoftParam --> SoftRange --> SoftImg

    StdImg --> HConcat
    NarrowImg --> HConcat
    SoftImg --> HConcat
    HConcat --> OutputShape --> SavePNG

    NormalPancreas --> Diff
    PDACTumor --> Diff
    Diff --> StandardContrast
    Diff --> NarrowContrast

    style Narrow fill:#ff9999
    style NarrowParam fill:#ff9999
    style NarrowContrast fill:#ff9999
    style WindowFormula fill:#ffcc99
```

---

## 三、SKILL: VLM视觉分析 - 技术实现详解

```mermaid
flowchart TB
    subgraph InputVLM["输入"]
        TiledImage["Tiled PNG图像<br/>1536×512像素<br/>三窗位并排"]
        QueryTemplate["查询模板<br/>结构化Prompt"]
    end

    subgraph Preprocess["预处理"]
        ReadImage["读取图像<br/>OpenCV/PIL"]
        Resize["尺寸检查<br/>max 10MB限制"]
        Base64["Base64编码<br/>data:image/png;base64,..."]
        MIME["MIME类型<br/>image/png"]
    end

    subgraph API["Qwen-VL-Plus API"]
        direction TB

        subgraph Request["请求构造"]
            SystemPrompt["System Prompt<br/>'你是专业放射科医生...'"]
            UserMessage["用户消息"]
            ContentList["内容列表<br/>[图像, 文本]"]

            subgraph MessageStruct["消息结构 ⭐"]
                ImageURL["type: image_url<br/>url: data:image/png;base64,{data}"]
                TextQuery["type: text<br/>text: {query}"]
            end
        end

        subgraph LLMInference["LLM推理 ⭐"]
            VisionEncoder["视觉编码器<br/>ViT架构"]
            LLM["大语言模型<br/>Qwen-VL-Plus"]
            Attention["跨模态注意力<br/>图像-文本对齐"]
            TokenGen["Token生成<br/>流式/非流式"]
        end

        subgraph OutputParse["输出解析"]
            RawOutput["原始文本输出"]
            StructureExtract["结构化提取<br/>正则匹配"]
            SuspicionLevel["怀疑度分级<br/>LOW/MODERATE/HIGH"]
            FindingsList["发现列表<br/>不规则轮廓/低衰减/占位效应"]
        end
    end

    subgraph VLMQuery["VLM查询模板 ⭐"]
        direction TB
        TemplateContent["""
        Analyze this pancreatic CT for PDAC.

        Image layout (left to right):
        - LEFT: Standard window (W:400, C:40)
        - CENTER: Narrow window (W:150, C:40)
        - RIGHT: Soft tissue window (W:250, C:50)

        Focus on CENTER (Narrow Window) for isodense tumors.

        Provide:
        1. Suspicion level (NONE/LOW/MODERATE/HIGH)
        2. Key findings (list)
        3. Confidence (DEFINITE/PROBABLE/POSSIBLE)
        """]
    end

    subgraph RealExample["CL-03356真实输出示例"]
        direction TB
        RealSuspicion["Suspicion: MODERATE"]
        RealFindings["""
        Key Findings:
        1. Irregular contour of pancreatic head
        2. Hypo-attenuating regions (narrow window)
        3. Heterogeneous texture
        4. Mass effect on SMV/SMA
        5. Loss of normal lobulation
        """]
        RealConfidence["Confidence: POSSIBLE"]
        InferenceTime["推理时间: 17,677ms"]
    end

    %% 连接
    TiledImage --> ReadImage --> Resize --> Base64 --> MIME
    QueryTemplate --> TextQuery

    MIME --> ImageURL
    ImageURL --> ContentList
    TextQuery --> ContentList
    ContentList --> UserMessage
    SystemPrompt --> UserMessage

    UserMessage --> VisionEncoder
    VisionEncoder --> Attention
    Attention --> LLM --> TokenGen

    TokenGen --> RawOutput --> StructureExtract
    StructureExtract --> SuspicionLevel
    StructureExtract --> FindingsList

    VLMQuery --> QueryTemplate

    SuspicionLevel --> RealSuspicion
    FindingsList --> RealFindings
    StructureExtract --> RealConfidence
    LLMInference --> InferenceTime

    style Narrow fill:#ff9999
    style VisionEncoder fill:#99ccff
    style Attention fill:#ffcc99
    style LLM fill:#99ff99
```

---

## 四、SKILL: 血管拓扑评估 - 技术实现详解

```mermaid
flowchart TB
    subgraph InputVascular["输入"]
        TumorMaskVasc["肿瘤Mask<br/>3D二值数组"]
        VesselMasksVasc["血管Masks<br/>aorta/SMV/portal_vein"]
        CTNiftiVasc["CT NIfTI<br/>空间参考"]
    end

    subgraph VascularAlgorithm["血管拓扑算法 ⭐"]
        direction TB

        subgraph Step1Vasc["步骤1: 血管分割与清理"]
            BinaryVessel["二值化血管<br/>阈值: >0.5"]
            MorphClean["形态学清理<br/>开运算去噪"]
            ConnectedComp["连通域分析<br/>保留最大连通域"]
        end

        subgraph Step2Vasc["步骤2: 中心线提取 ⭐"]
            Skeleton3D["3D Skeletonization<br/>细化算法"]
            CenterlinePoints["中心线点集<br/>{x,y,z}列表"]
            Interpolation["样条插值<br/>平滑曲线"]
        end

        subgraph Step3Vasc["步骤3: 分叉点检测"]
            BranchDetect["分支检测算法<br/>邻域分析"]
            BifurcationPoints["分叉点标记<br/>SMA主干/分支"]
            VesselTree["血管树结构<br/>图表示: G=(V,E)"]
        end

        subgraph Step4Vasc["步骤4: 包绕角度计算 ⭐⭐"]
            direction TB

            subgraph TumorRegion["肿瘤区域"]
                TumorCentroid["肿瘤质心计算<br/>mean(x,y,z)"]
                TumorConvexHull["凸包提取<br/>Convex Hull"]
            end

            subgraph AngleCalcDetail["角度计算详情 ⭐"]
                NormalPlane["法平面构造<br/>过质心垂直于血管"]
                IntersectEllipse["交线/椭圆<br/>肿瘤∩法平面"]
                AngleDefine["包绕角度定义<br/>血管圆心→椭圆切线"]

                subgraph AngleFormula["角度公式 ⭐"]
                    Theta["θ = arccos( (v⃗·t⃗) / (|v⃗||t⃗|) )"]
                    Where["v⃗: 血管方向向量<br/>t⃗: 肿瘤切线向量"]
                end
            end

            subgraph Classification["NCCN分级"]
                Angle180{"包绕角度?"}
                Resectable["< 180°<br/>Resectable<br/>可切除"]
                Borderline["= 180°<br/>Borderline<br/>边缘可切除"]
                Unresectable["> 180°<br/>Unresectable<br/>不可切除"]
            end
        end
    end

    subgraph VascularMetrics["血管评估指标"]
        direction LR

        subgraph SMA["SMA评估"]
            SMAAngle["SMA包绕角<br/>SMA-tumor angle"]
            SMAContact["接触长度<br/>mm"]
            SMAInvade["侵犯深度<br/>mm"]
        end

        subgraph SMV["SMV评估"]
            SMVAngle["SMV包绕角<br/>SMV-tumor angle"]
            SMVContact["接触长度<br/>mm"]
            SMVStenosis["狭窄程度<br/>%"]
        end

        subgraph OtherVessels["其他血管"]
            Celiac["腹腔干<br/>Celiac Trunk"]
            Portal["门静脉<br/>Portal Vein"]
            CHA["肝总动脉<br/>Common Hepatic A."]
        end
    end

    subgraph OutputVascular["输出"]
        VascularJSON["血管评估JSON<br/>结构化报告"]
        AngleVis["角度可视化<br/>3D渲染图"]
        MDTInput["MDT报告输入<br/>可切除性建议"]
    end

    %% 连接
    VesselMasksVasc --> BinaryVessel --> MorphClean --> ConnectedComp
    ConnectedComp --> Skeleton3D --> CenterlinePoints --> Interpolation
    Interpolation --> BranchDetect --> BifurcationPoints --> VesselTree

    TumorMaskVasc --> TumorCentroid
    TumorMaskVasc --> TumorConvexHull

    Interpolation --> NormalPlane
    TumorCentroid --> NormalPlane
    TumorConvexHull --> IntersectEllipse
    NormalPlane --> IntersectEllipse --> AngleDefine
    AngleDefine --> AngleFormula --> Theta

    Theta --> Angle180
    Angle180 -->|< 180°| Resectable
    Angle180 -->|= 180°| Borderline
    Angle180 -->|> 180°| Unresectable

    Theta --> SMAAngle
    Theta --> SMVAngle

    SMAAngle --> SMA
    SMVAngle --> SMV
    VesselTree --> OtherVessels

    SMA --> VascularJSON
    SMV --> VascularJSON
    OtherVessels --> VascularJSON
    VascularJSON --> AngleVis
    VascularJSON --> MDTInput

    style AngleCalcDetail fill:#ff9999
    style AngleFormula fill:#ffcc99
    style NormalPlane fill:#99ccff
    style Classification fill:#99ff99
```

---

## 五、认知失调检测机制

```mermaid
flowchart TB
    subgraph InputSources["输入源"]
        NnunetResult["nnU-Net结果<br/>定量: 0.00 ml"]
        VLMResult["VLM评估<br/>定性: MODERATE suspicion"]
        ClinicalContext["临床上下文<br/>患者症状/CA19-9"]
    end

    subgraph CognitiveProcess["认知失调检测流程 ⭐"]
        direction TB

        subgraph SemanticParse["语义解析"]
            NnunetParse["解析nnU-Net<br/>体积=0 → '无肿瘤'"]
            VLMParse["解析VLM<br/>MODERATE + 5特征 → '有可疑'"]
        end

        subgraph KnowledgeBase["医学知识库"]
            PDACChar["PDAC特征<br/>等密度假阴性常见"]
            IsoDense["等密度定义<br/>HU tumor ≈ HU pancreas"]
            ThresholdLimit["阈值分割局限<br/>HU阈值失效"]
        end

        subgraph ConflictDetection["冲突检测 ⭐"]
            Comparison["结果比较"]
            LogicCheck["逻辑检查<br/>'无' vs '有' = 矛盾"]
            ConfidenceWeight["置信度加权<br/>nnU-Net: HIGH<br/>VLM: POSSIBLE"]
        end

        subgraph MetaCognition["元认知判断 ⭐⭐"]
            AgentReasoning["Agent推理"]
            ReasoningContent["""
            'nnU-Net说没有肿瘤，
            但VLM说有可疑特征。

            根据PDAC的病理特征，
            等密度肿瘤会导致
            HU阈值方法失效。

            这是典型的
            假阴性场景。'
            """]
            ConflictFlag["冲突标记<br/>COGNITIVE_DIVERGENCE"]
        end
    end

    subgraph Action["行动决策"]
        TriggerDeepDrill["触发Deep Drill<br/>深度探查"]
        Escalate["升级处理<br/>建议人工复核"]
    end

    subgraph CL03356Example["CL-03356真实案例"]
        RealNnunet["nnU-Net: 0ml"]
        RealVLM["VLM: MODERATE<br/>5个形态特征"]
        RealDecision["Agent判定:<br/>ENDOGENOUS_FALSE_NEGATIVE"]
        RealAction["Deep Drill执行<br/>Tiled+VLM分析"]
    end

    %% 连接
    NnunetResult --> NnunetParse
    VLMResult --> VLMParse
    ClinicalContext --> KnowledgeBase

    NnunetParse --> Comparison
    VLMParse --> Comparison
    PDACChar --> Comparison
    IsoDense --> Comparison
    ThresholdLimit --> Comparison

    Comparison --> LogicCheck --> ConfidenceWeight
    ConfidenceWeight --> AgentReasoning
    KnowledgeBase --> AgentReasoning
    AgentReasoning --> ReasoningContent --> ConflictFlag

    ConflictFlag --> TriggerDeepDrill --> Escalate

    RealNnunet --> RealDecision
    RealVLM --> RealDecision
    RealDecision --> RealAction

    style MetaCognition fill:#ff9999
    style AgentReasoning fill:#ffcc99
    style ConflictDetection fill:#99ccff
```

---

## 六、系统数据流全景图

```mermaid
flowchart LR
    subgraph DataIn["原始数据输入"]
        DICOMFiles["DICOM文件<br/>*.dcm"]
    end

    subgraph Pipeline["处理流水线"]
        direction TB

        subgraph Stage1["Stage 1: 预处理"]
            D2N["DICOM→NIfTI<br/>SimpleITK"]
            Resample["重采样<br/>1mm各向同性"]
        end

        subgraph Stage2["Stage 2: 分割"]
            TotalSeg2["TotalSegmentator<br/>104类器官"]
            Nnunet2["nnU-Net<br/>肿瘤分割"]
        end

        subgraph Stage3["Stage 3: 增强 ⭐"]
            Tiled2["多窗位Tiled<br/>窄窗增强"]
        end

        subgraph Stage4["Stage 4: 分析 ⭐"]
            VLM2["Qwen-VL-Plus<br/>视觉分析"]
            Vascular2["血管拓扑<br/>3D角度计算"]
        end

        subgraph Stage5["Stage 5: 决策"]
            Cognitive2["认知失调检测"]
            Report2["MDT报告生成"]
        end
    end

    subgraph DataOut["输出产物"]
        direction TB
        NIfTIOut["NIfTI文件<br/>*.nii.gz"]
        PNGOut["Tiled图像<br/>*.png"]
        JSONOut["评估JSON<br/>*.json"]
        MDOut["MDT报告<br/>*.md"]
        LogOut["执行日志<br/>*.jsonl/*.log"]
    end

    %% 连接
    DICOMFiles --> D2N --> Resample
    Resample --> TotalSeg2 --> NIfTIOut
    Resample --> Nnunet2 --> NIfTIOut

    TotalSeg2 --> Tiled2
    Nnunet2 --> Tiled2
    Tiled2 --> PNGOut

    Tiled2 --> VLM2 --> JSONOut
    TotalSeg2 --> Vascular2 --> JSONOut
    Nnunet2 --> Vascular2

    VLM2 --> Cognitive2
    Vascular2 --> Cognitive2
    Cognitive2 --> Report2 --> MDOut

    Stage1 --> LogOut
    Stage2 --> LogOut
    Stage3 --> LogOut
    Stage4 --> LogOut
    Stage5 --> LogOut

    style Stage3 fill:#ff9999
    style Stage4 fill:#99ff99
    style Stage5 fill:#ffcc99
```

---

## 使用说明

以上Mermaid图表可以在以下平台渲染：

1. **GitHub/GitLab**: 直接嵌入Markdown，自动渲染
2. **Notion**: 使用Mermaid代码块
3. **Obsidian**: 安装Mermaid插件
4. **VS Code**: 安装Markdown Preview Mermaid Support插件
5. **在线编辑器**: https://mermaid.live

### 渲染技巧

- 使用`subgraph`组织相关节点
- 使用`style`突出显示关键技术点
- 使用`⭐`标记创新点
- 使用不同颜色区分不同类型的SKILL

### 颜色说明

| 颜色 | 含义 |
|------|------|
| 🟥 #ff9999 | 关键创新点/核心算法 |
| 🟧 #ffcc99 | 重要步骤/公式 |
| 🟦 #99ccff | 输入/数据流 |
| 🟩 #99ff99 | 输出/结果 |
| 🟪 #ffccff | 决策点/判断逻辑 |
