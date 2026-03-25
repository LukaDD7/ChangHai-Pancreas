# ChangHai PDA 项目配置文档

## 项目概述
胰腺导管腺癌（PDA）CT影像分析系统，采用 **Deep Agents 扁平化架构**，用于多模态AI肿瘤诊断和冲突检测。

## 架构核心 (Deep Agents v1.0)

### 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│              interactive_main.py (单一主控Agent)             │
│              ChangHai PDAC Agent v1.0                       │
├─────────────────────────────────────────────────────────────┤
│  L3 System Prompt                                           │
│  ├── Agent自主性原则 (无硬编码workflow)                      │
│  ├── 认知失调监测 (Agent语义判断，无硬编码阈值)              │
│  ├── 证据主权准则 (严禁编造测量数据)                          │
│  └── 物理溯源引用 (所有结果必须可追溯到脚本执行)              │
├─────────────────────────────────────────────────────────────┤
│  Tools (4个核心工具):                                        │
│  ├── execute (shell执行，跨conda环境)                        │
│  ├── read_file (读取/skills/和/memories/)                    │
│  ├── analyze_image (VLM视觉分析)                             │
│  └── submit_pdac_report (报告提交+引用验证)                   │
├─────────────────────────────────────────────────────────────┤
│  Skills (7个挂载技能):                                       │
│  ├── /skills/dicom_processor/          (DICOM→NIfTI)         │
│  ├── /skills/totalseg_segmentor/       (器官分割)             │
│  ├── /skills/nnunet_segmentor/         (肿瘤分割)             │
│  ├── /skills/master_slice_extractor/   (多窗位Tiled切片)      │
│  ├── /skills/llava_med_analyzer/       (VLM视觉分析)          │
│  ├── /skills/adw_ceo_reporter/         (CEO冲突检测)          │
│  └── /skills/vascular_topology/        (血管拓扑，可选)        │
├─────────────────────────────────────────────────────────────┤
│  Backend: CompositeBackend                                   │
│  ├── FilesystemBackend (sandbox隔离)                        │
│  └── StoreBackend (/memories/跨会话持久化)                   │
└─────────────────────────────────────────────────────────────┘
```

### 核心创新: 认知失调监测 (Cognitive Dissonance Monitoring)

**问题:** nnU-Net 对等密度肿瘤假阴性 (0ml)，但临床存在肿瘤

**传统方案:** 依赖金标准对比发现错误 (生产环境不可用)

**ADW方案:** 内源性冲突检测 (Agent自主判断)
- Agent比较: nnU-Net结果 vs VLM语义描述
- Agent判断: 是否存在临床意义上的冲突
- 无需硬编码阈值，Agent的语义理解本身就是检测机制
- 无需金标准，Agent自主决策

```
Agent的语义理解 = 冲突检测机制

IF (nnU-Net: "无肿瘤") AND (VLM: "可疑形态特征")
THEN Agent判断: 存在冲突 → ESCALATE_TO_RADIOLOGIST
```

### 使用方式 (Human-in-the-Loop)

```bash
# 启动主控Agent
conda run -n ChangHai python interactive_main.py

# Agent启动后进入交互模式，等待用户输入
🧑‍⚕️ Enter patient information: CL-03356

# Agent自主决定执行流程:
# - Agent自己决定读取哪个SKILL.md
# - Agent自己决定执行顺序
# - Agent自己判断是否检测到冲突
# - 无需预定义workflow，Agent根据临床情境自主决策
```

**关键特性：**
- **Human-in-the-loop**: 用户输入患者ID，Agent自主完成后续分析
- **Agent自主性**: 不强制检查点顺序，Agent自己判断何时使用哪个Skill
- **元思维**: Agent的语义理解本身就是冲突检测机制，无需硬编码阈值

### 多窗位Tiled策略 (等密度肿瘤检测)

**问题:** PDAC肿瘤在标准窗下与正常胰腺呈等密度

**解决方案:** Tiled三窗位拼接
- 标准窗 (W:400): 常规观察
- **窄窗 (W:150)**: HU范围[-35, 115]，20 HU范围→34灰度级(2.6x增强)
- 软组织窗 (W:250): 边界定义

**实现方式**: HU值重映射
```python
windowed = (raw_hu - center) / width * 255 + 128
windowed = np.clip(windowed, 0, 255)
```

**输出**: 1536×512 Tiled图像 (3窗口并排)

---

## 全量执行记录系统 (ExecutionLogger v1.0)

### 概述
完整的执行日志系统，提供从用户输入到最终报告的全链路追溯能力。

### 日志结构
```
workspace/sandbox/execution_logs/
└── {patient_id}_{session_id}/
    ├── execution_log.jsonl       # 结构化JSONL日志
    ├── execution.log              # 人类可读日志
    ├── artifacts.json             # 产物清单
    └── session_summary.json       # 会话摘要
```

### 记录内容

| 类型 | 记录内容 | 用途 |
|------|----------|------|
| USER_INPUT | 患者ID输入 | 追溯分析起点 |
| TOOL_CALL | 工具调用、参数、结果、耗时 | 审计追踪 |
| ARTIFACT | 生成的文件(NIfTI/PNG/JSON等) | 产物管理 |
| LLM_INTERACTION | VLM调用记录 | 模型行为追溯 |
| CHECKPOINT | 关键决策点 | 决策过程审查 |

### 自动产物检测
执行脚本时，系统自动解析stdout检测生成的文件：
- 匹配模式: "Saved to:", "Output:", "*.nii.gz", "*.png"等
- 自动记录文件路径、大小、类型、生成时间
- 无需Agent手动调用

### 与审计系统的区别
- **ExecutionAuditor**: 强审计闭环，用于引用验证
- **ExecutionLogger**: 全量记录，用于完整追溯和分析

### 引用验证
报告提交时自动验证所有引用：
- 检查 `[Script: xxx]` 是否在审计日志中
- 检查 `[Tool: xxx]` 是否有执行记录
- 未经验证的引用将导致报告被拒绝

---

## 虚拟环境配置

### 1. ChangHai (主系统环境)
```bash
conda activate ChangHai
```
- **用途**: 主系统开发环境，与专用模型环境隔离
- **Python**: 3.10.20
- **GCC**: 11.x (兼容旧服务器)
- **安装命令**:
```bash
conda create -n ChangHai python=3.10 gxx_linux-64=11.* gcc_linux-64=11.* gfortran_linux-64=11.* libstdcxx-ng libgcc-ng
```

### 2. totalseg (TotalSegmentator 专用)
```bash
conda run -n totalseg python script.py
```
- **用途**: TotalSegmentator 器官/血管分割
- **核心包**: nibabel, numpy, scipy, matplotlib, scikit-image
- **功能**: CT预处理、血管分割、可视化、评估报告生成

### 3. nnunetv2 (nnU-Net 专用)
```bash
conda activate nnunetv2
```
- **用途**: nnU-Net v1/v2 胰腺肿瘤分割
- **核心包**: nnunet (v1), PyTorch 2.0.1 with CUDA 11.8
- **重要补丁**: 3处 `torch.load()` 添加 `weights_only=False` 以兼容 PyTorch 2.6
- **环境变量自动注入**: `execute()` 工具检测到 nnU-Net 命令时自动设置：
  - `nnUNet_raw_data_base`
  - `nnUNet_preprocessed`
  - `RESULTS_FOLDER`

## 目录结构

```
/media/luzhenyang/project/ChangHai_PDA/data/
├── raw/                                    # 【原始数据】
│   ├── dicom/                             # DICOM 文件
│   │   └── dicom_data/
│   └── manifest-*.tcia                    # TCIA 清单文件
│
├── processed/                              # 【处理后数据】
│   ├── nifti/                             # NIfTI 格式 CT
│   │   └── nifti_output/
│   │       └── C3L-03348_CT.nii.gz
│   └── segmentations/                     # 分割结果
│       ├── nnunet_tumor_output/           # nnU-Net 肿瘤分割
│       │   ├── nnunet_raw_output/
│       │   │   └── C3L_03348.nii.gz      # 原始3类分割 (0=背景, 1=胰腺, 2=肿瘤)
│       │   ├── true_tumor_mask.nii.gz    # 提取的肿瘤Mask (二值)
│       │   └── pancreas_and_tumor_mask.nii.gz
│       └── segmentations/                 # TotalSegmentator 血管分割
│           └── C3L-03348/
│               ├── aorta.nii.gz
│               ├── portal_vein_and_splenic_vein.nii.gz
│               └── ...                    # 其他血管/器官分割文件
│
├── results/                                # 【评估结果】
│   ├── json/                              # JSON 报告
│   │   ├── final_assessment_result.json
│   │   ├── assessment_result.json
│   │   └── vascular_panel_*.json
│   └── images/                            # 可视化图片
│       ├── integrated_assessment.png
│       ├── clinical_evidence_overlay.png
│       ├── tumor_vessel_overlay.png
│       └── visualizations/
│
├── models/                                 # 【模型相关】
│   ├── LLaVA-Med/                        # LLaVA-Med 代码仓库
│   ├── llava-med-v1.5-mistral-7b/        # LLaVA-Med v1.5 模型权重 (~15GB)
│   │   ├── config.json
│   │   ├── model-0000{1,2,3,4}-of-00004.safetensors
│   │   ├── tokenizer.model
│   │   └── ...
│   └── nnunet/                           # nnU-Net 工作目录
│       ├── nnunet_v1_workspace/
│       │   ├── raw/nnUNet_raw_data/Task007_Pancreas/imagesTs/
│       │   ├── preprocessed/
│       │   └── results/nnUNet/3d_fullres/Task007_Pancreas/
│       │       └── nnUNetTrainerV2__nnUNetPlansv2.1/
│       │           ├── fold_0/model_final_checkpoint.model
│       │           └── plans.json
│       └── nnunet_workspace/
│
├── scripts/                                # 【Python 脚本】
│   ├── final_integrated_assessment.py    # 整合评估
│   ├── vascular_topology.py              # 血管包绕算法
│   ├── panel_vascular_assessment.py      # 血管评估面板
│   ├── run_nnunet_v1_inference.py        # nnU-Net 推理
│   ├── download_nnunet_weights.py        # 模型下载
│   ├── generate_diagnostic_figure.py     # 诊断图生成
│   ├── visualize_tumor_vessels.py        # 可视化
│   └── ...                               # 其他脚本
│
└── temp/                                   # 【临时文件】
    └── __pycache__/
```

## 核心代码文件

| 文件 | 路径 | 功能 | 所属环境 |
|------|------|------|----------|
| `final_integrated_assessment.py` | `scripts/` | 跨模态联调 (肿瘤+血管) | totalseg |
| `vascular_topology.py` | `scripts/` | 血管包绕角度计算 | totalseg |
| `panel_vascular_assessment.py` | `scripts/` | 血管评估面板 | totalseg |
| `run_nnunet_v1_inference.py` | `scripts/` | nnU-Net 肿瘤分割推理 | nnunetv2 |
| `download_nnunet_weights.py` | `scripts/` | 下载 MSD Task07 模型 | nnunetv2 |
| `generate_diagnostic_figure.py` | `scripts/` | 生成诊断可视化图 | totalseg |
| `visualize_tumor_vessels.py` | `scripts/` | 肿瘤血管可视化 | totalseg |

### 4. llava-med (LLaVA-Med 专用)
```bash
conda activate llava-med
```
- **用途**: LLaVA-Med v1.5 医学影像多模态推理
- **核心包**: transformers 4.36.2, torch 2.6.0+cu118
- **模型路径**: `/media/luzhenyang/project/ChangHai_PDA/data/models/llava-med-v1.5-mistral-7b`
- **模型大小**: ~15 GB
- **依赖修复**: triton==2.2.0 (解决 `No module named 'triton.ops'` 错误)

```bash
export nnUNet_raw_data_base="/media/luzhenyang/project/ChangHai_PDA/data/models/nnunet/nnunet_v1_workspace/raw"
export nnUNet_preprocessed="/media/luzhenyang/project/ChangHai_PDA/data/models/nnunet/nnunet_v1_workspace/preprocessed"
export RESULTS_FOLDER="/media/luzhenyang/project/ChangHai_PDA/data/models/nnunet/nnunet_v1_workspace/results"
```

## 技术栈与数据流

```
DICOM → NIfTI → TotalSegmentator (器官/血管) → 评估报告
                ↘ nnU-Net v1 (肿瘤) ↗
```

| 步骤 | 工具 | 输出 |
|------|------|------|
| 1. 器官分割 | TotalSegmentator | 胰腺、血管等正常结构 |
| 2. 肿瘤分割 | nnU-Net v1 (MSD Task07) | 真实肿瘤 Mask (5.25ml) |
| 3. 侵犯评估 | vascular_topology.py | 血管包绕角度 |
| 4. 临床报告 | final_integrated_assessment.py | 可切除性判定 |

## MSD Task07 模型

- **训练数据**: Medical Segmentation Decathlon (MSD) 胰腺数据集
- **标签定义**: 0=背景, 1=胰腺实质, 2=肿瘤
- **输入要求**: 静脉期 CT (Venous Phase)
- **预训练权重位置**: `/media/luzhenyang/project/ChangHai_PDA/data/models/nnunet/nnunet_v1_workspace/results/nnUNet/3d_fullres/Task007_Pancreas/`
- **大小**: 2.7GB

## PyTorch 2.6 兼容性补丁

在以下文件中为 `torch.load()` 添加 `weights_only=False` 参数:

1. `~/anaconda3/envs/nnunetv2/lib/python3.10/site-packages/nnunet/training/model_restore.py` (第147行)
2. `~/anaconda3/envs/nnunetv2/lib/python3.10/site-packages/nnunet/run/load_pretrained_weights.py` (第21行)
3. `~/anaconda3/envs/nnunetv2/lib/python3.10/site-packages/nnunet/training/network_training/network_trainer.py`

## 关键输出文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 肿瘤Mask | `data/processed/segmentations/nnunet_tumor_output/true_tumor_mask.nii.gz` | 二值化肿瘤掩膜 |
| 评估报告 | `data/results/json/final_assessment_result.json` | 结构化血管侵犯评估 |
| 临床可视化 | `data/results/images/integrated_assessment.png` | 300 DPI 诊断图 |
| DICOM数据 | `data/raw/dicom/` | 原始DICOM文件 |
| NIfTI数据 | `data/processed/nifti/nifti_output/` | 转换后的NIfTI文件 |
| 血管分割 | `data/processed/segmentations/segmentations/C3L-03348/` | TotalSegmentator输出 |

## 临床评估结果示例 (患者 C3L-03348)

- **肿瘤体积**: 5.25 ml
- **可切除性**: Resectable (可切除)
- **血管侵犯**: 无 (0/16 血管接触)
- **手术建议**: 适合行 Whipple 手术

## LLaVA-Med 本地推理

### 启动服务

```bash
conda activate llava-med

# 1. 启动控制器
python -m llava.serve.controller --host 0.0.0.0 --port 10000

# 2. 启动模型 worker (另一个终端)
python -m llava.serve.model_worker \
  --host 0.0.0.0 \
  --controller http://localhost:10000 \
  --port 40000 \
  --worker http://localhost:40000 \
  --model-path /media/luzhenyang/project/ChangHai_PDA/data/models/llava-med-v1.5-mistral-7b \
  --multi-modal
```

### 验证脚本

```bash
conda run -n llava-med python scripts/verify_llava_med.py
```

---
*最后更新: 2026-03-25*
