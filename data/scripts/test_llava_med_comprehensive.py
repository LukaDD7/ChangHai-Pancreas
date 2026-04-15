#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLaVA-Med 全面临床测试套件
基于官方使用方式：单窗位输入，非拼接

测试覆盖5层诊断流程：
1. 实性 vs 囊性鉴别
2. PDAC vs pNET 鉴别
3. 囊性肿瘤四分型 (SCN/MCN/IPMN/SPN)
4. 急性胰腺炎严重度评估
5. 慢性胰腺炎评估

官方使用规范：
- 单张图像输入 (非tiled拼接)
- 使用 vicuna_v1 对话模板
- 温度0.2，top_p 0.95
- max_new_tokens 512

注意：CUDA_VISIBLE_DEVICES 必须在导入torch前设置
"""

import os
import sys
import json
from PIL import Image
from datetime import datetime

# ============ 测试配置 ============
MODEL_PATH = "/media/luzhenyang/project/ChangHai_PDA/data/models/llava-med-v1.5-mistral-7b"
TEST_PATIENT = "CL-03356"
TEST_IMAGE_DIR = "/media/luzhenyang/project/ChangHai_PDA/data/results/archive/CL-03356"
OUTPUT_DIR = "/media/luzhenyang/project/ChangHai_PDA/data/results/llava_med_tests"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 延迟导入的模块（在设置CUDA_VISIBLE_DEVICES后导入）
torch = None
load_pretrained_model = None
get_model_name_from_path = None
process_images = None
tokenizer_image_token = None
conv_templates = None
LLAVA_AVAILABLE = False

def initialize_llava(gpu_id=6):
    """延迟初始化LLaVA-Med模块"""
    global torch, load_pretrained_model, get_model_name_from_path
    global process_images, tokenizer_image_token, conv_templates, LLAVA_AVAILABLE

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 现在导入torch和LLaVA
    import torch as _torch
    torch = _torch

    sys.path.insert(0, '/media/luzhenyang/project/ChangHai_PDA/data/models/LLaVA-Med')

    try:
        from llava.model.builder import load_pretrained_model as _load
        from llava.mm_utils import get_model_name_from_path as _get_name
        from llava.mm_utils import process_images as _proc
        from llava.mm_utils import tokenizer_image_token as _tokenize
        from llava.conversation import conv_templates as _conv

        load_pretrained_model = _load
        get_model_name_from_path = _get_name
        process_images = _proc
        tokenizer_image_token = _tokenize
        conv_templates = _conv
        LLAVA_AVAILABLE = True
        print(f"✅ LLaVA-Med 初始化成功 (GPU {gpu_id})")
    except ImportError as e:
        print(f"❌ LLaVA-Med 导入失败: {e}")
        LLAVA_AVAILABLE = False

    return LLAVA_AVAILABLE

TEST_CASES = {
    # ========== Layer 1: 实性 vs 囊性鉴别 ==========
    "L1_solid_cystic": {
        "name": "实性vs囊性肿瘤鉴别",
        "clinical_scenario": "胰腺占位性病变的初步定性",
        "window": "standard",  # 标准窗即可区分
        "prompt": """You are a radiologist specializing in pancreatic imaging. Analyze this CT image and classify the pancreatic lesion as solid or cystic.

Classification criteria:
- Solid tumor: CT density >20 HU, shows enhancement, soft tissue attenuation
- Cystic tumor: CT density <20 HU, fluid attenuation, no internal enhancement (wall/septation may enhance)

Provide your assessment in this format:
- Classification: [Solid / Cystic / Indeterminate]
- Confidence: [High / Moderate / Low]
- Key CT findings supporting your decision:
- Differential considerations if uncertain:""",
        "expected_output_fields": ["classification", "confidence", "ct_density", "enhancement_pattern"]
    },

    # ========== Layer 2: PDAC vs pNET 鉴别 ==========
    "L2_pdac_vs_pnet": {
        "name": "PDAC vs pNET 实性肿瘤鉴别",
        "clinical_scenario": "实性胰腺肿瘤的分型诊断",
        "window": "arterial",  # 动脉期对强化模式最关键
        "prompt": """You are a radiologist specializing in pancreatic imaging. Differentiate between Pancreatic Ductal Adenocarcinoma (PDAC) and Pancreatic Neuroendocrine Tumor (pNET) in this CT image.

Key distinguishing features:

PDAC characteristics:
- Hypovascular (hypo-enhancing relative to pancreas)
- Arterial phase: low attenuation
- Infiltrative margins, irregular borders
- Ductal involvement: double duct sign, abrupt cutoff
- Vascular encasement common

pNET characteristics:
- Hypervascular (hyper-enhancing in arterial phase)
- Arterial phase: bright enhancement
- Well-circumscribed, round/oval shape
- Displaces rather than encases vessels
- May have cystic degeneration or calcification

Analyze and provide:
- Primary diagnosis: [PDAC / pNET / Indeterminate]
- Confidence: [0-100%]
- Enhancement pattern: [Hypervascular / Hypovascular / Isovascular]
- Margin characteristics:
- Vessel relationship: [Encasement / Displacement / No involvement]
- Ductal findings:
- Supporting evidence:""",
        "expected_output_fields": ["diagnosis", "confidence", "enhancement", "margins", "vessel_relationship"]
    },

    # ========== Layer 3: 囊性肿瘤四分型 ==========
    "L3_cystic_subtypes": {
        "name": "囊性肿瘤四分型 (SCN/MCN/IPMN/SPN)",
        "clinical_scenario": "囊性胰腺肿瘤的精确分型",
        "window": "venous",  # 静脉期评估壁结节和分隔
        "prompt": """You are a radiologist specializing in pancreatic imaging. Classify this cystic pancreatic lesion into one of four subtypes: SCN, MCN, IPMN, or SPN.

Diagnostic criteria for each subtype:

1. Serous Cystadenoma (SCN):
   - Microcystic (multiple small cysts >6), lobulated appearance
   - Central scar with calcification (classic but not always present)
   - Thin walls, no enhancing mural nodules
   - No communication with pancreatic duct

2. Mucinous Cystic Neoplasm (MCN):
   - Macrocystic (fewer, larger cysts), round/oval shape
   - Thick wall (>2mm), may have enhancing mural nodules (≥5mm = concerning)
   - Typically body/tail location
   - No duct communication
   - Ovarian-type stroma (pathology)

3. Intraductal Papillary Mucinous Neoplasm (IPMN):
   - Communication with pancreatic duct is KEY feature
   - Main duct type: diffuse duct dilation (≥10mm)
   - Branch duct type: grape-like/cluster of cysts
   - May have enhancing mural nodules (high risk feature)
   - "Finger-like" projections into duct

4. Solid Pseudopapillary Neoplasm (SPN):
   - Young female demographic (20-40 years)
   - Mixed solid-cystic components
   - Large size often (>5cm)
   - Peripheral enhancing solid components
   - May have hemorrhage, calcification rare

Provide:
- Most likely diagnosis: [SCN / MCN / IPMN / SPN / Uncertain]
- Confidence level: [High / Moderate / Low]
- Key imaging features observed:
- Cyst number and size pattern:
- Wall characteristics:
- Duct communication: [Present / Absent / Unclear]
- Mural nodules: [Present (size) / Absent]
- Patient demographic considerations:""",
        "expected_output_fields": ["diagnosis", "confidence", "cyst_pattern", "duct_communication", "mural_nodule"]
    },

    # ========== Layer 4: 急性胰腺炎严重度评估 ==========
    "L4_acute_pancreatitis": {
        "name": "急性胰腺炎CT严重度指数 (CTSI)",
        "clinical_scenario": "急性胰腺炎影像学严重度评估",
        "window": "venous",  # 静脉期评估坏死
        "prompt": """You are a radiologist specializing in pancreatic imaging. Assess the severity of acute pancreatitis using the CT Severity Index (CTSI) / Modified CT Severity Index (MCTSI).

Balthazar CT Grade (assess extent of inflammation):
- Grade A: Normal pancreas (0 points)
- Grade B: Focal enlargement of pancreas (1 point)
- Grade C: Pancreatic gland abnormalities with peripancreatic inflammation (2 points)
- Grade D: Single fluid collection (3 points)
- Grade E: Multiple fluid collections or gas bubbles (4 points)

Necrosis Assessment (venous phase - look for non-enhancing areas):
- No necrosis: 0 points
- <30% necrosis: 2 points
- 30-50% necrosis: 4 points
- >50% necrosis: 6 points

Complications to identify:
- Acute Peripancreatic Fluid Collection (APFC): <4 weeks, no wall
- Acute Necrotic Collection (ANC): <4 weeks, heterogeneous
- Walled-off Necrosis (WON): >4 weeks, encapsulated
- Pseudocyst: >4 weeks, homogeneous fluid
- Infected necrosis: gas bubbles (diagnostic)

Evaluate and provide:
- Balthazar Grade: [A / B / C / D / E] (X points)
- Necrosis: [None / <30% / 30-50% / >50%] (X points)
- MCTSI Total Score: [0-10]
- Severity: [Mild (0-3) / Moderate (4-6) / Severe (7-10)]
- Complications identified: [List or "None"]
- Key findings:
- Pancreatic enhancement pattern:
- Peripancreatic collections: [Describe location, number, characteristics]""",
        "expected_output_fields": ["balthazar_grade", "necrosis_percentage", "mctsi_score", "severity", "complications"]
    },

    # ========== Layer 5: 慢性胰腺炎评估 ==========
    "L5_chronic_pancreatitis": {
        "name": "慢性胰腺炎影像评分",
        "clinical_scenario": "慢性胰腺炎严重度评估",
        "window": "standard",  # 标准窗评估钙化和萎缩
        "prompt": """You are a radiologist specializing in pancreatic imaging. Assess chronic pancreatitis using the Cambridge/MRCP criteria.

Key imaging findings to evaluate:

1. Pancreatic calcifications:
   - None: 0 points
   - Mild (<7 punctate calcifications): 1 point
   - Moderate (7-49 punctate or <7 coarse): 2 points
   - Severe (≥50 punctate or ≥7 coarse): 3 points

2. Main pancreatic duct changes:
   - Normal diameter (≤3mm body): Normal
   - Mild dilation (3-5mm): Mild
   - Moderate dilation (5-10mm): Moderate
   - Severe dilation (>10mm) or strictures: Severe
   - Beading/irregularity: Score +1

3. Side branch dilatation:
   - Number of dilated side branches (>1mm)

4. Pancreatic atrophy:
   - Mild: <25% volume loss
   - Moderate: 25-50% volume loss
   - Severe: >50% volume loss

5. Complications:
   - Pseudocysts (>4cm or persistent)
   - Biliary stricture/dilation
   - Splenic vein thrombosis
   - Portal hypertension

Provide:
- Calcification grade: [None / Mild / Moderate / Severe]
- Main duct status: [Normal / Mild / Moderate / Severe dilation], strictures [Present / Absent]
- Duct beading: [Present / Absent]
- Atrophy: [None / Mild / Moderate / Severe]
- Complications: [List or "None identified"]
- Overall severity: [Normal / Mild / Moderate / Severe chronic pancreatitis]
- Cambridge classification: [Normal / Equivocal / Mild / Moderate / Severe]""",
        "expected_output_fields": ["calcification_grade", "duct_dilation", "atrophy", "severity", "cambridge_class"]
    },

    # ========== 附加测试: 血管侵犯评估 ==========
    "L6_vascular_invasion": {
        "name": "血管侵犯与可切除性评估",
        "clinical_scenario": "PDAC可切除性判断",
        "window": "arterial",  # 动脉期评估血管
        "prompt": """You are a radiologist specializing in pancreatic imaging. Evaluate vascular involvement for pancreatic cancer resectability assessment (NCCN criteria).

Assess contact with these vessels:
- Superior Mesenteric Artery (SMA)
- Superior Mesenteric Vein (SMV)
- Celiac Artery (CA)
- Common Hepatic Artery (CHA)
- Portal Vein (PV)

Resectability criteria (NCCN 2025):

RESECTABLE:
- No arterial contact (SMA, CA, CHA)
- No venous involvement OR ≤180° contact without contour deformity

BORDERLINE RESECTABLE (BRPC):
- Solid tumor contact with CHA without extension to CA or HA bifurcation
- Solid tumor contact with SMA ≤180°
- Solid tumor contact with CA ≤180° (if body/tail tumor)
- Short-segment SMV/PV involvement amenable to reconstruction

LOCALLY ADVANCED (LAPC/Unresectable):
- Solid tumor contact with SMA >180°
- Solid tumor contact with CA >180° (any location)
- Aortic contact
- Unreconstructable SMV/PV involvement

Evaluate and provide:
- Tumor-vessel contact for each vessel:
  * SMA: [No contact / ≤180° / >180° / Encasement]
  * SMV: [No contact / ≤180° / >180° / Occlusion]
  * CA: [No contact / ≤180° / >180° / Encasement]
  * CHA: [No contact / ≤180° / >180° / Encasement]
  * PV: [No contact / ≤180° / >180° / Occlusion]
- Resectability assessment: [Resectable / Borderline Resectable / Locally Advanced / Metastatic]
- Recommended management: [Upfront surgery / Neoadjuvant therapy / Palliative care]
- Key concerning findings:""",
        "expected_output_fields": ["sma_contact", "smv_contact", "ca_contact", "resectability", "recommendation"]
    },

    # ========== 附加测试: 双管征检测 ==========
    "L7_double_duct_sign": {
        "name": "双管征检测 (胰管+胆管扩张)",
        "clinical_scenario": "PDAC特征性征象识别",
        "window": "standard",
        "prompt": """You are a radiologist specializing in pancreatic imaging. Evaluate for the "double duct sign" - a key indicator of pancreatic head pathology.

Double Duct Sign:
- Simultaneous dilation of both:
  1. Common bile duct (CBD)
  2. Main pancreatic duct (MPD)
- Indicates obstruction at the ampulla/pancreatic head
- Highly suggestive of malignancy (especially PDAC in pancreatic head)

Measure/Assess:
- Common bile duct diameter (normal ≤6mm, >10mm definitely dilated)
- Main pancreatic duct diameter (normal ≤3mm body, >5mm considered dilated)
- Level of obstruction
- Presence of abrupt cutoff vs gradual tapering

Provide:
- Double duct sign: [Present / Absent / Uncertain]
- CBD diameter: [Normal / Mildly dilated / Moderately dilated / Severely dilated] - estimate in mm if possible
- MPD diameter: [Normal / Mildly dilated / Moderately dilated / Severely dilated] - estimate in mm if possible
- Level of obstruction: [Pancreatic head / Ampulla / Distal CBD / Other]
- Associated findings:
- Clinical significance: [Highly suggestive of malignancy / Benign cause possible / Indeterminate]""",
        "expected_output_fields": ["double_duct_sign", "cbd_diameter", "mpd_diameter", "obstruction_level", "significance"]
    }
}

# ============ 全局模型缓存 ============
_global_model = None
_global_tokenizer = None
_global_image_processor = None
_global_context_len = None
_global_device_id = None

def load_model_once(model_path=MODEL_PATH, device_id=6):
    """只加载一次模型"""
    global _global_model, _global_tokenizer, _global_image_processor, _global_context_len, _global_device_id

    if _global_model is not None and _global_device_id == device_id:
        return _global_model, _global_tokenizer, _global_image_processor, _global_context_len

    # 如果有模型但设备不同，先清理
    if _global_model is not None:
        del _global_model
        del _global_tokenizer
        del _global_image_processor
        torch.cuda.empty_cache()
        _global_model = None

    device = f'cuda:0'  # 设置CUDA_VISIBLE_DEVICES后只有一个GPU可见
    print(f"  Loading model from {model_path} on GPU {device_id}...")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device=device
    )

    _global_model = model
    _global_tokenizer = tokenizer
    _global_image_processor = image_processor
    _global_context_len = context_len
    _global_device_id = device_id

    return model, tokenizer, image_processor, context_len

# ============ 测试执行函数 ============

def run_official_llava_med_inference(image_path, prompt, model_path=MODEL_PATH, device_id=6):
    """
    按照官方标准方式运行LLaVA-Med推理
    """
    if not LLAVA_AVAILABLE:
        return {"error": "LLaVA-Med not available"}

    # 检查文件
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}"}

    try:
        # 加载模型（只加载一次）
        model, tokenizer, image_processor, context_len = load_model_once(model_path, device_id)

        # 加载并处理图像
        print(f"  Processing image...")
        image = Image.open(image_path).convert('RGB')
        images_tensor = process_images([image], image_processor, model.config)
        images_tensor = images_tensor.to(model.device, dtype=torch.float16)

        # 准备对话 (官方 vicuna_v1 模板)
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], f"<image>\n{prompt}")
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # 编码输入
        input_ids = tokenizer_image_token(
            prompt_text,
            tokenizer,
            image_token_index=tokenizer.convert_tokens_to_ids("<image>"),
            return_tensors='pt'
        ).unsqueeze(0).to(model.device)

        # 生成 (官方参数)
        print(f"  Generating response...")
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                max_new_tokens=512,
                use_cache=True,
                temperature=0.2,
                top_p=0.95,
            )

        # 解码输出
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # 清理本次推理的临时变量（但保留模型）
        del images_tensor
        del input_ids
        del output_ids
        torch.cuda.empty_cache()

        return {
            "success": True,
            "response": outputs,
            "image_size": image.size,
            "model": "llava-med-v1.5-mistral-7b"
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


def run_single_test(test_key, test_config, image_path, device_id=6):
    """
    运行单个测试用例
    """
    print(f"\n{'='*70}")
    print(f"测试: {test_config['name']}")
    print(f"场景: {test_config['clinical_scenario']}")
    print(f"窗位: {test_config['window']}")
    print(f"设备: GPU {device_id}")
    print(f"{'='*70}")

    result = run_official_llava_med_inference(image_path, test_config['prompt'], device_id=device_id)

    if "error" in result:
        print(f"❌ 错误: {result['error']}")
        return {"test_key": test_key, "status": "error", "error": result['error']}

    print(f"\n{'='*70}")
    print("模型回复:")
    print(f"{'='*70}")
    print(result['response'])
    print(f"{'='*70}")

    return {
        "test_key": test_key,
        "test_name": test_config['name'],
        "status": "success",
        "response": result['response'],
        "image_path": image_path,
        "window": test_config['window']
    }


def run_comprehensive_tests():
    """
    运行全部临床测试
    """
    print("="*70)
    print("LLaVA-Med 全面临床测试套件")
    print("官方使用方式: 单窗位输入 | vicuna_v1 模板")
    print("="*70)
    print(f"测试患者: {TEST_PATIENT}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not LLAVA_AVAILABLE:
        print("\n❌ LLaVA-Med 库不可用，无法进行测试")
        return

    # 查找测试图像
    # 使用标准窗图像（非tiled）
    image_candidates = [
        f"{TEST_IMAGE_DIR}/{TEST_PATIENT}_master_slice.png",
        f"{TEST_IMAGE_DIR}/{TEST_PATIENT}_arterial.png",
        f"{TEST_IMAGE_DIR}/{TEST_PATIENT}_venous.png",
        f"{TEST_IMAGE_DIR}/{TEST_PATIENT}_standard.png",
    ]

    test_image = None
    for img_path in image_candidates:
        if os.path.exists(img_path):
            test_image = img_path
            print(f"\n✅ 使用测试图像: {img_path}")
            break

    if not test_image:
        print("\n❌ 未找到测试图像，请先生成切片图像")
        return

    # 运行所有测试
    results = []
    summary = []

    for test_key, test_config in TEST_CASES.items():
        result = run_single_test(test_key, test_config, test_image)
        results.append(result)

        if result['status'] == 'success':
            summary.append(f"✅ {test_config['name']}: 成功")
        else:
            summary.append(f"❌ {test_config['name']}: 失败 - {result.get('error', 'Unknown')}")

    # 保存完整结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_DIR}/llava_med_comprehensive_test_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_info": {
                "patient_id": TEST_PATIENT,
                "test_image": test_image,
                "model_path": MODEL_PATH,
                "timestamp": timestamp,
                "test_framework": "Official LLaVA-Med single-window input"
            },
            "test_cases": {k: {key: v[key] for key in ['name', 'clinical_scenario', 'window']} for k, v in TEST_CASES.items()},
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("测试摘要")
    print(f"{'='*70}")
    for s in summary:
        print(s)

    print(f"\n✅ 完整结果已保存: {output_file}")

    # 清理模型
    global _global_model, _global_tokenizer, _global_image_processor
    if _global_model is not None:
        del _global_model
        del _global_tokenizer
        del _global_image_processor
        _global_model = None
        _global_tokenizer = None
        _global_image_processor = None
        if torch is not None:
            torch.cuda.empty_cache()
        print("\n🧹 模型已清理，显存已释放")

    return results


def run_comprehensive_tests(device_id=6):
    """
    运行全部临床测试
    """
    # 先初始化LLaVA
    if not initialize_llava(device_id):
        print("\n❌ LLaVA-Med 初始化失败")
        return

    print("="*70)
    print("LLaVA-Med 全面临床测试套件")
    print("官方使用方式: 单窗位输入 | vicuna_v1 模板")
    print("="*70)
    print(f"测试患者: {TEST_PATIENT}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"推理设备: GPU {device_id}")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 查找测试图像
    # 使用标准窗图像（非tiled）
    image_candidates = [
        f"{TEST_IMAGE_DIR}/{TEST_PATIENT}_master_slice.png",
        f"{TEST_IMAGE_DIR}/{TEST_PATIENT}_arterial.png",
        f"{TEST_IMAGE_DIR}/{TEST_PATIENT}_venous.png",
        f"{TEST_IMAGE_DIR}/{TEST_PATIENT}_standard.png",
    ]

    test_image = None
    for img_path in image_candidates:
        if os.path.exists(img_path):
            test_image = img_path
            print(f"\n✅ 使用测试图像: {img_path}")
            break

    if not test_image:
        print("\n❌ 未找到测试图像，请先生成切片图像")
        return

    # 运行所有测试
    results = []
    summary = []

    for test_key, test_config in TEST_CASES.items():
        result = run_single_test(test_key, test_config, test_image, device_id=device_id)
        results.append(result)

        if result['status'] == 'success':
            summary.append(f"✅ {test_config['name']}: 成功")
        else:
            summary.append(f"❌ {test_config['name']}: 失败 - {result.get('error', 'Unknown')}")

    # 保存完整结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_DIR}/llava_med_comprehensive_test_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_info": {
                "patient_id": TEST_PATIENT,
                "test_image": test_image,
                "model_path": MODEL_PATH,
                "timestamp": timestamp,
                "test_framework": "Official LLaVA-Med single-window input",
                "device": f"GPU {device_id}"
            },
            "test_cases": {k: {key: v[key] for key in ['name', 'clinical_scenario', 'window']} for k, v in TEST_CASES.items()},
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("测试摘要")
    print(f"{'='*70}")
    for s in summary:
        print(s)

    print(f"\n✅ 完整结果已保存: {output_file}")

def run_comprehensive_tests(device_id=6):
    """
    运行全部临床测试
    """
    # 先初始化LLaVA
    if not initialize_llava(device_id):
        print("\n❌ LLaVA-Med 初始化失败")
        return

    print("="*70)
    print("LLaVA-Med 全面临床测试套件")
    print("官方使用方式: 单窗位输入 | vicuna_v1 模板")
    print("="*70)
    print(f"测试患者: {TEST_PATIENT}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"推理设备: GPU {device_id}")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 查找测试图像
    # 使用标准窗图像（非tiled）
    image_candidates = [
        f"{TEST_IMAGE_DIR}/{TEST_PATIENT}_master_slice.png",
        f"{TEST_IMAGE_DIR}/{TEST_PATIENT}_arterial.png",
        f"{TEST_IMAGE_DIR}/{TEST_PATIENT}_venous.png",
        f"{TEST_IMAGE_DIR}/{TEST_PATIENT}_standard.png",
    ]

    test_image = None
    for img_path in image_candidates:
        if os.path.exists(img_path):
            test_image = img_path
            print(f"\n✅ 使用测试图像: {img_path}")
            break

    if not test_image:
        print("\n❌ 未找到测试图像，请先生成切片图像")
        return

    # 运行所有测试
    results = []
    summary = []

    for test_key, test_config in TEST_CASES.items():
        result = run_single_test(test_key, test_config, test_image, device_id=device_id)
        results.append(result)

        if result['status'] == 'success':
            summary.append(f"✅ {test_config['name']}: 成功")
        else:
            summary.append(f"❌ {test_config['name']}: 失败 - {result.get('error', 'Unknown')}")

    # 保存完整结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_DIR}/llava_med_comprehensive_test_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_info": {
                "patient_id": TEST_PATIENT,
                "test_image": test_image,
                "model_path": MODEL_PATH,
                "timestamp": timestamp,
                "test_framework": "Official LLaVA-Med single-window input",
                "device": f"GPU {device_id}"
            },
            "test_cases": {k: {key: v[key] for key in ['name', 'clinical_scenario', 'window']} for k, v in TEST_CASES.items()},
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("测试摘要")
    print(f"{'='*70}")
    for s in summary:
        print(s)

    print(f"\n✅ 完整结果已保存: {output_file}")

    # 清理模型
    global _global_model, _global_tokenizer, _global_image_processor
    if _global_model is not None:
        del _global_model
        del _global_tokenizer
        del _global_image_processor
        _global_model = None
        _global_tokenizer = None
        _global_image_processor = None
        torch.cuda.empty_cache()
        print("\n🧹 模型已清理，显存已释放")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LLaVA-Med Comprehensive Clinical Tests')
    parser.add_argument('--gpu', type=int, default=6, help='GPU device ID (default: 6)')
    args = parser.parse_args()

    # run_comprehensive_tests会在内部设置CUDA_VISIBLE_DEVICES
    run_comprehensive_tests(device_id=args.gpu)
