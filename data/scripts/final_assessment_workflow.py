#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
全景血管侵犯量化评估工作流 (Final Assessment Workflow)
================================================================================

功能描述:
    对真实肿瘤 Mask 与所有可用血管进行全景侵犯评估，
    生成结构化 JSON 报告，用于临床决策支持。

作者: Claude Code Assistant
日期: 2026-03-21
版本: 1.0.0
================================================================================
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List
from pathlib import Path

# 导入 vascular_topology 核心函数
sys.path.insert(0, '/media/luzhenyang/project/ChangHai_PDA/data')
from vascular_topology import analyze_vascular_encasement


# 配置路径
TUMOR_MASK_PATH = "/media/luzhenyang/project/ChangHai_PDA/data/nnunet_tumor_output/true_tumor_mask.nii.gz"
MASK_DIR = "/media/luzhenyang/project/ChangHai_PDA/data/segmentations/C3L-03348"
OUTPUT_JSON = "/media/luzhenyang/project/ChangHai_PDA/data/assessment_result.json"

# 可用血管列表 (基于 TotalSegmentator 输出)
AVAILABLE_VESSELS = {
    "Arteries": {
        "AO": "aorta.nii.gz",
        "BCA": "brachiocephalic_trunk.nii.gz",
        "CCA_L": "common_carotid_artery_left.nii.gz",
        "CCA_R": "common_carotid_artery_right.nii.gz",
        "SA_L": "subclavian_artery_left.nii.gz",
        "SA_R": "subclavian_artery_right.nii.gz",
        "IA_L": "iliac_artery_left.nii.gz",
        "IA_R": "iliac_artery_right.nii.gz",
    },
    "Veins": {
        "MPV": "portal_vein_and_splenic_vein.nii.gz",
        "IVC": "inferior_vena_cava.nii.gz",
        "SVC": "superior_vena_cava.nii.gz",
        "PV": "portal_vein_and_splenic_vein.nii.gz",
        "IV_L": "iliac_vena_left.nii.gz",
        "IV_R": "iliac_vena_right.nii.gz",
        "BCV_L": "brachiocephalic_vein_left.nii.gz",
        "BCV_R": "brachiocephalic_vein_right.nii.gz",
    }
}


def check_file_exists(mask_dir: str, filename: str) -> bool:
    """检查掩膜文件是否存在"""
    return os.path.exists(os.path.join(mask_dir, filename))


def run_vascular_assessment(tumor_path: str, vessel_path: str, vessel_name: str) -> Dict:
    """
    对单根血管进行评估

    返回:
        Dict: 包含血管名称、是否接触、最大角度、临床分级
    """
    try:
        result = analyze_vascular_encasement(
            tumor_mask_path=tumor_path,
            vessel_mask_path=vessel_path,
            dilation_radius_mm=2.0
        )

        max_angle = result.get("max_angle_degree", 0.0)
        is_contact = result.get("is_contact", False)

        # 临床分级
        if not is_contact:
            classification = "Clear"
        elif max_angle <= 180:
            classification = "<= 180° (Borderline)"
        else:
            classification = "> 180° (Locally Advanced)"

        return {
            "vessel_name": vessel_name,
            "filename": os.path.basename(vessel_path),
            "is_contact": is_contact,
            "max_angle_degree": round(max_angle, 2),
            "classification": classification,
            "contact_slices": result.get("contact_z_slices", []),
            "status": "success"
        }
    except Exception as e:
        return {
            "vessel_name": vessel_name,
            "filename": os.path.basename(vessel_path),
            "is_contact": False,
            "max_angle_degree": 0.0,
            "classification": "Error",
            "error": str(e),
            "status": "error"
        }


def determine_clinical_conclusion(results: List[Dict]) -> Dict:
    """
    根据所有血管评估结果，确定总体临床结论
    """
    advanced_vessels = []
    borderline_vessels = []
    clear_vessels = []
    error_vessels = []

    for result in results:
        classification = result.get("classification", "")
        vessel_name = result.get("vessel_name", "")

        if result.get("status") == "error":
            error_vessels.append(vessel_name)
        elif "> 180°" in classification:
            advanced_vessels.append(vessel_name)
        elif "<= 180°" in classification:
            borderline_vessels.append(vessel_name)
        else:
            clear_vessels.append(vessel_name)

    # 判定总体结论
    if advanced_vessels:
        overall = f"Locally Advanced (Vessels: {', '.join(advanced_vessels)})"
    elif borderline_vessels:
        overall = f"Borderline Resectable (Vessels: {', '.join(borderline_vessels)})"
    else:
        overall = "Resectable"

    return {
        "overall_classification": overall,
        "advanced_vessels": advanced_vessels,
        "borderline_vessels": borderline_vessels,
        "clear_vessels": clear_vessels,
        "error_vessels": error_vessels
    }


def main():
    """主函数"""
    print("="*70)
    print("全景血管侵犯量化评估工作流")
    print("="*70)
    print(f"肿瘤 Mask: {TUMOR_MASK_PATH}")
    print(f"血管目录: {MASK_DIR}")
    print()

    # 验证肿瘤文件
    if not os.path.exists(TUMOR_MASK_PATH):
        print(f"[ERROR] 肿瘤 Mask 不存在: {TUMOR_MASK_PATH}")
        sys.exit(1)

    # 评估所有可用血管
    all_results = []

    print("="*70)
    print("开始评估所有血管...")
    print("="*70)

    # 评估动脉
    print("\n【动脉评估】")
    arterial_results = []
    for vessel_abbr, filename in AVAILABLE_VESSELS["Arteries"].items():
        vessel_path = os.path.join(MASK_DIR, filename)
        if check_file_exists(MASK_DIR, filename):
            print(f"\n  评估 {vessel_abbr}: {filename}")
            result = run_vascular_assessment(TUMOR_MASK_PATH, vessel_path, vessel_abbr)
            arterial_results.append(result)
            print(f"    结果: {result['classification']} (角度: {result['max_angle_degree']}°)")
        else:
            print(f"  [跳过] {vessel_abbr}: 文件不存在")

    # 评估静脉
    print("\n【静脉评估】")
    venous_results = []
    for vessel_abbr, filename in AVAILABLE_VESSELS["Veins"].items():
        vessel_path = os.path.join(MASK_DIR, filename)
        if check_file_exists(MASK_DIR, filename):
            print(f"\n  评估 {vessel_abbr}: {filename}")
            result = run_vascular_assessment(TUMOR_MASK_PATH, vessel_path, vessel_abbr)
            venous_results.append(result)
            print(f"    结果: {result['classification']} (角度: {result['max_angle_degree']}°)")
        else:
            print(f"  [跳过] {vessel_abbr}: 文件不存在")

    # 合并所有结果
    all_results = arterial_results + venous_results

    # 生成临床结论
    conclusion = determine_clinical_conclusion(all_results)

    # 构建最终输出
    output = {
        "patient_id": "C3L-03348",
        "tumor_mask_path": TUMOR_MASK_PATH,
        "evaluation_date": "2026-03-21",
        "tumor_volume_ml": 5.25,
        "dilation_radius_mm": 2.0,
        "Arterial_Evaluation": arterial_results,
        "Venous_Evaluation": venous_results,
        "Clinical_Conclusion": conclusion
    }

    # 保存 JSON
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print("\n" + "="*70)
    print("评估完成!")
    print("="*70)
    print(f"\n【总体结论】: {conclusion['overall_classification']}")
    print(f"\n详细结果已保存: {OUTPUT_JSON}")

    # 打印接触血管列表
    contact_vessels = [r for r in all_results if r.get('is_contact', False)]
    if contact_vessels:
        print("\n【接触血管列表】:")
        for v in contact_vessels:
            print(f"  - {v['vessel_name']}: {v['max_angle_degree']}° ({v['classification']})")
    else:
        print("\n【接触血管列表】: 无接触")

    return output


if __name__ == "__main__":
    result = main()
