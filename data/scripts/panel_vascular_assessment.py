#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
血管评价面板 (Vascular Panel Assessment)
================================================================================

功能描述:
    本脚本模拟长海医院的临床规范，对胰腺肿瘤患者的多根关键血管进行
    批量评估，并输出结构化的"血管评价面板"。这是 vascular_topology.py
    的高层集成封装，专门用于临床决策支持。

临床背景:
    在胰腺肿瘤手术规划中，肿瘤与周围血管的关系直接决定手术可切除性：
    - 动脉侵犯通常意味着更高的手术风险
    - 静脉侵犯在经验丰富中心可考虑切除重建
    - 多根血管受累需要综合评估

长海医院血管映射规范:
    本脚本内置了长海医院常用的血管缩写与 canonical vessel library
    输出文件名的对应关系，便于与临床术语对接。

可切除性判定逻辑:
    - 任何主要动脉 > 180°: Locally Advanced
    - 动脉 <= 180° 或仅静脉受累: Borderline Resectable
    - 所有血管 Clear: Resectable

技术架构:
    - 底层计算: vascular_topology.analyze_vascular_encasement
    - 批量处理: 自动遍历所有关键血管
    - 异常处理: 对缺失的血管分割安全降级
    - 输出格式: 结构化 JSON，便于 LLM/Qwen 直接解析填表

作者: Claude Code Assistant
日期: 2026-03-21
版本: 1.0.0
================================================================================
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# =============================================================================
# 引入底层拓扑计算函数
# =============================================================================
from vascular_topology import analyze_vascular_encasement


# =============================================================================
# 长海医院血管映射字典 (Changhai Vessel Panel Mapping)
# =============================================================================
# 将临床缩写映射到 canonical vessel library 的输出文件名
# 注意：下游统一依赖 canonical snake_case 文件名，而不是具体模型来源

CHANGHAI_VESSEL_PANEL = {
    "Arteries": {
        # 主要动脉 - 动脉侵犯通常意味着更高的手术风险
        "SMA": "superior_mesenteric_artery.nii.gz",      # 肠系膜上动脉
        "CA": "celiac_trunk.nii.gz",                      # 腹腔干
        "CHA": "common_hepatic_artery.nii.gz",           # 肝总动脉
        "SPLA": "splenic_artery.nii.gz",                  # 脾动脉
        "GDA": "gastroduodenal_artery.nii.gz",           # 胃十二指肠动脉
        "AO": "aorta.nii.gz",                             # 腹主动脉（参考血管）
        "IVC": "inferior_vena_cava.nii.gz",               # 下腔静脉（作为动脉组参考）
    },
    "Veins": {
        # 主要静脉 - 静脉侵犯可考虑切除重建
        "MPV": "portal_vein_and_splenic_vein.nii.gz",    # 门静脉+脾静脉
        "SMV": "superior_mesenteric_vein.nii.gz",        # 肠系膜上静脉
        "SV": "splenic_vein.nii.gz",                      # 脾静脉（如果单独分割）
        "IVV": "inferior_vena_cava.nii.gz",               # 下腔静脉（静脉组）
    }
}


# =============================================================================
# 临床分类常量定义
# =============================================================================

# 单个血管的侵犯分类
CLASSIFICATION_CLEAR = "Clear"
CLASSIFICATION_BORDERLINE = "<= 180 degrees"
CLASSIFICATION_ADVANCED = "> 180 degrees"
CLASSIFICATION_NOT_SEGMENTED = "Not Segmented"
CLASSIFICATION_ERROR = "Error"

# 总体可切除性分类
OVERALL_RESECTABLE = "Resectable"
OVERALL_BORDERLINE = "Borderline Resectable"
OVERALL_ADVANCED = "Locally Advanced"
OVERALL_ERROR = "Evaluation Error"


def check_vessel_mask_exists(mask_dir: str, vessel_filename: str) -> Optional[str]:
    """
    检查血管掩膜文件是否存在

    参数:
        mask_dir: 掩膜文件所在目录
        vessel_filename: 血管掩膜文件名

    返回:
        如果存在返回完整路径，否则返回 None
    """
    full_path = os.path.join(mask_dir, vessel_filename)
    if os.path.exists(full_path):
        return full_path
    return None


def evaluate_single_vessel(
    tumor_mask_path: str,
    vessel_mask_path: str,
    vessel_name: str,
    dilation_radius_mm: float = 2.0
) -> Dict:
    """
    评估单根血管的侵犯情况

    参数:
        tumor_mask_path: 肿瘤掩膜路径
        vessel_mask_path: 血管掩膜路径
        vessel_name: 血管名称（用于日志输出）
        dilation_radius_mm: 膨胀半径（毫米）

    返回:
        标准化的血管评估结果字典
    """
    print(f"\n  [评估] {vessel_name}: {os.path.basename(vessel_mask_path)}")

    try:
        # 调用底层拓扑分析函数
        result = analyze_vascular_encasement(
            tumor_mask_path=tumor_mask_path,
            vessel_mask_path=vessel_mask_path,
            dilation_radius_mm=dilation_radius_mm
        )

        # 标准化输出格式
        if result.get("status") == "error":
            return {
                "contact": False,
                "degree": 0,
                "classification": CLASSIFICATION_ERROR,
                "status": "Error",
                "error_message": result.get("error_message", "Unknown error")
            }

        # 根据角度确定分类
        max_angle = result.get("max_angle_degree", 0.0)
        is_contact = result.get("is_contact", False)

        if not is_contact:
            classification = CLASSIFICATION_CLEAR
        elif max_angle <= 180:
            classification = CLASSIFICATION_BORDERLINE
        else:
            classification = CLASSIFICATION_ADVANCED

        return {
            "contact": is_contact,
            "degree": round(max_angle, 1),
            "classification": classification,
            "contact_slices": result.get("contact_z_slices", []),
            "total_contact_slices": result.get("total_z_slices", 0)
        }

    except Exception as e:
        print(f"  [ERROR] 评估 {vessel_name} 时发生异常: {str(e)}")
        return {
            "contact": False,
            "degree": 0,
            "classification": CLASSIFICATION_ERROR,
            "status": "Error",
            "error_message": str(e)
        }


def determine_overall_resectability(
    arterial_results: Dict[str, Dict],
    venous_results: Dict[str, Dict]
) -> str:
    """
    根据所有血管评估结果，确定总体手术可切除性

    长海医院判定逻辑：
    1. 只要有一根主要动脉 > 180° -> Locally Advanced
    2. 动脉 <= 180° 或仅静脉受累 -> Borderline Resectable
    3. 所有血管 Clear -> Resectable

    参数:
        arterial_results: 动脉评估结果字典
        venous_results: 静脉评估结果字典

    返回:
        str: 总体可切除性分类
    """
    # 收集所有血管的分类状态
    all_results = {**arterial_results, **venous_results}

    # 检查是否有错误
    has_error = any(
        result.get("classification") == CLASSIFICATION_ERROR
        for result in all_results.values()
        if "classification" in result
    )
    if has_error:
        return OVERALL_ERROR

    # 统计各分类的血管数量
    advanced_vessels = []
    borderline_vessels = []
    clear_vessels = []

    for vessel_name, result in all_results.items():
        classification = result.get("classification", CLASSIFICATION_NOT_SEGMENTED)

        if classification == CLASSIFICATION_ADVANCED:
            advanced_vessels.append(vessel_name)
        elif classification == CLASSIFICATION_BORDERLINE:
            borderline_vessels.append(vessel_name)
        elif classification == CLASSIFICATION_CLEAR:
            clear_vessels.append(vessel_name)

    # 主要动脉列表（用于特殊判断）
    major_arteries = ["SMA", "CA", "CHA"]  # 最重要的三根动脉

    # 判定逻辑
    # 1. 检查主要动脉是否有 > 180° 的情况
    major_artery_advanced = [
        v for v in advanced_vessels
        if v in major_arteries
    ]

    if major_artery_advanced:
        return f"{OVERALL_ADVANCED} (Major Arteries: {', '.join(major_artery_advanced)})"

    # 2. 检查是否有任何动脉 > 180°
    any_artery_advanced = [
        v for v in advanced_vessels
        if v in arterial_results
    ]

    if any_artery_advanced:
        return f"{OVERALL_ADVANCED} (Arteries: {', '.join(any_artery_advanced)})"

    # 3. 检查是否有动脉 <= 180° 或静脉 > 180°
    if borderline_vessels:
        return f"{OVERALL_BORDERLINE} ({', '.join(borderline_vessels)})"

    # 4. 所有血管 Clear
    if len(clear_vessels) == len([v for v in all_results.values() if "classification" in v]):
        return OVERALL_RESECTABLE

    # 默认情况（部分血管未分割）
    return f"{OVERALL_BORDERLINE} (Partial Evaluation)"


def run_full_vascular_panel(
    tumor_mask_path: str,
    mask_dir: str,
    dilation_radius_mm: float = 2.0,
    custom_panel: Optional[Dict] = None
) -> Dict:
    """
    核心函数: 执行完整的血管评价面板评估

    这是本脚本的主入口函数，批量评估所有关键血管并生成结构化报告。

    参数:
        tumor_mask_path: 肿瘤掩膜文件路径（NIfTI格式）
        mask_dir: 血管掩膜文件所在目录
        dilation_radius_mm: 形态学膨胀半径（毫米），默认2mm
        custom_panel: 自定义血管面板字典（可选，默认为CHANGHAI_VESSEL_PANEL）

    返回:
        Dict: 结构化的血管评价面板结果，包含：
            - Arterial_Evaluation: 动脉评估结果
            - Venous_Evaluation: 静脉评估结果
            - Overall_Surgical_Resectability: 总体可切除性结论
            - Summary_Statistics: 统计摘要

    示例:
        >>> result = run_full_vascular_panel(
        ...     tumor_mask_path="tumor.nii.gz",
        ...     mask_dir="/path/to/segmentations/",
        ...     dilation_radius_mm=2.0
        ... )
        >>> print(json.dumps(result, indent=2, ensure_ascii=False))
    """
    print(f"\n{'='*70}")
    print("血管评价面板 - Vascular Panel Assessment")
    print(f"{'='*70}")
    print(f"[INFO] 肿瘤掩膜: {tumor_mask_path}")
    print(f"[INFO] 掩膜目录: {mask_dir}")
    print(f"[INFO] 膨胀半径: {dilation_radius_mm}mm")

    # 使用默认面板或自定义面板
    vessel_panel = custom_panel if custom_panel else CHANGHAI_VESSEL_PANEL

    # 验证肿瘤掩膜存在
    if not os.path.exists(tumor_mask_path):
        error_msg = f"肿瘤掩膜文件不存在: {tumor_mask_path}"
        print(f"[ERROR] {error_msg}")
        return {
            "status": "error",
            "error_message": error_msg
        }

    # ======================================================================
    # 动脉评估
    # ======================================================================
    print(f"\n{'-'*70}")
    print("动脉评估 - Arterial Evaluation")
    print(f"{'-'*70}")

    arterial_results = {}
    arterial_panel = vessel_panel.get("Arteries", {})

    for vessel_abbr, vessel_filename in arterial_panel.items():
        vessel_path = check_vessel_mask_exists(mask_dir, vessel_filename)

        if vessel_path is None:
            print(f"\n  [跳过] {vessel_abbr}: 掩膜文件不存在 ({vessel_filename})")
            arterial_results[vessel_abbr] = {
                "status": CLASSIFICATION_NOT_SEGMENTED,
                "filename": vessel_filename
            }
            continue

        # 评估该血管
        result = evaluate_single_vessel(
            tumor_mask_path=tumor_mask_path,
            vessel_mask_path=vessel_path,
            vessel_name=vessel_abbr,
            dilation_radius_mm=dilation_radius_mm
        )
        arterial_results[vessel_abbr] = result

    # ======================================================================
    # 静脉评估
    # ======================================================================
    print(f"\n{'-'*70}")
    print("静脉评估 - Venous Evaluation")
    print(f"{'-'*70}")

    venous_results = {}
    venous_panel = vessel_panel.get("Veins", {})

    for vessel_abbr, vessel_filename in venous_panel.items():
        vessel_path = check_vessel_mask_exists(mask_dir, vessel_filename)

        if vessel_path is None:
            print(f"\n  [跳过] {vessel_abbr}: 掩膜文件不存在 ({vessel_filename})")
            venous_results[vessel_abbr] = {
                "status": CLASSIFICATION_NOT_SEGMENTED,
                "filename": vessel_filename
            }
            continue

        # 评估该血管
        result = evaluate_single_vessel(
            tumor_mask_path=tumor_mask_path,
            vessel_mask_path=vessel_path,
            vessel_name=vessel_abbr,
            dilation_radius_mm=dilation_radius_mm
        )
        venous_results[vessel_abbr] = result

    # ======================================================================
    # 总体可切除性判定
    # ======================================================================
    print(f"\n{'-'*70}")
    print("总体可切除性判定 - Overall Surgical Resectability")
    print(f"{'-'*70}")

    overall_resectability = determine_overall_resectability(
        arterial_results,
        venous_results
    )

    print(f"\n[RESULT] 总体结论: {overall_resectability}")

    # ======================================================================
    # 统计摘要
    # ======================================================================
    all_results = {**arterial_results, **venous_results}

    segmented_vessels = [v for v in all_results.values() if "classification" in v]
    not_segmented_vessels = [k for k, v in all_results.items() if "status" in v and v["status"] == CLASSIFICATION_NOT_SEGMENTED]

    critical_pancreatic_vessels = ["SMA", "CA", "CHA", "SPLA", "GDA", "SMV", "SV", "MPV"]
    missing_critical_vessels = [
        vessel for vessel in critical_pancreatic_vessels
        if vessel in all_results and vessel in not_segmented_vessels
    ]

    summary_stats = {
        "total_vessels_in_panel": len(all_results),
        "segmented_vessels": len(segmented_vessels),
        "not_segmented_vessels": len(not_segmented_vessels),
        "not_segmented_list": not_segmented_vessels,
        "missing_critical_vessels": missing_critical_vessels,
        "vascular_reconstruction_ready": len(missing_critical_vessels) == 0,
        "reconstruction_limitation": "3D vascular geometry is incomplete without SMA/SMV/CA/CHA/SV masks"
        if missing_critical_vessels else "All critical vessels are available for 3D assessment",
        "advanced_vessels": [
            k for k, v in all_results.items()
            if v.get("classification") == CLASSIFICATION_ADVANCED
        ],
        "borderline_vessels": [
            k for k, v in all_results.items()
            if v.get("classification") == CLASSIFICATION_BORDERLINE
        ],
        "clear_vessels": [
            k for k, v in all_results.items()
            if v.get("classification") == CLASSIFICATION_CLEAR
        ]
    }

    # ======================================================================
    # 结果封装
    # ======================================================================
    final_result = {
        "status": "success",
        "tumor_mask": tumor_mask_path,
        "mask_directory": mask_dir,
        "dilation_radius_mm": dilation_radius_mm,
        "Arterial_Evaluation": arterial_results,
        "Venous_Evaluation": venous_results,
        "Overall_Surgical_Resectability": overall_resectability,
        "Summary_Statistics": summary_stats
    }

    print(f"\n{'='*70}")
    print("血管评价面板评估完成")
    print(f"{'='*70}")

    return final_result


# =============================================================================
# 测试入口 (Test Entrypoint)
# =============================================================================

if __name__ == "__main__":
    """
    本地调试和测试用例

    使用方法:
        1. 激活conda环境: conda activate totalseg
        2. 运行脚本: python panel_vascular_assessment.py
        3. 修改下方的路径为您的实际数据路径
    """

    import sys

    print("\n" + "="*70)
    print(" panel_vascular_assessment.py 本地测试模式")
    print("="*70)

    # -------------------------------------------------------------------------
    # 测试配置: 请根据实际情况修改以下路径
    # -------------------------------------------------------------------------

    # 测试用例: 使用患者 C3L-03348 的数据
    # 注意：需要准备真实的肿瘤掩膜和血管掩膜

    # 使用胰腺作为"模拟肿瘤"进行测试
    TEST_TUMOR_MASK = "/media/luzhenyang/project/ChangHai_PDA/data/segmentations/C3L-03348/pancreas.nii.gz"
    TEST_MASK_DIR = "/media/luzhenyang/project/ChangHai_PDA/data/segmentations/C3L-03348"

    # -------------------------------------------------------------------------
    # 执行测试
    # -------------------------------------------------------------------------

    if not os.path.exists(TEST_TUMOR_MASK):
        print(f"[ERROR] 肿瘤掩膜路径不存在: {TEST_TUMOR_MASK}")
        print("[INFO] 请修改脚本中的 TEST_TUMOR_MASK 为实际路径")
        sys.exit(1)

    if not os.path.exists(TEST_MASK_DIR):
        print(f"[ERROR] 掩膜目录不存在: {TEST_MASK_DIR}")
        print("[INFO] 请修改脚本中的 TEST_MASK_DIR 为实际路径")
        sys.exit(1)

    # 执行完整血管面板评估
    try:
        result = run_full_vascular_panel(
            tumor_mask_path=TEST_TUMOR_MASK,
            mask_dir=TEST_MASK_DIR,
            dilation_radius_mm=2.0
        )

        # 打印格式化后的JSON结果
        print("\n" + "="*70)
        print("最终输出 (JSON格式):")
        print("="*70)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # 保存结果到文件
        output_file = "/media/luzhenyang/project/ChangHai_PDA/data/vascular_panel_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] 结果已保存到: {output_file}")

    except Exception as e:
        print(f"\n[ERROR] 测试过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
