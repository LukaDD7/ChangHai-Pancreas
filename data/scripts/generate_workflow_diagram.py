#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ADW Workflow Diagram Generator
Shows complete pipeline from DICOM to CEO Decision
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(20, 28))
ax.set_xlim(0, 20)
ax.set_ylim(0, 28)
ax.axis('off')

# Color definitions
colors = {
    'input': '#E3F2FD',      # Light blue - input
    'process': '#FFF3E0',    # Light orange - process
    'model': '#E8F5E9',      # Light green - model
    'detection': '#FCE4EC',  # Light pink - detection
    'decision': '#F3E5F5',   # Light purple - decision
    'output': '#E0F7FA',     # Cyan - output
    'warning': '#FFEBEE',    # Red - warning/conflict
    'highlight': '#FFF9C4',  # Yellow - highlight
}

def draw_box(ax, x, y, width, height, text, color, text_color='black',
             fontsize=10, border_color=None, border_width=1.5, bold=False):
    """Draw rounded rectangle"""
    if border_color is None:
        border_color = '#333333'

    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=color,
        edgecolor=border_color,
        linewidth=border_width
    )
    ax.add_patch(box)

    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            color=text_color, weight=weight, wrap=True)

def draw_arrow(ax, start, end, color='#666666', style='->', width=2):
    """Draw arrow"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        linewidth=width,
        mutation_scale=15
    )
    ax.add_patch(arrow)

def draw_dashed_box(ax, x, y, width, height, label, color):
    """Draw dashed box (grouping)"""
    rect = Rectangle(
        (x, y), width, height,
        fill=False,
        edgecolor=color,
        linewidth=2,
        linestyle='--',
        alpha=0.7
    )
    ax.add_patch(rect)
    ax.text(x + width/2, y + height - 0.3, label,
            ha='center', va='top', fontsize=11,
            color=color, weight='bold', style='italic')

# ==================== Title ====================
ax.text(10, 27.5, 'ADW (AI Diagnostic Workflow) Technical Pipeline',
        ha='center', va='center', fontsize=18, weight='bold')
ax.text(10, 27.0, 'Patient CL-03356 (C3L-03356) | Stage IIB PDAC | 3.5cm Tumor',
        ha='center', va='center', fontsize=12, style='italic', color='#666')

# ==================== Layer 1: Raw Data ====================
draw_box(ax, 8, 25.5, 4, 1, 'Raw DICOM Data\n(CPTAC-PDA)', colors['input'], fontsize=11, bold=True)

# ==================== Layer 2: Preprocessing ====================
draw_dashed_box(ax, 0.5, 22.5, 19, 2.5, 'Step 1: Data Preprocessing & Spatial Standardization', '#1976D2')

draw_box(ax, 1, 23, 3.5, 1.2, 'DICOM Parsing\n• Metadata Extraction\n• Window Parameters', colors['process'])
draw_arrow(ax, (10, 25.5), (2.75, 24.2))

draw_box(ax, 5, 23, 4, 1.2, 'DICOM -> NIfTI\n• Format Conversion\n• RAS Coordinate', colors['process'])
draw_arrow(ax, (4.5, 23.6), (5, 23.6))

draw_box(ax, 9.5, 23, 4.5, 1.2, 'Spatial Standardization\n• Resample to 1.0mm³\n• Isotropic', colors['highlight'], border_color='#F57C00', border_width=2)
draw_arrow(ax, (9, 23.6), (9.5, 23.6))

draw_box(ax, 14.5, 23, 4.5, 1.2, 'Validation Output\n• Shape: (~360)³\n• HU: [-1024, 1365]', colors['process'])
draw_arrow(ax, (14, 23.6), (14.5, 23.6))

# ==================== Layer 3: Multi-modal Segmentation ====================
draw_dashed_box(ax, 0.5, 18.5, 19, 3.5, 'Step 2: Multi-modal Segmentation Pipeline', '#388E3C')

# TotalSegmentator branch
draw_box(ax, 1, 20, 4, 1.5, 'TotalSegmentator\n(Organ Segmentation)\n• Pancreas Parenchyma\n• Vascular Structures', colors['model'], border_color='#388E3C')
draw_arrow(ax, (2.75, 23), (3, 20))

draw_box(ax, 1, 18, 4, 1.2, 'Output: pancreas.nii.gz\nVolume: ~65ml', colors['output'])
draw_arrow(ax, (3, 20), (3, 19.2))

# nnU-Net branch
draw_box(ax, 6, 20, 4, 1.5, 'nnU-Net v1\n(MSD Task07)\n• Tumor Segmentation\n• Labels: 0,1,2', colors['model'], border_color='#388E3C')
draw_arrow(ax, (11.75, 23), (8, 20))

draw_box(ax, 6, 18, 4, 1.2, 'Output: FALSE NEGATIVE!\nTumor: 0ml', colors['warning'], border_color='#D32F2F', border_width=2)
draw_arrow(ax, (8, 20), (8, 19.2))

# Clinical Gold Standard (validation only)
draw_box(ax, 11, 20, 4, 1.5, 'Clinical Gold Std\n(Pathology TSV)\n• Tumor: 3.5cm\n• Stage IIB', colors['highlight'], border_color='#FBC02D')
draw_arrow(ax, (11.75, 25.5), (13, 20), color='#999', style='->', width=1)
ax.text(11.5, 22.5, '(validation only)', fontsize=8, color='#999', style='italic')

# Spatial consistency check
draw_box(ax, 16, 20, 3, 1.5, 'Spatial Check\n• RAS Coordinates\n• Layer Alignment', colors['highlight'], border_color='#F57C00')
draw_arrow(ax, (5, 18.6), (16, 20.5), color='#666', style='->', width=1.5)
draw_arrow(ax, (8, 18.6), (16, 20.5), color='#666', style='->', width=1.5)

# ==================== Layer 4: Master Slice Extraction ====================
draw_dashed_box(ax, 0.5, 14.5, 19, 3.5, 'Step 3: Master Slice Extraction (Multi-window Tiled)', '#F57C00')

draw_box(ax, 1, 16.5, 4, 1.2, 'Pancreas Centroid Calc\n• Per-layer Area Stats\n• Max Area Layer Loc', colors['process'])
draw_arrow(ax, (3, 18), (3, 16.5))

draw_box(ax, 6, 16.5, 3.5, 1.2, 'Z=145 Extraction\n• Max Pancreas Area\n• Tumor Location', colors['output'])
draw_arrow(ax, (5, 17.1), (6, 17.1))

# Multi-window processing
draw_box(ax, 10.5, 17.3, 4.5, 1.2, 'Multi-window Processing\n• Standard: 40/400\n• Narrow: 40/150 ★\n• Soft Tissue: 50/250', colors['highlight'], border_color='#F57C00', border_width=2)
draw_arrow(ax, (9.5, 17.1), (10.5, 17.8))

draw_box(ax, 10.5, 15.3, 4.5, 1.2, 'Tiled Image Concat\nHorizontal 1536×512\nContrast↑2.7x', colors['highlight'], border_color='#F57C00')
draw_arrow(ax, (12.75, 17.3), (12.75, 16.5))

draw_box(ax, 16, 16.3, 3, 1.5, 'Output:\nmaster_slice_tiled.png', colors['output'])
draw_arrow(ax, (15, 16.9), (16, 16.9))

# ==================== Layer 5: Vision-Language Analysis ====================
draw_dashed_box(ax, 0.5, 11, 19, 3, 'Step 4: LLaVA-Med Vision-Language Analysis', '#7B1FA2')

draw_box(ax, 1, 12, 3.5, 1.5, 'LLaVA-Med v1.5\nMistral-7B\nVision Encoder', colors['model'], border_color='#7B1FA2')
draw_arrow(ax, (17.5, 16.3), (2.75, 12.8))

draw_box(ax, 5.5, 12, 4, 1.5, 'Multi-window Input\nTiled Image\nTriple-window Analysis', colors['highlight'], border_color='#7B1FA2')
draw_arrow(ax, (4.5, 12.75), (5.5, 12.75))

draw_box(ax, 10.5, 12, 4.5, 1.5, 'Semantic Extraction\n• irregular\n• hypo-attenuating\n• mass effect\n• suspicious', colors['detection'], border_color='#C2185B')
draw_arrow(ax, (9.5, 12.75), (10.5, 12.75))

draw_box(ax, 16, 12, 3, 1.5, 'LLaVA Output:\nHighly Suspect PDAC\nScore: 3.4', colors['output'])
draw_arrow(ax, (15, 12.75), (16, 12.75))

# ==================== Layer 6: Conflict Detection ====================
draw_dashed_box(ax, 0.5, 6.5, 19, 4, 'Step 5: Endogenous Cognitive Dissonance Monitoring', '#C2185B')

# Input comparison
draw_box(ax, 1, 9, 3.5, 1.2, 'nnU-Net Result\nTumor: 0ml', colors['warning'])
draw_arrow(ax, (8, 18), (2.75, 9.6), color='#666', style='->', width=1.5)

draw_box(ax, 5.5, 9, 4, 1.2, 'LLaVA-Med Result\nSuspicion Score: 3.4/1.5\n✓ Exceeds Threshold', colors['highlight'], border_color='#F57C00')
draw_arrow(ax, (17.5, 12), (7.5, 9.6))

# Endogenous Detector
draw_box(ax, 10.5, 8.5, 4.5, 2, 'EndogenousConflictDetector\n━━━━━━━━━━━━━━━━━━━\n• Semantic Keyword Matching\n• Weighted Scoring System\n• No Gold Std Required\n━━━━━━━━━━━━━━━━━━━\nCognitive Dissonance', colors['detection'], border_color='#C2185B', border_width=3, bold=True)
draw_arrow(ax, (4.5, 9.6), (10.5, 9.5))
draw_arrow(ax, (9.5, 9.6), (10.5, 9.5))

# Detection result
draw_box(ax, 16, 8.5, 3, 2, 'Conflict Detected:\nENDOGENOUS_\nFALSE_\nNEGATIVE\n━━━━━━━━\nSeverity: HIGH', colors['warning'], border_color='#D32F2F', border_width=3, bold=True)
draw_arrow(ax, (15, 9.5), (16, 9.5))

# Trigger mechanism
draw_box(ax, 6, 7, 8, 1, 'Trigger: Suspicion Score > 1.5 AND Tumor Volume = 0  →  Auto Escalate', colors['highlight'], border_color='#F57C00')
draw_arrow(ax, (12.75, 8.5), (12.75, 8), color='#D32F2F', width=3)

# ==================== Layer 7: CEO Decision ====================
draw_dashed_box(ax, 0.5, 3.5, 19, 2.5, 'Step 6: ADW CEO Integrated Decision', '#6A1B9A')

draw_box(ax, 1, 4, 4, 1.5, 'Confidence Assessment\nnnU-Net: LOW\nLLaVA-Med: MEDIUM\nClinical: HIGH', colors['decision'], border_color='#6A1B9A')
draw_arrow(ax, (17.5, 8.5), (3, 4.8), color='#666', style='->', width=1.5)

draw_box(ax, 6, 4, 4, 1.5, 'Pathological Analysis\nDesmoplastic\nReaction\nScirrhous Carcinoma', colors['decision'], border_color='#6A1B9A')
draw_arrow(ax, (5, 4.75), (6, 4.75))

draw_box(ax, 11, 4, 4, 1.5, 'Final Diagnosis\nTumor: 3.5cm ✓\nStage IIB ✓\nPancreatic Head ✓', colors['output'], border_color='#6A1B9A', border_width=2)
draw_arrow(ax, (10, 4.75), (11, 4.75))

draw_box(ax, 16, 4, 3, 1.5, 'Recommendation:\nManual Review\nMulti-window CT', colors['decision'], border_color='#6A1B9A')
draw_arrow(ax, (15, 4.75), (16, 4.75))

# ==================== Bottom: Output Files ====================
draw_dashed_box(ax, 0.5, 0.5, 19, 2.5, 'Output Files & Reports', '#00695C')

outputs = [
    (1.5, 'master_slice_tiled.png', colors['output']),
    (5.5, 'semantic_features.json', colors['detection']),
    (9.5, 'dissonance_analysis.yaml', colors['warning']),
    (13.5, 'llava_med_report.txt', colors['model']),
    (17, 'conflict_report.txt', colors['decision']),
]

for i, (x, name, color) in enumerate(outputs):
    draw_box(ax, x, 1, 3, 1.2, name.replace('_', '\n'), color, fontsize=9)
    if i == 0:
        draw_arrow(ax, (17.5, 4), (x+1.5, 2.2))

# ==================== Right: Key Tech Annotations ====================
annotations = [
    (20.5, 23, '🔧 Spatial Std', '#F57C00'),
    (20.5, 20, '⚠️ False Neg', '#D32F2F'),
    (20.5, 17, '🎯 Multi-window', '#F57C00'),
    (20.5, 12, '🔍 Semantic Ext', '#C2185B'),
    (20.5, 9, '💡 Endogenous', '#C2185B'),
    (20.5, 4, '📊 CEO Decision', '#6A1B9A'),
]

for x, y, text, color in annotations:
    ax.text(x, y, text, fontsize=10, color=color, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))

# ==================== Bottom Left: Legend ====================
legend_items = [
    ('Input Data', colors['input']),
    ('Process Step', colors['process']),
    ('AI Model', colors['model']),
    ('Conflict Detection', colors['detection']),
    ('Decision Output', colors['decision']),
    ('Key Innovation', colors['highlight']),
    ('Warning/Error', colors['warning']),
]

for i, (label, color) in enumerate(legend_items):
    y_pos = 0.3 + i * 0.25
    rect = Rectangle((0.5, y_pos), 0.4, 0.2, facecolor=color, edgecolor='#333')
    ax.add_patch(rect)
    ax.text(1, y_pos + 0.1, label, fontsize=8, va='center')

plt.tight_layout()
plt.savefig('/media/luzhenyang/project/ChangHai_PDA/ADW_Workflow_Diagram.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Workflow diagram saved: /media/luzhenyang/project/ChangHai_PDA/ADW_Workflow_Diagram.png")

# ============= Generate Comparison Diagram =============
fig2, ax2 = plt.subplots(1, 1, figsize=(16, 20))
ax2.set_xlim(0, 16)
ax2.set_ylim(0, 20)
ax2.axis('off')

ax2.text(8, 19.5, 'Endogenous Conflict Detection: Innovation vs Traditional',
         ha='center', va='center', fontsize=16, weight='bold')

# Left: Traditional Pipeline
ax2.text(4, 18.5, 'X Traditional Pipeline', ha='center', fontsize=13, weight='bold', color='#D32F2F')
draw_dashed_box(ax2, 0.5, 12, 7, 5.5, 'Requires Gold Standard', '#D32F2F')

y_start = 17
rect = FancyBboxPatch((1, y_start), 6, 1, boxstyle="round,pad=0.02,rounding_size=0.15",
                       facecolor=colors['model'], edgecolor='#333', linewidth=1.5)
ax2.add_patch(rect)
ax2.text(4, y_start + 0.5, '1. nnU-Net Segmentation', ha='center', va='center', fontsize=10)

rect = FancyBboxPatch((1, y_start-1.5), 6, 1, boxstyle="round,pad=0.02,rounding_size=0.15",
                       facecolor=colors['model'], edgecolor='#333', linewidth=1.5)
ax2.add_patch(rect)
ax2.text(4, y_start-1.5 + 0.5, '2. LLaVA-Med Analysis', ha='center', va='center', fontsize=10)

rect = FancyBboxPatch((1, y_start-3), 6, 1, boxstyle="round,pad=0.02,rounding_size=0.15",
                       facecolor=colors['input'], edgecolor='#333', linewidth=1.5)
ax2.add_patch(rect)
ax2.text(4, y_start-3 + 0.5, '3. Compare with Clinical GS', ha='center', va='center', fontsize=10)

rect = FancyBboxPatch((1, y_start-4.5), 6, 1.2, boxstyle="round,pad=0.02,rounding_size=0.15",
                       facecolor=colors['warning'], edgecolor='#D32F2F', linewidth=1.5)
ax2.add_patch(rect)
ax2.text(4, y_start-4.5 + 0.6, '4. Conflict Detected?\n(Needs Gold Std!)', ha='center', va='center', fontsize=10)

ax2.text(4, 11, 'Problems:\n• No GS in production\n• Cannot auto-detect errors\n• Passive manual review',
         ha='center', fontsize=10, color='#D32F2F',
         bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))

# Right: ADW Autonomous Agent
ax2.text(12, 18.5, 'OK ADW Autonomous Agent', ha='center', fontsize=13, weight='bold', color='#388E3C')
draw_dashed_box(ax2, 8.5, 12, 7, 5.5, 'Endogenous Conflict Detection', '#388E3C')

rect = FancyBboxPatch((9, y_start), 6, 1, boxstyle="round,pad=0.02,rounding_size=0.15",
                       facecolor=colors['model'], edgecolor='#333', linewidth=1.5)
ax2.add_patch(rect)
ax2.text(12, y_start + 0.5, '1. nnU-Net Segmentation', ha='center', va='center', fontsize=10)

rect = FancyBboxPatch((9, y_start-1.5), 6, 1, boxstyle="round,pad=0.02,rounding_size=0.15",
                       facecolor=colors['model'], edgecolor='#333', linewidth=1.5)
ax2.add_patch(rect)
ax2.text(12, y_start-1.5 + 0.5, '2. LLaVA-Med Analysis', ha='center', va='center', fontsize=10)

rect = FancyBboxPatch((9, y_start-3), 6, 1.5, boxstyle="round,pad=0.02,rounding_size=0.15",
                       facecolor=colors['detection'], edgecolor='#C2185B', linewidth=1.5)
ax2.add_patch(rect)
ax2.text(12, y_start-3 + 0.75, '3. Semantic Keyword Extraction\n(No Gold Std!)', ha='center', va='center', fontsize=10)

rect = FancyBboxPatch((9, y_start-5), 6, 1.2, boxstyle="round,pad=0.02,rounding_size=0.15",
                       facecolor=colors['highlight'], edgecolor='#F57C00', linewidth=1.5)
ax2.add_patch(rect)
ax2.text(12, y_start-5 + 0.6, '4. Cognitive Dissonance Monitor\n(Auto Conflict Detection)', ha='center', va='center', fontsize=10)

ax2.text(12, 11, 'Advantages:\n• Semantic Score Auto-triggers\n• No pathology needed\n• Proactive Alert System',
         ha='center', fontsize=10, color='#388E3C',
         bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))

ax2.text(12, 11, 'Advantages:\n• Semantic Score Auto-triggers\n• No pathology needed\n• Proactive Alert System',
         ha='center', fontsize=10, color='#388E3C',
         bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))

# Comparison arrow
ax2.annotate('', xy=(8.3, 14), xytext=(7.7, 14),
            arrowprops=dict(arrowstyle='->', color='#666', lw=2))
ax2.text(8, 14.5, 'vs', ha='center', fontsize=12, weight='bold')

# Key innovation box
innovation_text = """Key Innovation: Cognitive Dissonance Monitoring

Trigger Logic:
  IF suspicion_score > 1.5 AND tumor_volume = 0
  THEN escalate_to_radiologist()

Keywords & Weights:
  • mass, tumor, lesion (1.0)
  • irregular, hypo-attenuating (0.8)
  • suspicious, abnormal (0.6)
  • pancreatic head, SMV (0.4)

CL-03356 Case:
  Score: 3.4 / 1.5 = 2.27x threshold
  -> ENDOGENOUS_FALSE_NEGATIVE detected"""

ax2.text(8, 4, innovation_text, ha='center', va='center', fontsize=10,
         family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4',
                  edgecolor='#F57C00', linewidth=2, alpha=0.9))

plt.tight_layout()
plt.savefig('/media/luzhenyang/project/ChangHai_PDA/ADW_Conflict_Mechanism.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Comparison diagram saved: /media/luzhenyang/project/ChangHai_PDA/ADW_Conflict_Mechanism.png")
