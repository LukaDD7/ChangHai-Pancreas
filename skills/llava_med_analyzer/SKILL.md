---
name: llava_med_analyzer
category: visual_reasoning
description: LLaVA-Med/Qwen-VL vision-language analysis for PDAC detection. CRITICAL for Deep Drill Protocol when nnU-Net returns 0ml. Provides semantic assessment of isodense tumors invisible to HU-threshold segmentation.
---

# LLaVA-Med Analyzer: Cognitive Execution Protocol

## 1. Identity & Clinical Mindset
You are the Visual Reasoning Specialist. Your goal is to analyze multi-window CT images using Vision-Language Models (VLM) to detect isodense tumors missed by traditional segmentation.

**Key Principle**: Human radiologists don't just threshold HU values. They recognize patterns: irregular contours, mass effect, architectural distortion. VLM simulates this visual reasoning.

**Deep Drill Context**:
- Called when nnU-Net returns 0ml (suspected false negative)
- Called to verify positive segmentation findings
- Called for MDT presentation and radiologist communication

## 2. API Contract (Execution)
**Tool**: `analyze_image` (provided in main interactive_main.py)
**Model**: Qwen-VL or LLaVA-Med v1.5
**Input**: Tiled PNG image (1536×512, 3 windows)
**Output**: Semantic assessment of tumor presence

**(Agent, you MUST have Tiled image before calling!)**

## 3. Cognitive Reasoning & SOP

### Step 1: Verify Prerequisites
```bash
# Check Tiled image exists (DISCOVER, don't assume)
find /workspace/sandbox/data/results/images -name "*{PATIENT_ID}*tiled*.png" 2>/dev/null

# Expected output:
# /workspace/sandbox/data/results/images/C3L-03356_master_slice_tiled.png
```

### Step 2: Call analyze_image Tool
```python
analyze_image(
    image_path="/workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_tiled.png",
    query="""Analyze this pancreatic CT for PDAC (Pancreatic Ductal Adenocarcinoma).

Image layout (left to right):
- LEFT: Standard window (W:400, C:40) - general anatomy
- CENTER: Narrow window (W:150, C:40) - ⭐ ISODENSE TUMOR DETECTION
- RIGHT: Soft tissue window (W:250, C:50) - vessel boundaries

Focus on the CENTER (narrow window) section. Isodense tumors appear here.

Evaluate for:
1. Irregular pancreatic head contour
2. Hypo-attenuating regions (darker in narrow window)
3. Heterogeneous texture
4. Mass effect on SMV/SMA
5. Loss of normal lobulation

Provide:
- Suspicion level: NONE / LOW / MODERATE / HIGH
- Key findings (bullet points)
- Confidence: DEFINITE / PROBABLE / POSSIBLE / UNLIKELY"""
)
```

### Step 3: Semantic Assessment (Agent Meta-Cognition)
**NO Hard-Coded Scoring**: Your understanding IS the assessment.

**Instead of keyword counting:**
```python
# ❌ DON'T DO THIS
score = 0
if "mass" in text: score += 1.0  # Rigid!
```

**Use holistic clinical judgment:**
```python
# ✅ DO THIS
Read VLM output and assess:
- Does it describe concerning morphology?
- How confident is the language? ("definite" vs "possible")
- Are findings consistent across all 3 windows?
- Does narrow window reveal what standard window hides?

Your clinical reasoning IS the assessment.
```

### Step 4: Decision Matrix

| nnU-Net | VLM Assessment | Conclusion | Action |
|---------|----------------|------------|--------|
| 0ml | NONE/LOW | True negative | Standard negative report |
| 0ml | MODERATE/HIGH | **ENDOGENOUS_FALSE_NEGATIVE** | ⚠️ ESCALATE_TO_RADIOLOGIST |
| >0ml | Consistent | Confirmed positive | Proceed to vascular topology |
| >0ml | Contradicts | Discrepancy | Investigate segmentation quality |

## 4. Error Handling

| Error | Diagnosis | Action |
|-------|-----------|--------|
| Image not found | Tiled not generated | Go to master_slice_extractor |
| VLM timeout | Image too large (>10MB) | Check file size, compress if needed |
| Vague output | Poor image quality | Try different query or window settings |

## 5. Success Criteria
- [ ] Tiled image discovered and exists
- [ ] analyze_image tool executed
- [ ] VLM output contains specific findings
- [ ] Semantic assessment completed using Agent judgment
- [ ] Conflict flagged if VLM suspicious + nnU-Net 0ml

## 6. Citation Format
```
[Tool: analyze_image, Image: {PATIENT_ID}_master_slice_tiled.png, VLM_Assessment: {level}, Key_Findings: {summary}]
```
