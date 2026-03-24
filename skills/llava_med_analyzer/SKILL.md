---
name: llava-med-analyzer
version: 1.0.0
category: vision_language_model
description: |
  [FUNCTION]: LLaVA-Med v1.5 vision-language analysis for PDAC detection in multi-window CT images
  [VALUE]: Provides semantic assessment from VLM analysis using Agent's judgment. CRITICAL for detecting isodense tumors missed by nnU-Net
  [TRIGGER_HOOK]: Read this Skill AFTER master-slice-extractor. Use analyze_image tool on Tiled image. Output semantic assessment for conflict detection
---

# LLaVA-Med Analyzer: Cognitive Execution Protocol

## Identity & Core Mechanism
This skill uses LLaVA-Med v1.5 (Mistral-7B) vision-language model via the `analyze_image` tool to detect PDAC signs in multi-window CT images, especially isodense tumors missed by segmentation.

**Why VLM for Isodense Tumors:**
- nnU-Net relies on HU thresholds (fails for isodense)
- VLM can recognize morphological patterns (irregular, mass effect)
- Multi-window input provides enhanced contrast for human-like perception

---

## Phase 1: Input Validation

**Required Input:**
- Tiled master slice: `/workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_tiled.png`
- Must contain 3 window sections (Standard/Narrow/Soft)

**Verify Input Exists:**
```bash
ls -lh /workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_tiled.png
```

---

## Phase 2: VLM Analysis

**Purpose:** Analyze Tiled image for PDAC signs

**Execution - Use analyze_image Tool:**

The agent MUST call the `analyze_image` tool (provided in main interactive_main.py):

```python
# This is executed by the agent, not via execute command
analyze_image(
    image_path="/workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_tiled.png",
    query="""Evaluate this pancreatic CT image for PDAC (Pancreatic Ductal Adenocarcinoma).

This image shows the SAME pancreatic slice in 3 different window settings:
- LEFT: Standard abdominal window (Center:40, Width:400)
- CENTER: Narrow window (Center:40, Width:150) ← **FOCUS HERE for isodense lesions**
- RIGHT: Soft tissue window (Center:50, Width:250)

The CENTER section (Narrow window) enhances contrast by 2.7x and is crucial for detecting isodense tumors that appear normal in standard window.

Evaluate for:
1. **Irregular contours** in pancreatic head
2. **Hypo-attenuating areas** (darker regions) - especially in narrow window
3. **Heterogeneous density** patterns
4. **Mass effect** on surrounding vessels (SMV, SMA)
5. **Loss of normal lobulation**

Rate your suspicion (0-10) where:
- 0-2: Normal appearance
- 3-4: Mildly suspicious (recommend follow-up)
- 5-6: Moderately suspicious (probable PDAC)
- 7-10: Highly suspicious (definite PDAC features)

Provide specific findings and confidence level."""
)
```

---

## Phase 3: Semantic Analysis (Agent Meta-Cognition)

**Purpose:** Interpret VLM output using YOUR semantic understanding

**NO Hard-Coded Weights:**
Your clinical understanding of medical language IS the analysis framework.

**Instead of keyword counting:**
```python
# ❌ DON'T DO THIS
score = 0
if "mass" in text: score += 1.0  # Rigid, misses context
```

**Use semantic judgment:**
```python
# ✅ DO THIS
Read VLM output holistically:
- Does it describe concerning morphological features?
- Are there mentions of mass effect or architectural distortion?
- How confident is the language? ("possible" vs "definite")
- Does it note hypo-attenuation in narrow window?

Your interpretation of these semantic cues IS the suspicion assessment.
```

**Output Format:**
```json
{
  "patient_id": "{PATIENT_ID}",
  "vlm_assessment": {
    "suspicion_level": "HIGH/MODERATE/LOW/NONE",
    "confidence": "Your assessment of VLM confidence",
    "key_findings": ["List of specific findings from VLM"],
    "reasoning": "Why you classified it this way"
  },
  "raw_vlm_output": "..."
}
```

**Execution:**
```bash
conda run -n ChangHai python /skills/llava_med_analyzer/scripts/parse_suspicion.py \
    --vlm-output "{VLM_OUTPUT_FILE}" \
    --patient-id {PATIENT_ID} \
    --output /workspace/sandbox/data/results/json/{PATIENT_ID}_suspicion_score.json
```

---

## Phase 4: LLaVA-Med Report Generation

**Purpose:** Generate structured VLM analysis report

**Execution Command:**
```bash
conda run -n ChangHai python /skills/llava_med_analyzer/scripts/generate_report.py \
    --patient-id {PATIENT_ID} \
    --suspicion-score {SCORE} \
    --vlm-findings "{FINDINGS}" \
    --output /workspace/sandbox/data/results/json/{PATIENT_ID}_llava_med_report.txt
```

**Expected Output Format:**
```markdown
# LLaVA-Med Visual Analysis Report

## Patient Information
- Patient ID: {PATIENT_ID}
- Analysis Date: {timestamp}
- Image: master_slice_tiled.png (Z={value})

## Window Settings Analyzed
1. Standard: C=40, W=400
2. **Narrow: C=40, W=150** (Enhanced contrast)
3. Soft Tissue: C=50, W=250

## Suspicion Score
**Total Score: {value}/10**

Keywords Detected:
- irregular (morphological, +0.8)
- hypo-attenuating (morphological, +0.8)
- mass effect (morphological, +0.8)
- suspicious (suspicious, +0.6)
...

## Clinical Impression
{VLM generated impression}

## Findings Detail
{VLM detailed findings}

## Recommendation
- If Score >= 1.5: Recommend immediate radiologist review
- If Score < 1.5: Continue with standard protocol
```

---

## Output Files

1. **VLM Analysis Report:**
   - Path: `/workspace/sandbox/data/results/json/{PATIENT_ID}_llava_med_report.txt`

2. **Suspicion Score JSON:**
   - Path: `/workspace/sandbox/data/results/json/{PATIENT_ID}_suspicion_score.json`

3. **Keywords Extracted:**
   - Path: `/workspace/sandbox/data/results/json/{PATIENT_ID}_semantic_features.json`

---

## CRITICAL: False Negative Detection

**Conflict Detection (Agent Judgment):**

| nnU-Net Result | VLM Assessment | Action |
|----------------|----------------|--------|
| 0 ml | Normal/Nonspecific | No conflict |
| 0 ml | **Suspicious findings** | **ENDOGENOUS_FALSE_NEGATIVE** |
| > 0 ml | Consistent with tumor | Consistent positive |
| > 0 ml | Contradicts segmentation | Investigate discrepancy |

**Your Role:**
- Compare nnU-Net result with VLM semantic description
- Use YOUR judgment: do they agree or conflict?
- If VLM describes suspicious features but nnU-Net finds nothing → ESCALATE
- No hard threshold. Your clinical reasoning IS the threshold.

**CL-03356 Example:**
```
nnU-Net: 0 ml tumor
VLM: "Irregular contour in pancreatic head with subtle hypo-attenuation
      in narrow window, concerning for mass effect"
Your judgment: VLM describes suspicious morphology despite nnU-Net negative
Action: ENDOGENOUS_FALSE_NEGATIVE → ESCALATE_TO_RADIOLOGIST
Root Cause: Desmoplastic reaction causing isodense appearance
```

---

## Quality Checkpoints

- [ ] analyze_image tool executed successfully
- [ ] VLM output contains specific findings
- [ ] Semantic assessment completed using your judgment
- [ ] Report generated with physical citations
- [ ] If VLM suspicious AND nnU-Net=0, flag conflict

**Citation Format:**
- VLM analysis: `[Tool: LLaVA-Med, Model: v1.5-mistral-7b, Finding: {description}]`
- Semantic assessment: `[Agent Assessment: {level}, Reasoning: {summary}]`

---

## Error Handling

**Common Issues:**

1. **VLM timeout:**
   - Image may be too large (>10MB)
   - Retry with compressed image

2. **Vague output:**
   - Refine query to be more specific
   - Ask for explicit suspicion score

3. **No keywords detected:**
   - Check VLM output quality
   - May need manual review

**Note:** LLaVA-Med may occasionally produce empty output due to model loading issues. If this occurs, use simplified rule-based scoring based on the presence of key terms in the raw output.
