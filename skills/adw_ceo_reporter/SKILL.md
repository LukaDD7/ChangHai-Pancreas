---
name: adw-ceo-reporter
version: 1.0.0
category: conflict_detection
description: |
  [FUNCTION]: ADW CEO (AI Diagnostic Workflow Chief Executive Officer) - Endogenous conflict detection and reconciliation
  [VALUE]: Detects cognitive dissonance between nnU-Net and LLaVA-Med WITHOUT gold standard using Agent's semantic judgment. Generates comprehensive conflict report with clinical recommendations
  [TRIGGER_HOOK]: Read this Skill AFTER nnunet-segmentor AND llava-med-analyzer. ALWAYS run for conflict detection. Activate when VLM findings conflict with nnU-Net result
---

# ADW CEO Reporter: Cognitive Execution Protocol

## Identity & Core Mechanism
This skill implements the Chief Executive Officer pattern for AI diagnostic workflows. It detects conflicts between segmentation models and vision-language models WITHOUT requiring clinical gold standard.

**Core Innovation: Endogenous Conflict Detection**
- Traditional: Requires gold standard to detect false negatives
- ADW CEO: Uses semantic analysis to detect conflicts autonomously
- Your Role: Compare nnU-Net result with VLM semantic findings using YOUR judgment
- Trigger: When VLM describes suspicious features but nnU-Net reports no tumor

---

## Phase 1: Input Aggregation

**Required Inputs:**
1. nnU-Net Result: `/workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/tumor_analysis.json`
2. LLaVA Result: `/workspace/sandbox/data/results/json/{PATIENT_ID}_suspicion_score.json`
3. Clinical Data (optional): TSV with gold standard

**Read Inputs:**
```bash
cat /workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/tumor_analysis.json
cat /workspace/sandbox/data/results/json/{PATIENT_ID}_suspicion_score.json
```

---

## Phase 2: Endogenous Conflict Detection

**Purpose:** Detect cognitive dissonance using your semantic understanding (NOT hard-coded weights)

**Meta-Cognitive Approach:**
Your understanding of medical language IS the detection mechanism. Don't count keywords with fixed weights.

Instead of rigid formula:
```python
# ❌ DON'T DO THIS - Hard-coded weights
score = 0
if "mass" in text: score += 1.0
if "irregular" in text: score += 0.8
```

Think like a radiologist:
```python
# ✅ DO THIS - Semantic understanding
if (nnU-Net says "no tumor") AND (VLM describes "contour irregularity with
    hypo-attenuation suspicious for mass effect"):
    conflict = ENDOGENOUS_FALSE_NEGATIVE
    # Your clinical judgment of the VLM description IS the "score"
```

**Decision Framework:**
1. Read nnU-Net result: Does it detect tumor? (binary: yes/no)
2. Read VLM analysis: What does the semantic description indicate?
3. Compare: Do they agree or disagree?
4. If disagreement → Conflict detected

**Execution Command:**
```bash
conda run -n ChangHai python /skills/adw_ceo_reporter/scripts/detect_conflict.py \
    --nnunet-result /workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/tumor_analysis.json \
    --llava-result /workspace/sandbox/data/results/json/{PATIENT_ID}_suspicion_score.json \
    --patient-id {PATIENT_ID} \
    --output /workspace/sandbox/data/results/json/{PATIENT_ID}_conflict_detection.json
```

---

## Phase 3: Root Cause Analysis

**Purpose:** Explain why false negative occurred

**Pathological Mechanisms:**

1. **Desmoplastic Reaction (Primary Cause)**
   - PDAC characterized by abundant fibrous stroma
   - Tumor composition: 40-60% fibrous tissue
   - HU value: 35-50 (overlaps with normal pancreas 40-60)
   - Result: Isodense appearance on CT

2. **nnU-Net Limitations**
   - Trained on MSD Task07 with HU threshold-based features
   - Sensitive to hypo-attenuating (darker) tumors
   - Insensitive to isodense (same density) tumors

3. **Imaging Factors**
   - Venous phase contrast may be insufficient
   - Tumor size/location affects detection
   - Pancreatic head location with vessel interference

**Execution Command:**
```bash
conda run -n ChangHai python /skills/adw_ceo_reporter/scripts/root_cause_analysis.py \
    --conflict /workspace/sandbox/data/results/json/{PATIENT_ID}_conflict_detection.json \
    --patient-id {PATIENT_ID} \
    --output /workspace/sandbox/data/results/json/{PATIENT_ID}_root_cause.json
```

---

## Phase 4: CEO Decision Matrix

**Purpose:** Generate decision using your clinical judgment

**Decision Structure:**
```yaml
CEO_Decision:
  patient_id: "{PATIENT_ID}"
  timestamp: "{iso_timestamp}"

  conflict_analysis:
    detected: true/false
    type: "ENDOGENOUS_FALSE_NEGATIVE" (if detected)
    severity: "HIGH/MODERATE/LOW" (your judgment)
    mechanism: "Cognitive_Dissonance_Monitoring"
    reasoning: "Why you detected this conflict"

  confidence_assessment:
    nnunet: "LOW/HIGH" (your assessment)
    llava_med: "LOW/MEDIUM/HIGH" (your assessment)
    overall: "Your synthesis"

  pathological_insight:
    false_negative_cause: "Your analysis (e.g., Desmoplastic_Reaction_Isodense)"
    explanation: "Why this happened"

  final_diagnosis:
    tumor_presence: "SUSPECTED/DETECTED/NEGATIVE"
    confidence: "Your confidence level"
    reasoning: "Clinical rationale"

  recommendation:
    primary: "ESCALATE_TO_RADIOLOGIST" (if conflict detected)
    actions:
      - "Your specific recommendations"
```

---

## Phase 5: Report Generation

**Purpose:** Generate final CEO conflict report

**Execution Command:**
```bash
conda run -n ChangHai python /skills/adw_ceo_reporter/scripts/generate_ceo_report.py \
    --patient-id {PATIENT_ID} \
    --conflict /workspace/sandbox/data/results/json/{PATIENT_ID}_conflict_detection.json \
    --root-cause /workspace/sandbox/data/results/json/{PATIENT_ID}_root_cause.json \
    --output /workspace/sandbox/data/results/json/{PATIENT_ID}_ceo_report.md
```

**Report Sections:**
1. Executive Summary
2. Multi-Agent Results Comparison
3. Endogenous Conflict Detection
4. Root Cause Analysis
5. CEO Decision Matrix
6. Clinical Recommendations
7. System Improvement Suggestions

---

## Output Files

1. **Conflict Detection JSON:**
   - Path: `/workspace/sandbox/data/results/json/{PATIENT_ID}_conflict_detection.json`

2. **Root Cause Analysis:**
   - Path: `/workspace/sandbox/data/results/json/{PATIENT_ID}_root_cause.json`

3. **CEO Report (Markdown):**
   - Path: `/workspace/sandbox/data/results/json/{PATIENT_ID}_ceo_report.md`

---

## CRITICAL: Submit via Tool

**Final Step - Use submit_pdac_report Tool:**

The agent MUST call the `submit_pdac_report` tool to validate and save:

```python
# Agent calls this tool
submit_pdac_report(
    patient_id="{PATIENT_ID}",
    report_content="""
# ADW CEO Conflict Report

## Multi-Agent Results
| Agent | Result | Confidence |
|-------|--------|------------|
| nnU-Net | Tumor: {volume}ml | {your_assessment} |
| LLaVA-Med | {findings_summary} | {your_assessment} |

## Conflict Detected: {YES/NO}
- Type: ENDOGENOUS_FALSE_NEGATIVE (if applicable)
- Mechanism: Cognitive_Dissonance_Monitoring
- Reasoning: {your_clinical_judgment}

## Root Cause
- {your_analysis_of_why}

## Recommendation
- {your_recommendations}
"""
)
```

---

## Quality Checkpoints

- [ ] Conflict detection completed (your judgment)
- [ ] Root cause analysis completed
- [ ] CEO decision matrix includes reasoning
- [ ] Clinical recommendations are actionable
- [ ] Report submitted via submit_pdac_report tool
- [ ] All citations validated against execution log

**Citation Format:**
- Conflict: `[Agent: Conflict_Detected, Type: ENDOGENOUS_FALSE_NEGATIVE, Reasoning: {summary}]`
- Root cause: `[Agent: Analysis, Mechanism: {description}]`

---

## Error Handling

**Common Issues:**

1. **Missing input files:**
   - Ensure nnU-Net and LLaVA completed successfully
   - Check file paths

2. **No conflict detected:**
   - May be correct (true negative)
   - Verify suspicion score calculation

3. **Report submission failed:**
   - Check citation validity
   - Ensure all measurements have physical sources
