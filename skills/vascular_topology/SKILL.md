---
name: vascular-topology
version: 1.0.0
category: vascular_analysis
description: |
  [FUNCTION]: Vascular topology analysis for PDAC resectability assessment
  [VALUE]: Evaluates SMA/SMV involvement, vessel encasement, and resectability classification
  [TRIGGER_HOOK]: CONDITIONAL - Only if tumor detected AND surgery considered. Use AFTER nnunet-segmentor (if tumor>0) or adw-ceo-reporter (if conflict detected). Optional for complete MDT workflow
---

# Vascular Topology: Cognitive Execution Protocol

## Identity & Core Mechanism
This skill analyzes vascular relationships for PDAC resectability assessment. Evaluates Superior Mesenteric Artery (SMA), Superior Mesenteric Vein (SMV), and portal vein involvement.

**When to Use:**
- Tumor detected (>0ml) AND surgery being considered
- Conflict detected (suspected tumor) AND need vascular assessment
- Complete MDT workflow requiring resectability classification

---

## Phase 1: Input Validation

**Required Inputs:**
1. CT Volume: `/workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz`
2. TotalSegmentator Vessels: `/workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/`
   - `superior_mesenteric_vein.nii.gz`
   - `portal_vein_and_splenic_vein.nii.gz`
   - `aorta.nii.gz`
3. Tumor Mask (if available): `/workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/true_tumor_mask.nii.gz`

**Verify Inputs:**
```bash
ls -lh /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/superior_mesenteric_vein.nii.gz
ls -lh /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/portal_vein_and_splenic_vein.nii.gz
```

---

## Phase 2: Vascular Segmentation Enhancement

**Purpose:** Ensure high-quality vessel segmentation

**Execution Command:**
```bash
conda run -n ChangHai python /skills/vascular_topology/scripts/enhance_vessels.py \
    --ct /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz \
    --vessel-dir /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/ \
    --output /workspace/sandbox/data/processed/vascular/{PATIENT_ID}/
```

---

## Phase 3: Vascular Topology Analysis

**Purpose:** Calculate vessel-tumor relationships

**Metrics:**
- **Contact Length**: mm of vessel circumference in contact with tumor
- **Encasement Angle**: degrees of vessel circumference surrounded
- **Involvement Score**: 0 (none), 1 (<180°), 2 (>180°), 3 (occlusion)

**Execution Command:**
```bash
conda run -n ChangHai python /skills/vascular_topology/scripts/analyze_topology.py \
    --patient-id {PATIENT_ID} \
    --ct /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz \
    --vessel-dir /workspace/sandbox/data/processed/vascular/{PATIENT_ID}/ \
    --tumor-mask /workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/true_tumor_mask.nii.gz \
    --output /workspace/sandbox/data/results/vascular/{PATIENT_ID}_vascular_assessment.json
```

**Expected Output:**
```json
{
  "patient_id": "{PATIENT_ID}",
  "vessels_analyzed": {
    "SMA": {
      "contact_length_mm": 12.5,
      "encasement_degrees": 135,
      "involvement_score": 1,
      "status": "borderline_resectable"
    },
    "SMV": {
      "contact_length_mm": 18.3,
      "encasement_degrees": 225,
      "involvement_score": 2,
      "status": "unresectable"
    },
    "portal_vein": {
      "contact_length_mm": 8.2,
      "encasement_degrees": 90,
      "involvement_score": 1,
      "status": "borderline_resectable"
    }
  },
  "overall_classification": "BORDERLINE_RESECTABLE",
  "recommendation": "Neoadjuvant therapy followed by re-evaluation"
}
```

---

## Phase 4: Resectability Classification

**NCCN Criteria:**

| Classification | Criteria |
|----------------|----------|
| **Resectable** | No arterial involvement, clear fat plane around SMA/CHA |
| **Borderline** | <180° SMA/HA involvement, or reconstructible SMV/PV |
| **Unresectable** | >180° SMA/HA encasement, or occluded SMV/PV without reconstruction |

**Execution Command:**
```bash
conda run -n ChangHai python /skills/vascular_topology/scripts/classify_resectability.py \
    --vascular-assessment /workspace/sandbox/data/results/vascular/{PATIENT_ID}_vascular_assessment.json \
    --output /workspace/sandbox/data/results/vascular/{PATIENT_ID}_resectability.md
```

---

## Phase 5: Visualization (Optional)

**Purpose:** Generate diagnostic figures

**Execution Command:**
```bash
conda run -n ChangHai python /skills/vascular_topology/scripts/visualize_vessels.py \
    --patient-id {PATIENT_ID} \
    --ct /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz \
    --vascular-assessment /workspace/sandbox/data/results/vascular/{PATIENT_ID}_vascular_assessment.json \
    --output /workspace/sandbox/data/results/images/{PATIENT_ID}_vascular_overlay.png
```

---

## Output Files

1. **Vascular Assessment JSON:**
   - Path: `/workspace/sandbox/data/results/vascular/{PATIENT_ID}_vascular_assessment.json`

2. **Resectability Report:**
   - Path: `/workspace/sandbox/data/results/vascular/{PATIENT_ID}_resectability.md`

3. **Visualization (Optional):**
   - Path: `/workspace/sandbox/data/results/images/{PATIENT_ID}_vascular_overlay.png`

---

## Integration with CEO Report

**When Conflict Detected:**
If `ENDOGENOUS_FALSE_NEGATIVE` detected, vascular assessment may be limited:
- Tumor mask is empty (0ml)
- Cannot calculate precise vessel involvement
- Can still assess baseline vessel anatomy
- Note in report: "Vascular assessment limited due to undetected tumor on CT"

**When Tumor Detected:**
Include vascular assessment in CEO report:
```markdown
## Vascular Assessment
- SMA: {contact_length}mm, {encasement}°
- SMV: {contact_length}mm, {encasement}°
- Classification: {RESECTABLE/BORDERLINE/UNRESECTABLE}
```

---

## Quality Checkpoints

- [ ] Vessel masks exist and non-empty
- [ ] SMA and SMV both segmented
- [ ] Contact lengths calculated
- [ ] Resectability classification assigned
- [ ] Clinical recommendation provided

**Citation Format:**
- Vessels: `[Local: superior_mesenteric_vein.nii.gz, Contact: {value}mm]`
- Classification: `[Tool: VascularTopology, SMA: {value}°, Classification: {value}]`

---

## Conditional Execution

**Skip This Skill If:**
- Patient is not a surgical candidate
- Tumor is clearly metastatic (liver, peritoneum)
- Only diagnostic confirmation needed

**Always Include If:**
- Surgical resection being considered
- Borderline resectable case
- Need vascular involvement documentation
