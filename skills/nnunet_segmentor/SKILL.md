---
name: nnunet-segmentor
version: 1.0.0
category: medical_ai_segmentation
description: |
  [FUNCTION]: nnU-Net v1 pancreatic tumor segmentation (MSD Task07 model)
  [VALUE]: Provides tumor volume and location (Label 2) for PDAC detection. May produce FALSE NEGATIVES for isodense tumors
  [TRIGGER_HOOK]: Read this Skill AFTER totalseg-segmentor. Use for tumor detection, but CHECK for false negatives via LLaVA-Med
  [WARNING]: This model may produce FALSE NEGATIVES (0ml tumor) for isodense tumors. Always cross-reference with LLaVA-Med suspicion score!
---

# nnU-Net Tumor Segmentation: Cognitive Execution Protocol

## Identity & Core Mechanism
This skill runs nnU-Net v1 (MSD Task07 Pancreas model) via conda environment `nnunetv2`. Trained on Medical Segmentation Decathlon (MSD) pancreatic tumor dataset.

**Model Characteristics:**
- Labels: 0=background, 1=pancreas parenchyma, 2=tumor
- Input: Venous phase CT, isotropic 1.0mm³
- Output: 3-class segmentation mask
- Known Limitation: May miss isodense tumors (HU overlap with normal pancreas)

---

## Phase 1: Environment Setup

**Purpose:** Configure nnU-Net environment variables

**CRITICAL:** These environment variables MUST be set before execution:

```bash
export nnUNet_raw_data_base="/workspace/sandbox/data/models/nnunet/nnunet_v1_workspace/raw"
export nnUNet_preprocessed="/workspace/sandbox/data/models/nnunet/nnunet_v1_workspace/preprocessed"
export RESULTS_FOLDER="/workspace/sandbox/data/models/nnunet/nnunet_v1_workspace/results"
export TMPDIR="/workspace/sandbox/tmp"
```

---

## Phase 2: Input Preparation

**Purpose:** Copy input to nnU-Net raw data directory

**Execution Command:**
```bash
mkdir -p ${nnUNet_raw_data_base}/nnUNet_raw_data/Task007_Pancreas/imagesTs
cp /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz \
   ${nnUNet_raw_data_base}/nnUNet_raw_data/Task007_Pancreas/imagesTs/{PATIENT_ID}_0000.nii.gz
```

**Naming Convention:**
- nnU-Net requires `{CASE_ID}_0000.nii.gz` suffix
- `_0000` indicates single modality (CT only)

---

## Phase 3: nnU-Net Inference

**Purpose:** Run tumor segmentation

**CRITICAL: Check for cached results**
Before execution, check if output exists:
- Path: `/workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/{PATIENT_ID}.nii.gz`
- If EXISTS: Skip inference, proceed to analysis
- If NOT EXISTS: Proceed

**Execution Command:**
```bash
conda run -n nnunetv2 nnUNet_predict \
    -i ${nnUNet_raw_data_base}/nnUNet_raw_data/Task007_Pancreas/imagesTs \
    -o /workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/ \
    -t Task007_Pancreas \
    -m 3d_fullres \
    --num_threads_preprocessing 4 \
    --num_threads_nifti_save 4
```

**Expected Duration:** 5-10 minutes on GPU

---

## Phase 4: Output Analysis (CRITICAL)

**Purpose:** Extract tumor mask and calculate volume

**Execution Command:**
```bash
conda run -n ChangHai python /skills/nnunet_segmentor/scripts/analyze_tumor.py \
    --input /workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/{PATIENT_ID}.nii.gz \
    --patient-id {PATIENT_ID} \
    --output-dir /workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/
```

**Expected Output:**
```json
{
  "patient_id": "{PATIENT_ID}",
  "labels_detected": [0, 1, 2],
  "tumor_volume_ml": 0.0,
  "tumor_voxels": 0,
  "pancreas_volume_ml": 65.47,
  "has_tumor": false,
  "warning": "FALSE_NEGATIVE_POSSIBLE",
  "model": "nnU-Net v1 MSD Task07"
}
```

**⚠️ FALSE NEGATIVE DETECTION:**
If `tumor_volume_ml` == 0.0:
1. This is a POTENTIAL false negative
2. nnU-Net failed to detect tumor (common for isodense tumors)
3. MUST cross-reference with LLaVA-Med suspicion score
4. If LLaVA score > 1.5 → ENDOGENOUS_FALSE_NEGATIVE detected

---

## Phase 5: Extract Binary Tumor Mask

**Purpose:** Create binary mask for downstream analysis

**Execution Command:**
```bash
conda run -n ChangHai python /skills/nnunet_segmentor/scripts/extract_tumor_mask.py \
    --input /workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/{PATIENT_ID}.nii.gz \
    --output /workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/true_tumor_mask.nii.gz
```

**Output:**
- Binary mask (0=background, 1=tumor)
- If no tumor detected: All zeros

---

## Output Files

1. **Raw Segmentation (3-class):**
   - Path: `/workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/{PATIENT_ID}.nii.gz`
   - Labels: 0, 1, 2

2. **Binary Tumor Mask:**
   - Path: `/workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/true_tumor_mask.nii.gz`
   - Binary: 0 or 1

3. **Analysis Report:**
   - Path: `/workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/tumor_analysis.json`

---

## Quality Checkpoints

- [ ] Output file exists and is readable
- [ ] Labels are [0, 1] or [0, 1, 2]
- [ ] Tumor volume calculated (may be 0)
- [ ] Pancreas volume matches TotalSegmentator output (±5ml)

**Citation Format:**
- Segmentation: `[Tool: nnU-Net, Labels: {values}, Volume: {value}ml]`
- If false negative: `[Tool: nnU-Net, Volume: 0ml, Warning: FALSE_NEGATIVE_POSSIBLE]`

---

## False Negative Protocol

**When tumor_volume_ml == 0:**

1. **DO NOT conclude "no tumor"**
2. **Flag for cognitive dissonance monitoring**
3. **Proceed to LLaVA-Med analysis**
4. **Compare semantic suspicion score**

**Example (CL-03356):**
```
nnU-Net Result: 0ml tumor
LLaVA-Med Suspicion Score: 3.4/1.5
Conclusion: ENDOGENOUS_FALSE_NEGATIVE
Reason: Desmoplastic reaction causing isodense appearance
```

---

## Error Handling

**Common Issues:**

1. **Model not found:**
   - Check RESULTS_FOLDER path
   - Verify MSD Task07 weights downloaded

2. **Out of memory:**
   - Reduce `--num_threads_preprocessing`
   - Ensure GPU has >8GB VRAM

3. **Wrong labels:**
   - Verify input is venous phase CT
   - Check HU range [-1024, 2000]
