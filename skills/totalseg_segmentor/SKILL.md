---
name: totalseg-segmentor
version: 1.0.0
category: medical_ai_segmentation
description: |
  [FUNCTION]: TotalSegmentator organ and vessel segmentation for abdominal CT
  [VALUE]: Provides pancreas mask, vascular structures, and abdominal organs for spatial localization and master slice selection
  [TRIGGER_HOOK]: Read this Skill AFTER dicom-processor. Use for pancreas localization BEFORE tumor segmentation
---

# TotalSegmentator Segmentation: Cognitive Execution Protocol

## Identity & Core Mechanism
This skill runs TotalSegmentator AI model via conda environment `totalseg` to segment 104 abdominal organs and vessels. Essential for pancreas localization and master slice extraction.

---

## Phase 1: Input Validation

**Purpose:** Verify standardized NIfTI input exists

**Required Input:**
- Path: `/workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz`
- Must be spatially standardized (1.0mm³ isotropic)

**Execution Command:**
```bash
ls -lh /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz
```

---

## Phase 2: TotalSegmentator Execution

**Purpose:** Segment pancreas and abdominal organs

**CRITICAL: Check for cached results**
Before execution, check if segmentation already exists:
- Path: `/workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz`
- If EXISTS: Skip segmentation, use existing results
- If NOT EXISTS: Proceed

**Execution Command:**
```bash
conda run -n totalseg TotalSegmentator \
    -i /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz \
    -o /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/ \
    --fast \
    --verbose
```

**Parameters:**
- `--fast`: Use fast mode (sufficient for pancreas localization)
- `--verbose`: Output detailed logging
- No `--task` flag: Use default total (104 classes)

---

## Phase 3: Output Verification

**Purpose:** Validate segmentation outputs

**Expected Output Structure:**
```
/workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/
├── pancreas.nii.gz                    # ← PRIMARY: Pancreas mask
├── aorta.nii.gz                       # Aorta
├── portal_vein_and_splenic_vein.nii.gz # Portal + splenic veins
├── superior_mesenteric_vein.nii.gz    # SMV (critical for PDAC)
├── inferior_vena_cava.nii.gz          # IVC
├── liver.nii.gz                       # Liver
├── spleen.nii.gz                      # Spleen
└── ... (104 classes total)
```

**Execution Command - Verify Outputs:**
```bash
ls -lh /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz
conda run -n ChangHai python /skills/totalseg_segmentor/scripts/verify_segmentation.py \
    --pancreas-mask /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz
```

---

## Phase 4: Pancreas Volume Analysis

**Purpose:** Calculate pancreas volume and identify maximum area slice

**Execution Command:**
```bash
conda run -n ChangHai python /skills/totalseg_segmentor/scripts/analyze_pancreas.py \
    --pancreas-mask /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz \
    --output-json /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas_analysis.json
```

**Expected Output:**
```json
{
  "patient_id": "{PATIENT_ID}",
  "total_volume_ml": 65.47,
  "max_area_slice": 145,
  "max_area_voxels": 2847,
  "z_range": [98, 187],
  "centroid": [256.3, 245.7, 145.3],
  "bounding_box": {"x": [120, 380], "y": [130, 360], "z": [98, 187]}
}
```

**CRITICAL: Master Slice Location**
- Max area slice Z=145 is the PRIMARY candidate for tumor location
- This is where pancreatic head is typically largest
- Use this Z for master slice extraction

---

## Output Files

1. **Pancreas Mask:**
   - Path: `/workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz`
   - Binary mask (0=background, 1=pancreas)

2. **Vessel Masks:**
   - SMV: `superior_mesenteric_vein.nii.gz`
   - Portal vein: `portal_vein_and_splenic_vein.nii.gz`
   - Aorta: `aorta.nii.gz`

3. **Analysis JSON:**
   - Path: `/workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas_analysis.json`

---

## Quality Checkpoints

- [ ] Pancreas mask exists and is non-empty
- [ ] Volume is reasonable (40-100ml typical)
- [ ] Max area slice identified
- [ ] Z-range covers pancreatic head-to-tail
- [ ] SMV and portal vein segmented (for vascular assessment)

**Citation Format:**
- Pancreas mask: `[Local: pancreas.nii.gz, Volume: {value}ml, MaxSlice: Z{value}]`
- Vessel masks: `[Local: superior_mesenteric_vein.nii.gz]`

---

## Error Handling

**Common Issues:**

1. **Empty pancreas mask:**
   - Pancreas may be outside CT field of view
   - Check Z-range of input NIfTI

2. **Poor segmentation quality:**
   - Try without `--fast` flag (slower but more accurate)
   - Check if CT phase is venous (required)

3. **Missing vessels:**
   - Small vessels may not be segmented
   - Manual verification may be needed
