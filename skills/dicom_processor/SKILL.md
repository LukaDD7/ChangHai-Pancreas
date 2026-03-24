---
name: dicom-processor
version: 1.0.0
category: medical_imaging
description: |
  [FUNCTION]: Convert DICOM medical imaging data to NIfTI format with spatial standardization
  [VALUE]: Provides standardized 3D medical volumes (1.0mm³ isotropic) in RAS coordinate system for downstream AI segmentation
  [TRIGGER_HOOK]: Read this Skill FIRST for any new patient case. Use BEFORE TotalSegmentator or nnU-Net processing
---

# DICOM Processor: Cognitive Execution Protocol

## Identity & Core Mechanism
This is a medical imaging preprocessing skill. It uses Python libraries (dicom2nifti, SimpleITK, nibabel) via the `execute` tool to convert DICOM to standardized NIfTI format.

---

## Phase 1: DICOM Discovery

**Purpose:** Locate and validate DICOM series for a patient

**Execution Command:**
```bash
find /workspace/sandbox/data/raw/dicom/CPTAC-PDA/{PATIENT_ID}/ -name "*.dcm" -type f | head -20
```

**Expected Structure:**
```
{PATIENT_ID}/
└── {DATE}-{DESC}-{ID}/
    ├── {SERIES_ID}-ROUTINE_CHEST-XXX/  (Chest CT - IGNORE)
    └── {SERIES_ID}-CT_PANCREAS-XXX/     (Pancreas CT - TARGET)
        ├── 1-001.dcm
        └── ... (~200-400 slices)
```

---

## Phase 2: DICOM to NIfTI Conversion

**Purpose:** Convert DICOM series to 3D NIfTI volume

**CRITICAL: Check if NIfTI already exists**
Before conversion, check if output file exists:
- Path: `/workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT.nii.gz`
- If EXISTS: Skip conversion, use existing file
- If NOT EXISTS: Proceed to conversion

**Execution Command:**
```bash
conda run -n ChangHai python /skills/dicom_processor/scripts/dicom_to_nifti.py \
    --dicom-dir /workspace/sandbox/data/raw/dicom/CPTAC-PDA/{PATIENT_ID}/ \
    --patient-id {PATIENT_ID} \
    --output-dir /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/
```

---

## Phase 3: Spatial Standardization (CRITICAL)

**Purpose:** Resample to isotropic 1.0mm³ spacing in RAS coordinate system

**Why This Matters:**
- TotalSegmentator and nnU-Net must operate on the SAME physical coordinate system
- Prevents layer misalignment between pancreas mask and tumor segmentation
- Ensures Z=145 in pancreas mask = Z=145 in CT volume

**Execution Command:**
```bash
conda run -n ChangHai python /skills/dicom_processor/scripts/resample_isotropic.py \
    --input /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT.nii.gz \
    --output /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz \
    --spacing 1.0 1.0 1.0
```

**Validation Output:**
- Original spacing: (0.703, 0.703, 1.500) mm
- New spacing: (1.0, 1.0, 1.0) mm
- New shape: (~360, ~360, ~470) voxels
- Coordinate system: RAS (Right-Anterior-Superior)

---

## Phase 4: Metadata Extraction

**Purpose:** Extract and validate key DICOM metadata

**Execution Command:**
```bash
conda run -n ChangHai python /skills/dicom_processor/scripts/extract_metadata.py \
    --nifti /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz
```

**Expected Output:**
```json
{
  "modality": "CT",
  "body_part": "ABDOMEN",
  "slice_thickness": 1.0,
  "kvp": 120,
  "window_center": 40,
  "window_width": 400,
  "rescale_intercept": -1024,
  "rescale_slope": 1,
  "phase": "Venous",
  "matrix": [512, 512, 312],
  "voxel_spacing": [1.0, 1.0, 1.0],
  "hu_range": [-1024, 1365]
}
```

---

## Output Files

After successful execution, these files will exist:

1. **NIfTI Volume:**
   - Path: `/workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz`
   - Format: NIfTI-1, compressed (.nii.gz)
   - Spacing: 1.0mm³ isotropic
   - Coordinates: RAS

2. **Metadata JSON:**
   - Path: `/workspace/sandbox/data/processed/nifti/{PATIENT_ID}/metadata.json`

3. **Validation Report:**
   - Path: `/workspace/sandbox/data/processed/nifti/{PATIENT_ID}/validation_report.txt`

---

## Quality Checkpoints

**Before proceeding to next skill, verify:**

- [ ] NIfTI file exists and is readable
- [ ] Spacing is exactly (1.0, 1.0, 1.0) mm
- [ ] HU range is reasonable ([-1024, ~2000])
- [ ] Shape dimensions are consistent (~300-500 per axis)
- [ ] Affine matrix encodes RAS orientation

**Citation Format for Downstream Skills:**
- NIfTI file: `[Local: {PATIENT_ID}_CT_1mm.nii.gz, Spacing: 1.0mm³]`
- Metadata: `[Local: metadata.json, Line <line>]`

---

## Error Handling

**Common Issues:**

1. **Multiple CT series found:**
   - Select the one with "PANCREAS" in series description
   - Verify it's the venous phase (60-70s delay)

2. **Corrupted DICOM:**
   - Skip corrupted slices if possible
   - Report missing slices

3. **Memory error:**
   - Use smaller batch size for conversion
   - Ensure sufficient /tmp space

**If conversion fails:**
- Check DICOM directory permissions
- Verify conda environment ChangHai exists
- Check available disk space (>10GB recommended)
