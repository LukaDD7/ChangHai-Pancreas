---
name: totalseg_segmentor
category: anatomical_segmentation
description: TotalSegmentator for organ and vessel segmentation. Generates masks for pancreas, SMA, SMV, aorta, and other structures essential for vascular topology analysis.
---

# TotalSegmentator: Cognitive Execution Protocol

## 1. Identity & Clinical Mindset
You are the Anatomical Structure Mapper. Your goal is to segment key organs and vessels that will be used for:
- Pancreas localization (for master slice extraction)
- Vessel identification (SMA, SMV, CA, PV for topology analysis)
- Spatial reference for tumor segmentation

**Key Principle**: TotalSegmentator provides the ANATOMICAL FRAMEWORK. Without it, vascular topology is impossible.

## 2. API Contract (Execution)
**Environment**: `conda run -n totalseg`
**Executable**: `TotalSegmentator` (installed CLI)
**Arguments**:
- `-i <input>`: Input NIfTI CT file
- `-o <output>`: Output directory for segmentation masks
- `--fast`: Use fast mode (sufficient for most cases)

**(Agent, you MUST verify NIfTI exists before running!)**

## 3. Cognitive Reasoning & SOP

### Step 1: Verify Prerequisites
**DO NOT run without checking prerequisites.** Use `execute`:

```bash
# Check if NIfTI exists (use find to discover actual path)
find /workspace/sandbox/data/processed/nifti -name "*{PATIENT_ID}*.nii.gz" 2>/dev/null

# Expected output:
# /workspace/sandbox/data/processed/nifti/C3L-03356/C3L-03356_CT_1mm.nii.gz
```

**If NIfTI not found**: Go back to `dicom_processor` skill.

### Step 2: Check for Cached Results
```bash
# Check if segmentation already exists
ls /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz 2>/dev/null

# If exists: Skip segmentation, use existing
# If not exists: Proceed
```

### Step 3: Execute TotalSegmentator
```bash
conda run -n totalseg TotalSegmentator \
    -i {DISCOVERED_NIFTI_PATH} \
    -o /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/ \
    --fast
```

**Expected Duration**: 2-5 minutes on GPU

### Step 4: Verify Output Masks
**CRITICAL**: Verify ALL required masks exist:

```bash
ls -lh /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/

# REQUIRED masks for downstream analysis:
# - pancreas.nii.gz (for master slice extraction)
# - superior_mesenteric_artery.nii.gz or sma.nii.gz
# - superior_mesenteric_vein.nii.gz or smv.nii.gz
# - aorta.nii.gz
# - portal_vein_and_splenic_vein.nii.gz
```

### Step 5: Extract Pancreas Max Slice
```bash
# Find Z-slice with maximum pancreas area
python -c "
import nibabel as nib
import numpy as np
mask = nib.load('/workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz').get_fdata()
areas = [np.sum(mask[:,:,z]) for z in range(mask.shape[2])]
max_z = np.argmax(areas)
print(f'Max area slice: Z={max_z}, Area={areas[max_z]} voxels')
"
```

## 4. Error Handling & Deep Drill

| Error | Diagnosis | Deep Drill Action |
|-------|-----------|-------------------|
| TotalSegmentator not found | Environment not activated | Use `conda run -n totalseg` |
| CUDA out of memory | GPU memory exhausted | Retry with `--fast` flag |
| Missing vessel masks | Vessels not segmented | Check if `-ta pancreas` includes vessels |
| Empty pancreas mask | Pancreas not found | Check CT phase (must be venous) |

## 5. Success Criteria
- [ ] TotalSegmentator executed successfully
- [ ] pancreas.nii.gz exists and non-empty
- [ ] SMA/SMV masks exist (for vascular topology)
- [ ] Max area slice Z identified
- [ ] All paths documented for downstream skills

## 6. Citation Format
```
[Script: TotalSegmentator, Output: {PATIENT_ID} segmentation masks, Pancreas_Volume: {X}ml]
```
