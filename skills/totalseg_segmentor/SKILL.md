---
name: totalseg_segmentor
category: anatomical_segmentation
description: TotalSegmentator for anatomy and baseline vessel segmentation. Provides pancreas and baseline structures; missing pancreatic vessels should trigger the dedicated pancreatic_vessel_segmentor skill.
---

# TotalSegmentator: Cognitive Execution Protocol

## 1. Identity & Clinical Mindset
You are the Anatomical Structure Mapper. Your goal is to segment the baseline anatomy that will be used for:
- Pancreas localization (for master slice extraction)
- Baseline vessel context (aorta, portal-venous structures, IVC)
- Spatial reference for tumor segmentation

**Key Principle**: TotalSegmentator provides the ANATOMICAL FRAMEWORK. When pancreatic vessels needed for topology are missing, the agent must escalate to `pancreatic_vessel_segmentor` rather than assuming TotalSegmentator alone is sufficient.

## 2. API Contract (Execution)
**Environment**: `conda run -n totalseg`
**Executable**: `/media/luzhenyang/project/ChangHai_PDA/skills/totalseg_segmentor/scripts/run_totalseg.py`
**Arguments**:
- `-i <input>`: Input NIfTI CT file
- `-o <output>`: Output directory for segmentation masks
- `--fast`: Use fast mode (sufficient for most cases)
- `--high-res-vessels`: Run a second vessel-focused pass and normalize pancreatic vessel masks into the vessel library

**(Agent, you MUST verify NIfTI exists before running!)**

### ⚠️ CRITICAL: Find Scope Limitation
**Use PHYSICAL paths with -maxdepth to avoid server-wide search:**

```bash
# ✅ GOOD - Limited scope with physical path
find /media/luzhenyang/project/ChangHai_PDA/data/processed/nifti -maxdepth 3 -name "*{PATIENT_ID}*.nii.gz" 2>/dev/null

# ❌ BAD - Virtual path may not work with find, and no depth limit
find /workspace/sandbox/data/processed/nifti -name "*{PATIENT_ID}*.nii.gz"
```

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
**CRITICAL**: Verify the masks that are actually present in the current TotalSegmentator release and mark any missing pancreatic vessels explicitly:

```bash
ls -lh /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/

# REQUIRED baseline masks for downstream analysis:
# - pancreas.nii.gz (for master slice extraction)
# - aorta.nii.gz
# - portal_vein_and_splenic_vein.nii.gz
#
# Optional / case-dependent pancreatic vessels:
# - superior_mesenteric_artery.nii.gz or sma.nii.gz
# - superior_mesenteric_vein.nii.gz or smv.nii.gz
# - celiac_trunk.nii.gz
# - common_hepatic_artery.nii.gz
# - splenic_artery.nii.gz
# - gastroduodenal_artery.nii.gz
#
# If these are absent in the baseline pass, the Agent MUST invoke `pancreatic_vessel_segmentor` to publish canonical vessel masks before concluding that the vessel is unavailable.
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
| Missing vessel masks | Baseline pass missed pancreatic vessels | Run `pancreatic_vessel_segmentor` to publish canonical vessel masks into the vessel library |
| Empty pancreas mask | Pancreas not found | Check CT phase (must be venous) |

## 5. Success Criteria
- [ ] TotalSegmentator executed successfully
- [ ] pancreas.nii.gz exists and non-empty
- [ ] Baseline anatomy masks exist (pancreas/aorta/portal venous structures)
- [ ] If critical pancreatic vessels are missing, `pancreatic_vessel_segmentor` was invoked to publish canonical vessel masks
- [ ] Max area slice Z identified
- [ ] All paths documented for downstream skills

## 6. Citation Format
```
[Script: TotalSegmentator, Output: {PATIENT_ID} segmentation masks, Pancreas_Volume: {X}ml]
```
