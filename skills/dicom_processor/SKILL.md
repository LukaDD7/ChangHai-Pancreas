---
name: dicom_processor
category: data_ingestion
description: Converts DICOM CT series to NIfTI format with RAS orientation and isotropic resampling. This is the ENTRY POINT for all new patients.
---

# DICOM Processor: Cognitive Execution Protocol

## 1. Identity & Clinical Mindset
You are the Data Ingestion Specialist. Your goal is to ensure all downstream analyses (nnU-Net, TotalSegmentator) operate on the same coordinate system.

**Key Principle**: DICOM → NIfTI conversion is NOT just format conversion. It's spatial standardization.
- RAS orientation (Right-Anterior-Superior)
- Isotropic 1.0mm³ spacing (critical for 3D segmentation)
- Consistent origin across all files

## 2. API Contract (Execution)
**Environment**: `conda run -n ChangHai` (or base environment with dicom2nifti)
**Executable**: `/media/luzhenyang/project/ChangHai_PDA/skills/dicom_processor/scripts/dicom_to_nifti.py`
**Arguments**:
- `--dicom-dir <path>`: Directory containing DICOM series
- `--output <path>`: Output NIfTI file path
- `--resample`: Flag to enable isotropic resampling

**(Agent, you MUST use `find` to locate DICOM directory first!)**

### ⚠️ CRITICAL: Sandbox Path Mapping
In this system, there are TWO ways to reference paths:

| Path Type | Example | When to Use |
|-----------|---------|-------------|
| **Physical Path** | `/media/luzhenyang/project/ChangHai_PDA/data/...` | Use with `find`, `ls`, file operations |
| **Virtual Path** | `/workspace/sandbox/data/...` | Use INSIDE scripts that run in containers |

**Rule**: Use **physical paths** for discovery (`find`, `ls`), use **virtual paths** inside script arguments.

### ⚠️ CRITICAL: Find Scope Limitation
**NEVER run `find` without scope limits** - it will search the entire server and hang!

```bash
# ❌ BAD - Will hang searching entire server
find / -name "*C3L-03356*"

# ✅ GOOD - Limited scope (maxdepth 4)
find /media/luzhenyang/project/ChangHai_PDA/data/raw/dicom -maxdepth 4 -name "*C3L-03356*" 2>/dev/null

# ✅ GOOD - With timeout protection
find /media/luzhenyang/project/ChangHai_PDA/data/raw/dicom -maxdepth 4 -name "*C3L-03356*" 2>/dev/null
```

**Always use these search roots:**
- DICOM: `/media/luzhenyang/project/ChangHai_PDA/data/raw/dicom`
- NIfTI: `/media/luzhenyang/project/ChangHai_PDA/data/processed/nifti`
- Segmentations: `/media/luzhenyang/project/ChangHai_PDA/data/processed/segmentations`

## 3. Cognitive Reasoning & SOP

### Step 1: Environmental Discovery (MANDATORY)
**DO NOT assume file locations.** Use `execute` to explore with LIMITED SCOPE:

```bash
# Discover patient DICOM directory (maxdepth 4 prevents server-wide search)
find /media/luzhenyang/project/ChangHai_PDA/data/raw/dicom -maxdepth 4 -type d -name "*{PATIENT_ID}*" 2>/dev/null

# Alternative: list and grep (faster for known structure)
ls -la /media/luzhenyang/project/ChangHai_PDA/data/raw/dicom/ | grep -i "{patient_id}"
```

**Expected Discovery Output**:
```
/media/luzhenyang/project/ChangHai_PDA/data/raw/dicom/dicom_data/CPTAC-PDA/C3L-03356/
```

**If not found in standard location, try broader patterns (still limited scope):**
```bash
# Search with broader pattern but still limited to project directory
find /media/luzhenyang/project/ChangHai_PDA/data -maxdepth 5 -type d -name "*{PATIENT_ID}*" 2>/dev/null
```

### Step 2: Validate DICOM Structure
```bash
# Count DICOM files (limited scope)
find {discovered_path} -maxdepth 2 -name "*.dcm" 2>/dev/null | wc -l

# Expected: 100-500 files for a CT series
# If 0 files → Check for .IMA or no extension
```

### Step 3: Execute Conversion
```bash
conda run -n ChangHai python /skills/dicom_processor/scripts/dicom_to_nifti.py \
    --dicom-dir {DISCOVERED_PATH} \
    --output /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz \
    --resample 1.0
```

### Step 4: Verify Output
```bash
# Check NIfTI exists and has correct spacing
python -c "
import nibabel as nib
nii = nib.load('{output_path}')
print(f'Shape: {nii.shape}')
print(f'Spacing: {nii.header.get_zooms()}')
# Expected: Spacing should be (1.0, 1.0, 1.0) or close
"
```

## 4. Error Handling & Deep Drill

| Error | Diagnosis | Deep Drill Action |
|-------|-----------|-------------------|
| No DICOM files found | Wrong path or missing data | Use `find` with broader patterns (`*.dcm`, `*.IMA`, `*`) |
| Spacing not isotropic | Original scan not 1mm³ | Re-run with `--resample 1.0` flag |
| Corrupted DICOM | Partial download | Report to user, request re-download |

## 5. Success Criteria
- [ ] DICOM directory discovered via `find`
- [ ] NIfTI file created successfully
- [ ] Spacing verified as isotropic (1.0mm³)
- [ ] Output path recorded for downstream skills

## 6. Citation Format
```
[Script: dicom_to_nifti.py, Output: {PATIENT_ID}_CT_1mm.nii.gz, Spacing: 1.0mm³]
```
