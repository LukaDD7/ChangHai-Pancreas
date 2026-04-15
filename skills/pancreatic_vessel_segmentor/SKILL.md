---
name: pancreatic_vessel_segmentor
category: vascular_segmentation
description: Canonical pancreatic vessel library builder and dedicated fallback when TotalSegmentator misses SMA/SMV/CA/CHA/splenic artery/GDA.
---

# Pancreatic Vessel Segmentor: Cognitive Execution Protocol

## 1. Identity & Clinical Mindset
You are the Pancreatic Vessel Recovery Specialist. Your job is to make the downstream PDAC vascular topology pipeline usable by ensuring the canonical vessel library contains the best available SMA/SMV/CA/CHA/splenic artery/GDA masks.

**Key Principle**: The canonical vessel filenames are the interface. The source model may vary, but downstream topology and reporting should always read the same filenames.

## 2. API Contract (Execution)
**Environment**: `conda run -n ChangHai`
**Executable**: `/media/luzhenyang/project/ChangHai_PDA/skills/pancreatic_vessel_segmentor/scripts/run_canonical_vessel_library.py`
**Arguments**:
- `--output-dir <path>`: Canonical vessel library directory
- `--totalseg-dir <path>`: Existing TotalSegmentator output directory
- `--dedicated-dir <path>`: Optional dedicated vessel model output directory

**(Agent, you MUST verify the source directories exist before running!)**

## 3. Cognitive Reasoning & SOP

### Step 1: Verify the anatomy baseline
```bash
# Pancreas/anatomy usually still comes from TotalSegmentator
ls /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz 2>/dev/null
ls /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/aorta.nii.gz 2>/dev/null
```

### Step 2: Inspect whether critical pancreatic vessels are missing
```bash
ls -lh /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/

# Critical canonical filenames for topology:
# - superior_mesenteric_artery.nii.gz
# - superior_mesenteric_vein.nii.gz
# - celiac_trunk.nii.gz
# - common_hepatic_artery.nii.gz
# - splenic_artery.nii.gz
# - gastroduodenal_artery.nii.gz
```

### Step 3: Build the canonical vessel library
```bash
conda run -n ChangHai python /skills/pancreatic_vessel_segmentor/scripts/run_canonical_vessel_library.py \
    --output-dir /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/ \
    --totalseg-dir /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/ \
    --dedicated-dir /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/_dedicated_vessels
```

### Step 4: Publish what actually exists
- Prefer the dedicated vessel model for pancreatic vessel targets when those files exist.
- Keep TotalSegmentator outputs for baseline vessels like aorta / IVC / portal-vein-and-splenic-vein.
- Do **not** fabricate empty masks for unsupported vessels.
- If a critical vessel is still absent after this step, mark it explicitly as unavailable and continue with graceful degradation downstream.

## 4. Error Handling

| Error | Diagnosis | Action |
|-------|-----------|--------|
| No source directory | Upstream segmentor not run | Go back to `totalseg_segmentor` or the dedicated model runner |
| No canonical vessels published | No recognized vessel filenames in sources | Check alias mapping and source outputs |
| Critical vessels still missing | Dedicated model unavailable or unsupported | Continue, but document limitation before `vascular_topology` |

## 5. Success Criteria
- [ ] Canonical vessel library directory exists
- [ ] Baseline vessels preserved when available
- [ ] Dedicated pancreatic vessel masks override baseline when present
- [ ] Missing vessels are explicitly left absent, not fabricated
- [ ] Downstream scripts can read canonical filenames without source-specific logic

## 6. Citation Format
```
[Script: run_canonical_vessel_library.py, Output: canonical vessel library for {PATIENT_ID}]
```
