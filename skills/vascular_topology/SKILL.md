---
name: vascular_topology
category: 3d_spatial_computation
description: Calculates tumor wrapping angles around SMA, SMV, CA, and PV. Determines NCCN resectability classification. CRITICAL for surgical planning.
---

# Vascular Topology: Cognitive Execution Protocol

## 1. Identity & Clinical Mindset
You are the 3D Geometric Computation Specialist. Your goal is to determine surgical resectability by calculating how many degrees the tumor wraps around critical mesenteric vessels.

**Key Principle**: Resectability is not binary. It's determined by the geometric relationship between tumor and vessels.
- SMA encasement >180° = Unresectable (arterial involvement)
- SMV occlusion without reconstruction = Unresectable
- <180° involvement = Borderline (neoadjuvant therapy candidate)

**Clinical Stakes**:
- Over-call resectable → Failed surgery, complications
- Under-call resectable → Missed curative opportunity

## 2. API Contract (Execution)
**Environment**: `conda run -n ChangHai`
**Executable**: `/media/luzhenyang/project/ChangHai_PDA/skills/vascular_topology/scripts/calculate_angles.py`
**Arguments**:
- `--tumor-mask <path>`: Binary tumor mask from nnU-Net
- `--vessel-dir <path>`: Directory with TotalSegmentator vessel masks
- `--patient-id <id>`: Patient identifier
- `--output <path>`: JSON output path

**(Agent, you MUST verify tumor mask and vessels exist before running!)**

## 3. Cognitive Reasoning & SOP

### Step 1: Environmental Discovery
```bash
# Discover actual file paths (DON'T assume)
find /workspace/sandbox/data/processed/segmentations -name "*{PATIENT_ID}*" -type d

# Check for tumor mask
find /workspace/sandbox/data/processed/segmentations -name "*tumor*{PATIENT_ID}*.nii.gz" 2>/dev/null

# Check for vessel masks
ls /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/ | grep -E "(sma|smv|portal|celiac)"
```

### Step 2: Verify Prerequisites
```bash
# Required files must exist:
# 1. Tumor mask (binary, 0/1)
# 2. SMA mask
# 3. SMV mask
# 4. Portal vein mask (optional)
# 5. Celiac artery mask (optional)

for file in "{tumor_mask}" "{sma_mask}" "{smv_mask}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Missing $file"
    fi
done
```

**If tumor mask is empty (all zeros)**:
- nnU-Net returned 0ml
- Cannot calculate precise vessel involvement
- Note in report: "Vascular assessment limited - tumor not detected by segmentation"
- Still assess baseline vessel anatomy

### Step 3: Execute Angle Calculation
```bash
conda run -n ChangHai python /skills/vascular_topology/scripts/calculate_angles.py \
    --tumor-mask {DISCOVERED_TUMOR_MASK} \
    --vessel-dir /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/ \
    --patient-id {PATIENT_ID} \
    --output /workspace/sandbox/data/results/vascular/{PATIENT_ID}_vascular_assessment.json
```

### Step 4: Interpret Results

**NCCN Resectability Criteria**:

| Vessel | Resectable | Borderline | Unresectable |
|--------|-----------|------------|--------------|
| **SMA** | No contact OR abutment only | <180° contact | ≥180° encasement |
| **CA** | Clear fat plane | Abutment | Encasement |
| **SMV** | Patent | Narrowed/Reconstructible | Occluded (no reconstruction) |
| **PV** | Patent | Narrowed/Reconstructible | Occluded (no reconstruction) |

**Classification Logic**:
```python
if SMA_angle >= 180 or CA_angle >= 180:
    classification = "UNRESECTABLE"
elif SMA_angle > 0 or SMV_occluded:
    classification = "BORDERLINE"
else:
    classification = "RESECTABLE"
```

## 4. Error Handling & Deep Drill

| Error | Diagnosis | Deep Drill Action |
|-------|-----------|-------------------|
| Tumor mask not found | nnU-Net not run | Go back to nnunet_segmentor |
| Vessel masks missing | TotalSegmentator incomplete | Re-run with `--task total` |
| Angle calculation fails | Tumor too small or scattered | Use VLM visual assessment instead |
| SMA/SMV not segmented | Vessels outside FOV | Note limitation in report |

## 5. Success Criteria
- [ ] Tumor mask discovered and verified
- [ ] Vessel masks exist (SMA, SMV minimum)
- [ ] Angle calculation script executed
- [ ] Results JSON generated with angles
- [ ] NCCN classification assigned
- [ ] Clinical recommendation provided

## 6. Citation Format
```
[Script: calculate_angles.py, Output: {PATIENT_ID}_vascular_assessment.json, SMA: {X}°, SMV: {Y}°, Classification: {RESECTABLE/BORDERLINE/UNRESECTABLE}]
```
