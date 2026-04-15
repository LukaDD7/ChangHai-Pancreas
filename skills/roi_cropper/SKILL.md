---
name: roi_cropper
category: image_preprocessing
description: Crops CT to the pancreas ROI using the pancreas mask and a fixed 20-voxel margin. Improves nnU-Net efficiency and keeps the pancreas-centered input aligned.
---

# ROI Cropper: Cognitive Execution Protocol

## 1. Identity & Clinical Mindset
You are the ROI Preprocessing Specialist. Your goal is to reduce the CT search space before nnU-Net by cropping tightly around the pancreas mask.

**Key Principle**: ROI cropping is not just speed optimization. It focuses downstream tumor segmentation on the anatomical region of interest while preserving enough context for safe inference.

## 2. API Contract (Execution)
**Environment**: `conda run -n ChangHai`
**Executable**: `/media/luzhenyang/project/ChangHai_PDA/skills/roi_cropper/scripts/crop_ct.py`
**Arguments**:
- `--ct <path>`: Input CT NIfTI file
- `--pancreas-mask <path>`: Pancreas mask NIfTI file
- `--output <path>`: Cropped CT output path
- `--mask-output <path>`: Optional cropped pancreas mask output path
- `--margin <int>`: Margin in voxels on each side (default: 20)

**(Agent, you MUST verify the pancreas mask exists before running!)**

## 3. Cognitive Reasoning & SOP
### Step 1: Verify Prerequisites
```bash
# Use physical paths to discover the pancreas mask
find /media/luzhenyang/project/ChangHai_PDA/data/processed/segmentations -maxdepth 3 -name "*{PATIENT_ID}*pancreas*.nii.gz" 2>/dev/null
```

### Step 2: Verify Shape Compatibility
```bash
python -c "
import nibabel as nib
ct = nib.load('{DISCOVERED_CT_PATH}')
mask = nib.load('{DISCOVERED_MASK_PATH}')
print(ct.shape, mask.shape)
"
```

### Step 3: Execute Crop
```bash
conda run -n ChangHai python /skills/roi_cropper/scripts/crop_ct.py \
    --ct {DISCOVERED_CT_PATH} \
    --pancreas-mask {DISCOVERED_MASK_PATH} \
    --output /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_roi.nii.gz \
    --mask-output /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas_roi.nii.gz \
    --margin 20
```

### Step 4: Verify Output
```bash
python -c "
import nibabel as nib
ct = nib.load('/workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_roi.nii.gz')
print(ct.shape)
print(ct.header.get_zooms())
"
```

## 4. Error Handling & Deep Drill
| Error | Diagnosis | Action |
|-------|-----------|--------|
| Mask not found | TotalSegmentator not run | Run `totalseg_segmentor` first |
| Shape mismatch | Input files not aligned | Verify both files are from the same coordinate space |
| Crop too small | Pancreas mask incomplete | Fall back to full CT for nnU-Net |

## 5. Success Criteria
- [ ] Pancreas mask discovered and verified
- [ ] Cropped CT generated
- [ ] Outside-ROI voxels set to -1000 HU
- [ ] Output ready for nnU-Net input preparation

## 6. Citation Format
```
[Script: crop_ct.py, Output: {PATIENT_ID}_CT_roi.nii.gz, Margin: 20 voxels]
```
