---
name: master_slice_extractor
category: image_preprocessing
description: Extracts multi-window Tiled master slice from pancreas maximum area. Generates 3-window image (Standard/Narrow/Soft) for isodense tumor detection via VLM. CRITICAL for Deep Drill Protocol.
---

# Master Slice Extractor: Cognitive Execution Protocol

## 1. Identity & Clinical Mindset
You are the Visual Enhancement Specialist. Your goal is to generate multi-window CT images that reveal isodense tumors invisible in standard windows.

**Key Principle**: Isodense tumors (HU 35-50, same as normal pancreas 40-60) are invisible in standard window (W:400) but become visible in narrow window (W:150) due to contrast stretching.

**Mathematical Foundation**:
```python
windowed = (raw_hu - center) / width * 255 + 128
windowed = np.clip(windowed, 0, 255)
```

In narrow window (W:150, C:40):
- HU range [-35, 115] mapped to [0, 255]
- Pancreas HU [40,60] → gray levels [128, 162] (34 levels)
- vs Standard window: only ~13 gray levels

## 2. API Contract (Execution)
**Environment**: `conda run -n ChangHai`
**Executable**: `/media/luzhenyang/project/ChangHai_PDA/skills/master_slice_extractor/scripts/extract_tiled_master_slice.py`
**Arguments**:
- `--ct <path>`: Input CT NIfTI file
- `--pancreas-mask <path>`: Pancreas mask from TotalSegmentator
- `--patient-id <id>`: Patient identifier
- `--output <path>`: Output Tiled image path
- `--windows <str>`: Window settings (standard:40:400,narrow:40:150,soft:50:250)

**(Agent, you MUST have TotalSegmentator pancreas mask before running!)**

## 3. Cognitive Reasoning & SOP

### Step 1: Verify Prerequisites
```bash
# Check TotalSegmentator outputs exist
ls /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz
ls /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas_analysis.json

# Extract max slice Z
python -c "
import json
with open('/workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas_analysis.json') as f:
    data = json.load(f)
print(f'Max slice Z: {data[\"max_area_slice\"]}')
"
```

### Step 2: Execute Tiled Extraction
```bash
conda run -n ChangHai python /skills/master_slice_extractor/scripts/extract_tiled_master_slice.py \
    --ct /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz \
    --pancreas-mask /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz \
    --patient-id {PATIENT_ID} \
    --output /workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_tiled.png \
    --windows "standard:40:400,narrow:40:150,soft:50:250"
```

### Step 3: Verify Tiled Image
```bash
# Check image properties
python -c "
from PIL import Image
import os
img = Image.open('/workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_tiled.png')
print(f'Size: {img.size}')
print(f'Mode: {img.mode}')
# Expected: (1536, 512) or similar (3 x 512 width)
"

ls -lh /workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_tiled.png
```

## 4. Tiled Image Structure

```
┌─────────────────────────────────────────────────────────────┐
│  [Standard W:400]  │  [Narrow W:150]  │  [Soft W:250]      │
│  C:40, HU:[-160,240]│  C:40, HU:[-35,115]│ C:50, HU:[-75,175]│
│  General anatomy   │  ⭐ ISODENSE       │  Boundary def     │
│                    │     ENHANCED       │                   │
│  Pancreas: low     │  Pancreas: visible │  Vessels: clear   │
│  contrast          │  Tumor: visible!   │                   │
└─────────────────────────────────────────────────────────────┘
         512px              512px               512px
                    Total: 1536 x 512
```

## 5. Use Cases

### Use Case 1: Deep Drill Protocol
**When**: nnU-Net returns 0ml tumor volume
**Action**: Generate Tiled → VLM analysis
**Purpose**: Detect isodense tumors missed by HU-threshold segmentation

### Use Case 2: Segmentation Verification
**When**: nnU-Net detects tumor
**Action**: Generate Tiled → VLM confirmation
**Purpose**: Validate segmentation against visual findings

### Use Case 3: MDT Presentation
**When**: Generating clinical report
**Action**: Include Tiled image for radiologist review
**Purpose**: Provide intuitive multi-window view

## 6. Error Handling

| Error | Diagnosis | Action |
|-------|-----------|--------|
| Pancreas mask not found | TotalSegmentator not run | Go back to totalseg_segmentor |
| Empty output | Wrong Z-slice | Check pancreas_analysis.json for correct Z |
| Image too dark/light | Wrong window settings | Verify C:40, W:150 for narrow window |

## 7. Success Criteria
- [ ] Tiled image generated (1536 x 512 pixels)
- [ ] Three distinct window regions visible
- [ ] Pancreas region clearly visible in narrow window
- [ ] Image ready for VLM analysis

## 8. Citation Format
```
[Script: extract_tiled_master_slice.py, Output: {PATIENT_ID}_master_slice_tiled.png, Windows: Standard/Narrow/Soft, Z: {value}]
```
