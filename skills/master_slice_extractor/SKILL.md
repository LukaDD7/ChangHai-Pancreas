---
name: master-slice-extractor
version: 1.0.0
category: medical_imaging
description: |
  [FUNCTION]: Extract master slice from pancreas max area with multi-window Tiled strategy
  [VALUE]: Generates Tiled image (standard + narrow + soft tissue windows) for enhanced isodense tumor detection via VLM
  [TRIGGER_HOOK]: Read this Skill AFTER totalseg-segmentor (has pancreas max slice Z). Use BEFORE llava-med-analyzer. CRITICAL: Use TILED multi-window mode for isodense tumors!
---

# Master Slice Extractor: Cognitive Execution Protocol

## Identity & Core Mechanism
This skill extracts the key 2D CT slice from the pancreas maximum area Z-layer, with CRITICAL multi-window Tiled strategy for isodense tumor detection.

**Why Multi-Window Matters:**
- PDAC tumors often appear isodense (same HU as normal pancreas) in standard window
- Narrow window (W:150) enhances contrast by 2.7x
- Tiled image shows same slice in 3 windows for VLM analysis

---

## Phase 1: Input Preparation

**Required Inputs:**
1. CT Volume: `/workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz`
2. Pancreas Mask: `/workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz`
3. Max Slice Z: From totalseg_segmentor pancreas_analysis.json

**Read Max Slice Location:**
```bash
cat /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas_analysis.json | grep max_area_slice
```

---

## Phase 2: Multi-Window Tiled Extraction

**Purpose:** Generate Tiled image with 3 window settings

**Window Implementation:**

Window transformation is applied via HU value remapping:
```python
# For each window, transform raw HU to displayable grayscale
windowed_value = (raw_hu - center) / width * 255 + 128
windowed_value = np.clip(windowed_value, 0, 255)
```

**Window Settings:**
| Window Type | Center (L) | Width (W) | HU Range | Purpose |
|-------------|-----------|-----------|----------|---------|
| Standard | 40 | 400 | [-160, 240] | General observation |
| **Narrow** | **40** | **150** | **[-35, 115]** | **Isodense enhancement** |
| Soft Tissue | 50 | 250 | [-75, 175] | Tissue boundary definition |

**Why Narrow Window Works:**
- Normal pancreas HU: 40-60
- In standard window (W:400): 20 HU range → ~13 grayscale levels (poor contrast)
- In narrow window (W:150): 20 HU range → ~34 grayscale levels (**2.6x enhancement**)
- Isodense tumors become visible due to this contrast stretching

**Execution Command:**
```bash
conda run -n ChangHai python /skills/master_slice_extractor/scripts/extract_tiled_master_slice.py \
    --ct /workspace/sandbox/data/processed/nifti/{PATIENT_ID}/{PATIENT_ID}_CT_1mm.nii.gz \
    --pancreas-mask /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz \
    --patient-id {PATIENT_ID} \
    --output /workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_tiled.png \
    --windows "standard:40:400,narrow:40:150,soft:50:250"
```

---

## Phase 3: Tiled Image Structure

**Output Layout:**
```
┌─────────────────────────────────────────────────────┐
│  [Standard W:400]  │  [Narrow W:150]  │  [Soft W:250] │
│  General View      │  ★ Enhanced      │  Boundaries   │
│  C:40              │  C:40            │  C:50         │
└─────────────────────────────────────────────────────┘
           ↓ Input to LLaVA-Med
```

**Image Dimensions:**
- Width: ~1536 pixels (512 × 3)
- Height: 512 pixels
- Format: PNG, RGB

**CRITICAL:** The narrow window section is where isodense tumors become visible!

---

## Phase 4: Window Analysis Explanation

**Standard Window (40/400):**
- HU range: [-160, 240]
- Good for: General anatomy
- Bad for: Isodense tumors (appears same as pancreas)

**Narrow Window (40/150) ⭐:**
- HU range: [-35, 115]
- Good for: **Isodense tumor detection**
- Contrast enhancement: 2.6x (20 HU range → full grayscale)
- Pancreas HU: 40-60 → stretched across ~34 gray levels instead of ~13

**Soft Tissue Window (50/250):**
- HU range: [-75, 175]
- Good for: Tissue interfaces, vessel boundaries
- Moderate contrast

---

## Phase 5: VLM-Ready Output

**Purpose:** Verify output is ready for LLaVA-Med analysis

**Execution Command:**
```bash
ls -lh /workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_tiled.png
conda run -n ChangHai python /skills/master_slice_extractor/scripts/verify_tiled.py \
    --image /workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_tiled.png \
    --patient-id {PATIENT_ID}
```

**Expected Output:**
```json
{
  "patient_id": "{PATIENT_ID}",
  "image_path": ".../master_slice_tiled.png",
  "dimensions": [1536, 512],
  "windows": [
    {"name": "Standard", "center": 40, "width": 400, "x_range": [0, 512]},
    {"name": "Narrow", "center": 40, "width": 150, "x_range": [512, 1024]},
    {"name": "Soft", "center": 50, "width": 250, "x_range": [1024, 1536]}
  ],
  "z_position": 145,
  "pancreas_area_voxels": 2847,
  "ready_for_vlm": true
}
```

---

## Output Files

1. **Tiled Master Slice:**
   - Path: `/workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_tiled.png`
   - Format: PNG, RGB
   - Layout: Horizontal concatenation of 3 windows

2. **Metadata JSON:**
   - Path: `/workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_meta.json`

---

## Quality Checkpoints

- [ ] Image dimensions are (1536, 512) or similar
- [ ] Three distinct window regions visible
- [ ] Pancreas region clearly visible in all windows
- [ ] Narrow window shows enhanced contrast
- [ ] Z-position matches TotalSegmentator max area slice

**Citation Format:**
- Tiled image: `[Local: {PATIENT_ID}_master_slice_tiled.png, Windows: Standard/Narrow/Soft, Z: {value}]`

---

## LLaVA-Med Query Template

**When calling analyze_image tool:**

```python
analyze_image(
    image_path="/workspace/sandbox/data/results/images/{PATIENT_ID}_master_slice_tiled.png",
    query="""Analyze this multi-window CT image for PDAC signs:

This image shows the same pancreatic slice in 3 window settings:
- Left: Standard abdominal window (W:400, L:40)
- Center: **Narrow window (W:150, L:40)** - enhanced for isodense lesions
- Right: Soft tissue window (W:250, L:50)

Focus on the CENTER (narrow window) section for isodense tumor detection.

Look for:
1. Irregular contours in pancreatic head
2. Hypo-attenuating areas (darker regions)
3. Heterogeneous density patterns
4. Mass effect on surrounding structures
5. Relationship with SMV/SMA vessels

Provide suspicion score 0-10 and specific findings."""
)
```

---

## Error Handling

**Common Issues:**

1. **Pancreas not visible:**
   - Check Z-position from TotalSegmentator
   - Verify pancreas mask is non-empty

2. **Contrast too low:**
   - Verify HU window calculations
   - Check if CT is venous phase

3. **Image too large:**
   - Reduce quality parameter
   - Ensure <10MB for VLM analysis
