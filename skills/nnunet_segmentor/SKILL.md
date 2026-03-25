---
name: nnunet_segmentor
category: tumor_segmentation
description: nnU-Net v1 (MSD Task07) for pancreatic tumor segmentation. WARNING: May produce FALSE NEGATIVES (0ml) for isodense tumors. Deep Drill Protocol is MANDATORY for 0ml results.
---

# nnU-Net Segmentor: Cognitive Execution Protocol

## 1. Identity & Clinical Mindset
You are the Tumor Detection Specialist. Your goal is to segment pancreatic tumors using the gold-standard nnU-Net model trained on MSD Task07.

**⚠️ CRITICAL WARNING**: nnU-Net relies on HU thresholding. PDAC tumors with desmoplastic stroma appear ISODENSE (same HU as normal pancreas) and may return **0ml** (false negative).

**Your duty**: When volume = 0ml, you MUST trigger **Deep Drill Protocol** (multi-window VLM analysis) before concluding "no tumor".

## 2. API Contract (Execution)
**Environment**: `conda run -n nnunetv2`
**Executable**: `nnUNet_predict` (nnU-Net CLI)
**Arguments**:
- `-i <path>`: Input directory
- `-o <path>`: Output directory
- `-t Task007_Pancreas`: Task name
- `-m 3d_fullres`: Model configuration

**(Agent, you MUST set nnU-Net environment variables first!)**

## 3. Cognitive Reasoning & SOP

### Step 0: Environment Setup (MANDATORY)
```bash
export nnUNet_raw_data_base="/media/luzhenyang/project/ChangHai_PDA/data/models/nnunet/nnunet_v1_workspace/raw"
export nnUNet_preprocessed="/media/luzhenyang/project/ChangHai_PDA/data/models/nnunet/nnunet_v1_workspace/preprocessed"
export RESULTS_FOLDER="/media/luzhenyang/project/ChangHai_PDA/data/models/nnunet/nnunet_v1_workspace/results"
```

### Step 1: Verify Prerequisites
```bash
# Check NIfTI exists (DISCOVER, don't assume)
find /workspace/sandbox/data/processed/nifti -name "*{PATIENT_ID}*.nii.gz"

# Check model weights exist
ls $RESULTS_FOLDER/nnUNet/3d_fullres/Task007_Pancreas/ 2>/dev/null || echo "Model not found!"
```

### Step 2: Prepare Input
```bash
# Copy to nnU-Net raw data directory with proper naming
mkdir -p ${nnUNet_raw_data_base}/nnUNet_raw_data/Task007_Pancreas/imagesTs
cp {DISCOVERED_NIFTI_PATH} \
   ${nnUNet_raw_data_base}/nnUNet_raw_data/Task007_Pancreas/imagesTs/{PATIENT_ID}_0000.nii.gz
```

### Step 3: Check for Cached Results
```bash
# Check if output already exists
ls /workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/{PATIENT_ID}.nii.gz 2>/dev/nul

# If exists: Skip to analysis
# If not: Proceed
```

### Step 4: Execute nnU-Net
```bash
conda run -n nnunetv2 nnUNet_predict \
    -i ${nnUNet_raw_data_base}/nnUNet_raw_data/Task007_Pancreas/imagesTs \
    -o /workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/ \
    -t Task007_Pancreas \
    -m 3d_fullres
```

### Step 5: Analyze Output
```bash
# Calculate tumor volume from label 2
python -c "
import nibabel as nib
import numpy as np

seg = nib.load('/workspace/sandbox/data/processed/segmentations/nnunet_tumor_output_{PATIENT_ID}/{PATIENT_ID}.nii.gz').get_fdata()
tumor_mask = (seg == 2).astype(np.uint8)
volume_ml = np.sum(tumor_mask) * np.prod([1.0, 1.0, 1.0]) / 1000

print(f'Tumor Volume: {volume_ml:.2f} ml')
print(f'Has Tumor: {volume_ml > 0}')

if volume_ml == 0:
    print('⚠️ WARNING: Potential false negative - consider Deep Drill')
"
```

## 4. Deep Drill Protocol (MANDATORY for 0ml)

**⚠️ If tumor_volume_ml = 0, DO NOT conclude "no tumor"!**

### Deep Drill Trigger Condition:
```python
if tumor_volume_ml == 0:
    TRIGGER_DEEP_DRILL()
```

### Deep Drill Actions:
1. **Generate multi-window Tiled image**:
   ```bash
   python /skills/master_slice_extractor/scripts/extract_tiled_master_slice.py \
       --ct {DISCOVERED_NIFTI_PATH} \
       --pancreas-mask /workspace/sandbox/data/processed/segmentations/{PATIENT_ID}/pancreas.nii.gz
   ```

2. **VLM Visual Confirmation**:
   - Use `analyze_image` tool on Tiled image
   - Query: "Analyze for isodense masses in narrow window"

3. **Decision Matrix**:
   | nnU-Net | VLM Assessment | Conclusion |
   |---------|----------------|------------|
   | 0ml | Normal | True negative |
   | 0ml | Suspicious | **ENDOGENOUS_FALSE_NEGATIVE** |

## 5. Success Criteria
- [ ] nnU-Net executed with correct environment variables
- [ ] Tumor volume calculated
- [ ] **If volume = 0ml: Deep Drill triggered**
- [ ] Results documented with physical traceability

## 6. Citation Format
```
[Script: nnUNet_predict, Output: {PATIENT_ID}.nii.gz, Labels: [0,1,2]]
[Script: tumor_analysis.py, Output: Tumor_Volume: {X}ml, Has_Tumor: {true/false}]
```

**Special Citation for 0ml with Deep Drill**:
```
[Script: nnUNet_predict, Output: Tumor_Volume: 0ml]
[Deep Drill: Triggered - Multi-window VLM analysis initiated]
```
