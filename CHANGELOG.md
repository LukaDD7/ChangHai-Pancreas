# ChangHai PDA Agent Changelog

All notable changes to the ChangHai PDA Agent project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2026-03-30

### LLaVA-Med Comprehensive Clinical Testing Framework

#### Added
- **Comprehensive test suite** (`test_llava_med_comprehensive.py`):
  - Official LLaVA-Med inference pattern (single-window, non-tiled)
  - vicuna_v1 conversation template
  - GPU device selection support (`--gpu` flag, default: GPU 6)
  - Model caching to avoid repeated loading

- **7 Clinical Test Cases** covering 5-layer diagnostic workflow:
  1. **L1 - Solid vs Cystic Classification**: Initial characterization of pancreatic masses
  2. **L2 - PDAC vs pNET Differentiation**: Solid tumor subtype classification
  3. **L3 - Cystic Tumor Subtyping**: SCN/MCN/IPMN/SPN classification
  4. **L4 - Acute Pancreatitis CTSI**: CT Severity Index scoring (Atlanta 2024)
  5. **L5 - Chronic Pancreatitis Scoring**: Cambridge/MRCP severity assessment
  6. **L6 - Vascular Invasion Assessment**: NCCN resectability criteria (SMA/SMV/CA/CHA/PV)
  7. **L7 - Double Duct Sign Detection**: PDAC-specific sign (CBD + MPD dilation)

- **Structured prompts** for each diagnostic task:
  - Clinical guidelines embedded (NCCN 2025, Atlanta 2024, WHO 2024)
  - Specific evaluation criteria
  - Standardized output format (diagnosis, confidence, key findings)

#### Changed
- **LLaVA-Med model loading** (`builder.py`):
  - Fixed: `low_cpu_mem_usage=False` → `True` to support device_map
  - Enables proper multi-GPU allocation

- **Delayed initialization pattern**:
  - CUDA_VISIBLE_DEVICES set before torch import
  - Prevents GPU allocation conflicts

#### Usage
```bash
# Run all 7 clinical tests on specified GPU
CUDA_VISIBLE_DEVICES=6 conda run -n llava-med python \
  data/scripts/test_llava_med_comprehensive.py

# Output saved to:
# data/results/llava_med_tests/llava_med_comprehensive_test_{timestamp}.json
```

#### Status
- ⚠️ **Blocked**: All GPUs occupied (wupengli: train_softcot9_22.py)
- **Ready to run** when GPU resources available

---

## [1.1.1] - 2026-03-26

### TotalSegmentator Version and Weight Audit

#### Verified
- **Installed package**: `TotalSegmentator 2.13.0`
- **Installed weights** are v2-style dataset series under `~/.totalsegmentator/nnunet/results/`, including:
  - `Dataset291_TotalSegmentator_part1_organs_1559subj`
  - `Dataset292_TotalSegmentator_part2_vertebrae_1532subj`
  - `Dataset293_TotalSegmentator_part3_cardiac_1559subj`
  - `Dataset294_TotalSegmentator_part4_muscles_1559subj`
  - `Dataset295_TotalSegmentator_part5_ribs_1559subj`
  - `Dataset297_TotalSegmentator_total_3mm_1559subj`
  - `Dataset298_TotalSegmentator_total_6mm_1559subj`

#### Changed
- Confirmed the current installation is **not** a downgraded v1.5 setup.
- Confirmed that removing `--fast` from the `total` task does **not** produce SMA/SMV/CA masks for CL-03356.

#### Impact
- The missing PDAC-critical vessels are now attributed to current TotalSegmentator task/class coverage rather than a package downgrade.
- Pan-Agent still requires a dedicated pancreatic vessel-capable model or an alternative segmentation pass to reach the target standard for SMA, SMV, PV, and CA analysis.

---

## [1.0.0] - 2026-03-24

### Major Architecture Refactoring

#### Changed
- **Agent Autonomy**: Removed hard-coded workflow and mandatory checkpoints
  - Agent now decides which Skill to read and execute based on clinical context
  - No pre-defined execution order - Agent adapts to each patient's unique situation
  - Agent can skip skills if clinically irrelevant or add steps if necessary

- **Meta-Cognitive Conflict Detection**: Replaced hard-coded keyword weights with Agent's semantic judgment
  - Removed rigid scoring formula: `score = sum(keyword_weights)`
  - Agent uses clinical reasoning to detect conflicts between nnU-Net and VLM results
  - No hard threshold (e.g., `score > 1.5`) - Agent's judgment IS the threshold
  - Conflict detection: IF (nnU-Net: "no tumor") AND (VLM: "suspicious morphology") THEN Agent decides

- **Multi-Window Implementation**: Documented HU window transformation formula
  - Implementation: `windowed = (raw_hu - center) / width * 255 + 128`
  - Narrow window (W:150): 20 HU range → 34 grayscale levels (2.6x enhancement)
  - Purpose: Detect isodense tumors invisible in standard window

#### Removed
- Hard-coded 5 mandatory checkpoints from L3 System Prompt
- Keyword weight scoring system (mass:1.0, irregular:0.8, etc.)
- Suspicion score threshold (>1.5) for conflict triggering
- Pre-defined Skill execution sequences

#### Added
- Agent-driven execution model
- Semantic understanding as conflict detection mechanism
- Meta-cognitive reasoning in L3 System Prompt
- Clinical judgment-based decision making
- Skills directory with 7 modular capabilities:
  - dicom_processor
  - totalseg_segmentor
  - nnunet_segmentor
  - master_slice_extractor
  - llava_med_analyzer
  - adw_ceo_reporter
  - vascular_topology

#### Architecture
- Deep Agents flat architecture (NO Subagents)
- CompositeBackend: FilesystemBackend + StoreBackend
- Human-in-the-loop: User provides patient ID, Agent autonomously completes analysis
- Evidence sovereignty: All measurements must have physical traceability

---

## [1.1.0] - 2026-03-25

### Full Execution Logging System (ExecutionLogger)

#### Added
- **ExecutionLogger class**: Complete traceability of all Agent actions
  - Per-patient log directories: `execution_logs/{patient_id}_{session_id}/`
  - Structured JSONL logs (`execution_log.jsonl`) + human-readable logs (`execution.log`)
  - Artifact manifest (`artifacts.json`) tracking all generated files
  - Session summary with execution statistics

- **Automatic artifact detection**: `_detect_and_log_artifacts()` function
  - Parses stdout for generated files (NIfTI, PNG, JSON, etc.)
  - Automatically logs artifacts without Agent intervention
  - Supports common output patterns: "Saved to:", "Output:", file extensions

- **Tool call logging**: All 4 core tools now log to ExecutionLogger
  - `execute`: Logs command, arguments, result, duration, auto-detects artifacts
  - `read_file`: Logs file reads with path and line count
  - `analyze_image`: Logs VLM calls and tracks image artifacts
  - `submit_mdt_report`: Logs report submission with citation validation

- **User input tracking**: `log_user_input()` captures patient ID entry

#### Enhanced
- **nnU-Net environment auto-injection**: `execute()` tool now automatically sets
  required environment variables when running nnU-Net commands:
  - `nnUNet_raw_data_base`
  - `nnUNet_preprocessed`
  - `RESULTS_FOLDER`

- **Security enhancements**:
  - Blocked server-wide `find` without `-maxdepth` limit
  - Dangerous pattern detection (curl, wget, ssh, etc.)
  - Command whitelist enforcement

- **Virtual path handling**: Improved `/skills/` and `/workspace/sandbox/` conversion
  to actual filesystem paths in `execute()` tool

#### Architecture
- Deep Agents flat architecture with full audit trail
- Evidence sovereignty: All measurements have physical traceability via logs
- Citation validation: Reports rejected if citations lack execution records

---

### Initial Architecture Setup

#### Added
- Interactive main entry point: `interactive_main.py`
- L3 System Prompt with 5 mandatory checkpoints
- Hard-coded workflow with keyword weights for conflict detection
- Multi-window Tiled strategy documentation
- 4 Core Tools: execute, read_file, analyze_image, submit_pdac_report
- Execution logger with audit trail
- Citation validation system

---

## Comparison: v0.9 vs v1.0

| Aspect | v0.9 (Pre-Refactor) | v1.0 (Current) |
|--------|----------------------|----------------|
| **Workflow** | Hard-coded sequence | Agent decides |
| **Conflict Detection** | Keyword weights + threshold | Agent semantic judgment |
| **Checkpoints** | 5 mandatory | None (Agent decides) |
| **Scoring** | `score = sum(weights)` | Agent understanding |
| **Threshold** | `> 1.5` hard-coded | Agent clinical reasoning |
| **Skills** | Must follow order | Agent selects adaptively |
| **Philosophy** | State machine | Autonomous Agent |

---

## Migration Notes

### For Developers

If you were relying on the hard-coded scoring system:

**Before:**
```python
# Hard-coded in system prompt
keywords = {'mass': 1.0, 'irregular': 0.8}
if suspicion_score > 1.5:
    trigger_conflict()
```

**After:**
```
# Agent's semantic understanding
Agent reads VLM output and nnU-Net result
Agent applies clinical reasoning to detect conflicts
No formula - Agent judgment IS the detection mechanism
```

### For Users

No changes to usage:
```bash
conda run -n ChangHai python interactive_main.py
# Enter patient ID
# Agent autonomously completes analysis
```

The Agent now makes more intelligent, context-aware decisions.

---

*Last Updated: 2026-03-30*
