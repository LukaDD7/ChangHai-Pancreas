# ChangHai PDA Agent Changelog

All notable changes to the ChangHai PDA Agent project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [0.9.0] - 2026-03-20

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

*Last Updated: 2026-03-24*
