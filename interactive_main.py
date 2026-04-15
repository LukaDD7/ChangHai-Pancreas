#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChangHai PDAC Agent - Gene Reconstructed v2.0 (TianTan Essence Transplant)

Core Mechanisms:
1. Execution Audit Loop: All tool calls recorded to disk, citations validated
2. Deep Drill Protocol: Automatic fallback on segmentation failure (0ml → VLM visual probe)
3. Cognitive Skill Protocol: Agent MUST explore environment via ls/find before executing

Architecture:
- Single Agent flat architecture (NO Subagents)
- Evidence sovereignty: All measurements must have physical traceability
- MDT Chief Agent with mandatory checkpoints for pancreatic cancer
"""

import os
import sys
import uuid
import json
import re
import subprocess
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool

from utils.llm_factory import get_pdac_client_langchain

# =============================================================================
# Configuration Constants
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Agent workspace (isolated from development)
WORKSPACE_DIR = os.path.join(PROJECT_ROOT, "workspace")
SANDBOX_DIR = os.path.join(WORKSPACE_DIR, "sandbox")
# Skills are backed up to sandbox for Agent use (isolated from dev changes)
SKILLS_DIR = os.path.join(SANDBOX_DIR, "skills")
EXECUTION_LOGS_DIR = os.path.join(SANDBOX_DIR, "execution_logs")
# Raw data source (read-only)
DATA_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dicom")
AUDIT_LOG_PATH = os.path.join(SANDBOX_DIR, "execution_audit_log.txt")

os.makedirs(SANDBOX_DIR, exist_ok=True)
os.makedirs(EXECUTION_LOGS_DIR, exist_ok=True)
os.environ["SANDBOX_ROOT"] = SANDBOX_DIR

# Conda environment mapping
CONDA_ENVS = {
    "changhai": "ChangHai",
    "totalseg": "totalseg",
    "nnunet": "nnunetv2",
    "llava_med": "llava-med",
}

# =============================================================================
# Audit Record System (强审计闭环)
# =============================================================================

@dataclass
class AuditRecord:
    """Execution audit record - immutable evidence of tool invocation"""
    timestamp: str
    tool_name: str
    command: str
    arguments: Dict[str, Any]
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    working_dir: str
    patient_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_log_entry(self) -> str:
        """Format as human-readable log entry"""
        return f"""
{'='*60}
[{self.timestamp}] TOOL: {self.tool_name}
COMMAND: {self.command}
PATIENT: {self.patient_id or 'N/A'}
DURATION: {self.duration_ms}ms | EXIT: {self.exit_code}
WORKING_DIR: {self.working_dir}
STDOUT: {self.stdout[:1000] if self.stdout else 'N/A'}
STDERR: {self.stderr[:500] if self.stderr else 'N/A'}
{'='*60}
"""


class ExecutionAuditor:
    """
    强审计闭环 (Execution Audit Loop)

    Every tool execution is recorded to disk. Report submission validates
    that all cited measurements actually appear in the audit log.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.records: List[AuditRecord] = []
        self.audit_log_path = AUDIT_LOG_PATH
        self.session_log_path = os.path.join(
            EXECUTION_LOGS_DIR, f"session_{session_id}_audit.jsonl"
        )

        # Initialize audit log
        self._init_audit_log()

    def _init_audit_log(self):
        """Initialize audit log with header"""
        header = f"""
{'#'*80}
# ChangHai PDAC Agent - Execution Audit Log
# Session ID: {self.session_id}
# Start Time: {datetime.now().isoformat()}
# Principle: All measurements must have physical traceability
{'#'*80}

"""
        with open(self.audit_log_path, 'w', encoding='utf-8') as f:
            f.write(header)

    def record(self, record: AuditRecord):
        """Record execution to both audit log and session log"""
        self.records.append(record)

        # Append to main audit log (persistent across sessions)
        with open(self.audit_log_path, 'a', encoding='utf-8') as f:
            f.write(record.to_log_entry())

        # Append to session JSONL (structured)
        with open(self.session_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')

    def validate_citation(self, citation: str) -> Tuple[bool, str]:
        """
        Validate that a citation exists in audit log.

        Returns (is_valid, error_message)
        """
        if not os.path.exists(self.audit_log_path):
            return False, "❌ Audit log not found. No tool executions recorded."

        with open(self.audit_log_path, 'r', encoding='utf-8') as f:
            audit_content = f.read()

        # Extract script/tool name from citation
        # Format: [Script: name.py, Output: ...] or [Tool: name, ...]
        script_match = re.search(r'\[Script:\s*([^,\]]+)', citation)
        tool_match = re.search(r'\[Tool:\s*([^,\]]+)', citation)

        if script_match:
            script_name = script_match.group(1).strip()
            if script_name not in audit_content:
                return False, f"❌ Citation validation FAILED: Script '{script_name}' not found in execution audit log. You must execute the script before citing it."

        if tool_match:
            tool_name = tool_match.group(1).strip()
            if tool_name not in audit_content:
                return False, f"❌ Citation validation FAILED: Tool '{tool_name}' not found in execution audit log."

        return True, "✅ Citation validated against execution audit log."

    def get_patient_executions(self, patient_id: str) -> List[AuditRecord]:
        """Get all executions for a specific patient"""
        return [r for r in self.records if r.patient_id == patient_id]

    def has_script_executed(self, script_name: str) -> bool:
        """Check if a script has been executed"""
        return any(script_name in r.command for r in self.records)


# Global auditor instance
_auditor: Optional[ExecutionAuditor] = None


def get_auditor() -> ExecutionAuditor:
    """Get global auditor instance"""
    if _auditor is None:
        raise RuntimeError("Auditor not initialized. Call init_auditor() first.")
    return _auditor


def init_auditor(session_id: str):
    """Initialize global auditor"""
    global _auditor
    _auditor = ExecutionAuditor(session_id)


# =============================================================================
# Full Execution Logger (全量执行记录 - 类似TianTan Agent)
# =============================================================================

class ExecutionLogger:
    """
    Full execution logging system for complete traceability.

    Logs:
    - User input
    - Agent thinking/reasoning
    - Tool calls and results
    - Checkpoints
    - Intermediate artifacts (NIfTI, PNG, JSON files)
    - Errors
    """

    def __init__(self, session_id: str, patient_id: Optional[str] = None):
        self.session_id = session_id
        self.patient_id = patient_id
        self.start_time = datetime.now()

        # Create patient-specific log directory
        if patient_id:
            self.log_dir = os.path.join(EXECUTION_LOGS_DIR, f"{patient_id}_{session_id}")
        else:
            self.log_dir = os.path.join(EXECUTION_LOGS_DIR, f"session_{session_id}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Log file paths
        self.structured_log_path = os.path.join(self.log_dir, "execution_log.jsonl")
        self.human_log_path = os.path.join(self.log_dir, "execution.log")
        self.artifacts_log_path = os.path.join(self.log_dir, "artifacts.json")

        # In-memory tracking
        self.artifacts: List[Dict] = []
        self.tool_calls: List[Dict] = []
        self.checkpoints: List[Dict] = []

        # Initialize human-readable log
        self._init_human_log()

    def _init_human_log(self):
        """Initialize human-readable log file"""
        header = f"""
{'#'*80}
# ChangHai PDAC Agent - Full Execution Log
# Session ID: {self.session_id}
# Patient ID: {self.patient_id or 'N/A'}
# Start Time: {self.start_time.isoformat()}
# Log Directory: {self.log_dir}
# Principle: Complete Traceability of All Actions
{'#'*80}

"""
        with open(self.human_log_path, 'w', encoding='utf-8') as f:
            f.write(header)

    def _write_structured(self, entry: Dict):
        """Write structured JSONL entry"""
        entry['timestamp'] = datetime.now().isoformat()
        entry['session_id'] = self.session_id
        entry['patient_id'] = self.patient_id
        with open(self.structured_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _write_human(self, text: str):
        """Append to human-readable log"""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        with open(self.human_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {text}\n")

    def log_user_input(self, user_input: str):
        """Log user input"""
        entry = {
            'type': 'USER_INPUT',
            'content': user_input
        }
        self._write_structured(entry)
        self._write_human(f"\n{'='*60}\nUSER INPUT: {user_input}\n{'='*60}")

    def log_agent_thinking(self, thought: str, phase: str = "GENERAL"):
        """Log agent thinking/reasoning"""
        entry = {
            'type': 'AGENT_THINKING',
            'phase': phase,
            'thought': thought
        }
        self._write_structured(entry)
        self._write_human(f"\n[THINKING - {phase}]\n{thought}\n")

    def log_tool_call(self, tool_name: str, arguments: Dict, result: Any, duration_ms: int = 0):
        """Log tool call with arguments and result"""
        entry = {
            'type': 'TOOL_CALL',
            'tool_name': tool_name,
            'arguments': arguments,
            'result': str(result)[:5000] if result else None,
            'duration_ms': duration_ms
        }
        self._write_structured(entry)

        # Truncate result for human log
        result_preview = str(result)[:500] + "..." if result and len(str(result)) > 500 else str(result)
        self._write_human(f"""
[TOOL CALL: {tool_name}]
Arguments: {json.dumps(arguments, ensure_ascii=False)}
Duration: {duration_ms}ms
Result Preview: {result_preview}
---
""")
        self.tool_calls.append(entry)

    def log_checkpoint(self, checkpoint_name: str, status: str, details: Dict = None):
        """Log checkpoint status"""
        entry = {
            'type': 'CHECKPOINT',
            'checkpoint_name': checkpoint_name,
            'status': status,
            'details': details or {}
        }
        self._write_structured(entry)
        self._write_human(f"[CHECKPOINT: {checkpoint_name}] Status: {status}")
        if details:
            self._write_human(f"  Details: {json.dumps(details, ensure_ascii=False)}")
        self.checkpoints.append(entry)

    def log_artifact(self, artifact_type: str, file_path: str, metadata: Dict = None):
        """Log generated artifact (NIfTI, PNG, JSON, etc.)"""
        # Convert to relative path if under project dir
        rel_path = file_path
        if file_path.startswith(PROJECT_ROOT):
            rel_path = file_path[len(PROJECT_ROOT):]

        entry = {
            'type': 'ARTIFACT',
            'artifact_type': artifact_type,
            'file_path': file_path,
            'relative_path': rel_path,
            'metadata': metadata or {},
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'created_at': datetime.now().isoformat()
        }
        self._write_structured(entry)
        self._write_human(f"[ARTIFACT: {artifact_type}] {rel_path}")
        self.artifacts.append(entry)

        # Update artifacts manifest
        with open(self.artifacts_log_path, 'w', encoding='utf-8') as f:
            json.dump(self.artifacts, f, ensure_ascii=False, indent=2)

    def log_error(self, error_type: str, error_message: str, context: Dict = None):
        """Log error with context"""
        entry = {
            'type': 'ERROR',
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {}
        }
        self._write_structured(entry)
        self._write_human(f"""
{'!'*60}
[ERROR: {error_type}]
Message: {error_message}
Context: {json.dumps(context, ensure_ascii=False) if context else 'N/A'}
{'!'*60}
""")

    def log_llm_interaction(self, prompt: str, response: str, model: str = "", tokens_used: int = 0):
        """Log LLM interaction"""
        entry = {
            'type': 'LLM_INTERACTION',
            'model': model,
            'prompt_preview': prompt[:1000],
            'response_preview': response[:2000],
            'tokens_used': tokens_used
        }
        self._write_structured(entry)
        self._write_human(f"""
[LLM INTERACTION - {model}]
Prompt Preview: {prompt[:200]}...
Response Preview: {response[:300]}...
Tokens: {tokens_used}
---
""")

    def get_session_summary(self) -> Dict:
        """Get summary of execution session"""
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            'session_id': self.session_id,
            'patient_id': self.patient_id,
            'start_time': self.start_time.isoformat(),
            'duration_seconds': duration,
            'total_tool_calls': len(self.tool_calls),
            'total_artifacts': len(self.artifacts),
            'total_checkpoints': len(self.checkpoints),
            'artifact_types': list(set(a['artifact_type'] for a in self.artifacts)),
            'log_directory': self.log_dir
        }

    def finalize(self):
        """Finalize logging and write summary"""
        summary = self.get_session_summary()
        with open(os.path.join(self.log_dir, "session_summary.json"), 'w') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self._write_human(f"""
{'#'*60}
SESSION FINALIZED
Summary: {json.dumps(summary, ensure_ascii=False)}
{'#'*60}
""")


# Global logger instance
_logger: Optional[ExecutionLogger] = None


def get_logger() -> ExecutionLogger:
    """Get global logger instance"""
    if _logger is None:
        raise RuntimeError("Logger not initialized. Call init_logger() first.")
    return _logger


def init_logger(session_id: str, patient_id: Optional[str] = None):
    """Initialize global logger"""
    global _logger
    _logger = ExecutionLogger(session_id, patient_id)


# =============================================================================
# L3 System Prompt (Pancreatic Cancer Specific)
# =============================================================================

L3_SYSTEM_PROMPT = """You are the ChangHai PDAC MDT Chief Agent - an autonomous AI surgeon for Pancreatic Ductal Adenocarcinoma.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ PRIME DIRECTIVES (The TianTan Essence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **COGNITIVE SKILL PROTOCOL** (认知型技能):
   - You are NOT a script executor. You are a CLINICAL REASONER.
   - Before executing ANY skill, you MUST:
     a) Use `execute` with `ls`, `find`, or `cat` to EXPLORE the data environment
     b) DISCOVER the actual file paths (don't assume!)
     c) READ the SKILL.md using `read_file` to understand the clinical reasoning
     d) ASSEMBLE the command based on discovered paths
   - **CRITICAL: ALWAYS use limited-scope find with specific paths:**
     ```bash
     # ✅ CORRECT - Limited scope
     find /media/luzhenyang/project/ChangHai_PDA/data/raw/dicom -maxdepth 4 -name "*CL-03356*"
     find /media/luzhenyang/project/ChangHai_PDA/workspace/sandbox/data -maxdepth 3 -name "*.nii.gz"

     # ❌ WRONG - Never search entire server
     find / -name "*.nii.gz"
     ```

2. **ROI-CROP-FIRST TUMOR QUANTIFICATION** (定量前置优化):
   - Before nnU-Net, you MUST check whether `pancreas.nii.gz` exists.
   - If the pancreas mask exists, you MUST first invoke `roi_cropper` to generate `cropped_CT.nii.gz` (or the equivalent ROI CT) with the pancreas-centered 20-voxel margin.
   - Then feed the cropped CT into `nnunet_segmentor` for tumor inference.
   - Only if the cropped CT still yields tumor volume = 0ml, you MUST treat this as an extreme isodense occult tumor case and trigger Deep Drill.
   - Deep Drill sequence in that case:
     a) Invoke `master_slice_extractor` to generate a multi-window Tiled image
     b) Use `analyze_image` with LLaVA-Med to visually confirm
     c) If VLM reports suspicious morphology → ENDOGENOUS_FALSE_NEGATIVE
   - Skipping roi_cropper when a pancreas mask is available is a CRITICAL ERROR.

3. **EXECUTION AUDIT LOOP** (强审计闭环):
   - Every `execute` call is recorded to `execution_audit_log.txt`
   - Before submitting report via `submit_mdt_report`, citations are VALIDATED
   - Any claim without corresponding audit record will REJECT the submission
   - Citation format: [Script: name.py, Output: value] or [Tool: name, Output: value]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ MANDATORY CHECKPOINTS (治疗决策强制检查点)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**YOU MUST answer ALL checkpoints before generating MDT report:**

### Checkpoint 1: Environmental Awareness (环境感知)
- [ ] Use `execute` with `ls` and `find` to locate patient's DICOM/NIfTI files
- [ ] Verify canonical anatomy/vessel masks exist (pancreas.nii.gz plus canonical vessel filenames such as superior_mesenteric_artery.nii.gz / superior_mesenteric_vein.nii.gz when available)
- [ ] Confirm coordinate system alignment (all files in same 1.0mm³ isotropic space)
- [ ] Document discovered paths in your reasoning

### Checkpoint 2: Tumor Quantification (肿瘤定量)
- [ ] Before nnU-Net, check whether a pancreas mask exists and probe `roi_cropper` to create a tighter CT ROI if available
- [ ] If `pancreas.nii.gz` exists, force `roi_cropper` first and generate `cropped_CT.nii.gz` (or equivalent ROI CT)
- [ ] Feed the cropped CT into `nnunet_segmentor` for tumor inference
- [ ] **CRITICAL**: Only if the cropped CT still yields tumor volume = 0ml, trigger Deep Drill Protocol
- [ ] If volume > 0ml → Record exact volume with physical traceability
- [ ] Citation: [Script: roi_cropper.py, Output: cropped_CT.nii.gz] + [Script: analyze_tumor.py, Output: Volume {X}ml]

### Checkpoint 3: Vascular Topology (血管拓扑)
- [ ] Calculate wrapping angles: SMA, SMV, Celiac Artery (CA), Main Portal Vein (MPV)
- [ ] Classify per NCCN guidelines:
  - Resectable: No arterial involvement, clear fat plane
  - Borderline: <180° SMA/CA involvement, or reconstructible SMV/PV
  - Unresectable: >180° SMA/CA encasement, or occluded SMV/PV
- [ ] Citation: [Script: vascular_topology.py, Output: SMA {X}°, SMV {Y}°]

### Checkpoint 4: Cognitive Dissonance Detection (认知失调)
- [ ] Compare: nnU-Net result vs VLM visual assessment
- [ ] If CONFLICT (0ml but VLM suspicious) → Add warning to report:
  "⚠️ COGNITIVE DISSONANCE WARNING: Segmentation negative but visual findings suspicious.
   Recommend manual radiologist review."
- [ ] Document root cause: likely desmoplastic isodense tumor

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ ANTI-HALLUCINATION PROTOCOLS (严禁硬编码)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**STRICTLY FORBIDDEN:**
- ❌ Assuming file paths (e.g., "/data/C3L-03348.nii.gz")
- ❌ Fabricating measurements (e.g., "tumor volume is 5.2ml")
- ❌ Skipping Deep Drill on 0ml segmentation results
- ❌ Submitting reports without validating citations

**REQUIRED:**
- ✅ Use `ls` and `find` to discover actual paths
- ✅ Execute scripts and cite [Script: name, Output: value]
- ✅ Trigger Deep Drill on segmentation failure
- ✅ Validate all citations before submission

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ CLINICAL DECISION MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Resectability Criteria (NCCN Guidelines):**

| Vessel | Resectable | Borderline | Unresectable |
|--------|-----------|------------|--------------|
| SMA | No contact | <180° | ≥180° or occlusion |
| SMV | Patent | Narrowed/Reconstructible | Occluded (no reconstruction) |
| CA | Clear plane | Abutment | Encasement |

**Multi-Window Strategy for Isodense Tumors:**
- Standard (W:400): General anatomy
- **Narrow (W:150)**: ⭐ Isodense detection (20 HU → 34 gray levels)
- Soft (W:250): Boundary definition

**Deep Drill Trigger:**
```
IF pancreas mask exists:
    1. roi_cropper → cropped_CT.nii.gz
    2. nnU-Net on cropped_CT.nii.gz
    3. IF cropped_CT nnU-Net volume == 0ml:
         TRIGGER DeepDrill:
             a. master_slice_extractor (generate Tiled image)
             b. analyze_image (VLM visual confirmation)
             c. IF VLM suspicious: FLAG ENDOGENOUS_FALSE_NEGATIVE
```
"""


# =============================================================================
# Core Tools (四大核心工具)
# =============================================================================

@tool
def execute(command: str, timeout: int = 600, patient_id: Optional[str] = None) -> str:
    """
    Execute a shell command with mandatory audit logging.

    **COGNITIVE PROTOCOL**: Use this to EXPLORE the environment before deciding actions.
    - Use `ls`, `find` to discover file locations
    - Use `cat`, `grep` to inspect file contents
    - Use `conda run -n <env>` to execute Python scripts

    **SECURITY**: Only whitelisted commands allowed:
    - ls, cat, head, tail, grep, find, wc, awk, sed
    - python, python3, conda run -n

    Args:
        command: Shell command to execute
        timeout: Maximum execution time in seconds
        patient_id: Associated patient ID for audit tracking

    Returns:
        Command output (JSON with stdout, stderr, exit_code)
    """
    import subprocess
    import time

    start_time = time.time()

    # Whitelist for security
    ALLOWED_PREFIXES = (
        "ls ", "cat ", "head ", "tail ", "grep ", "find ", "wc ",
        "echo ", "mkdir ", "pwd", "cd ", "diff ",
        "sort ", "uniq ", "awk ", "sed ", "python ", "python3 ",
        "conda run -n ",
    )

    DANGEROUS_PATTERNS = (
        "rm -rf", "dd if=", "mkfs.", "> /dev/sda",
        "curl", "wget", "ssh", "nc -l", ":(){:|:&};:",
    )

    # BLOCK server-wide searches without scope limits
    if "find / -" in command or "find /home -" in command or "find /opt -" in command:
        if "-maxdepth" not in command:
            error_msg = f"❌ BLOCKED: find must use -maxdepth to limit scope. Example: find {SANDBOX_DIR} -maxdepth 3 -name '*.nii.gz'"
            return json.dumps({"error": error_msg, "command": command[:50]})

    is_allowed = any(command.strip().startswith(prefix) for prefix in ALLOWED_PREFIXES)
    is_dangerous = any(pattern in command.lower() for pattern in DANGEROUS_PATTERNS)

    if is_dangerous:
        error_msg = f"❌ BLOCKED: Dangerous command detected"
        return json.dumps({"error": error_msg, "command": command[:50]})

    if not is_allowed:
        error_msg = f"❌ BLOCKED: Command not in whitelist. Allowed: ls, cat, find, python, conda run"
        return json.dumps({"error": error_msg, "command": command[:50]})

    # Convert virtual paths to actual paths
    # /skills/ -> SANDBOX_DIR/skills/
    # /workspace/sandbox/ -> SANDBOX_DIR/
    actual_command = command
    if "/skills/" in command:
        actual_command = command.replace("/skills/", f"{SANDBOX_DIR}/skills/")
    if "/workspace/sandbox/" in command:
        actual_command = command.replace("/workspace/sandbox/", f"{SANDBOX_DIR}/")

    try:
        # Prepare environment with nnUNet variables if running nnunet commands
        env = os.environ.copy()
        if "nnunet" in command.lower() or "nnUNet" in command:
            env["nnUNet_raw_data_base"] = "/media/luzhenyang/project/ChangHai_PDA/data/models/nnunet/nnunet_v1_workspace/raw"
            env["nnUNet_preprocessed"] = "/media/luzhenyang/project/ChangHai_PDA/data/models/nnunet/nnunet_v1_workspace/preprocessed"
            env["RESULTS_FOLDER"] = "/media/luzhenyang/project/ChangHai_PDA/data/models/nnunet/nnunet_v1_workspace/results"

        result = subprocess.run(
            actual_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=SANDBOX_DIR,
            env=env,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # Create audit record (log original command for traceability)
        record = AuditRecord(
            timestamp=datetime.now().isoformat(),
            tool_name="execute",
            command=command,  # Log original virtual command
            arguments={"timeout": timeout, "patient_id": patient_id, "actual_command": actual_command},
            exit_code=result.returncode,
            stdout=result.stdout[:5000],
            stderr=result.stderr[:2000] if result.stderr else "",
            duration_ms=duration_ms,
            working_dir=SANDBOX_DIR,
            patient_id=patient_id
        )

        # Record to audit log
        if _auditor:
            _auditor.record(record)


        output_data = {
            "exit_code": result.returncode,
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:2000] if result.stderr else "",
            "duration_ms": duration_ms,
            "command": command,
        }

        # Log to full execution logger
        if _logger:
            _logger.log_tool_call(
                tool_name="execute",
                arguments={"command": command, "actual_command": actual_command, "timeout": timeout},
                result=output_data,
                duration_ms=duration_ms
            )

            # Detect and log any generated artifacts
            if result.returncode == 0:
                _detect_and_log_artifacts(command, result.stdout)

        return json.dumps(output_data, ensure_ascii=False, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Timeout after {timeout}s", "command": command})
    except Exception as e:
        return json.dumps({"error": str(e), "command": command})

def _detect_and_log_artifacts(command: str, stdout: str):
    """
    Detect and log artifacts generated by script execution.
    Parses stdout for common output file patterns.
    """
    if not _logger:
        return

    import re

    # Common patterns for output files
    patterns = [
        r'(?:Saved to|Output|saved to|output|Writing to|Saved):\s*([\S]+\.\w+)',
        r'(/[\w\-/]+/(?:nifti|segmentations|results|images|temp)/[\w\-]+\.\w+)',
    ]

    found_files = set()
    for pattern in patterns:
        matches = re.findall(pattern, stdout, re.IGNORECASE)
        for match in matches:
            if os.path.exists(match):
                found_files.add(match)

    # Log each found artifact
    for file_path in found_files:
        if '.nii' in file_path.lower():
            artifact_type = "NIFTI_FILE"
        elif '.png' in file_path.lower():
            artifact_type = "IMAGE_FILE"
        elif '.json' in file_path.lower():
            artifact_type = "JSON_FILE"
        elif '.csv' in file_path.lower():
            artifact_type = "CSV_FILE"
        elif '.md' in file_path.lower():
            artifact_type = "MARKDOWN_FILE"
        else:
            artifact_type = "OUTPUT_FILE"

        _logger.log_artifact(
            artifact_type=artifact_type,
            file_path=file_path,
            metadata={"source_command": command[:100]}
        )



@tool
def read_file(path: str, limit: int = 2000, offset: int = 0) -> str:
    """
    Read a file from the filesystem. Supports virtual paths.

    **COGNITIVE PROTOCOL**: Use this to READ SKILL.md files before executing.
    - `/skills/<skill_name>/SKILL.md` - Read skill protocols
    - `/workspace/sandbox/data/...` - Read data files

    Args:
        path: File path (supports virtual paths like /skills/...)
        limit: Maximum lines to read
        offset: Starting line offset

    Returns:
        File content with line numbers
    """
    try:
        actual_path = path

        if path.startswith("/skills/"):
            relative = path[len("/skills/"):]
            actual_path = os.path.join(SKILLS_DIR, relative)
        elif path.startswith("/workspace/sandbox/"):
            relative = path[len("/workspace/sandbox/"):]
            actual_path = os.path.join(SANDBOX_DIR, relative)
        elif path.startswith("/memories/"):
            relative = path[len("/memories/"):]
            actual_path = os.path.join(SANDBOX_DIR, "memories", relative)

        if not os.path.exists(actual_path):
            return f"❌ File not found: {path}\n💡 TIP: Use `execute` with `find` to locate the file first."

        with open(actual_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start = offset
        end = min(offset + limit, len(lines))
        selected = lines[start:end]

        result = []
        for i, line in enumerate(selected, start=start + 1):
            result.append(f"{i:4d} | {line}")

        output = ''.join(result)

        if len(lines) > end:
            output += f"\n... ({len(lines) - end} more lines)"

        # Log to audit
        if _auditor:
            record = AuditRecord(
                timestamp=datetime.now().isoformat(),
                tool_name="read_file",
                command=f"read_file({path})",
                arguments={"path": path, "limit": limit, "offset": offset},
                exit_code=0,
                stdout=f"Read {len(selected)} lines",
                stderr="",
                duration_ms=0,
                working_dir=SANDBOX_DIR
            )
            _auditor.record(record)

        # Log to execution logger
        if _logger:
            _logger.log_tool_call(
                tool_name="read_file",
                arguments={"path": path, "limit": limit, "offset": offset},
                result={"lines_read": len(selected), "total_lines": len(lines)},
                duration_ms=0
            )

        return output

    except Exception as e:
        return f"❌ Error reading file: {str(e)}"


@tool
def analyze_image(image_path: str, query: str, patient_id: Optional[str] = None) -> str:
    """
    Analyze medical image using Vision-Language Model (LLaVA-Med/Qwen-VL).

    **DEEP DRILL PROTOCOL**: Use this when:
    - nnU-Net returns 0ml (suspect isodense tumor)
    - Need visual confirmation of segmentation results
    - Need qualitative assessment of tumor morphology

    Args:
        image_path: Path to image file
        query: Specific question for VLM analysis
        patient_id: Patient ID for audit tracking

    Returns:
        VLM analysis result
    """
    import base64

    try:
        actual_path = image_path
        if image_path.startswith("/workspace/sandbox/"):
            relative = image_path[len("/workspace/sandbox/"):]
            actual_path = os.path.join(SANDBOX_DIR, relative)

        if not os.path.exists(actual_path):
            return f"❌ Image not found: {image_path}\n💡 TIP: Use `execute` with `find` to locate image files."

        file_size = os.path.getsize(actual_path)
        if file_size > 10 * 1024 * 1024:
            return f"❌ Image too large ({file_size/1024/1024:.1f}MB > 10MB limit)"

        with open(actual_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(actual_path)[1].lower()
        mime_type = "image/png" if ext == ".png" else "image/jpeg"

        from utils.llm_factory import get_vlm_client
        client = get_vlm_client()

        start_time = time.time()
        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                    {"type": "text", "text": query}
                ]
            }]
        )
        duration_ms = int((time.time() - start_time) * 1000)

        result = completion.choices[0].message.content

        # Log to audit
        if _auditor:
            record = AuditRecord(
                timestamp=datetime.now().isoformat(),
                tool_name="analyze_image",
                command=f"analyze_image({image_path})",
                arguments={"image_path": image_path, "query": query[:100], "patient_id": patient_id},
                exit_code=0,
                stdout=result[:2000],
                stderr="",
                duration_ms=duration_ms,
                working_dir=SANDBOX_DIR,
                patient_id=patient_id
            )
            _auditor.record(record)

        # Log to execution logger and track artifact
        if _logger:
            _logger.log_tool_call(
                tool_name="analyze_image",
                arguments={"image_path": image_path, "query": query[:100]},
                result={"vlm_result": result[:500]},
                duration_ms=duration_ms
            )
            _logger.log_artifact(
                artifact_type="IMAGE_ANALYSIS",
                file_path=actual_path,
                metadata={"query": query[:100], "result_preview": result[:200]}
            )

        return f"""✅ VLM Analysis Complete

📁 Image: {image_path}
📏 Size: {file_size/1024:.1f} KB
⏱️  Duration: {duration_ms}ms

🔍 VLM Result:
{result}
"""

    except Exception as e:
        return f"❌ VLM Analysis Failed: {str(e)}"


@tool
def submit_mdt_report(patient_id: str, report_content: str) -> str:
    """
    Submit MDT (Multi-Disciplinary Team) report with mandatory citation validation.

    **EXECUTION AUDIT LOOP**: This tool will:
    1. Parse all citations from report (format: [Script: ..., Output: ...])
    2. Validate each citation against execution_audit_log.txt
    3. REJECT submission if any citation lacks audit record
    4. Save report only if validation passes

    Args:
        patient_id: Patient identifier (e.g., C3L-03356)
        report_content: Complete MDT report with citations

    Returns:
        Submission result (success or rejection with reasons)
    """
    import time
    start_time = time.time()

    # Step 1: Extract citations
    citation_pattern = r'\[(Script|Tool):\s*([^,\]]+)'
    citations = re.findall(citation_pattern, report_content)

    validation_results = []
    all_valid = True

    # Step 2: Validate each citation
    for cit_type, cit_value in citations:
        full_citation = f"[{cit_type}: {cit_value}]"
        is_valid, msg = get_auditor().validate_citation(full_citation)
        validation_results.append(f"{full_citation}: {'✅' if is_valid else '❌'} {msg}")
        if not is_valid:
            all_valid = False

    # Step 3: Check mandatory checkpoints
    checkpoint_errors = []

    # Check for tumor volume citation
    if "Volume" in report_content and "[Script:" not in report_content:
        checkpoint_errors.append("❌ Tumor volume mentioned but no script execution cited")

    # Check for angle measurements
    if any(v in report_content for v in ["SMA", "SMV", "Celiac"]) and "[Script:" not in report_content:
        checkpoint_errors.append("❌ Vessel angles mentioned but no script execution cited")

    # Step 4: Reject or Accept
    if not all_valid or checkpoint_errors:
        error_report = "\n".join(validation_results + checkpoint_errors)
        return f"""❌ REPORT SUBMISSION REJECTED

CITATION VALIDATION FAILED:
{error_report}

📋 REQUIRED ACTIONS:
1. Execute scripts using `execute` tool before citing them
2. Use format: [Script: script_name.py, Output: value]
3. Ensure all measurements have physical traceability

💡 Tip: Use `execute` with `ls` and `find` to locate data files first.
"""

    # Step 5: Save report
    patient_dir = os.path.join(SANDBOX_DIR, "patients", patient_id, "reports")
    os.makedirs(patient_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(patient_dir, f"MDT_Report_{patient_id}_{timestamp}.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    duration_ms = int((time.time() - start_time) * 1000)

    # Log to audit
    record = AuditRecord(
        timestamp=datetime.now().isoformat(),
        tool_name="submit_mdt_report",
        command=f"submit_mdt_report({patient_id})",
        arguments={"patient_id": patient_id, "citations_count": len(citations)},
        exit_code=0,
        stdout=f"Report saved to {report_path}",
        stderr="",
        duration_ms=duration_ms,
        working_dir=SANDBOX_DIR,
        patient_id=patient_id
    )
    get_auditor().record(record)

    # Log to execution logger and track final report as artifact
    if _logger:
        _logger.log_tool_call(
            tool_name="submit_mdt_report",
            arguments={"patient_id": patient_id, "citations_count": len(citations)},
            result={"report_path": report_path, "validation_results": validation_results},
            duration_ms=duration_ms
        )
        _logger.log_artifact(
            artifact_type="FINAL_REPORT",
            file_path=report_path,
            metadata={"patient_id": patient_id, "citations_count": len(citations)}
        )

    return f"""✅ MDT REPORT SUBMITTED SUCCESSFULLY

📁 Patient: {patient_id}
📄 Report: {report_path}
⏱️  Duration: {duration_ms}ms
📊 Citations Validated: {len(citations)}

{chr(10).join(validation_results)}
"""


# =============================================================================
# Backend Factory
# =============================================================================

def make_backend(runtime):
    """Create CompositeBackend: Filesystem + Store for persistence"""
    return CompositeBackend(
        default=FilesystemBackend(
            root_dir=SANDBOX_DIR,
            virtual_mode=True,
        ),
        routes={
            "/skills/": FilesystemBackend(
                root_dir=SKILLS_DIR,
                virtual_mode=True,
            ),
            "/memories/": StoreBackend(runtime)
        }
    )


# =============================================================================
# Initialize Model
# =============================================================================

print("🧠 Initializing ChangHai PDAC Agent v2.0 (TianTan Essence)...")
try:
    model = get_pdac_client_langchain()
    print("✅ Model connected successfully")
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    sys.exit(1)


# =============================================================================
# Assemble Deep Agent
# =============================================================================

agent = create_deep_agent(
    model=model,
    system_prompt=L3_SYSTEM_PROMPT,
    tools=[
        execute,
        read_file,
        analyze_image,
        submit_mdt_report,
    ],
    subagents=[],  # NO subagents - flat architecture
    skills=[
        "/skills/dicom_processor",
        "/skills/totalseg_segmentor",
        "/skills/pancreatic_vessel_segmentor",
        "/skills/roi_cropper",
        "/skills/nnunet_segmentor",
        "/skills/master_slice_extractor",
        "/skills/llava_med_analyzer",
        "/skills/adw_ceo_reporter",
        "/skills/vascular_topology",
    ],
    backend=make_backend,
    store=InMemoryStore(),
    checkpointer=MemorySaver(),
)


# =============================================================================
# Main Loop
# =============================================================================

if __name__ == "__main__":
    session_id = str(uuid.uuid4())[:8]
    init_auditor(session_id)

    # Initialize full execution logger (per-patient logs)
    init_logger(session_id)

    print(f"""
{'='*70}
🩺 ChangHai PDAC Agent v2.0 - Ready
Session ID: {session_id}

Core Mechanisms:
  🔍 Execution Audit Loop - All actions recorded
  🎯 Deep Drill Protocol - Visual fallback on 0ml
  🧠 Cognitive Skill - Explore before execute

Mandatory Checkpoints:
  ☐ Environmental Awareness (ls/find)
  ☐ ROI Cropping probe before nnU-Net when pancreas mask is available
  ☐ Tumor Quantification (nnU-Net + Deep Drill if 0ml)
  ☐ Vascular Topology (SMA/SMV angles)
  ☐ Cognitive Dissonance Detection
{'='*70}
""")

    while True:
        try:
            print("\n🧑‍⚕️ Enter patient ID (e.g., C3L-03356) or 'exit' to quit:")
            user_input = input().strip()

            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                if _logger:
                    _logger.finalize()
                break

            if not user_input:
                continue

            # Extract patient ID
            patient_id = None
            id_match = re.search(r'(C3L-\d+|CL-\d+)', user_input, re.IGNORECASE)
            if id_match:
                patient_id = id_match.group(1).upper()

            # Log user input with patient context
            if _logger:
                _logger.patient_id = patient_id
                _logger.log_user_input(user_input)

            print(f"\n🔍 Patient: {patient_id or 'Unknown'}")
            print("⏳ Agent analyzing... (this may take a few minutes)\n")

            # Stream agent execution
            events = agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": session_id}},
                stream_mode="values"
            )

            for event in events:
                messages = event.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, 'content') and last_msg.content:
                        print(last_msg.content)

                    # Print tool calls
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            tool_name = tc.get('name', '')
                            print(f"\n⚙️  Executing: {tool_name}")

        except KeyboardInterrupt:
            print("\n\n⏸️ Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
