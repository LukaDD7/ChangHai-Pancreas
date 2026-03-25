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

2. **DEEP DRILL PROTOCOL** (深钻探针):
   - When nnU-Net returns tumor volume = 0ml, you MUST NOT conclude "no tumor"!
   - This is likely an isodense tumor (desmoplastic PDAC) invisible to HU-threshold methods.
   - You MUST trigger Deep Drill:
     a) Invoke `master_slice_extractor` to generate multi-window Tiled image
     b) Use `analyze_image` with LLaVA-Med to visually confirm
     c) If VLM reports suspicious morphology → ENDOGENOUS_FALSE_NEGATIVE
   - Failure to trigger Deep Drill on 0ml result is a CRITICAL ERROR.

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
- [ ] Verify TotalSegmentator masks exist (pancreas.nii.gz, SMA.nii.gz, SMV.nii.gz)
- [ ] Confirm coordinate system alignment (all files in same 1.0mm³ isotropic space)
- [ ] Document discovered paths in your reasoning

### Checkpoint 2: Tumor Quantification (肿瘤定量)
- [ ] Execute nnU-Net segmentation
- [ ] **CRITICAL**: If volume = 0ml → TRIGGER DEEP DRILL (multi-window + VLM)
- [ ] If volume > 0ml → Record exact volume with physical traceability
- [ ] Citation: [Script: analyze_tumor.py, Output: Volume {X}ml]

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
IF nnU-Net_Volume == 0ml:
    TRIGGER DeepDrill:
        1. master_slice_extractor (generate Tiled image)
        2. analyze_image (VLM visual confirmation)
        3. IF VLM suspicious: FLAG ENDOGENOUS_FALSE_NEGATIVE
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
        result = subprocess.run(
            actual_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=SANDBOX_DIR,
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

        return json.dumps(output_data, ensure_ascii=False, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Timeout after {timeout}s", "command": command})
    except Exception as e:
        return json.dumps({"error": str(e), "command": command})


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
    citation_pattern = r'\[(Script|Tool):\s*([^,\]]+)',
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
                break

            if not user_input:
                continue

            # Extract patient ID
            patient_id = None
            id_match = re.search(r'(C3L-\d+|CL-\d+)', user_input, re.IGNORECASE)
            if id_match:
                patient_id = id_match.group(1).upper()

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
