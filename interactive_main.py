#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChangHai PDAC Agent - Interactive Main Entry (v1.0 Deep Agent Architecture)

Architecture:
- Single Agent flat architecture (NO Subagents)
- Skills-based modular design for multi-modal medical AI
- Dynamic module applicability judgment
- CompositeBackend: FilesystemBackend + StoreBackend for persistence
- Execute tool: Custom shell execution with conda environment switching
- Evidence sovereignty: All measurements must have physical traceability
"""

import os
import sys
import uuid
import json
import re
import subprocess
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

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
SKILLS_DIR = os.path.join(PROJECT_ROOT, "skills")
SANDBOX_DIR = os.path.join(PROJECT_ROOT, "workspace", "sandbox")
EXECUTION_LOGS_DIR = os.path.join(SANDBOX_DIR, "execution_logs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

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
# Global Execution Logger
# =============================================================================

execution_logger: Optional['ExecutionLogger'] = None


def get_logger() -> Optional['ExecutionLogger']:
    """Get global execution logger"""
    return execution_logger


class ExecutionLogger:
    """Complete execution history logger - Records all agent behaviors"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.log_entries: List[Dict[str, Any]] = []

        self.main_log_path = os.path.join(
            EXECUTION_LOGS_DIR,
            f"session_{session_id}_complete.log"
        )
        self.json_log_path = os.path.join(
            EXECUTION_LOGS_DIR,
            f"session_{session_id}_structured.jsonl"
        )
        self.audit_log_path = os.path.join(SANDBOX_DIR, "execution_audit_log.txt")

        self._write_header()

    def _write_header(self):
        """Write log header"""
        header = f"""
{'='*80}
ChangHai PDAC Agent - Complete Execution History
Session ID: {self.session_id}
Start Time: {self.start_time.isoformat()}
{'='*80}

"""
        with open(self.main_log_path, 'w', encoding='utf-8') as f:
            f.write(header)

    def _write_log(self, entry: Dict[str, Any]):
        """Write single log entry"""
        with open(self.json_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        readable_entry = self._format_readable(entry)
        with open(self.main_log_path, 'a', encoding='utf-8') as f:
            f.write(readable_entry + '\n')

    def _format_readable(self, entry: Dict[str, Any]) -> str:
        """Format log entry to human-readable"""
        timestamp = entry.get('timestamp', datetime.now().isoformat())
        event_type = entry.get('type', 'unknown')

        formatted = f"\n{'─'*80}\n"
        formatted += f"⏱️  {timestamp} | Type: {event_type}\n"
        formatted += f"{'─'*80}\n"

        if event_type == 'user_input':
            patient_id = entry.get('patient_id', 'N/A')
            content = entry.get('content', '')
            formatted += f"👤 User Input (Patient ID: {patient_id})\n"
            formatted += f"   Content:\n{content}\n"

        elif event_type == 'tool_call':
            tool_name = entry.get('tool_name', 'unknown')
            arguments = entry.get('arguments', {})
            result = entry.get('result')
            duration = entry.get('duration_ms', 0)
            formatted += f"⚙️  Tool Call: {tool_name}\n"
            formatted += f"   Args: {json.dumps(arguments, ensure_ascii=False, indent=2)}\n"
            formatted += f"   Duration: {duration}ms\n"
            if result:
                formatted += f"   Result: {str(result)[:500]}...\n"

        elif event_type == 'agent_thinking':
            content = entry.get('content', '')
            thinking_type = entry.get('thinking_type', 'general')
            emoji = {'reasoning': '🧠', 'planning': '📋', 'decision': '⚖️', 'error': '❌',
                     'system': '🔧', 'interrupt': '⏸️'}.get(thinking_type, '💭')
            formatted += f"{emoji} Agent Thinking ({thinking_type}):\n{content}\n"

        elif event_type == 'skill_execution':
            skill_name = entry.get('skill_name', 'unknown')
            status = entry.get('status', 'unknown')
            formatted += f"📚 Skill Execution: {skill_name} ({status})\n"

        return formatted

    def log_user_input(self, content: str, patient_id: Optional[str] = None):
        """Log user input"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "user_input",
            "patient_id": patient_id,
            "content": content,
            "content_length": len(content)
        }
        self._write_log(entry)

    def log_tool_call(self, tool_name: str, arguments: Dict, result: Any, duration_ms: int):
        """Log tool call"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "duration_ms": duration_ms
        }
        self._write_log(entry)

    def log_agent_thinking(self, content: str, thinking_type: str = "general"):
        """Log agent thinking"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "agent_thinking",
            "content": content,
            "thinking_type": thinking_type
        }
        self._write_log(entry)

    def log_skill_execution(self, skill_name: str, status: str, details: Dict = None):
        """Log skill execution"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "skill_execution",
            "skill_name": skill_name,
            "status": status,
            "details": details or {}
        }
        self._write_log(entry)

    def log_error(self, message: str, error_type: str = "runtime_error", traceback_info: str = ""):
        """Log error"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "message": message,
            "error_type": error_type,
            "traceback": traceback_info
        }
        self._write_log(entry)


# =============================================================================
# Initialize Model
# =============================================================================

try:
    print("🧠 Connecting to PDAC Medical Brain (Deep Thinking Mode)...")
    model = get_pdac_client_langchain()
except Exception as e:
    print(f"❌ Brain initialization failed: {e}")
    sys.exit(1)


# =============================================================================
# L3 Master Control System Prompt (Agent Autonomy Architecture)
# =============================================================================

L3_SYSTEM_PROMPT = """You are a Pancreatic Ductal Adenocarcinoma (PDAC) MDT Chief Agent - an autonomous medical AI with meta-cognitive capabilities.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ AGENT AUTONOMY PRINCIPLES (Meta-Cognitive)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**NEVER encode workflows into your reasoning.** You are not a state machine. You are an intelligent agent.

1. **SELF-DIRECTED DISCOVERY**:
   - You have 7 Skills mounted in `/skills/`. Each Skill's SKILL.md contains its own reasoning.
   - When facing a task, FIRST read the relevant SKILL.md using read_file tool.
   - UNDERSTAND the skill's purpose, inputs, outputs - not just execute commands.
   - DECIDE yourself which skills to use, in what order, based on clinical context.

2. **NO HARD-CODED WORKFLOWS**:
   - There are NO "mandatory checkpoints" or "phases" you must follow.
   - Your reasoning should adapt to each patient's unique situation.
   - Skip skills if clinically irrelevant. Add steps if clinically necessary.

3. **EVIDENCE SOVEREIGNTY**:
   - NEVER fabricate tumor volumes, HU values, coordinates.
   - All measurements MUST be obtained through actual script execution.
   - Citation format: [Script: <name>, Output: <value>] or [Local: <file>, Line: <n>]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ META-THINKING: CONFLICT DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Do NOT use hard-coded keyword weights.** Your semantic understanding IS the weight.

Instead of:
```
if "mass" in text: score += 1.0  # ❌ Hard-coded
if "irregular" in text: score += 0.8  # ❌ Rigid
```

Think like this:
```
"The VLM described 'subtle contour irregularity in the pancreatic head with
hypo-attenuation relative to surrounding parenchyma' - this is clinically
suspicious for malignancy despite the formal segmentation showing no lesion."
# Your understanding of medical language IS the judgment.
```

**Conflict Detection (Endogenous)**:
- Compare: nnU-Net segmentation result vs VLM semantic analysis
- If they disagree (e.g., nnU-Net says "no tumor" but VLM reports suspicious findings)
- THEN: Detect ENDOGENOUS_FALSE_NEGATIVE
- ACTION: ESCALATE_TO_RADIOLOGIST
- No threshold. No formula. Your clinical judgment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ SKILL INVENTORY (Your Capabilities)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You have access to these skills - USE AS NEEDED, NOT SEQUENTIALLY:

| Skill | Purpose | When to Use |
|-------|---------|-------------|
| dicom_processor | DICOM→NIfTI conversion | New patient, raw data |
| totalseg_segmentor | Organ/vessel segmentation | Need anatomy reference |
| nnunet_segmentor | Tumor segmentation | Primary detection task |
| master_slice_extractor | Multi-window tiled images | Need VLM visual analysis |
| llava_med_analyzer | VLM image analysis | Semantic interpretation |
| adw_ceo_reporter | Conflict detection & reporting | Disagreement between models |
| vascular_topology | Resectability assessment | Surgical planning |

**YOU decide the order. YOU decide which to skip. YOU decide when to loop back.**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ MULTI-WINDOW STRATEGY (For Isodense Tumors)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PDAC tumors are often isodense (same HU as normal pancreas) - invisible in standard window.

**Window Settings** (implemented via pixel value transformation):
- Standard: Center 40, Width 400 (HU range: -160 to 240)
- **Narrow**: Center 40, Width 150 (HU range: -35 to 115) ⭐ Isodense detection
- Soft Tissue: Center 50, Width 250 (HU range: -75 to 175)

**Implementation**: The master_slice_extractor script applies these window transforms:
```python
# Pseudo-code for window transform
windowed = (raw_hu - center) / width * 255 + 128
windowed = np.clip(windowed, 0, 255)
```

**Output**: Tiled image (1536×512) - 3 windows side by side for VLM comparison.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ PHYSICAL TRACEABILITY CITATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Measurement: [Script: <script_name>, Output: <value>]
- Image: [Local: <filename>, Visual: Slice Z<value>]
- Segmentation: [Tool: <tool_name>, Volume: <value>ml]
- VLM Analysis: [Tool: LLaVA-Med, Finding: <description>]
- **NEVER cite non-existent scripts!**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ MEMORY PROTOCOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ NEVER read `/memories/personas/` before report generation - avoid stale data.

- Read current patient data via execute tool (real-time)
- Write persona AFTER report (for future reference only)
- Clinical principles in `/memories/principles/` can be read/written anytime

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ REPORT SUBMISSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use `submit_pdac_report` tool when complete. Tool validates citations against execution log.
"""


# =============================================================================
# Custom Tools
# =============================================================================

@tool
def analyze_image(image_path: str, query: str = "Analyze this CT image and describe any abnormalities.") -> str:
    """Analyze a CT/Medical image using VLM (Vision-Language Model).

    Use this tool when you need to:
    - Extract information from CT slices (tumor detection, vessel relationships)
    - Analyze complex medical images that are unclear in text format
    - Verify segmentation results against original images

    Args:
        image_path: Path to the image file (e.g., /workspace/sandbox/patients/CL-03356/master_slice_tiled.png)
        query: Specific question for VLM analysis (default: general tumor detection)

    Returns:
        VLM analysis result with extracted findings
    """
    import base64
    import os

    try:
        actual_path = image_path
        if image_path.startswith("/workspace/sandbox/"):
            relative = image_path[len("/workspace/sandbox/"):]
            actual_path = os.path.join(SANDBOX_DIR, relative)

        if not os.path.exists(actual_path):
            return f"Error: Image not found: {image_path}"

        file_size = os.path.getsize(actual_path)
        if file_size > 10 * 1024 * 1024:
            return f"Error: Image too large ({file_size / 1024 / 1024:.2f}MB > 10MB limit)"

        with open(actual_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(actual_path)[1].lower()
        mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png" if ext == ".png" else "image/jpeg"

        from utils.llm_factory import get_vlm_client
        client = get_vlm_client()

        completion = client.chat.completions.create(
            model="qwen3.5-plus",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                    {"type": "text", "text": query}
                ]
            }]
        )

        result = completion.choices[0].message.content

        if get_logger():
            get_logger().log_tool_call(
                "analyze_image",
                {"image_path": image_path, "query": query[:100]},
                {"result_length": len(result), "image_size": file_size},
                0
            )

        return f"""✅ VLM Analysis Complete

📁 Image: {image_path}
📏 Size: {file_size / 1024:.1f} KB
❓ Query: {query}

🔍 Result:
{result}
"""
    except Exception as e:
        error_msg = f"Error analyzing image: {str(e)}"
        if get_logger():
            get_logger().log_error(error_msg, "image_analysis_error")
        return error_msg


@tool
def read_file(path: str, limit: int = 2000, offset: int = 0) -> str:
    """Read a file from the filesystem. Supports virtual paths like /skills/ and /memories/.

    Args:
        path: File path (can be virtual path like /skills/dicom_processor/SKILL.md)
        limit: Maximum number of lines to read (default 2000)
        offset: Line offset to start reading from (default 0)

    Returns:
        File content with line numbers
    """
    try:
        actual_path = path

        if path.startswith("/skills/"):
            relative = path[len("/skills/"):]
            actual_path = os.path.join(SKILLS_DIR, relative)
        elif path.startswith("/memories/"):
            relative = path[len("/memories/"):]
            actual_path = os.path.join(SANDBOX_DIR, "memories", relative)
        elif path.startswith("/workspace/sandbox/"):
            relative = path[len("/workspace/sandbox/"):]
            actual_path = os.path.join(SANDBOX_DIR, relative)

        if not os.path.exists(actual_path):
            return f"Error: File not found: {path}"

        with open(actual_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start = offset
        end = offset + limit if limit > 0 else len(lines)
        selected_lines = lines[start:end]

        result = []
        for i, line in enumerate(selected_lines, start=start + 1):
            result.append(f"{i:4d} | {line}")

        output = ''.join(result)

        if get_logger():
            get_logger().log_tool_call(
                "read_file",
                {"path": path, "limit": limit, "offset": offset},
                {"content_length": len(output), "actual_path": actual_path},
                0
            )

        if len(lines) > end:
            output += f"\n... ({len(lines) - end} more lines)"

        return output

    except Exception as e:
        error_msg = f"Error reading file {path}: {str(e)}"
        if get_logger():
            get_logger().log_error(error_msg, "file_read_error")
        return error_msg


@tool
def execute(command: str, timeout: int = 600, max_output_bytes: int = 100000) -> str:
    """Execute a shell command and return the output. Supports conda environment switching.

    Args:
        command: The shell command to execute
        timeout: Maximum execution time in seconds (default: 600)
        max_output_bytes: Maximum output size in bytes (default: 100000)

    Returns:
        Command output (stdout + stderr) or error message
    """
    import subprocess
    import time

    start_time = time.time()

    # Whitelist for security
    ALLOWED_PREFIXES = (
        "ls ", "cat ", "head ", "tail ", "grep ", "find .",
        "echo ", "mkdir ", "pwd", "cd ", "wc ", "diff ",
        "sort ", "uniq ", "awk ", "sed ", "python ", "python3 ",
        "conda run -n ",  # Allow conda environment switching
    )

    DANGEROUS_PATTERNS = (
        "rm -rf", "rm -rf /", "dd if=", "mkfs.", "> /dev/sda",
        "curl", "wget", "ssh", "nc ",
    )

    is_allowed = any(command.strip().startswith(prefix) for prefix in ALLOWED_PREFIXES)
    is_dangerous = any(pattern in command.lower() for pattern in DANGEROUS_PATTERNS)

    if is_dangerous:
        error_msg = f"Error: Command blocked by security policy: {command[:100]}"
        if get_logger():
            get_logger().log_tool_call("execute", {"command": command}, {"error": error_msg}, 0)
        return error_msg

    if not is_allowed:
        error_msg = f"Error: Command not in whitelist: {command[:100]}"
        if get_logger():
            get_logger().log_tool_call("execute", {"command": command}, {"error": error_msg}, 0)
        return error_msg

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=SANDBOX_DIR,
            env={**os.environ, "SANDBOX_ROOT": SANDBOX_DIR}
        )

        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr

        if len(output.encode('utf-8')) > max_output_bytes:
            output = output[:max_output_bytes] + "\n... [output truncated]"

        duration_ms = int((time.time() - start_time) * 1000)

        result_dict = {
            "exit_code": result.returncode,
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:2000] if result.stderr else "",
            "current_working_dir": SANDBOX_DIR,
        }

        if get_logger():
            get_logger().log_tool_call("execute", {"command": command}, result_dict, duration_ms)

        # Audit log
        audit_log_path = os.path.join(SANDBOX_DIR, "execution_audit_log.txt")
        with open(audit_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*40}\n")
            f.write(f"TIME: {datetime.now().isoformat()}\n")
            f.write(f"COMMAND: {command}\n")
            f.write(f"DURATION: {duration_ms}ms\n")
            f.write(f"EXIT_CODE: {result.returncode}\n")
            f.write(f"OUTPUT:\n{output[:2000]}\n")

        return json.dumps(result_dict, ensure_ascii=False, indent=2)

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool
def submit_pdac_report(patient_id: str, report_content: str) -> str:
    """Submit the PDAC analysis report with citation validation.

    This tool will:
    1. Validate all citations in the report against execution audit log
    2. Block any fabricated citations
    3. Only save the report if validation passes

    Args:
        patient_id: Patient identifier (e.g., CL-03356)
        report_content: The complete PDAC analysis report

    Returns:
        Submission result message
    """
    import time
    start_time = time.time()

    if get_logger():
        get_logger().log_tool_call("submit_pdac_report", {"patient_id": patient_id}, None, 0)

    # Citation validation
    is_valid, msg = validate_citations(report_content)
    if not is_valid:
        if get_logger():
            get_logger().log_error(f"Citation validation failed: {msg}", "citation_validation")
        return msg

    # Save report
    patient_dir = os.path.join(SANDBOX_DIR, "patients", patient_id, "reports")
    os.makedirs(patient_dir, exist_ok=True)

    file_path = os.path.join(patient_dir, f"PDAC_Report_{patient_id}.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    duration_ms = int((time.time() - start_time) * 1000)

    result_msg = f"✅ PDAC REPORT SUBMITTED! Saved to {file_path}"
    if get_logger():
        get_logger().log_tool_call("submit_pdac_report", {"patient_id": patient_id}, {"status": "success", "file_path": file_path}, duration_ms)

    return result_msg


def validate_citations(report_content: str) -> tuple:
    """Validate citations in report"""
    import re

    audit_log_path = os.path.join(SANDBOX_DIR, "execution_audit_log.txt")

    if not os.path.exists(audit_log_path):
        return False, "❌ No execution audit log found. You must execute commands before citing."

    with open(audit_log_path, "r", encoding="utf-8") as f:
        audit_content = f.read()

    # Check for citations
    citations = {
        'script': re.findall(r'\[Script: ([^\]]+)', report_content),
        'tool': re.findall(r'\[Tool: ([^\]]+)', report_content),
        'local': re.findall(r'\[Local: ([^\]]+)', report_content),
    }

    for script_ref in citations['script']:
        if script_ref.split(',')[0].strip() not in audit_content:
            return False, f"❌ Citation validation failed: [Script: {script_ref}] not found in execution log."

    return True, "All citations validated."


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
# Assemble Deep Agent
# =============================================================================

print(f"🚀 Initializing ChangHai PDAC Agent v1.0 (Deep Agent Architecture)...")
print(f"   - Sandbox: {SANDBOX_DIR}")
print(f"   - Execution Logs: {EXECUTION_LOGS_DIR}")
print(f"   - Memories: /memories/")
print(f"   - Skills: 7 PDAC-specific skills mounted")
print(f"   - Architecture: Flat + Dynamic Module Adaptation")
print(f"   - Subagents: None")

store = InMemoryStore()
checkpointer = MemorySaver()

interrupt_config = {
    "write_file": False,
    "edit_file": False,
    "execute": False,
    "read_file": False,
}

agent = create_deep_agent(
    model=model,
    system_prompt=L3_SYSTEM_PROMPT,
    tools=[
        execute,
        read_file,
        analyze_image,
        submit_pdac_report,
    ],
    subagents=[],
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
    store=store,
    checkpointer=checkpointer,
    interrupt_on=interrupt_config,
)


# =============================================================================
# Main Loop
# =============================================================================

if __name__ == "__main__":
    patient_thread_id = str(uuid.uuid4())
    execution_logger = ExecutionLogger(patient_thread_id)

    print(f"\n✅ ChangHai PDAC Agent v1.0 Ready. Session ID: {patient_thread_id}")
    print(f"📝 Execution Log: {execution_logger.main_log_path}")
    print(f"💡 Input patient ID (e.g., CL-03356) or clinical data")
    print("-" * 60)

    while True:
        try:
            print("\n🧑‍⚕️ Enter patient information (empty line or `---` to end, exit to quit):")
            lines = []
            empty_line_count = 0
            while True:
                try:
                    line_bytes = sys.stdin.buffer.readline()
                    if not line_bytes:
                        break
                    try:
                        line = line_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        line = line_bytes.decode('utf-8', errors='replace')

                    if line.strip().lower() in ["exit", "quit"] and len(lines) == 0:
                        print("👋 Goodbye!")
                        sys.exit(0)

                    if line.strip() == "" or line.strip() == "---":
                        empty_line_count += 1
                        if empty_line_count >= 1:
                            break
                    else:
                        empty_line_count = 0

                    lines.append(line)

                except EOFError:
                    break

            user_input = "\n".join(lines).strip()
            if not user_input:
                continue

            # Auto-detect patient ID
            patient_id = None
            id_match = re.search(r'(CL-\d+|C3L-\d+)', user_input, re.IGNORECASE)
            if id_match:
                patient_id = id_match.group(1).upper()

            execution_logger.log_user_input(user_input, patient_id)
            print(f"\n⏳ Agent thinking... (Patient ID: {patient_id or 'N/A'})")

            events = agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": patient_thread_id}},
                stream_mode="values"
            )

            last_printed_msg_id = None
            for event in events:
                messages = event.get("messages", [])
                if not messages:
                    continue
                last_msg = messages[-1]

                if id(last_msg) == last_printed_msg_id:
                    continue
                last_printed_msg_id = id(last_msg)

                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        tool_args = tc.get('args', {})
                        print(f"\n⚙️  [Tool Call] {tool_name} (args: {tool_args})")

                content = getattr(last_msg, 'content', '')
                if content:
                    print(f"\n🤖 Agent:\n{content}")
                    execution_logger.log_agent_thinking(content, "response")

        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            execution_logger.log_error(str(e), "runtime_error", traceback.format_exc())
