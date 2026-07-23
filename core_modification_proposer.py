"""
CORE MODIFICATION PROPOSER   Manages AI-Proposed Changes to Core System Files

The AI uses this module to propose modifications to its own core files.
Every proposal is routed through self_evolution_guard.py which applies the
6-stage validation pipeline before any change takes effect.

This is Phase 3 of the AGI-trajectory architecture: the AI can now improve
its own reasoning loop, memory system, and tool suite   not just its solutions.
"""

import json
import os
import time
from typing import Optional, Tuple

PROPOSAL_LOG_PATH = "sandbox/modification_proposals.json"


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def submit_modification(
        target_file: str,
        proposed_code: str,
        rationale: str,
        epoch: int) -> Tuple[bool, str]:
    """
    Submit a proposed modification to a core file.
    Routes through self_evolution_guard.validate_and_apply().

    Args:
        target_file:   Relative path to the file to modify
        proposed_code: Full new content of the file
        rationale:     Short description of why this change is proposed
        epoch:         Current epoch number

    Returns:
        (accepted: bool, report: str)
    """
    # Log the proposal before validation
    log = _load_json(PROPOSAL_LOG_PATH, [])
    log.append({
        "epoch": epoch,
        "target": target_file,
        "rationale": rationale,
        "code_length": len(proposed_code),
        "timestamp": time.time(),
        "status": "pending",
    })
    _save_json(PROPOSAL_LOG_PATH, log)

    # Import guard here to avoid circular imports
    try:
        from self_evolution_guard import validate_and_apply
    except ImportError as e:
        return False, f"[PROPOSER] Cannot import self_evolution_guard: {e}"

    accepted, report = validate_and_apply(target_file, proposed_code, rationale, epoch)

    # Update log entry with result
    log = _load_json(PROPOSAL_LOG_PATH, [])
    if log:
        log[-1]["status"] = "accepted" if accepted else "rejected"
        log[-1]["report_summary"] = report[:200]
    _save_json(PROPOSAL_LOG_PATH, log)

    return accepted, report


def build_self_modification_prompt(
        target_file: str,
        current_code: str,
        bottleneck_description: str,
        performance_trends: str,
        epoch: int) -> str:
    """
    Build the prompt given to the AI when asking it to propose a self-modification.
    The AI must respond with the FULL new file content inside <proposed_code>...</proposed_code> tags.
    """
    return f"""
You are the Self-Evolution Engine of an Autonomous AI system.

You have permission to propose modifications to your own core files.
Your proposal will be reviewed by an automated guard that validates:
- Python syntax correctness
- No forbidden patterns (no disabling security checks, no network calls)
- No benchmark regression (your change must not lower anchor scores)

TARGET FILE TO MODIFY: {target_file}

CURRENT FILE CONTENT:
```python
{current_code[:3000]}
```
{'...[truncated]' if len(current_code) > 3000 else ''}

IDENTIFIED BOTTLENECK:
{bottleneck_description}

PERFORMANCE TRENDS:
{performance_trends}

YOUR TASK:
1. Analyze the current implementation
2. Identify the specific function or logic causing the bottleneck
3. Write an improved version of the ENTIRE file
4. Wrap your proposed new file content inside <proposed_code>...</proposed_code> tags
5. Write a one-line rationale inside <rationale>...</rationale> tags

CRITICAL RULES:
- Your proposed code must be a COMPLETE replacement of the file (not a diff)
- Do NOT remove any safety checks or immutability guards
- Do NOT add any network requests, file system access outside sandbox/, or subprocess calls
- The change must be a genuine improvement, not just reformatting
- If you cannot identify a clear improvement, write <proposed_code>NO_CHANGE</proposed_code>

Epoch: {epoch}
"""


def parse_ai_proposal(ai_response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the AI's response to extract proposed code and rationale.
    Returns (proposed_code, rationale) or (None, None) if parsing fails.
    """
    if not ai_response:
        return None, None

    # Extract proposed code
    if "<proposed_code>" not in ai_response:
        return None, None

    try:
        code_start = ai_response.index("<proposed_code>") + len("<proposed_code>")
        code_end = ai_response.index("</proposed_code>")
        proposed_code = ai_response[code_start:code_end].strip()
    except ValueError:
        return None, None

    if proposed_code == "NO_CHANGE" or not proposed_code:
        return None, "AI indicated no change needed"

    # Extract rationale
    rationale = "Self-modification (no rationale provided)"
    if "<rationale>" in ai_response and "</rationale>" in ai_response:
        try:
            r_start = ai_response.index("<rationale>") + len("<rationale>")
            r_end = ai_response.index("</rationale>")
            rationale = ai_response[r_start:r_end].strip()
        except ValueError:
            pass

    return proposed_code, rationale


def get_proposal_summary() -> str:
    """Human-readable summary of proposal history."""
    log = _load_json(PROPOSAL_LOG_PATH, [])
    if not log:
        return "No modification proposals submitted yet."
    accepted = sum(1 for e in log if e.get("status") == "accepted")
    rejected = sum(1 for e in log if e.get("status") == "rejected")
    pending  = sum(1 for e in log if e.get("status") == "pending")
    return (f"Proposals: {len(log)} total | "
            f"{accepted} accepted | {rejected} rejected | {pending} pending")


def get_evolvable_files_list() -> str:
    """Return a formatted list of files the AI is allowed to modify."""
    try:
        from self_evolution_guard import list_evolvable_files
        files = list_evolvable_files()
        return "\n".join(f"  {f}" for f in files)
    except ImportError:
        return "  (self_evolution_guard not available)"


if __name__ == "__main__":
    print("Core Modification Proposer")
    print(f"\nEvolvable files:\n{get_evolvable_files_list()}")
    print(f"\n{get_proposal_summary()}")
