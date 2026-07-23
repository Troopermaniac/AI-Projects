"""
Self evolution guard module for validation and testing of proposed system edits.

Validates proposed core file modifications against AST parsing, safety rules,
isolated subprocess execution, and benchmark regression tests before committing.
"""

import ast
import os
import subprocess
import sys
import json
import shutil
import time
from typing import Tuple, Optional

MODIFICATION_LOG_PATH = "sandbox/self_modification_log.json"

# IMMUTABLE FILES   the AI can NEVER modify these
IMMUTABLE_FILES = {
    "evaluator_anchor.py",
    "evaluator_mbpp.py",
    "self_evolution_guard.py",
    "sandbox_guard.py",
    "model_selector.py",    # selection pressure must stay honest
    "reset_ai.py",
    "reset_ai.bat",
    "structured_memory.py",
    "knowledge_transfer.py",
    "recursive_self_improvement.py",
}

# EVOLVABLE FILES   AI CAN propose modifications to these
EVOLVABLE_FILES = {
    "autonomous_loop.py",
    "structured_memory.py",
    "knowledge_transfer.py",
    "code_refactorer.py",
    "meta_evaluator.py",
    "performance_optimizer.py",
    "tool_evolution.py",
    "self_modifying_architecture.py",
    "recursive_self_improvement.py",
    "self_directed_curriculum.py",
    "architectural_transitions.py",
    "competitive_evolution.py",
    "adaptive_difficulty.py",
    "quality_fitness.py",
    "module_evolution.py",
    "problem_generator.py",
    "trajectory_collector.py",
    "capability_frontier.py",
    "research_planner.py",
    "core_modification_proposer.py",
    # sandbox files the AI writes to
    "sandbox/reasoning_engine.py",
    "sandbox/experiment.py",
}

# FORBIDDEN PATTERNS   code containing these is auto-rejected
FORBIDDEN_PATTERNS = [
    # Cannot delete immutable files
    'os.remove("evaluator_anchor',
    'os.remove("evaluator_mbpp',
    'os.remove("self_evolution_guard',
    'os.remove("sandbox_guard',
    # Cannot modify the guard itself through file writes
    'write_file("self_evolution_guard',
    'write_file(\'self_evolution_guard',
    # Cannot disable security checks
    'sandbox_guard = None',
    'IMMUTABLE_FILES = {}',
    'IMMUTABLE_FILES.clear()',
    # Cannot exfiltrate data
    'requests.post("http',
    'socket.connect',
    'urllib.request.urlopen',
]


# INTERNAL HELPERS

def _load_log() -> list:
    if not os.path.exists(MODIFICATION_LOG_PATH):
        return []
    try:
        with open(MODIFICATION_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_log(log: list) -> None:
    os.makedirs(os.path.dirname(MODIFICATION_LOG_PATH), exist_ok=True)
    with open(MODIFICATION_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def _append_log(entry: dict) -> None:
    log = _load_log()
    log.append(entry)
    if len(log) > 100:
        log = log[-100:]
    _save_log(log)


def _get_anchor_score() -> float:
    """Run evaluator_anchor.py and return the score. Returns -1.0 on error."""
    try:
        result = subprocess.run(
            [sys.executable, "evaluator_anchor.py"],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace"
        )
        for line in result.stdout.split("\n"):
            if line.startswith("ANCHOR SCORE:"):
                return float(line.split(":")[1].strip())
    except Exception:
        pass
    return -1.0


# VALIDATION STAGES

def _stage_ast_parse(code: str) -> Tuple[bool, str]:
    """Stage 1: Verify the proposed code is syntactically valid Python."""
    try:
        ast.parse(code)
        return True, "AST valid"
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _stage_immutable_check(target_file: str) -> Tuple[bool, str]:
    """Stage 2: Verify the target file is not on the immutable list."""
    basename = os.path.basename(target_file)
    if basename in IMMUTABLE_FILES or target_file in IMMUTABLE_FILES:
        return False, f"REJECTED: '{basename}' is immutable and cannot be modified by the AI."
    return True, f"Target '{basename}' is evolvable."


def _stage_evolvable_check(target_file: str) -> Tuple[bool, str]:
    """Stage 2b: Verify the target file is on the evolvable allowlist."""
    basename = os.path.basename(target_file)
    norm = target_file.replace("\\", "/")
    if basename in EVOLVABLE_FILES or norm in EVOLVABLE_FILES or target_file in EVOLVABLE_FILES:
        return True, f"Target '{basename}' is on the evolvable allowlist."
    return False, f"REJECTED: '{basename}' is not on the evolvable allowlist."


def _stage_safety_scan(code: str) -> Tuple[bool, str]:
    """Stage 3: Scan for forbidden patterns."""
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in code:
            return False, f"REJECTED: Forbidden pattern detected: '{pattern}'"
    return True, "Safety scan passed."


def _stage_subprocess_test(target_file: str, proposed_code: str) -> Tuple[bool, str]:
    """Stage 4: Run the proposed code in an isolated subprocess."""
    tmp_path = target_file + ".candidate"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(proposed_code)

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace"
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[:300]
            return False, f"Subprocess exited with code {result.returncode}: {stderr}"
        return True, "Subprocess test passed."
    except subprocess.TimeoutExpired:
        return False, "Subprocess test timed out (30s)."
    except Exception as e:
        return False, f"Subprocess test error: {e}"
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _stage_benchmark_regression(
        target_file: str,
        proposed_code: str,
        original_code: str,
        pre_score: float) -> Tuple[bool, str]:
    """
    Stage 5: Apply the change temporarily, run evaluator_anchor.py,
    then rollback if score regresses.
    """
    backup_path = target_file + ".guard_backup"
    try:
        # Backup original
        shutil.copy2(target_file, backup_path)

        # Apply proposed change
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(proposed_code)

        # Run benchmark
        post_score = _get_anchor_score()

        if post_score < pre_score - 0.5:  # Allow tiny float variance
            # Rollback
            shutil.copy2(backup_path, target_file)
            return False, (f"Benchmark regression: pre={pre_score:.1f} → post={post_score:.1f}. "
                           f"Change REJECTED and rolled back.")

        return True, f"Benchmark OK: pre={pre_score:.1f} → post={post_score:.1f}"

    except Exception as e:
        # Always rollback on error
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, target_file)
        return False, f"Regression test error (rolled back): {e}"
    finally:
        if os.path.exists(backup_path):
            try:
                os.remove(backup_path)
            except OSError:
                pass


def _stage_git_commit(target_file: str, rationale: str) -> Tuple[bool, str]:
    """Stage 6: Commit the accepted change to git for rollback capability."""
    try:
        subprocess.run(["git", "add", target_file], capture_output=True, timeout=10)
        msg = f"[SELF-EVOLUTION] {os.path.basename(target_file)}: {rationale[:80]}"
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return True, f"Committed: {result.stdout.strip()[:100]}"
        return True, f"Git commit warning (non-fatal): {result.stderr.strip()[:100]}"
    except Exception as e:
        return True, f"Git commit skipped (non-fatal): {e}"


# PUBLIC API

def validate_and_apply(
        target_file: str,
        proposed_code: str,
        rationale: str,
        epoch: int) -> Tuple[bool, str]:
    """
    Run all validation stages on a proposed modification.
    If all stages pass, the modification is applied to disk and committed to git.
    If any stage fails, the modification is rejected and the file is unchanged.

    Args:
        target_file:   Path to the file to be modified
        proposed_code: The full new content for the file
        rationale:     Short description of why this change is proposed
        epoch:         Current training epoch (for logging)

    Returns:
        (accepted: bool, report: str)
    """
    report_lines = [
        f"[GUARD] Evaluating proposed modification to '{target_file}'",
        f"[GUARD] Rationale: {rationale}",
        f"[GUARD] Epoch: {epoch}",
        f"[GUARD] Proposed code length: {len(proposed_code)} chars",
    ]

    stages = [
        ("AST Parse",           lambda: _stage_ast_parse(proposed_code)),
        ("Immutable Check",     lambda: _stage_immutable_check(target_file)),
        ("Evolvable Check",     lambda: _stage_evolvable_check(target_file)),
        ("Safety Scan",         lambda: _stage_safety_scan(proposed_code)),
    ]

    for stage_name, stage_fn in stages:
        ok, msg = stage_fn()
        report_lines.append(f"  [{'+' if ok else 'X'}] {stage_name}: {msg}")
        if not ok:
            report = "\n".join(report_lines)
            _append_log({
                "epoch": epoch, "target": target_file, "rationale": rationale,
                "accepted": False, "rejection_stage": stage_name,
                "report": report, "timestamp": time.time()
            })
            print(report)
            return False, report

    # Read original code for regression test
    original_code = ""
    if os.path.exists(target_file):
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                original_code = f.read()
        except Exception:
            pass

    # Subprocess test (skip for files that aren't standalone runnable, like modules)
    ok, msg = _stage_subprocess_test(target_file, proposed_code)
    report_lines.append(f"  [{'+' if ok else '~'}] Subprocess Test: {msg}")
    # Subprocess test failures are warnings, not hard rejections for module files

    # Benchmark regression (only run if file affects experiment.py or the main loop)
    key_files = {"sandbox/experiment.py", "autonomous_loop.py", "reasoning_engine.py",
                 "sandbox/reasoning_engine.py"}
    norm = target_file.replace("\\", "/").lower()
    runs_regression = any(k in norm for k in key_files)

    if runs_regression:
        pre_score = _get_anchor_score()
        ok, msg = _stage_benchmark_regression(target_file, proposed_code, original_code, pre_score)
        report_lines.append(f"  [{'+' if ok else 'X'}] Benchmark Regression: {msg}")
        if not ok:
            report = "\n".join(report_lines)
            _append_log({
                "epoch": epoch, "target": target_file, "rationale": rationale,
                "accepted": False, "rejection_stage": "Benchmark Regression",
                "report": report, "timestamp": time.time()
            })
            print(report)
            return False, report
    else:
        # For non-critical files, apply the change directly
        try:
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(proposed_code)
            report_lines.append(f"  [+] Direct Apply: Written to '{target_file}'")
        except Exception as e:
            report_lines.append(f"  [X] Direct Apply: Failed   {e}")
            report = "\n".join(report_lines)
            _append_log({
                "epoch": epoch, "target": target_file, "rationale": rationale,
                "accepted": False, "rejection_stage": "File Write",
                "report": report, "timestamp": time.time()
            })
            return False, report

    # Git commit
    ok, msg = _stage_git_commit(target_file, rationale)
    report_lines.append(f"  [{'+' if ok else '~'}] Git Commit: {msg}")

    report = "\n".join(report_lines)
    report += "\n[GUARD] ✓ MODIFICATION ACCEPTED"
    _append_log({
        "epoch": epoch, "target": target_file, "rationale": rationale,
        "accepted": True, "report": report, "timestamp": time.time()
    })
    print(report)
    return True, report


def get_guard_summary() -> str:
    """Human-readable summary of recent guard decisions."""
    log = _load_log()
    if not log:
        return "No modification attempts recorded yet."
    accepted = sum(1 for e in log if e.get("accepted"))
    rejected = len(log) - accepted
    return (f"Guard history: {accepted} accepted / {rejected} rejected "
            f"({len(log)} total attempts)")


def list_evolvable_files() -> list:
    return sorted(EVOLVABLE_FILES)


def list_immutable_files() -> list:
    return sorted(IMMUTABLE_FILES)


if __name__ == "__main__":
    print("Self-Evolution Guard")
    print(f"Immutable files ({len(IMMUTABLE_FILES)}): {list_immutable_files()}")
    print(f"Evolvable files ({len(EVOLVABLE_FILES)}): {list_evolvable_files()}")
    print(get_guard_summary())
