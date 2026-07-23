"""
Trajectory collector for recording code generation attempts and building DPO pairs.

Records successful and failed code generation attempts, then pairs them
into DPO preference pairs (chosen=passed, rejected=failed) for fine-tuning.
"""

import json
import os
import time
from typing import Optional, List, Dict

SFT_BUFFER_PATH  = "sandbox/training_trajectories.json"
DPO_BUFFER_PATH  = "sandbox/dpo_pairs.json"
ATTEMPT_LOG_PATH = "sandbox/attempt_log.json"

SFT_TRIGGER_THRESHOLD  = 30   # trigger SFT training after N quality trajectories
DPO_TRIGGER_THRESHOLD  = 25   # trigger DPO training after N quality pairs


def is_sufficiently_complex(problem_description: str, generated_code: str) -> bool:
    """
    Filter out trivial 1-liners and basic arithmetic from entering SFT/DPO training buffers.
    Ensures fine-tuning data consists of meaningful algorithmic trajectories.
    """
    if not generated_code or not isinstance(generated_code, str):
        return False
    
    code = generated_code.strip()
    # Require at least 60 characters
    if len(code) < 60:
        return False

    lines = [line.strip() for line in code.splitlines() if line.strip() and not line.strip().startswith('#')]
    # Require at least 3 non-comment lines
    if len(lines) < 3:
        return False

    # Check for meaningful programming constructs
    complexity_keywords = [
        "for ", "while ", "if ", "elif ", "def ", "class ", "import ",
        "lambda", "yield", "try:", "except", "sorted(", ".sort(", "zip(",
        "enumerate(", "map(", "filter(", "math.", "append(", "pop("
    ]

    found_constructs = sum(1 for kw in complexity_keywords if kw in code)
    if found_constructs < 2:
        return False

    return True


# Helper functions

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


# Attempt logging

def record_attempt(problem_description: str,
                   generated_code: str,
                   passed_all_tests: bool,
                   n_tests_passed: int,
                   n_tests_total: int,
                   epoch: int) -> None:
    """
    Record a single problem-solving attempt.
    Called after every engineering phase regardless of outcome.
    """
    log = _load_json(ATTEMPT_LOG_PATH, [])
    log.append({
        "problem": problem_description,
        "code": generated_code,
        "passed_all": passed_all_tests,
        "n_passed": n_tests_passed,
        "n_total": n_tests_total,
        "epoch": epoch,
        "timestamp": time.time(),
    })
    # Keep last 200 attempts
    if len(log) > 200:
        log = log[-200:]
    _save_json(ATTEMPT_LOG_PATH, log)


# SFT buffer management

def add_sft_trajectory(problem_description: str,
                        generated_code: str) -> int:
    """
    Add a successful trajectory to the SFT buffer if it meets complexity requirements.
    Returns the current buffer size.
    """
    if not is_sufficiently_complex(problem_description, generated_code):
        return get_sft_buffer_size()

    buffer = _load_json(SFT_BUFFER_PATH, [])

    buffer.append({
        "prompt": problem_description,
        "solution": generated_code,
        "timestamp": time.time(),
    })
    _save_json(SFT_BUFFER_PATH, buffer)
    return len(buffer)


def get_sft_buffer_size() -> int:
    return len(_load_json(SFT_BUFFER_PATH, []))


def clear_sft_buffer() -> None:
    _save_json(SFT_BUFFER_PATH, [])


# DPO buffer management

def _build_dpo_pairs_from_log() -> List[Dict]:
    """
    Scan the attempt log for problems that have both failed and
    successful attempts. Pair the best success against the worst failure.
    """
    log = _load_json(ATTEMPT_LOG_PATH, [])

    # Group by problem description
    by_problem: Dict[str, List[Dict]] = {}
    for entry in log:
        key = entry["problem"]
        by_problem.setdefault(key, []).append(entry)

    pairs = []
    for problem, attempts in by_problem.items():
        successes = [a for a in attempts if a["passed_all"] and a["code"]]
        failures  = [a for a in attempts if not a["passed_all"] and a["code"]]

        if not successes or not failures:
            continue

        # chosen: most recent success
        chosen = sorted(successes, key=lambda x: x["timestamp"])[-1]
        # rejected: attempt with fewest tests passed
        rejected = sorted(failures, key=lambda x: x["n_passed"])[0]

        # Require chosen solution to be sufficiently complex
        if not is_sufficiently_complex(problem, chosen["code"]):
            continue

        # Skip if rejected attempt passed 80% or more of tests
        if rejected["n_total"] > 0:
            reject_rate = rejected["n_passed"] / rejected["n_total"]
            if reject_rate >= 0.8:
                continue

        pairs.append({
            "prompt": problem,
            "chosen": chosen["code"],
            "rejected": rejected["code"],
            "chosen_pass_rate": 1.0,
            "rejected_pass_rate": (rejected["n_passed"] / max(rejected["n_total"], 1)),
            "timestamp": time.time(),
        })

    return pairs


def refresh_dpo_buffer() -> int:
    """
    Rebuild the DPO pairs buffer from the current attempt log.
    Returns the number of pairs now available.
    """
    pairs = _build_dpo_pairs_from_log()

    # Merge with existing pairs, deduplicate by prompt
    existing = _load_json(DPO_BUFFER_PATH, [])
    existing_prompts = {p["prompt"] for p in existing}

    new_pairs = [p for p in pairs if p["prompt"] not in existing_prompts]
    combined = existing + new_pairs

    # Keep 100 most recent pairs
    if len(combined) > 100:
        combined = combined[-100:]

    _save_json(DPO_BUFFER_PATH, combined)
    return len(combined)


def get_dpo_buffer_size() -> int:
    return len(_load_json(DPO_BUFFER_PATH, []))


def clear_dpo_buffer() -> None:
    _save_json(DPO_BUFFER_PATH, [])


# Training trigger logic

def should_trigger_training(has_improved: bool = False, is_singularity: bool = False) -> Optional[str]:
    """
    Check whether training should be triggered.
    Triggers automatically when all challenges and tiers are mastered (is_singularity=True)
    or when enough trajectory pairs have been collected from benchmark improvements.
    """
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not cfg.get("training", {}).get("auto_train", True):
            return None
    except Exception:
        pass

    dpo_size = get_dpo_buffer_size()
    sft_size = get_sft_buffer_size()

    # Trigger fine-tuning when Singularity/Mastery is reached and trajectories exist
    if is_singularity and (sft_size > 0 or dpo_size > 0):
        return "dpo" if dpo_size >= sft_size else "sft"

    if not has_improved:
        return None

    if dpo_size >= DPO_TRIGGER_THRESHOLD:
        return "dpo"
    if sft_size >= SFT_TRIGGER_THRESHOLD:
        return "sft"
    return None


def get_collector_summary() -> str:
    """Return status summary of the trajectory collector."""
    sft = get_sft_buffer_size()
    dpo = get_dpo_buffer_size()
    log = _load_json(ATTEMPT_LOG_PATH, [])
    total = len(log)
    successes = sum(1 for a in log if a.get("passed_all"))
    return (f"Attempts logged: {total} ({successes} successes) | "
            f"SFT buffer: {sft}/{SFT_TRIGGER_THRESHOLD} | "
            f"DPO pairs: {dpo}/{DPO_TRIGGER_THRESHOLD}")


if __name__ == "__main__":
    print("Trajectory Collector self-test:")
    print(get_collector_summary())
    dpo = _build_dpo_pairs_from_log()
    print(f"DPO pairs from current log: {len(dpo)}")

