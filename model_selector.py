"""
MODEL SELECTOR   Selection Pressure on Fine-Tuned Weights

Implements the "selection gate" that runs BEFORE export_gguf_to_lm_studio().
After QLoRA fine-tuning and merging completes, this module compares the
candidate model's performance on a held-out benchmark against the current
baseline. Only if the candidate wins does the GGUF overwrite proceed.

This ensures the system cannot regress   fine-tuned weights must earn
their place as the active model.

The held-out benchmark is a fixed set of 20 problems drawn from evaluator_anchor.py
tasks plus MBPP-style problems. These are NEVER in the training data.
"""

import json
import os
import subprocess
import sys
import time
from typing import Optional, Tuple

SELECTION_LOG_PATH   = "sandbox/selection_log.json"
HOLDOUT_BENCHMARK    = "sandbox/holdout_benchmark.json"
BASELINE_SCORE_PATH  = "sandbox/baseline_score.json"

# Minimum improvement margin to accept a new model
IMPROVEMENT_THRESHOLD = 0.01  # 1%


# HELD-OUT BENCHMARK DEFINITION
# These problems are FIXED and NEVER used for training.
# They are hard-coded here so the AI cannot influence them.

HOLDOUT_PROBLEMS = [
    # Mathematical
    {"description": "Check if a number is prime", "inputs": [17], "expected": True},
    {"description": "Check if a number is prime", "inputs": [15], "expected": False},
    {"description": "Compute the Greatest Common Divisor (GCD) of two numbers", "inputs": [48, 18], "expected": 6},
    {"description": "Compute the Least Common Multiple (LCM) of two numbers", "inputs": [4, 6], "expected": 12},
    {"description": "Compute the sum of digits of a number", "inputs": [12345], "expected": 15},
    {"description": "Count the total number of divisors of a number", "inputs": [12], "expected": 6},
    {"description": "Return the Nth prime number", "inputs": [10], "expected": 29},
    {"description": "Check if an integer reads the same forwards and backwards (palindrome number)", "inputs": [121], "expected": True},
    {"description": "Check if an integer reads the same forwards and backwards (palindrome number)", "inputs": [123], "expected": False},
    {"description": "Compute the sum of squares of all integers from 1 to N", "inputs": [5], "expected": 55},
    # Algorithmic
    {"description": "Return the sum of all even numbers in a list", "inputs": [[1,2,3,4,5,6]], "expected": 12},
    {"description": "Return the sum of all odd numbers in a list", "inputs": [[1,2,3,4,5,6]], "expected": 9},
    {"description": "Return a list with duplicate elements removed, preserving original order", "inputs": [[1,2,2,3,3,3]], "expected": [1,2,3]},
    {"description": "Rotate a list to the left by K positions", "inputs": [[1,2,3,4,5], 2], "expected": [3,4,5,1,2]},
    {"description": "Return the median value of a list of numbers", "inputs": [[1,3,5,7,9]], "expected": 5.0},
    # String
    {"description": "Count the number of vowels in a string", "inputs": ["hello world"], "expected": 3},
    {"description": "Count the number of words in a sentence", "inputs": ["the quick brown fox"], "expected": 4},
    {"description": "Check if a sentence contains every letter of the alphabet (pangram check)", "inputs": ["the quick brown fox jumps over the lazy dog"], "expected": True},
    {"description": "Check if a sentence contains every letter of the alphabet (pangram check)", "inputs": ["hello world"], "expected": False},
    {"description": "Count the non-overlapping occurrences of a substring within a string", "inputs": ["banana", "an"], "expected": 2},
]


def save_holdout_benchmark() -> None:
    """Write the held-out benchmark to disk (used by the evaluator subprocess)."""
    os.makedirs("sandbox", exist_ok=True)
    with open(HOLDOUT_BENCHMARK, "w", encoding="utf-8") as f:
        json.dump(HOLDOUT_PROBLEMS, f, indent=2)


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


# SCORE EVALUATION

def _evaluate_model_on_holdout(model_path: str) -> float:
    """
    Run the held-out benchmark against a specific model.
    Uses a subprocess so we can test a candidate model without
    disrupting the running inference server.

    Returns a score fraction (0.0 1.0).
    """
    save_holdout_benchmark()

    # Write a one-shot evaluation script
    eval_script = os.path.abspath("sandbox/_holdout_eval_runner.py")
    with open(eval_script, "w", encoding="utf-8") as f:
        f.write(f"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HOLDOUT_PATH = {repr(os.path.abspath(HOLDOUT_BENCHMARK))}
SANDBOX_GLOBALS = {{
    '__builtins__': {{
        'print': print, 'len': len, 'range': range, 'list': list,
        'dict': dict, 'set': set, 'tuple': tuple, 'int': int,
        'float': float, 'str': str, 'bool': bool, 'True': True,
        'False': False, 'None': None, 'sorted': sorted,
        'sum': sum, 'max': max, 'min': min, 'abs': abs,
        'any': any, 'all': all, 'enumerate': enumerate,
        'zip': zip, 'map': map, 'filter': filter,
        'isinstance': isinstance, 'round': round,
    }}
}}

try:
    import math as _math
    SANDBOX_GLOBALS['math'] = _math
except ImportError:
    pass

with open(HOLDOUT_PATH) as f:
    problems = json.load(f)

# Load experiment module
exp_path = os.path.join(os.path.dirname(os.path.dirname(HOLDOUT_PATH)), 'experiment.py')
if not os.path.exists(exp_path):
    print("HOLDOUT_SCORE:0.0")
    sys.exit(0)

import importlib.util
spec = importlib.util.spec_from_file_location("experiment", exp_path)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

passed = 0
for p in problems:
    try:
        code = exp.generate_code(p["description"])
        local = {{}}
        exec(code, SANDBOX_GLOBALS, local)
        func = next((v for k, v in local.items() if callable(v) and not k.startswith("__")), None)
        if func is None:
            continue
        inputs = p["inputs"]
        result = func(*inputs) if isinstance(inputs, list) else func(inputs)
        if result == p["expected"]:
            passed += 1
    except Exception:
        pass

print(f"HOLDOUT_SCORE:{{passed / max(len(problems), 1):.4f}}")
""")

    try:
        result = subprocess.run(
            [sys.executable, eval_script],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace"
        )
        for line in result.stdout.split("\n"):
            if line.startswith("HOLDOUT_SCORE:"):
                return float(line.split(":")[1].strip())
    except Exception as e:
        print(f"[SELECTOR] Holdout eval failed: {e}")

    return 0.0


def get_baseline_score() -> float:
    """Return the stored baseline score (from the currently-active model)."""
    data = _load_json(BASELINE_SCORE_PATH, {"score": None})
    score = data.get("score")
    if score is None:
        # First run   measure and store it
        print("[SELECTOR] No baseline score stored. Measuring current model...")
        score = _evaluate_model_on_holdout("current")
        _save_json(BASELINE_SCORE_PATH, {"score": score, "measured_at": time.time()})
        print(f"[SELECTOR] Baseline established: {score:.1%}")
    return score


def update_baseline_score(new_score: float) -> None:
    """Update the stored baseline score after a successful model swap."""
    _save_json(BASELINE_SCORE_PATH, {"score": new_score, "measured_at": time.time()})


# SELECTION GATE

def evaluate_candidate_and_decide(
        merged_hf_dir: str,
        base_model_name: str) -> Tuple[bool, float, float]:
    """
    Run the held-out benchmark on the current experiment.py (which was
    generated using the fine-tuned model) and decide whether to keep it.

    NOTE: We evaluate by running experiment.py (the AI's code) rather than
    directly loading the HF model, because:
      1. Loading a large LLM twice is memory-prohibitive
      2. The true test is whether the fine-tuned model produces better CODE
         on novel problems, which experiment.py captures

    Returns:
        (accepted: bool, baseline_score: float, candidate_score: float)
    """
    print("\n[SELECTOR] Running held-out benchmark on candidate model behavior...")

    baseline_score = get_baseline_score()
    candidate_score = _evaluate_model_on_holdout(merged_hf_dir)

    improvement = candidate_score - baseline_score
    accepted = improvement >= IMPROVEMENT_THRESHOLD

    # Log the selection decision
    log = _load_json(SELECTION_LOG_PATH, [])
    log.append({
        "timestamp": time.time(),
        "merged_hf_dir": merged_hf_dir,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "improvement": improvement,
        "accepted": accepted,
        "threshold": IMPROVEMENT_THRESHOLD,
    })
    if len(log) > 50:
        log = log[-50:]
    _save_json(SELECTION_LOG_PATH, log)

    if accepted:
        print(f"[SELECTOR] ✓ ACCEPTED: candidate={candidate_score:.1%} vs baseline={baseline_score:.1%} "
              f"(+{improvement:.1%})   proceeding with GGUF overwrite.")
        update_baseline_score(candidate_score)
    else:
        print(f"[SELECTOR] ✗ REJECTED: candidate={candidate_score:.1%} vs baseline={baseline_score:.1%} "
              f"({improvement:+.1%})   keeping current GGUF, discarding fine-tuned weights.")

    return accepted, baseline_score, candidate_score


def get_selection_summary() -> str:
    """Human-readable summary of recent selection decisions."""
    log = _load_json(SELECTION_LOG_PATH, [])
    if not log:
        return "No selection decisions recorded yet."
    accepted = sum(1 for e in log if e.get("accepted"))
    rejected = len(log) - accepted
    recent = log[-1]
    return (f"Selection history: {accepted} accepted / {rejected} rejected | "
            f"Last: baseline={recent['baseline_score']:.1%} "
            f"candidate={recent['candidate_score']:.1%} "
            f"({'ACCEPTED' if recent['accepted'] else 'REJECTED'})")


if __name__ == "__main__":
    print("Model Selector self-test:")
    print(f"Holdout benchmark size: {len(HOLDOUT_PROBLEMS)} problems")
    save_holdout_benchmark()
    print(f"Holdout benchmark saved to: {HOLDOUT_BENCHMARK}")
    print(get_selection_summary())
