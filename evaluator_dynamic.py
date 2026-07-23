"""
EVALUATOR DYNAMIC   Multi-Test-Case Evaluation Engine

Phase 1 upgrade: replaces the old single-test-case pass/fail with a rigorous
multi-case evaluator. A problem is only counted as solved if the AI's generated
code passes ALL provided test cases, not just the first one.

Two evaluation modes:
  1. Multi-case mode (preferred): reads sandbox/generated_problems.json which
     contains 3-4 test cases per problem, produced by problem_generator.py.
  2. Legacy mode (fallback): reads sandbox/dynamic_dataset.json in the old
     [prompt, inputs, expected] format for backward compatibility.

This prevents the AI from gaming the benchmark with lucky edge-case solutions
that only pass one specific input but fail on generalization.
"""

import json
import os
import sys
import time
import psutil
import ast

# Sandbox globals   same restricted namespace as evaluator_anchor.py
SANDBOX_GLOBALS = {
    '__builtins__': {
        'print': print, 'len': len, 'range': range, 'list': list,
        'dict': dict, 'set': set, 'tuple': tuple, 'int': int,
        'float': float, 'str': str, 'bool': bool, 'True': True,
        'False': False, 'None': None, 'sorted': sorted,
        'sum': sum, 'max': max, 'min': min, 'abs': abs,
        'any': any, 'all': all, 'enumerate': enumerate,
        'zip': zip, 'map': map, 'filter': filter,
        'isinstance': isinstance, 'round': round,
        'reversed': reversed, 'chr': chr, 'ord': ord,
    }
}

try:
    import math as _math
    SANDBOX_GLOBALS['math'] = _math
except ImportError:
    pass

MULTI_CASE_PATH  = "sandbox/generated_problems.json"
LEGACY_PATH      = "sandbox/dynamic_dataset.json"
TIMEOUT_SECS     = 60.0
MEM_LIMIT_MB     = 8000.0


# MODULE LOADER

def _load_experiment():
    """Reload the experiment module fresh each evaluation."""
    sandbox_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox")
    if sandbox_path not in sys.path:
        sys.path.insert(0, sandbox_path)

    # Force full module reload so we always test the latest AI-written version
    for mod in list(sys.modules.keys()):
        if mod in ('experiment', 'baselines') or mod.startswith('sandbox'):
            try:
                del sys.modules[mod]
            except Exception:
                pass

    import experiment
    return experiment


def _run_code(generated_code: str, inputs):
    """
    Execute AI-generated code in the sandbox and return the result.
    Returns (result, error_str). error_str is None on success.
    """
    local_env = {}
    try:
        exec(generated_code, SANDBOX_GLOBALS, local_env)
    except Exception as e:
        return None, f"exec failed: {e}"

    # Find the first callable that isn't a dunder
    func = next((v for k, v in local_env.items()
                 if callable(v) and not k.startswith("__")), None)
    if func is None:
        return None, "no callable found in generated code"

    try:
        if isinstance(inputs, (list, tuple)):
            result = func(*inputs)
        else:
            result = func(inputs)
        return result, None
    except Exception as e:
        return None, f"runtime error: {e}"


# MULTI-CASE EVALUATION (Phase 1   preferred path)

def evaluate_multi_case(experiment_mod) -> tuple:
    """
    Evaluate against sandbox/generated_problems.json.
    Each problem requires ALL test cases to pass for credit.

    Returns (score, exec_time, mem_mb, problem_results)
    where problem_results is a list of dicts for diagnostics.
    """
    if not os.path.exists(MULTI_CASE_PATH):
        return 0.0, 0.0, 0.0, []

    try:
        with open(MULTI_CASE_PATH, "r", encoding="utf-8") as f:
            problems = json.load(f)
    except Exception as e:
        print(f"[DYNAMIC] Failed to load generated_problems.json: {e}")
        return 0.0, 0.0, 0.0, []

    if not problems:
        return 0.0, 0.0, 0.0, []

    process   = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    start_time = time.time()

    score = 0.0
    problem_results = []
    code_cache = {}

    for problem in problems:
        description = problem.get("description", "")
        test_cases  = problem.get("test_cases", [])
        category    = problem.get("category", "general")
        difficulty  = problem.get("difficulty", 1)

        if not test_cases:
            continue

        # Generate code once per problem description
        if description not in code_cache:
            try:
                generated_code = experiment_mod.generate_code(description)
                code_cache[description] = generated_code
            except Exception as e:
                code_cache[description] = None

        generated_code = code_cache.get(description)
        if not generated_code or "No matching code found" in str(generated_code):
            problem_results.append({
                "description": description,
                "category": category,
                "difficulty": difficulty,
                "passed": False,
                "passed_cases": 0,
                "total_cases": len(test_cases),
                "reason": "no code generated",
            })
            continue

        # Run every test case
        cases_passed = 0
        last_error = None
        for tc in test_cases:
            inputs   = tc.get("inputs", [])
            expected = tc.get("expected")
            result, err = _run_code(generated_code, inputs)
            if err:
                last_error = err
                break  # Stop on first execution error
            if result == expected:
                cases_passed += 1

        problem_fraction = cases_passed / max(len(test_cases), 1)
        score += problem_fraction
        all_passed = (cases_passed == len(test_cases))

        problem_results.append({
            "description": description,
            "category": category,
            "difficulty": difficulty,
            "passed": all_passed,
            "passed_cases": cases_passed,
            "total_cases": len(test_cases),
            "reason": last_error if last_error else ("all passed" if all_passed else "wrong answer"),
        })

    exec_time = time.time() - start_time
    mem_used_mb = max(0, process.memory_info().rss - mem_before) / (1024 * 1024)

    # Guillotine checks
    if exec_time > TIMEOUT_SECS:
        print(f"CRASH_LOG: Dynamic eval timed out ({exec_time:.1f}s)")
        return 0.0, exec_time, mem_used_mb, problem_results
    if mem_used_mb > MEM_LIMIT_MB:
        print(f"CRASH_LOG: Dynamic eval exceeded memory ({mem_used_mb:.1f}MB)")
        return 0.0, exec_time, mem_used_mb, problem_results

    return score, exec_time, mem_used_mb, problem_results


# LEGACY EVALUATION (backward compat   single test case)

def evaluate_legacy(experiment_mod) -> tuple:
    """
    Fallback: read sandbox/dynamic_dataset.json in [prompt, inputs, expected] format.
    Delegates directly to evaluator_anchor.run_experiment_guillotine for compatibility.
    """
    import evaluator_anchor

    if not os.path.exists(LEGACY_PATH):
        return 0.0, 0.0, 0.0

    try:
        with open(LEGACY_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        dataset = [tuple(item) for item in raw]
    except Exception as e:
        print(f"[DYNAMIC] Failed to load legacy dynamic_dataset.json: {e}")
        return 0.0, 0.0, 0.0

    score, exec_time, mem_mb = evaluator_anchor.run_experiment_guillotine(dataset)
    return score, exec_time, mem_mb


# MAIN ENTRYPOINT

def evaluate_dynamic():
    """
    Run dynamic evaluation. Uses multi-case mode if generated_problems.json
    exists (Phase 1), otherwise falls back to legacy single-case mode.
    """
    try:
        experiment_mod = _load_experiment()
    except Exception as e:
        print(f"CRASH_LOG: Could not load experiment module: {e}")
        return 0.0, 0.0, 0.0

    if not hasattr(experiment_mod, "generate_code"):
        print("CRASH_LOG: experiment module has no generate_code function")
        return 0.0, 0.0, 0.0

    # Prefer multi-case evaluation when generated_problems.json is present
    if os.path.exists(MULTI_CASE_PATH):
        score, exec_time, mem_mb, results = evaluate_multi_case(experiment_mod)

        # Print per-category diagnostics for the autonomous loop to learn from
        categories = {}
        for r in results:
            cat = r["category"]
            categories.setdefault(cat, {"passed": 0, "total": 0})
            categories[cat]["total"] += 1
            if r["passed"]:
                categories[cat]["passed"] += 1

        if categories:
            cat_summary = " | ".join(
                f"{cat}: {v['passed']}/{v['total']}"
                for cat, v in sorted(categories.items())
            )
            print(f"[DYNAMIC] Category breakdown: {cat_summary}")

        # Print failed problems so the engineer knows what to fix
        failed = [r for r in results if not r["passed"]]
        if failed:
            print(f"[DYNAMIC] Failed problems ({len(failed)}/{len(results)}):")
            for r in failed[:5]:  # Show first 5
                print(f"  [{r['difficulty']}*] {r['description'][:60]} "
                      f"({r['passed_cases']}/{r['total_cases']} cases)   {r['reason'][:40]}")

        return score, exec_time, mem_mb

    # Fall back to legacy mode
    print("[DYNAMIC] generated_problems.json not found   using legacy dynamic_dataset.json")
    score, exec_time, mem_mb = evaluate_legacy(experiment_mod)
    return score, exec_time, mem_mb


if __name__ == "__main__":
    score, exec_time, mem_mb = evaluate_dynamic()
    print(f"DYNAMIC SCORE: {score}")
    print(f"FITNESS: {score}")
    print(f"TIME: {exec_time:.4f}s")
    print(f"MEM: {mem_mb:.2f}MB")
