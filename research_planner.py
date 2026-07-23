"""
RESEARCH PLANNER   The AI's Autonomous Research Agenda

Reads the capability frontier to identify the AI's current weaknesses, then
designs a targeted curriculum to push the capability boundary forward.

This is the highest-level component: the AI is no longer just solving problems
that a human designed   it is deciding what to study next based on its own
performance data. This is a key step toward autonomous self-directed learning.

Every 20 epochs, the planner:
  1. Reads the frontier map   identifies partial/untested domains
  2. Generates problems specifically targeting those gaps
  3. Injects them into the epoch curriculum
  4. Evaluates whether the capability expanded after the research cycle
"""

import json
import os
import time
import random
from typing import List, Dict, Optional

from capability_frontier import (
    load_frontier, get_frontier_gaps, get_mastered_domains,
    get_frontier_summary, get_overall_progress_pct
)
from problem_generator import generate_epoch_problems, get_problem_stats

RESEARCH_LOG_PATH = "sandbox/research_plan_log.json"
RESEARCH_PLAN_PATH = "sandbox/current_research_plan.json"


# DOMAIN → CATEGORY MAPPING (for problem generator targeting)

DOMAIN_TO_CATEGORY = {
    "mathematical": "mathematical",
    "string":       "string",
    "algorithmic":  "algorithmic",
    "logical":      "algorithmic",
    "meta":         "algorithmic",
}

SUBDOMAIN_FOCUS_KEYWORDS = {
    "prime_numbers":          "prime",
    "divisibility":           "divisor",
    "gcd_lcm":                "gcd lcm",
    "digit_operations":       "digit sum",
    "sequences_fibonacci":    "fibonacci collatz sequence",
    "combinatorics":          "sum of squares combinations",
    "number_theory":          "perfect number",
    "geometry":               "triangle area",
    "statistics_median_mean": "median average",
    "basic_string_ops":       "vowel count words",
    "palindromes":            "palindrome",
    "anagram_detection":      "anagram",
    "encoding_cipher":        "caesar binary",
    "run_length_encoding":    "run length encoding",
    "substring_search":       "substring occurrence",
    "character_frequency":    "most common character",
    "string_validation":      "pangram hamming",
    "sorting_searching":      "sort search",
    "list_transformations":   "product clamp filter",
    "deduplication":          "unique duplicate",
    "rotation_shifting":      "rotate shift",
    "dynamic_programming":    "optimal subproblem",
    "recursion":              "recursive base case",
    "boolean_logic":          "xor and or logic",
    "set_operations":         "union intersection",
    "classification":         "classify categorize",
    "pattern_recognition":    "detect pattern",
    "code_generalization":    "general reusable abstract",
    "novel_problem_adaptation": "novel unseen adapt",
    "multi_step_reasoning":   "pipeline compose steps",
}


# INTERNAL HELPERS

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


# RESEARCH PLAN GENERATION

def generate_research_plan(epoch: int, n_focus_areas: int = 3) -> Dict:
    """
    Analyze the capability frontier and generate a targeted research plan.

    Returns a dict describing:
     , Which capability gaps are being targeted
     , What types of problems to generate
     , Success criteria for the research cycle
    """
    gaps = get_frontier_gaps()
    mastered = get_mastered_domains()
    progress = get_overall_progress_pct()

    if not gaps:
        # Everything is mastered   generate harder variants
        return {
            "type": "frontier_push",
            "epoch": epoch,
            "message": "All tracked capabilities mastered. Pushing to harder problem variants.",
            "focus_areas": [],
            "target_categories": ["mathematical", "algorithmic", "string"],
            "difficulty_bias": 4,
            "created_at": time.time(),
        }

    # Focus on top-priority gaps (partial first, then attempted, then untested)
    focus_gaps = gaps[:n_focus_areas]

    focus_areas = []
    target_categories = set()
    for gap in focus_gaps:
        domain = gap["domain"]
        subdomain = gap["subdomain"]
        keywords = SUBDOMAIN_FOCUS_KEYWORDS.get(subdomain, subdomain.replace("_", " "))
        category = DOMAIN_TO_CATEGORY.get(domain, "algorithmic")
        target_categories.add(category)
        focus_areas.append({
            "domain": domain,
            "subdomain": subdomain,
            "state": gap["state"],
            "keywords": keywords,
            "category": category,
            "current_pass_rate": gap.get("pass_rate"),
            "target_pass_rate": 0.8,
        })

    plan = {
        "type": "targeted_research",
        "epoch": epoch,
        "overall_progress": progress,
        "mastered_count": len(mastered),
        "gap_count": len(gaps),
        "focus_areas": focus_areas,
        "target_categories": list(target_categories),
        "difficulty_bias": 2 if progress < 0.3 else 3 if progress < 0.6 else 4,
        "created_at": time.time(),
        "research_cycle_length": 20,  # epochs until next plan revision
        "success_criteria": f"Move {n_focus_areas} gap(s) from current state to next state",
    }

    _save_json(RESEARCH_PLAN_PATH, plan)
    return plan


def generate_targeted_problems(plan: Dict, epoch: int, n: int = 5) -> List[Dict]:
    """
    Generate problems specifically targeting the capability gaps in the research plan.
    Falls back to general generation if targeted generation doesn't produce enough problems.
    """
    from problem_generator import PROBLEM_TEMPLATES, _gen_is_prime, _gen_gcd, _gen_lcm
    import random as _rng

    target_categories = plan.get("target_categories", ["mathematical"])
    focus_areas = plan.get("focus_areas", [])

    # Use the standard generator but filter to targeted categories
    rng = random.Random(epoch * 7919 + 12345)

    # Try generating more than needed, then filter
    all_problems = generate_epoch_problems(epoch, n_problems=30)
    targeted = [p for p in all_problems if p["category"] in target_categories]
    general = [p for p in all_problems if p["category"] not in target_categories]

    # Fill with targeted first, then general
    result = targeted[:n]
    if len(result) < n:
        result += general[:n - len(result)]

    return result[:n]


def log_research_cycle_results(
        plan: Dict,
        epoch_start: int,
        epoch_end: int,
        problems_attempted: int,
        problems_solved: int,
        frontier_changes: List[Dict]) -> None:
    """Log the results of a completed research cycle."""
    log = _load_json(RESEARCH_LOG_PATH, [])
    log.append({
        "plan_type": plan.get("type"),
        "epoch_start": epoch_start,
        "epoch_end": epoch_end,
        "focus_areas": [a["subdomain"] for a in plan.get("focus_areas", [])],
        "problems_attempted": problems_attempted,
        "problems_solved": problems_solved,
        "solve_rate": problems_solved / max(problems_attempted, 1),
        "frontier_changes": frontier_changes,
        "timestamp": time.time(),
    })
    if len(log) > 50:
        log = log[-50:]
    _save_json(RESEARCH_LOG_PATH, log)


def get_current_plan() -> Optional[Dict]:
    """Load the current active research plan."""
    return _load_json(RESEARCH_PLAN_PATH, None)


def get_research_summary() -> str:
    """Human-readable summary of research planning activity."""
    plan = get_current_plan()
    log = _load_json(RESEARCH_LOG_PATH, [])
    frontier_str = get_frontier_summary()

    if plan is None:
        return f"No active research plan. {frontier_str}"

    focus_str = ", ".join(
        f"{a['subdomain']} [{a['state']}]"
        for a in plan.get("focus_areas", [])
    )
    progress = plan.get("overall_progress", 0.0)
    cycles = len(log)

    return (f"Research Plan (epoch {plan.get('epoch')}): "
            f"progress={progress:.1%} | focus=[{focus_str}] | "
            f"completed_cycles={cycles} | {frontier_str}")


def should_revise_plan(current_epoch: int) -> bool:
    """Returns True if it's time to generate a new research plan."""
    plan = get_current_plan()
    if plan is None:
        return True
    plan_epoch = plan.get("epoch", 0)
    cycle_length = plan.get("research_cycle_length", 20)
    return current_epoch >= plan_epoch + cycle_length


if __name__ == "__main__":
    print("Research Planner self-test:")
    plan = generate_research_plan(epoch=1)
    print(f"Plan type: {plan['type']}")
    print(f"Focus areas: {[a['subdomain'] for a in plan.get('focus_areas', [])]}")
    print(get_research_summary())
