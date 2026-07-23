"""
CAPABILITY FRONTIER   Tracks What the AI Can and Cannot Do

Maintains a map of the AI's demonstrated capabilities across all problem domains.
After each epoch, the map is updated based on which problem categories were
solved vs failed. This drives the research planner's autonomous curriculum.

Capability states:
  "mastered"    solved consistently across multiple epochs (>= 80% pass rate)
  "partial"     sometimes solved, sometimes not (20-80% pass rate)
  "attempted"   tried but mostly failing (< 20% pass rate)
  "untested"    no problems in this area have been attempted yet
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

FRONTIER_PATH = "sandbox/capability_frontier.json"


# CAPABILITY MAP DEFINITION
# All domains the system tracks. New ones can be added freely.

DEFAULT_FRONTIER: Dict[str, Dict] = {
    # Mathematical Reasoning
    "mathematical": {
        "basic_arithmetic":          {"state": "untested", "attempts": 0, "successes": 0},
        "prime_numbers":             {"state": "untested", "attempts": 0, "successes": 0},
        "divisibility":              {"state": "untested", "attempts": 0, "successes": 0},
        "gcd_lcm":                   {"state": "untested", "attempts": 0, "successes": 0},
        "digit_operations":          {"state": "untested", "attempts": 0, "successes": 0},
        "sequences_fibonacci":       {"state": "untested", "attempts": 0, "successes": 0},
        "combinatorics":             {"state": "untested", "attempts": 0, "successes": 0},
        "number_theory":             {"state": "untested", "attempts": 0, "successes": 0},
        "geometry":                  {"state": "untested", "attempts": 0, "successes": 0},
        "statistics_median_mean":    {"state": "untested", "attempts": 0, "successes": 0},
    },
    # String / Text Reasoning
    "string": {
        "basic_string_ops":          {"state": "untested", "attempts": 0, "successes": 0},
        "palindromes":               {"state": "untested", "attempts": 0, "successes": 0},
        "anagram_detection":         {"state": "untested", "attempts": 0, "successes": 0},
        "encoding_cipher":           {"state": "untested", "attempts": 0, "successes": 0},
        "run_length_encoding":       {"state": "untested", "attempts": 0, "successes": 0},
        "substring_search":          {"state": "untested", "attempts": 0, "successes": 0},
        "character_frequency":       {"state": "untested", "attempts": 0, "successes": 0},
        "string_validation":         {"state": "untested", "attempts": 0, "successes": 0},
    },
    # Algorithmic / Data Structures
    "algorithmic": {
        "sorting_searching":         {"state": "untested", "attempts": 0, "successes": 0},
        "list_transformations":      {"state": "untested", "attempts": 0, "successes": 0},
        "deduplication":             {"state": "untested", "attempts": 0, "successes": 0},
        "rotation_shifting":         {"state": "untested", "attempts": 0, "successes": 0},
        "dynamic_programming":       {"state": "untested", "attempts": 0, "successes": 0},
        "recursion":                 {"state": "untested", "attempts": 0, "successes": 0},
        "graph_algorithms":          {"state": "untested", "attempts": 0, "successes": 0},
        "tree_traversal":            {"state": "untested", "attempts": 0, "successes": 0},
    },
    # Logical Reasoning
    "logical": {
        "boolean_logic":             {"state": "untested", "attempts": 0, "successes": 0},
        "set_operations":            {"state": "untested", "attempts": 0, "successes": 0},
        "classification":            {"state": "untested", "attempts": 0, "successes": 0},
        "constraint_satisfaction":   {"state": "untested", "attempts": 0, "successes": 0},
        "pattern_recognition":       {"state": "untested", "attempts": 0, "successes": 0},
    },
    # Meta-Capabilities
    "meta": {
        "code_generalization":       {"state": "untested", "attempts": 0, "successes": 0},
        "novel_problem_adaptation":  {"state": "untested", "attempts": 0, "successes": 0},
        "multi_step_reasoning":      {"state": "untested", "attempts": 0, "successes": 0},
    },
}

# Category keywords → frontier domain mapping
CATEGORY_TO_DOMAIN = {
    "mathematical": "mathematical",
    "string":       "string",
    "algorithmic":  "algorithmic",
    "logical":      "logical",
}

# Problem description keywords → sub-domain mapping
KEYWORD_TO_SUBDOMAIN = {
    "prime":             ("mathematical", "prime_numbers"),
    "divisor":           ("mathematical", "divisibility"),
    "gcd":               ("mathematical", "gcd_lcm"),
    "lcm":               ("mathematical", "gcd_lcm"),
    "digit":             ("mathematical", "digit_operations"),
    "fibonacci":         ("mathematical", "sequences_fibonacci"),
    "collatz":           ("mathematical", "sequences_fibonacci"),
    "palindrome":        ("string", "palindromes"),
    "anagram":           ("string", "anagram_detection"),
    "caesar":            ("string", "encoding_cipher"),
    "cipher":            ("string", "encoding_cipher"),
    "run-length":        ("string", "run_length_encoding"),
    "vowel":             ("string", "basic_string_ops"),
    "word":              ("string", "basic_string_ops"),
    "substring":         ("string", "substring_search"),
    "pangram":           ("string", "string_validation"),
    "hamming":           ("string", "string_validation"),
    "rotate":            ("algorithmic", "rotation_shifting"),
    "duplicate":         ("algorithmic", "deduplication"),
    "unique":            ("algorithmic", "deduplication"),
    "median":            ("mathematical", "statistics_median_mean"),
    "sort":              ("algorithmic", "sorting_searching"),
    "clamp":             ("algorithmic", "list_transformations"),
    "product":           ("algorithmic", "list_transformations"),
    "triangle":          ("mathematical", "geometry"),
    "perfect number":    ("mathematical", "number_theory"),
    "sum of squares":    ("mathematical", "combinatorics"),
    "square of sum":     ("mathematical", "combinatorics"),
    "binary":            ("string", "encoding_cipher"),
}


# LOAD / SAVE

def load_frontier() -> Dict:
    if not os.path.exists(FRONTIER_PATH):
        return {k: {sk: dict(sv) for sk, sv in v.items()} for k, v in DEFAULT_FRONTIER.items()}
    try:
        with open(FRONTIER_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Merge with default to ensure new sub-domains are always present
        for domain, subdomains in DEFAULT_FRONTIER.items():
            data.setdefault(domain, {})
            for subdomain, defaults in subdomains.items():
                data[domain].setdefault(subdomain, dict(defaults))
        return data
    except Exception:
        return {k: {sk: dict(sv) for sk, sv in v.items()} for k, v in DEFAULT_FRONTIER.items()}


def save_frontier(frontier: Dict) -> None:
    os.makedirs(os.path.dirname(FRONTIER_PATH), exist_ok=True)
    with open(FRONTIER_PATH, "w", encoding="utf-8") as f:
        json.dump(frontier, f, indent=2)


# STATE COMPUTATION

def _compute_state(attempts: int, successes: int) -> str:
    if attempts == 0:
        return "untested"
    rate = successes / attempts
    if rate >= 0.8:
        return "mastered"
    if rate >= 0.2:
        return "partial"
    return "attempted"


def _find_subdomain(problem_description: str, category: str) -> Optional[Tuple[str, str]]:
    """Map a problem description to a (domain, subdomain) pair."""
    desc_lower = problem_description.lower()
    for keyword, (domain, subdomain) in KEYWORD_TO_SUBDOMAIN.items():
        if keyword in desc_lower:
            return domain, subdomain
    # Fall back to category-level domain
    domain = CATEGORY_TO_DOMAIN.get(category, "algorithmic")
    return domain, "list_transformations"  # generic fallback


# PUBLIC API

def record_problem_result(
        problem_description: str,
        category: str,
        passed: bool) -> None:
    """
    Record the result of a single problem attempt.
    Updates the capability frontier map.
    """
    frontier = load_frontier()
    mapping = _find_subdomain(problem_description, category)
    if mapping is None:
        return
    domain, subdomain = mapping

    if domain not in frontier:
        frontier[domain] = {}
    if subdomain not in frontier[domain]:
        frontier[domain][subdomain] = {"state": "untested", "attempts": 0, "successes": 0}

    node = frontier[domain][subdomain]
    node["attempts"] += 1
    if passed:
        node["successes"] += 1
    node["state"] = _compute_state(node["attempts"], node["successes"])
    node["last_updated"] = time.time()

    save_frontier(frontier)


def get_frontier_gaps() -> List[Dict]:
    """
    Return sub-domains that are NOT yet mastered, sorted by priority.
    Priority order: partial > attempted > untested
    """
    frontier = load_frontier()
    gaps = []
    for domain, subdomains in frontier.items():
        for subdomain, data in subdomains.items():
            state = data.get("state", "untested")
            if state != "mastered":
                priority = {"partial": 0, "attempted": 1, "untested": 2}.get(state, 3)
                gaps.append({
                    "domain": domain,
                    "subdomain": subdomain,
                    "state": state,
                    "priority": priority,
                    "attempts": data.get("attempts", 0),
                    "successes": data.get("successes", 0),
                    "pass_rate": (data["successes"] / data["attempts"])
                                 if data.get("attempts", 0) > 0 else None,
                })
    gaps.sort(key=lambda x: (x["priority"], -x.get("attempts", 0)))
    return gaps


def get_mastered_domains() -> List[str]:
    """Return list of fully mastered domain+subdomain pairs."""
    frontier = load_frontier()
    mastered = []
    for domain, subdomains in frontier.items():
        for subdomain, data in subdomains.items():
            if data.get("state") == "mastered":
                mastered.append(f"{domain}/{subdomain}")
    return mastered


def get_frontier_summary() -> str:
    """Human-readable summary of current capability frontier."""
    frontier = load_frontier()
    counts = {"mastered": 0, "partial": 0, "attempted": 0, "untested": 0}
    for domain, subdomains in frontier.items():
        for subdomain, data in subdomains.items():
            state = data.get("state", "untested")
            counts[state] = counts.get(state, 0) + 1
    total = sum(counts.values())
    return (f"Frontier ({total} sub-domains): "
            f"mastered={counts['mastered']} | "
            f"partial={counts['partial']} | "
            f"attempted={counts['attempted']} | "
            f"untested={counts['untested']}")


def get_overall_progress_pct() -> float:
    """Return the percentage of sub-domains at 'mastered' state (0.0, 1.0)."""
    frontier = load_frontier()
    total, mastered = 0, 0
    for domain, subdomains in frontier.items():
        for subdomain, data in subdomains.items():
            total += 1
            if data.get("state") == "mastered":
                mastered += 1
    return mastered / max(total, 1)


if __name__ == "__main__":
    print("Capability Frontier self-test:")
    print(get_frontier_summary())
    gaps = get_frontier_gaps()
    print(f"\nTop 5 frontier gaps:")
    for g in gaps[:5]:
        rate_str = f"{g['pass_rate']:.0%}" if g["pass_rate"] is not None else "no data"
        print(f"  {g['domain']}/{g['subdomain']} [{g['state']}] pass_rate={rate_str}")
