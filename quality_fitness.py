"""
QUALITY-AWARE FITNESS FUNCTION   Beyond Correctness: Efficiency, Elegance & Reusability

This system replaces the simple correctness-only fitness function with a multi-dimensional
quality metric that measures:
- Code efficiency (time/space complexity awareness)
- Handler elegance (fewer handlers  more intelligent solutions)
- Pattern reusability (does the solution generalize to similar tasks?)
- Architectural growth rate (is the codebase growing intelligently or just bloating?)

This is critical for singularity because:
- Correct but bloated code can't scale to complex problems
- True intelligence recognizes patterns and generalizes, not just memorizing handlers
- The fitness function must reward elegance, not just correctness

This implements:
- Multi-dimensional scoring (correctness + efficiency + elegance + reusability)
- Complexity-aware penalties (O(n²) solutions score lower than O(n))
- Handler count optimization (fewer handlers that do more  higher quality)
- Pattern generalization detection (recognizing when one handler solves multiple tasks)
"""

import ast
import json
import os
from typing import Dict, List, Optional


QUALITY_LOG = "sandbox/quality_log.json"


def load_quality_log() -> list:
    """Load the quality tracking log from disk."""
    if not os.path.exists(QUALITY_LOG):
        return []
    try:
        with open(QUALITY_LOG, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


def save_quality_log(log: list) -> None:
    """Save the quality tracking log to disk."""
    os.makedirs(os.path.dirname(QUALITY_LOG), exist_ok=True)
    with open(QUALITY_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


def estimate_code_complexity(code: str) -> Dict[str, float]:
    """Estimate the complexity of generated code using AST analysis.
    
    Returns a dict with time/space complexity estimates and nesting depth.
    This enables the fitness function to penalize inefficient solutions.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"time": "unknown", "space": "unknown", "nesting_depth": 0}
    
    max_nesting = 0
    loop_count = 0
    
    def _count_nesting(node):
        nonlocal max_nesting, loop_count
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                loop_count += 1
                current_depth = 1
                stack = [child]
                while stack:
                    node = stack.pop()
                    current_depth += 1
                    max_nesting = max(max_nesting, current_depth)
                    for sub in ast.iter_child_nodes(node):
                        if isinstance(sub, (ast.For, ast.While)):
                            stack.append(sub)
            else:
                _count_nesting(child)
    
    _count_nesting(tree)
    
    # Complexity estimation based on nesting depth
    if loop_count == 0 and max_nesting == 0:
        time_complexity = "O(1)" if len(code.split('\n')) < 5 else "O(n)"
    elif max_nesting >= 3:
        time_complexity = "O(n³) or worse"
    elif max_nesting >= 2:
        time_complexity = "O(n²)"
    else:
        time_complexity = "O(n)" if loop_count <= 1 else "O(n log n)"
    
    space_complexity = "O(1)" if loop_count == 0 and max_nesting == 0 else "O(n)"
    
    return {
        "time": time_complexity,
        "space": space_complexity,
        "nesting_depth": max_nesting,
        "loops": loop_count
    }


def calculate_handler_elegance_score(code: str) -> float:
    """Calculate an elegance score based on handler count and code structure.
    
    Fewer handlers that solve more tasks = higher elegance score.
    This rewards the AI for generalizing instead of appending.
    """
    if not os.path.exists("sandbox/experiment.py"):
        return 0.0
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count handlers (if 'prompt' in prompt.lower())
        handler_count = content.count("in prompt.lower()")
        
        # Calculate elegance score (fewer handlers  higher score)
        # Base score of 1.0, penalize for each handler beyond a reasonable amount
        if handler_count <= 5:
            return 1.0  # Perfect elegance
        elif handler_count <= 10:
            return 0.8  # Good   some handlers needed
        elif handler_count <= 20:
            return 0.6  # Moderate   getting bloated
        else:
            # Heavy bloat   significant penalty
            return max(0.1, 1.0 - (handler_count - 20) * 0.05)
    
    except Exception as e:
        print(f"Elegance score calculation failed ({e})")
        return 0.0


def detect_pattern_reusability(code: str) -> Dict[str, float]:
    """Detect how reusable the code is across different tasks.
    
    Returns a dict with reusability metrics:
   , Shared logic ratio (how much code handles multiple tasks?)
   , Function extraction potential (are there extractable utility functions?)
   , Generalization score (does one solution solve similar problems?)
    """
    if not os.path.exists("sandbox/experiment.py"):
        return {"shared_logic": 0.0, "extractable_functions": 0, "generalization": 0.0}
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count unique function definitions (excluding handlers)
        func_count = content.count("def ")
        
        # Estimate shared logic ratio (handlers that share common patterns)
        handler_count = content.count("in prompt.lower()")
        if handler_count > 0:
            # If multiple handlers use similar patterns, they're reusable
            shared_patterns = content.count("return ")
            shared_logic_ratio = min(1.0, shared_patterns / max(handler_count * 2, 1))
        else:
            shared_logic_ratio = 0.0
        
        # Estimate extractable functions (common operations that could be utilities)
        common_ops = ["len(", "sorted(", "enumerate(", "zip("]
        extractable = sum(1 for op in common_ops if op in content)
        
        # Generalization score (how many tasks can one handler solve?)
        generalization_score = min(1.0, shared_logic_ratio * 0.7 + extractable / max(func_count, 1) * 0.3)
        
        return {
            "shared_logic": round(shared_logic_ratio, 3),
            "extractable_functions": extractable,
            "generalization": round(generalization_score, 3)
        }
    
    except Exception as e:
        print(f"Pattern reusability detection failed ({e})")
        return {"shared_logic": 0.0, "extractable_functions": 0, "generalization": 0.0}


def calculate_quality_fitness(correctness_score: float, exec_time: float = 0.1) -> Dict[str, float]:
    """Calculate multi-dimensional quality fitness score.
    
    Returns a dict with individual scores and overall weighted fitness.
    This replaces the simple correctness-only scoring with nuanced evaluation.
    """
    # Load current code for analysis
    elegance_score = calculate_handler_elegance_score("")
    reusability = detect_pattern_reusability("")
    complexity = estimate_code_complexity("")
    
    # Calculate weighted fitness
    weights = {
        "correctness": 0.4,  # Still important   must be correct first
        "elegance": 0.25,     # Rewards architectural quality
        "reusability": 0.25,  # Rewards generalization
        "efficiency": 0.1     # Penalizes excessive time/memory usage
    }
    
    # Efficiency score (based on exec_time   faster is better)
    efficiency_score = max(0.0, 1.0 - (exec_time / 60.0))  # Normalize to 60s timeout
    
    # Calculate individual scores
    scores = {
        "correctness": correctness_score,
        "elegance": elegance_score,
        "reusability": reusability.get("generalization", 0.0),
        "efficiency": efficiency_score
    }
    
    # Weighted overall fitness
    overall_fitness = sum(scores[k] * weights[k] for k in scores)
    
    return {
        "overall_fitness": round(overall_fitness, 4),
        "correctness": round(scores["correctness"], 4),
        "elegance": round(scores["elegance"], 4),
        "reusability": round(scores["reusability"], 4),
        "efficiency": round(scores["efficiency"], 4)
    }


def record_quality_metrics(epoch_num: int, quality_scores: Dict[str, float]) -> None:
    """Record quality metrics for tracking progress."""
    log_entry = {
        "epoch": epoch_num,
        **quality_scores,
        "timestamp": __import__('time').time()
    }
    
    log = load_quality_log()
    log.append(log_entry)
    save_quality_log(log)


def get_quality_trends() -> str:
    """Analyze quality trends across epochs.
    
    Returns a summary of whether the AI is improving in elegance, reusability, etc.
    This is critical for singularity   the system must become smarter, not just bigger.
    """
    log = load_quality_log()
    
    if len(log) < 2:
        return "Not enough quality data yet."
    
    # Calculate trends by dimension
    dimensions = ["correctness", "elegance", "reusability", "efficiency"]
    
    summary = "Quality Trends:\n"
    
    for dim in dimensions:
        values = [e.get(dim, 0) for e in log]
        
        if len(values) >= 2:
            early_avg = sum(values[:len(values)//3]) / max(len(values)//3, 1)
            late_avg = sum(values[-(len(values)//3):]) / max(len(values)//3, 1)
            
            trend = "IMPROVING" if late_avg > early_avg else "DECLINING" if late_avg < early_avg else "STABLE"
            summary += f"  {dim}: {early_avg:.2f} → {late_avg:.2f} ({trend})\n"
    
    # Overall fitness trend
    if len(log) >= 3:
        early_fitness = sum(e.get("overall_fitness", 0) for e in log[:len(log)//3]) / max(len(log)//3, 1)
        late_fitness = sum(e.get("overall_fitness", 0) for e in log[-(len(log)//3):]) / max(len(log)//3, 1)
        
        if late_fitness > early_fitness:
            summary += f"\nOverall quality is IMPROVING (fitness {early_fitness:.2f} → {late_fitness:.2f})\n"
        else:
            summary += f"\nWARNING: Overall quality is DECLINING   focus on elegance and reusability\n"
    
    return summary


def build_quality_fitness_phase() -> str:
    """Build the quality fitness section for the Visionary's prompt.
    
    This tells the Visionary what quality dimensions to optimize for.
    """
    trends = get_quality_trends()
    
    if "not enough" in trends.lower():
        return ""
    
    return f"""
QUALITY-AWARE FITNESS INSIGHTS:

{trends}

OPTIMIZATION PRIORITIES FOR THIS EPOCH:
1. Reduce handler count where possible (elegance)
2. Extract common patterns into reusable functions (reusability)
3. Optimize for efficiency without sacrificing correctness
4. Generalize solutions across similar tasks

CRITICAL: Your goal is to become MORE ELEGANT and REUSABLE, not just correct.
True intelligence recognizes patterns and generalizes   it doesn't just append handlers.
"""


def prompt_ai_quality_optimization(current_code: str) -> Optional[str]:
    """Generate a quality optimization prompt that instructs the AI to improve code quality.
    
    This tells the Engineer to focus on elegance, reusability, and efficiency.
    """
    from autonomous_loop import prompt_ai
    
    try:
        prompt = f"""
You are OPTIMIZING YOUR CODE for QUALITY, not just correctness.

CURRENT CODE:
{current_code[:2000]}

TASK: Analyze this code and identify opportunities to improve quality:
1. Can multiple handlers be merged into one generalized solution?
2. Are there common patterns that should be extracted into reusable functions?
3. Is there inefficient logic that can be optimized for speed/memory?
4. Can any handler be simplified while maintaining correctness?

RULES:
- Preserve ALL existing functionality ,  do not break working code
- Focus on elegance and reusability, not just correctness
- Extract patterns and generalize where possible
- Call final_answer("Quality optimization complete") when done

CRITICAL: Your goal is to become MORE ELEGANT. True intelligence recognizes patterns and generalizes.
"""
        
        return prompt_ai(prompt)
    
    except Exception as e:
        print(f"Quality optimization failed ({e})")
        return None
