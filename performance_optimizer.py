"""
PERFORMANCE OPTIMIZER   Efficiency-Aware Evolution

This system adds performance-aware scoring to the fitness function. Instead of just
checking correctness, it evaluates:
- Time complexity awareness (does the AI optimize for efficiency?)
- Memory usage tracking across epochs
- Complexity class recognition (can the AI identify O(n) vs O(n²) patterns?)
- Performance regression detection (did a change make things slower?)

This prevents the AI from creating correct but inefficient code   a critical step
toward building systems that can handle increasingly complex tasks.
"""

import ast
import json
import os
import time
from typing import Dict, List, Optional

PERFORMANCE_LOG = "sandbox/performance_log.json"


def load_performance_log() -> list:
    """Load the performance log from disk."""
    if not os.path.exists(PERFORMANCE_LOG):
        return []
    try:
        with open(PERFORMANCE_LOG, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


def save_performance_log(log: list) -> None:
    """Save the performance log to disk."""
    os.makedirs(os.path.dirname(PERFORMANCE_LOG), exist_ok=True)
    with open(PERFORMANCE_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


def estimate_complexity(code: str) -> Dict[str, float]:
    """Estimate the time and space complexity of generated code.
    
    Returns a dict with estimated complexity classes (O(1), O(n), O(n²), etc.)
    This is an approximation based on AST analysis.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"time": "unknown", "space": "unknown", "loops": 0, "recursions": 0}
    
    # Count nested loops (depth of loop nesting)
    max_loop_depth = 0
    current_loop_depth = 0
    
    def _count_loops(node):
        nonlocal max_loop_depth, current_loop_depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                current_loop_depth += 1
                max_loop_depth = max(max_loop_depth, current_loop_depth)
                _count_loops(child)
                current_loop_depth -= 1
            else:
                _count_loops(child)
    
    loop_count = 0
    recursion_count = 0
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            loop_count += 1
        elif isinstance(node, ast.FunctionDef):
            # Check for recursive calls within function body
            func_name = node.name
            for inner_node in ast.walk(node):
                if isinstance(inner_node, ast.Call) and hasattr(inner_node.func, 'name'):
                    if inner_node.func.name == func_name:
                        recursion_count += 1
    
    _count_loops(tree)
    
    # Complexity estimation based on nesting depth (not just total count)
    if loop_count == 0 and recursion_count == 0:
        time_complexity = "O(1)" if len(code.split('\n')) < 5 else "O(n)"
    elif max_loop_depth >= 3 or (max_loop_depth >= 2 and recursion_count > 0):
        # Deep nesting  O(n³) or worse
        time_complexity = "O(n³) or worse"
    elif max_loop_depth >= 2:
        time_complexity = "O(n²)"
    elif loop_count <= 1 and recursion_count == 0:
        time_complexity = "O(n)"
    else:
        time_complexity = "O(n log n)" if recursion_count > 0 else "O(n)"
    
    # Space complexity estimation (considering recursion depth)
    space_complexity = "O(1)" if loop_count == 0 and recursion_count == 0 else "O(n)"
    if max_loop_depth >= 3:
        space_complexity = "O(n²)"
    
    return {
        "time": time_complexity,
        "space": space_complexity,
        "loops": loop_count,
        "recursions": recursion_count,
        "max_nesting": max_loop_depth
    }


def record_performance(task_prompt: str, code: str, score: float, exec_time: float) -> None:
    """Record performance metrics for a task solution."""
    complexity = estimate_complexity(code)
    
    entry = {
        "task": task_prompt,
        "timestamp": __import__('time').time(),
        "score": score,
        "exec_time": round(exec_time, 4),
        "complexity": complexity,
        "code_length": len(code.split('\n'))
    }
    
    log = load_performance_log()
    log.append(entry)
    save_performance_log(log)


def get_performance_trends() -> str:
    """Analyze performance trends across epochs.
    
    Returns a summary of whether the AI is becoming more efficient over time.
    This is critical for singularity   the system must not only be correct,
    but also scalable to increasingly complex tasks.
    """
    log = load_performance_log()
    if len(log) < 2:
        return "Not enough performance data yet."
    
    # Group by task type (extract from prompt keywords)
    by_type = {}
    for entry in log:
        task = entry.get("task", "")
        # Simple categorization
        if "matrix" in task.lower() or "determinant" in task.lower():
            category = "linear_algebra"
        elif "prime" in task.lower() or "gcd" in task.lower():
            category = "number_theory"
        else:
            category = "general"
        
        if category not in by_type:
            by_type[category] = []
        by_type[category].append(entry)
    
    trends = "Performance Trends:\n"
    
    for category, entries in by_type.items():
        avg_time = sum(e.get("exec_time", 0) for e in entries) / len(entries) if entries else 0
        avg_score = sum(e.get("score", 0) for e in entries) / len(entries) if entries else 0
        
        trends += f"\n{category} ({len(entries)} solutions):\n"
        trends += f"  Average score: {avg_score:.2f}\n"
        trends += f"  Average exec time: {avg_time:.4f}s\n"
    
    # Check for complexity improvement over time
    if len(log) >= 3:
        early = log[:len(log)//3]
        late = log[-(len(log)//3):]
        
        early_avg = sum(e.get("exec_time", 0) for e in early) / max(len(early), 1)
        late_avg = sum(e.get("exec_time", 0) for e in late) / max(len(late), 1)
        
        if late_avg < early_avg:
            trends += f"\nPerformance is IMPROVING (avg time decreased from {early_avg:.4f}s to {late_avg:.4f}s)\n"
        elif late_avg > early_avg:
            trends += f"\nWARNING: Performance is DECLINING   code may be getting slower\n"
    
    return trends


def prompt_ai_optimization(current_code: str, task_prompt: str) -> Optional[str]:
    """Generate an optimization prompt for the AI.
    
    This instructs the Engineer to optimize existing solutions for efficiency.
    """
    from autonomous_loop import prompt_ai
    
    complexity = estimate_complexity(current_code)
    
    prompt = f"""
You are optimizing YOUR OWN CODE for performance.

CURRENT CODE:
{current_code[:2000]}

COMPLEXITY ANALYSIS: {json.dumps(complexity)}

TASK: Analyze this code and identify optimization opportunities:
1. Can nested loops be replaced with hash-based lookups? (O(n²) → O(n))
2. Can redundant calculations be cached or memoized?
3. Are there unnecessary iterations or repeated computations?

RULES:
- Do NOT change the behavior ,  only optimize for speed and memory
- Preserve ALL existing functionality
- Focus on reducing time complexity where possible
- If code is already optimal, explain why no further optimization is needed

CRITICAL: Your goal is to make solutions SCALABLE. Correct but slow code will fail as tasks grow harder.
"""
    
    try:
        response = prompt_ai(prompt)
        
        # Log this optimization attempt
        log_entry = {
            "timestamp": __import__('time').time(),
            "task": task_prompt,
            "action": "optimization_analysis",
            "complexity": complexity,
            "result": response[:500] if response else "failed"
        }
        
        log = load_performance_log()
        log.append(log_entry)
        save_performance_log(log)
        
        return response
    except Exception as e:
        print(f"Optimization prompt failed ({e})")
        return None


def build_optimization_phase() -> str:
    """Build the optimization section for the Engineer's prompt.
    
    This tells the AI when to focus on efficiency vs correctness.
    """
    log = load_performance_log()
    
    # Check if we have enough data to recommend optimization
    if len(log) < 3:
        return ""
    
    # Get recent performance trends
    recent_avg_time = sum(e.get("exec_time", 0) for e in log[-5:]) / max(len(log[-5:]), 1)
    
    if recent_avg_time > 0.5:  # If average execution time is high
        return f"""
PERFORMANCE OPTIMIZATION NOTE:

Recent solutions are taking {recent_avg_time:.4f}s on average. As tasks grow more complex,
you must prioritize EFFICIENCY alongside correctness.

When writing new handlers or refactoring existing ones:
- Use hash-based lookups instead of linear searches where possible
- Cache repeated calculations (memoization)
- Avoid unnecessary iterations through data structures
- Consider whether O(n²) can be reduced to O(n)

CRITICAL: A correct solution that's too slow will fail on harder tasks.
"""
    
    return ""


def get_efficiency_score(current_code: str, exec_time: float) -> float:
    """Calculate an efficiency score (0-100) based on complexity and execution time.
    
    This can be used as part of the fitness function to reward efficient code.
    """
    complexity = estimate_complexity(current_code)
    
    # Base score from complexity class
    complexity_scores = {
        "O(1)": 100,
        "O(n)": 80,
        "O(n log n)": 60,
        "O(n²)": 30,
        "unknown": 0
    }
    
    base_score = complexity_scores.get(complexity["time"], 50)
    
    # Penalize slow execution
    time_penalty = max(0, (exec_time - 0.1) * 20)  # Penalty starts after 0.1s
    
    efficiency = max(0, base_score - time_penalty)
    return min(100, efficiency)
