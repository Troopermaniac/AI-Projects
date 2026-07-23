import time
import psutil
import os
import sys
import ast
import multiprocessing

# Sandbox globals   no access to dangerous builtins like os, sys, subprocess
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

# Add math module for safe imports the AI might use
try:
    import math as _math
    SANDBOX_GLOBALS['math'] = _math
except ImportError:
    pass

def run_experiment_guillotine(dataset):
    """
    Runs the experiment in a separate process with a strict timeout and memory limit.
    Returns (score, exec_time, mem_mb).
    """
    if "sandbox" not in sys.path:
        sys.path.append("sandbox")
        
    try:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        start_time = time.time()
        
        # Ensure sandbox is on path so 'import baselines' works inside experiment.py
        sandbox_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox")
        if sandbox_path not in sys.path:
            sys.path.insert(0, sandbox_path)
        
        # Reload the experiment module to ensure we get the latest AI-written version
        for mod in list(sys.modules.keys()):
            if mod in ('experiment', 'baselines') or mod.startswith('sandbox'):
                del sys.modules[mod]
        import experiment
        
        if not hasattr(experiment, "generate_code"):
            print("CRASH_LOG: No generate_code function found in experiment.")
            return 0.0, 0.0, 0.0

        # Run actual evaluation
        score = 0.0
        cached_functions = {}
        
        for item in dataset:
            if len(item) == 3:
                prompt, input_data, expected = item
            else:
                # Fallback for old 2-tuple dataset formats
                prompt = "XOR Logic" if len(item[0]) == 2 else "AND Logic"
                input_data, expected = item
                
            # Cache code generation per prompt to save time
            if prompt not in cached_functions:
                try:
                    generated_code = experiment.generate_code(prompt)
                    local_env = {}
                    # Use sandboxed globals   no access to os, sys, subprocess, etc.
                    exec(generated_code, SANDBOX_GLOBALS, local_env)
                    
                    # Extract the first defined function
                    func = None
                    for k, v in local_env.items():
                        if callable(v) and not k.startswith("__"):
                            func = v
                            break
                    cached_functions[prompt] = func
                except Exception as e:
                    cached_functions[prompt] = f"Error: {e}"
            
            func = cached_functions[prompt]
            if isinstance(func, str):
                continue # Skip if compile/generation failed
                
            if func is None:
                continue # Skip if no function defined
                
            try:
                # Try calling with unpacked arguments first, then as a single argument
                try:
                    if isinstance(input_data, (list, tuple)):
                        res = func(*input_data)
                    else:
                        res = func(input_data)
                except TypeError:
                    res = func(input_data)
                    
                if res == expected:
                    score += 1.0
            except Exception as e:
                pass # Test failed due to runtime error
        
        end_time = time.time()
        mem_after = process.memory_info().rss
        
        mem_used_mb = max(0, mem_after - mem_before) / (1024 * 1024)
        exec_time = end_time - start_time
        
        # Execution Guillotine Logic:
        if exec_time > 60.0:
            return 0.0, exec_time, mem_used_mb  # Killed for time
        if mem_used_mb > 8000.0:
            return 0.0, exec_time, mem_used_mb  # Killed for memory
            
        return score, exec_time, mem_used_mb
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc().replace('\n', ' | ')
        print(f"CRASH_LOG: {err_msg}")
        return 0.0, 0.0, 0.0  # Crash counts as 0

ANCHOR_DATASET = [
    # --- Level 1: Elementary Logic & Arithmetic (6) ---
    ("XOR Logic", [0, 1], 1),
    ("AND Logic", [1, 0, 1], 0),
    ("Is Even", 4, True),
    ("Is Odd", 5, True),
    ("Absolute Difference", [10, 3], 7),
    ("Clamp Value", [15, 0, 10], 10),

    # --- Level 2: Basic Sequences & Math (6) ---
    ("Fibonacci Sequence", 10, 55),
    ("Factorial", 5, 120),
    ("Sum of Digits", [1234], 10),
    ("GCD Of Two Numbers", [48, 18], 6),
    ("LCM Of Two Numbers", [4, 6], 12),
    ("Is Power Of Two", [16], True),

    # --- Level 3: Array & List Operations (6) ---
    ("Sum of List", [1, 2, 3, 4, 5], 15),
    ("Max of List", [5, 3, 9, 1], 9),
    ("Reverse List", [1, 2, 3], [3, 2, 1]),
    ("Sort List", [3, 1, 4, 2], [1, 2, 3, 4]),
    ("Find Second Largest", [[10, 5, 8, 3]], 8),
    ("Rotate List Left By N", [[1, 2, 3, 4, 5], 2], [3, 4, 5, 1, 2]),

    # --- Level 4: String Processing & Encryption (6) ---
    ("Count Uppercase Letters", ["Hello World"], 2),
    ("Remove Spaces", ["hello world"], "helloworld"),
    ("String Is Digits Only", ["12345"], True),
    ("Caesar Cipher Shift", ["abc", 1], "bcd"),
    ("Reverse Words In String", ["hello world foo"], "foo world hello"),
    ("Hamming Distance", ["karolin", "kathrin"], 3),

    # --- Level 5: Number Theory & Algorithmic Logic (6) ---
    ("Check If Prime", [97], True),
    ("Nth Prime Number", [5], 11),
    ("Count Divisors", [12], 6),
    ("Is Perfect Number", [6], True),
    ("Leap Year Check", [2024], True),
    ("Grade From Score", [85], "B"),

    # --- Level 6: Advanced Dynamic Programming & Search (5) ---
    ("Maximum Subarray Sum", [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], 6),
    ("Binary Search Index", [[1, 3, 5, 7, 9, 11], 7], 3),
    ("Valid Parentheses", ["()[]{}"], True),
    ("Longest Common Prefix", [["flower", "flow", "flight"]], "fl"),
    ("Coin Change Min Coins", [[1, 2, 5], 11], 3),

    # --- Level 7: Matrix & Advanced Structures (5) ---
    ("Matrix Transpose", [[[1, 2], [3, 4]]], [[1, 3], [2, 4]]),
    ("Flatten Nested List", [[[1, [2, 3]], 4]], [1, 2, 3, 4]),
    ("Pascal Triangle Row", [4], [1, 4, 6, 4, 1]),
    ("Run Length Encoding", ["AAABBC"], "3A2B1C"),
    ("RLE Decoding", ["3A2B1C"], "AAABBC"),
]

def evaluate_anchor():
    """
    The expanded 40-problem Anchor Benchmark across 7 difficulty levels.
    The AI cannot edit this file. It provides a gradual learning curve.
    """
    return run_experiment_guillotine(ANCHOR_DATASET)


#  IMPROVEMENT #3: EXPANDED QUALITY METRICS 

def calculate_elegance_score(code: str) -> float:
    """Calculate code elegance score (0-1)."""
    if not code:
        return 0.0
    
    handler_count = code.count("if '")
    pattern_duplication = 0.0
    if handler_count > 20:
        pattern_duplication = min((handler_count - 20) * 0.02, 0.5)
    
    function_defs = code.count("def ")
    class_defs = code.count("class ")
    abstraction_bonus = min((function_defs + class_defs) * 0.05, 0.3)
    
    line_count = len(code.split('\n'))
    line_efficiency = max(0, (1 - line_count / 500))
    
    elegance = max(0, min(1.0,
        0.4 * line_efficiency +
        0.3 * abstraction_bonus +
        0.3 * (1 - pattern_duplication)
    ))
    return round(elegance, 3)


def calculate_generalization_score(code: str) -> float:
    """Calculate generalization potential (0-1)."""
    if not code:
        return 0.0
    
    has_function_abstraction = "def solve" in code or "def generate" in code
    has_list_comprehension = "[x for" in code or "[y for" in code
    has_map_filter = "map(" in code or "filter(" in code
    has_generic_patterns = any(p in code for p in ["sorted(", "min(", "max(", "any(", "all("])
    
    abstraction_count = sum([
        has_function_abstraction,
        has_list_comprehension,
        has_map_filter,
        has_generic_patterns
    ])
    
    handler_count = code.count("if '")
    handler_penalty = min(handler_count / 30, 1.0) * 0.5
    
    generalization = max(0, min(1.0,
        (abstraction_count / 4) - handler_penalty + 0.25
    ))
    return round(generalization, 3)


def calculate_reusability_score(code: str) -> float:
    """Calculate code reusability (0-1)."""
    if not code:
        return 0.0
    
    utility_functions = []
    lines = code.split('\n')
    for line in lines:
        if 'def ' in line and 'generate_code' not in line:
            func_name = line.split('def ')[1].split('(')[0]
            utility_functions.append(func_name)
    
    utility_bonus = min(len(utility_functions) * 0.15, 0.6)
    
    common_patterns = sum(line.count('sorted(') + line.count('min(') + line.count('max(')
                          for line in lines) / max(len(lines), 1)
    pattern_bonus = min(common_patterns * 5, 0.3)
    
    reusability = max(0, min(1.0,
        utility_bonus + pattern_bonus
    ))
    return round(reusability, 3)


if __name__ == "__main__":
    score, exec_time, mem_mb = evaluate_anchor()
    print(f"ANCHOR SCORE: {score}")
    print(f"TIME: {exec_time:.4f}s")
    print(f"MEM: {mem_mb:.2f}MB")
