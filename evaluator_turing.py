import time
import os
import sys

# Sandbox globals   no access to dangerous builtins like os, sys, subprocess
SANDBOX_GLOBALS = {
    '__builtins__': {'print': print, 'len': len, 'range': range, 'list': list,
                     'dict': dict, 'set': set, 'tuple': tuple, 'int': int,
                     'float': float, 'str': str, 'bool': bool, 'True': True,
                     'False': False, 'None': None}
}

try:
    import math as _math
    SANDBOX_GLOBALS['math'] = _math
except ImportError:
    pass

def evaluate_turing():
    """
    Evaluates the Child Brain's ability to generate code for new tasks.
    Passes prompts from the dynamic dataset and checks if the AI generates
    executable code that actually produces correct results.
    
    This is the SINGULARITY CHECK: if the AI can generate correct code for
    tasks it has never seen before, it has transcended its baseline.
    """
    turing_score = 0
    start_time = time.time()
    
    if not os.path.exists("sandbox/experiment.py"):
        return 0, 0, "sandbox/experiment.py does not exist yet."
    if not os.path.exists("sandbox/dynamic_dataset.json"):
        return 0, 0, "sandbox/dynamic_dataset.json does not exist yet."

    try:
        import importlib
        import json

        # Ensure sandbox is on path so 'import baselines' works inside experiment.py
        sandbox_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox")
        if sandbox_path not in sys.path:
            sys.path.insert(0, sandbox_path)

        # Remove cached modules to force reload
        for mod in list(sys.modules.keys()):
            if mod in ('experiment', 'baselines') or mod.startswith('sandbox'):
                del sys.modules[mod]
        
        if 'sandbox' not in sys.path:
            sys.path.insert(0, 'sandbox')

        import experiment as exp
        importlib.reload(exp)
        
        if not hasattr(exp, "generate_code"):
            return 0, 0, "No generate_code function found."
        
        with open("sandbox/dynamic_dataset.json", "r") as f:
            raw = json.load(f)
        
        # Filter to only proper 3-tuple items, skip corrupted entries
        dataset = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                dataset.append(tuple(item))
        
        if not dataset:
            return 0, 0, "Dynamic dataset is empty or corrupted."
        
        # Try each prompt in the dynamic dataset
        passed = 0
        tested = 0
        cache = {}
        for prompt, input_data, expected in dataset:
            if prompt not in cache:
                try:
                    generated_code = exp.generate_code(prompt)
                    local_env = {}
                    # Use sandboxed globals   no access to os, sys, subprocess, etc.
                    exec(generated_code, SANDBOX_GLOBALS, local_env)
                    func = None
                    for k, v in local_env.items():
                        if callable(v) and not k.startswith("__"):
                            func = v
                            break
                    cache[prompt] = func
                except Exception as e:
                    cache[prompt] = None
            
            func = cache[prompt]
            if func is None:
                tested += 1
                continue
            
            try:
                if isinstance(input_data, (list, tuple)):
                    res = func(*input_data)
                else:
                    res = func(input_data)
                tested += 1
                if res == expected:
                    passed += 1
            except Exception:
                tested += 1

        if tested == 0:
            return 0, 0, "No valid test cases in dynamic dataset."
        
        # Turing score  fraction of dynamic tasks solved * 1000
        fraction = passed / tested
        turing_score = int(fraction * 1000)
        
        err_msg = None
        if turing_score == 0:
            err_msg = f"Solved 0/{tested} dynamic tasks. The AI must add handlers for new prompt types."
        
    except Exception as e:
        import traceback
        return 0, 0, f"Error in Turing eval: {traceback.format_exc()}"
    
    exec_time = time.time() - start_time
    
    # Output format must match other evaluators for run_evaluator() to parse correctly
    print(f"TURING SCORE: {turing_score}")
    print(f"DYNAMIC_SCORE: {int(turing_score / 10)}")
    print(f"FITNESS: {fraction:.4f}")
    if err_msg:
        print(f"CRASH_LOG: {err_msg}")
    
    return turing_score, exec_time, err_msg

if __name__ == "__main__":
    score, exec_time, err = evaluate_turing()
    print(f"TURING SCORE: {score}")
    if err:
        print(f"ERROR: {err}")
    print(f"TIME: {exec_time:.4f}s")
