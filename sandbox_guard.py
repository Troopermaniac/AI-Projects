import sys
import os
import ast
import runpy

# Safe imports the AI is allowed to use (math, random, functools, etc.)
SAFE_IMPORTS = {'math', 'random', 'functools', 'itertools', 'collections', 'operator', 'string', 're', 'datetime', 'time'}

DANGEROUS_IMPORTS = {'os', 'subprocess', 'shutil', 'pathlib', 'socket', 'urllib', 'requests'}

# Core files the AI can NEVER overwrite   modifying these would break the security cage
# or allow the AI to game its own benchmarks.
FORBIDDEN_WRITE_FILES = {
    'main.py', 'autonomous_loop.py', 'sandbox_guard.py',
    'bootstrap.py', 'start_ai.bat', 'evaluator_mbpp.py',
    'evaluator_anchor.py',   # Lock the anchor evaluator
    'self_evolution_guard.py',  # Phase 3: immutable safety gate
    'model_selector.py',     # Phase 4: selection pressure must stay honest
}

class SecurityException(Exception):
    pass

def check_ast(script_path):
    with open(script_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split('.')[0]
                # Allow safe math/logic imports the AI needs
                if base in SAFE_IMPORTS:
                    continue
                # sys is allowed   it's needed for path manipulation and testing
                if base == 'sys':
                    continue
                if base in DANGEROUS_IMPORTS:
                    raise SecurityException(f"AST Blocked: Dangerous import '{alias.name}' is not allowed.")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in DANGEROUS_IMPORTS:
                raise SecurityException(f"AST Blocked: Dangerous import from '{node.module}' is not allowed.")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # Block eval() and __import__ everywhere   truly dangerous
                if node.func.id in ('eval', '__import__'):
                    raise SecurityException(f"AST Blocked: Dangerous function '{node.func.id}()' is not allowed.")
                # Allow exec() for evaluators AND experiment.py (they need it to run AI code)
                if node.func.id == 'exec':
                    allow_patterns = ['evaluator_', 'test_eval', 'experiment']
                    if any(p in script_path for p in allow_patterns):
                        continue
                    raise SecurityException(f"AST Blocked: Dangerous function '{node.func.id}()' is not allowed.")
            elif isinstance(node.func, ast.Attribute):
                # Allow exec() for evaluators AND experiment.py
                if node.func.attr == 'exec':
                    allow_patterns = ['evaluator_', 'test_eval', 'experiment']
                    if any(p in script_path for p in allow_patterns):
                        continue
                    raise SecurityException(f"AST Blocked: Dangerous function attribute '{node.func.attr}()' is not allowed.")

def audit_hook(event, args):
    blocked_events = {
        "os.system", "subprocess.Popen", "os.spawn", "os.exec", 
        "os.remove", "os.rename", "os.rmdir", "shutil.rmtree", "socket.connect", "urllib.Request"
    }
    if event in blocked_events:
        raise SecurityException(f"Audit Blocked: Dangerous operation '{event}' is forbidden.")

def sandboxed_exec(code):
    """Execute code in a restricted namespace with dangerous builtins removed."""
    import time
    
    # Create isolated globals/locals   no access to os, sys, subprocess, etc.
    restricted_globals = {
        '__builtins__': {
            'print': print,
            'len': len,
            'range': range,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'True': True,
            'False': False,
            'None': None,
        },
    }
    
    # Add math and other safe modules if needed
    try:
        import math as _math
        restricted_globals['math'] = _math
    except ImportError:
        pass
    
    start_time = time.time()
    try:
        exec(code, restricted_globals)
        elapsed = time.time() - start_time
        return True, "Executed successfully", elapsed
    except Exception as e:
        return False, str(e), time.time() - start_time


def main():
    if len(sys.argv) < 2:
        print("Usage: python sandbox_guard.py [--check-only] <script_to_run>")
        sys.exit(1)
        
    check_only = False
    script_path = sys.argv[1]
    if sys.argv[1] == "--check-only":
        check_only = True
        script_path = sys.argv[2]
        
    # 1. AST Check
    try:
        check_ast(script_path)
    except SecurityException as e:
        print(f"SECURITY VIOLATION (AST): {e}", file=sys.stderr)
        sys.exit(1)
    except SyntaxError as e:
        print(f"SyntaxError in {script_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading script: {e}", file=sys.stderr)
        sys.exit(1)
        
    if check_only:
        print("AST Check passed.")
        sys.exit(0)
        
    # 2. Install Audit Hook
    sys.addaudithook(audit_hook)
    
    # 3. Execute with SANDBOX_GLOBALS   restrict builtins like evaluators do
    restricted_globals = {
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
        },
    }
    
    try:
        import runpy as _runpy
        _runpy.run_path(script_path, run_name="__main__", init_globals=restricted_globals)
    except SecurityException as e:
        print(f"SECURITY VIOLATION (RUNTIME): {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
