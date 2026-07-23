"""
CODE REFACTORER   Beyond Appending: True Code Evolution

This system enables the AI to refactor, consolidate, and generalize its codebase.
Instead of just appending new handlers (which leads to bloat), it can:
- Identify redundant or overlapping patterns across evolved handlers
- Merge similar handlers into generalized abstractions
- Extract common sub-patterns into reusable utility functions
- Reorganize code structure for better maintainability

This is the missing piece that prevents the AI from creating a bloated, unmaintainable
codebase. It's what turns "adding more rules" into "building smarter systems."
"""

import ast
import json
import os
import time
from typing import List, Dict, Optional

REFACTOR_LOG = "sandbox/refactor_log.json"


def load_refactor_log() -> list:
    """Load the refactor log from disk."""
    if not os.path.exists(REFACTOR_LOG):
        return []
    try:
        with open(REFACTOR_LOG, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


def save_refactor_log(log: list) -> None:
    """Save the refactor log to disk."""
    os.makedirs(os.path.dirname(REFACTOR_LOG), exist_ok=True)
    with open(REFACTOR_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


def analyze_handler_patterns(code: str) -> Dict:
    """Analyze experiment.py code to find patterns and redundancies.
    
    Returns a dict with:
   , handlers: List of detected handler functions/patterns
   , duplicates: Groups of handlers that could be merged
   , utilities: Common operations that should be extracted
   , suggestions: Refactoring recommendations
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"error": "Invalid syntax   cannot analyze"}
    
    # Extract all if-statements in generate_code (potential handlers)
    handler_patterns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Try to extract the condition pattern
            if hasattr(node.test, 'op') and isinstance(node.test.op, ast.In):
                try:
                    left = ast.dump(node.test.left)
                    right = ast.dump(node.test.right) if hasattr(node.test.right, 'value') else ast.dump(node.test.right)
                    handler_patterns.append({
                        "pattern": f"if '{right}' in {left}",
                        "line": node.lineno
                    })
                except Exception:
                    pass
    
    return {
        "handlers": len(handler_patterns),
        "patterns": handler_patterns,
        "suggestions": []
    }


def prompt_ai_refactoring(current_code: str, task_prompt: str) -> Optional[str]:
    """Generate a refactoring prompt and call the AI.
    
    This instructs the Engineer to refactor existing code instead of just appending.
    """
    from autonomous_loop import prompt_ai
    
    # Analyze current patterns
    analysis = analyze_handler_patterns(current_code)
    
    prompt = f"""
You are the Refactorer of an Autonomous AI system. Your job is to IMPROVE existing code, not add more.

CURRENT CODE:
{current_code[:2000]}

ANALYSIS: {json.dumps(analysis)}

TASK: Analyze this code and identify opportunities to REFACTOR:
1. Are there handlers that do similar things? Merge them into a single generalized handler.
2. Is there repeated logic (e.g., importing math, checking types)? Extract it into utility functions.
3. Can conditional chains be replaced with data structures or patterns?

RULES:
- Do NOT change the behavior of existing handlers ,  only reorganize and generalize
- Preserve ALL existing functionality ,  this is refactoring, not rewriting
- Focus on reducing code duplication and improving maintainability
- If no refactoring is needed, say "No refactoring required" and explain why

CRITICAL: Your goal is to make the codebase SMARTER, not bigger.
"""
    
    try:
        response = prompt_ai(prompt)
        
        # Log this refactoring attempt
        log_entry = {
            "timestamp": time.time(),
            "task": task_prompt,
            "action": "refactor_analysis",
            "result": response[:500] if response else "failed"
        }
        
        log = load_refactor_log()
        log.append(log_entry)
        save_refactor_log(log)
        
        return response
    except Exception as e:
        print(f"Refactoring prompt failed ({e})")
        return None


def build_refactoring_prompt(current_code: str, unsolved_prompts: List[str]) -> str:
    """Build a refactoring section for the Engineer's prompt.
    
    This tells the AI when to refactor vs when to add new handlers.
    """
    if len(unsolved_prompts) == 0:
        # No new tasks   focus on refactoring existing code
        return f"""
REFACTORING PHASE (No new tasks to solve):

Current experiment.py has evolved handlers for many tasks. Instead of adding more,
focus on IMPROVING what exists:

1. Look for redundant patterns across handlers
2. Extract common operations into utility functions  
3. Generalize specific handlers into broader abstractions

Read sandbox/experiment.py and identify 2-3 refactoring opportunities.
Write the improved version using write_file().
"""
    
    return ""


def prompt_ai_consolidation() -> Optional[str]:
    """Generate a consolidation prompt that merges related handlers."""
    from autonomous_loop import prompt_ai
    
    if not os.path.exists("sandbox/experiment.py"):
        return None
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            current_code = f.read()
        
        # Extract evolved handlers section
        parts = current_code.split("#  EVOLVED HANDLERS")
        if len(parts) < 2:
            return None
        
        evolved_section = parts[1][:3000]
        
        prompt = f"""
You are consolidating the evolved handlers in experiment.py.

EVOLVED SECTION:
{evolved_section}

TASK: Identify groups of handlers that solve similar problems and could be merged.

For example:
- If you have separate handlers for "sum", "average", "max", "min" ,  consider a single "list_operations" handler
- If multiple handlers use the same pattern (e.g., iterating through lists) ,  extract it into a utility function

RULES:
1. Read the current code with read_file()
2. Identify 2-3 consolidation opportunities
3. Write the consolidated version with write_file()
4. Test with run_python_script()

CRITICAL: Preserve ALL existing functionality. Consolidation means reducing duplication, not removing features.
"""
        
        return prompt_ai(prompt)
    except Exception as e:
        print(f"Consolidation failed ({e})")
        return None


def get_refactoring_suggestions(current_code: str) -> List[str]:
    """Return concrete refactoring suggestions based on code analysis."""
    suggestions = []
    
    # Check for repeated imports
    math_imports = current_code.count("import math")
    if math_imports > 5:
        suggestions.append(f"Found {math_imports} 'import math' statements   consider a single utility module.")
    
    # Check for deeply nested conditionals (code smell)
    max_indent = 0
    for line in current_code.split('\n'):
        indent = len(line) - len(line.lstrip())
        if indent > max_indent:
            max_indent = indent
    
    if max_indent > 100:  # Rough heuristic for deep nesting
        suggestions.append("Deeply nested conditionals detected   consider extracting to separate functions.")
    
    # Check for handler count (bloat indicator)
    handler_count = current_code.count("if '")
    if handler_count > 20:
        suggestions.append(f"Found {handler_count} conditional handlers   high duplication risk. Consider generalization.")
    
    return suggestions
