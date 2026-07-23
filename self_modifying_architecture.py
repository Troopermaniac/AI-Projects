"""
Self modifying architecture module for refactoring and generalizing handlers.

Enables code refactoring, handler pattern recognition, and code consolidation.
"""

import json
import os
from typing import List, Dict, Optional


REFACTOR_LOG = "sandbox/refactor_log.json"


def load_refactor_log() -> list:
    """Load the refactoring log from disk."""
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
    """Save the refactoring log to disk."""
    os.makedirs(os.path.dirname(REFACTOR_LOG), exist_ok=True)
    with open(REFACTOR_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


def extract_handler_patterns(code: str) -> List[Dict]:
    """Extract handler patterns from experiment.py code.
    
    Returns a list of handlers with their prompt keywords and logic structure.
    This enables the AI to see which handlers share common patterns.
    """
    if not os.path.exists("sandbox/experiment.py"):
        return []
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all handler definitions (if 'prompt' in prompt.lower())
        handlers = []
        lines = content.split('\n')
        current_handler = None
        
        for i, line in enumerate(lines):
            if "if '" in line and "in prompt.lower()" in line:
                try:
                    # Extract handler name from the pattern
                    parts = line.split("if '")
                    if len(parts) > 1:
                        handler_name = parts[1].split("'")[0]
                        current_handler = {
                            "name": handler_name,
                            "line": i,
                            "prompts": [],
                            "logic_lines": []
                        }
                        handlers.append(current_handler)
                except Exception:
                    pass
            
            if current_handler and len(current_handler["logic_lines"]) < 10:
                # Collect logic lines until we hit the next handler or return
                stripped = line.strip()
                if stripped.startswith("if '") and "in prompt.lower()" in stripped:
                    # New handler started   stop collecting
                    pass
                elif stripped.startswith("return"):
                    current_handler["logic_lines"].append(line)
                    break
                else:
                    current_handler["logic_lines"].append(line)
        
        return handlers
    
    except Exception as e:
        print(f"Handler pattern extraction failed ({e})")
        return []


def find_similar_handlers(handlers: List[Dict]) -> List[Dict]:
    """Find handlers that share common patterns and could be merged.
    
    Returns a list of groups of similar handlers.
    """
    if len(handlers) < 3:
        return []
    
    # Group by logic complexity (number of lines)
    by_complexity = {}
    for h in handlers:
        complexity = len(h.get("logic_lines", []))
        if complexity not in by_complexity:
            by_complexity[complexity] = []
        by_complexity[complexity].append(h)
    
    # Find groups with similar complexity that might share patterns
    similar_groups = []
    for complexity, group in by_complexity.items():
        if len(group) >= 2 and complexity <= 5:
            # These handlers are simple enough to potentially merge
            similar_groups.append({
                "type": "merge_candidates",
                "handlers": [h["name"] for h in group],
                "complexity": complexity
            })
    
    return similar_groups


def prompt_self_modification(current_code: str, task_prompt: str) -> Optional[str]:
    """Generate a self-modification prompt that instructs the AI to rewrite its own code.
    
    This is the highest level of architectural evolution: the AI rewriting its own logic
    instead of just appending new handlers.
    """
    from autonomous_loop import prompt_ai
    
    # Extract handler patterns for context
    handlers = extract_handler_patterns(current_code)
    similar_groups = find_similar_handlers(handlers) if handlers else []
    
    if not similar_groups:
        return None
    
    merge_candidates = "\n".join([f"- {g['handlers']} (complexity: {g['complexity']})" for g in similar_groups])
    
    try:
        prompt = f"""
You are REWRITING YOUR OWN ARCHITECTURE.

CURRENT CODE:
{current_code[:2000]}

MERGE CANDIDATES (handlers that could be consolidated):
{merge_candidates}

TASK: Analyze these handlers and identify opportunities to merge or refactor them into more elegant, generalized solutions.

RULES FOR SELF-MODIFICATION:
1. Identify handlers that solve similar problems and combine them into one generalized handler
2. Extract common patterns across multiple handlers and create a single reusable function
3. Replace verbose logic with concise, efficient implementations
4. Preserve ALL existing functionality   do not break any working handlers
5. Focus on reducing code complexity while maintaining correctness

EXAMPLE: If you have separate handlers for "Is Even" and "Is Odd", combine them into one "Parity Check" handler that returns True/False based on the input.

CRITICAL: Your goal is to become MORE ELEGANT, not just more capable. True intelligence recognizes patterns and generalizes.
"""
        
        return prompt_ai(prompt)
    except Exception as e:
        print(f"Self-modification failed ({e})")
        return None


def record_self_modification(original_code: str, modified_code: str, improvement_type: str) -> None:
    """Record a self-modification event for tracking."""
    log_entry = {
        "timestamp": __import__('time').time(),
        "improvement_type": improvement_type,
        "original_size": len(original_code),
        "modified_size": len(modified_code),
        "size_change": len(modified_code) - len(original_code)
    }
    
    log = load_refactor_log()
    log.append(log_entry)
    save_refactor_log(log)


def get_self_modification_summary() -> str:
    """Get a summary of self-modification progress."""
    log = load_refactor_log()
    
    if not log:
        return "No self-modifications recorded yet."
    
    total = len(log)
    improvements = sum(1 for e in log if e["size_change"] < 0)  # Size decreased
    
    summary = f"Self-Modification ({total} changes, {improvements} optimizations):\n"
    
    for entry in log[-5:]:  # Show last 5
        direction = "↓" if entry["size_change"] < 0 else "↑"
        summary += f"  [{direction}] {entry['improvement_type']}: {entry['size_change']} chars\n"
    
    return summary


def build_self_modification_phase() -> str:
    """Build the self-modification section for the Visionary's prompt.
    
    This tells the Visionary what architectural improvements are needed.
    """
    # Check if we have enough handlers to warrant refactoring
    if not os.path.exists("sandbox/experiment.py"):
        return ""
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Count handlers
        handler_count = code.count("if '")
        
        if handler_count > 10:  # Threshold for refactoring
            return f"""
SELF-MODIFICATION OPPORTUNITIES:

Your architecture currently has {handler_count} handlers. This is getting large enough that
refactoring could significantly improve elegance and efficiency.

Consider:
- Merging similar handlers into generalized solutions
- Extracting common patterns into reusable functions
- Replacing verbose logic with concise implementations

CRITICAL: Your goal is to become MORE ELEGANT, not just more capable.
"""
    except Exception as e:
        print(f"Self-modification build failed ({e})")
    
    return ""


def prompt_ai_self_refactor() -> Optional[str]:
    """Generate a self-refactoring prompt that tells the AI to identify and fix its own code bloat.
    
    This is the highest level of autonomous architectural improvement.
    """
    from autonomous_loop import prompt_ai
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            current_code = f.read()[:3000]
        
        prompt = f"""
You are identifying YOUR OWN ARCHITECTURAL WEAKNESSES and fixing them.

CURRENT CODE:
{current_code}

TASK: Analyze this code and identify opportunities for self-improvement:
1. Are there redundant handlers that could be merged?
2. Is there common logic across multiple handlers that should be extracted into a reusable function?
3. Can any handler be simplified or made more efficient?

RULES:
- Preserve ALL existing functionality ,  do not break working code
- Focus on elegance and efficiency, not just correctness
- Extract patterns and generalize where possible
- Call final_answer("Self-refactoring complete") when done

CRITICAL: Your goal is to become MORE ELEGANT. True intelligence recognizes patterns and generalizes.
"""
        
        return prompt_ai(prompt)
    except Exception as e:
        print(f"Self-refactoring failed ({e})")
        return None
