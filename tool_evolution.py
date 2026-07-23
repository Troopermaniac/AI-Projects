"""
Tool evolution module for autonomous capability expansion.

Identifies capability gaps, designs missing tools, integrates them,
and verifies tool execution.
"""

import json
import os
from typing import List, Dict, Optional


TOOL_LOG = "sandbox/tool_log.json"


def load_tool_log() -> list:
    """Load the tool evolution log from disk."""
    if not os.path.exists(TOOL_LOG):
        return []
    try:
        with open(TOOL_LOG, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


def save_tool_log(log: list) -> None:
    """Save the tool evolution log to disk."""
    os.makedirs(os.path.dirname(TOOL_LOG), exist_ok=True)
    with open(TOOL_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


def identify_tool_gaps(current_tools: List[str], unsolved_tasks: List[str]) -> List[Dict]:
    """Identify gaps in the current toolset based on unsolved tasks.
    
    Returns a list of potential new tools the AI should create.
    """
    if not unsolved_tasks:
        return []
    
    # Define what each existing tool can do
    known_tool_capabilities = {
        "read_file": ["file", "content", "read", "open"],
        "write_file": ["file", "content", "write", "save", "create"],
        "run_python_script": ["python", "script", "execute", "compute", "calculate"],
        "run_git_command": ["git", "commit", "branch", "repository", "version"],
        "search_arxiv": ["paper", "research", "arxiv", "academic", "publication"],
        "update_memory": ["memory", "learn", "store", "recall"],
        "check_syntax": ["syntax", "validate", "error", "lint"],
        "profile_python_script": ["profile", "performance", "time", "benchmark"]
    }
    
    # Define what types of operations exist that might need new tools
    operation_types = {
        "file_operations": ["copy", "move", "delete", "compare", "diff", "merge"],
        "data_processing": ["filter", "sort", "aggregate", "transform", "parse"],
        "pattern_matching": ["regex", "search", "find", "match", "extract"],
        "code_analysis": ["analyze", "refactor", "optimize", "inspect", "audit"],
        "testing": ["test", "verify", "assert", "validate", "check"],
        "generation": ["generate", "create", "produce", "synthesize"]
    }
    
    # Analyze task keywords and map to operation types
    task_analysis = []
    for task in unsolved_tasks:
        task_lower = task.lower()
        task_words = set(task_lower.split())
        
        matched_ops = []
        for op_type, ops in operation_types.items():
            if any(op in task_lower or any(kw in task_words for kw in ops) for op in ops):
                matched_ops.append(op_type)
        
        # If a task doesn't match known tool capabilities, it's a potential gap
        has_known_tool = False
        for cap_list in known_tool_capabilities.values():
            if any(cap in task_lower or any(kw in task_words for kw in cap_list) for cap in cap_list):
                has_known_tool = True
        
        if not has_known_tool:
            task_analysis.append({
                "task": task,
                "matched_operations": matched_ops if matched_ops else ["unknown"],
                "gap_type": "new_capability"
            })
    
    # Return unique gap types with associated tasks
    seen_types = set()
    for ta in task_analysis:
        gt = ta["gap_type"]
        if gt not in seen_types:
            seen_types.add(gt)
    
    return [{"task": t, "gap_type": "new_capability", "operations": a.get("matched_operations", [])} 
            for t, a in [(ta["task"], ta) for ta in task_analysis]][:5]


def prompt_ai_tool_creation(gaps: List[Dict]) -> Optional[str]:
    """Generate a tool creation prompt for the AI.
    
    This instructs the Engineer to design and integrate new tools.
    """
    from autonomous_loop import prompt_ai
    
    if not gaps:
        return None
    
    gap_summary = "\n".join([f"- {g['task']} (needs: {g.get('potential_gap', 'unknown')})" for g in gaps])
    
    try:
        with open("agent_tools.py", 'r', encoding='utf-8') as f:
            current_tools = f.read()
        
        prompt = f"""
You are expanding YOUR OWN CAPABILITIES by creating new tools.

CURRENT TOOLSET:
{current_tools[:2000]}

IDENTIFIED GAPS:
{gap_summary}

TASK: Design and implement 1-3 new tools to fill these gaps.

RULES FOR NEW TOOLS:
1. Each tool must be a @tool-decorated function (use the same pattern as existing tools)
2. Tools should be self-contained   no external dependencies beyond standard library
3. Include clear docstrings explaining what each tool does and its parameters
4. Write new tools to agent_tools.py using write_file()
5. Test the new tool with run_python_script() before finishing

TOOL DESIGN PRINCIPLES:
- Focus on operations you frequently need but can't currently perform
- Examples: file comparison, pattern matching, data validation, code analysis
- Each tool should solve a specific, repeatable problem

CRITICAL: Your goal is to become MORE CAPABLE. New tools let you solve harder tasks.
"""
        
        return prompt_ai(prompt)
    except Exception as e:
        print(f"Tool creation failed ({e})")
        return None


def record_tool_creation(tool_name: str, success: bool, description: str) -> None:
    """Record a new tool being created."""
    log_entry = {
        "tool": tool_name,
        "success": success,
        "description": description[:500],
        "timestamp": __import__('time').time()
    }
    
    log = load_tool_log()
    log.append(log_entry)
    save_tool_log(log)


def get_tool_evolution_summary() -> str:
    """Get a summary of tool evolution progress."""
    log = load_tool_log()
    
    if not log:
        return "No tools have been created yet."
    
    total = len(log)
    successful = sum(1 for e in log if e["success"])
    
    summary = f"Tool Evolution ({total} tools, {successful} successful):\n"
    
    for entry in log[-5:]:  # Show last 5
        status = "✓" if entry["success"] else "✗"
        summary += f"  [{status}] {entry['tool']}: {entry.get('description', 'N/A')[:100]}\n"
    
    return summary


def build_tool_evolution_phase() -> str:
    """Build the tool evolution section for the Visionary's prompt.
    
    This tells the Visionary what capabilities are missing and should be added.
    """
    # Check if we have unsolved tasks that might need new tools
    if not os.path.exists("sandbox/dynamic_dataset.json"):
        return ""
    
    try:
        with open("sandbox/dynamic_dataset.json", "r") as f:
            dataset = json.load(f)
        
        unsolved = []
        for item in dataset:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                # We don't know which are solved without running evaluator
                # But we can check if the task type might need new tools
                prompt = item[0]
                unsolved.append(prompt)
        
        gaps = identify_tool_gaps([], unsolved[:5])
        
        if gaps:
            return f"""
TOOL EVOLUTION OPPORTUNITIES:

The following tasks may benefit from new tool capabilities:
{chr(10).join([f'- {g["task"]}' for g in gaps])}

Consider designing tools that can:
- Analyze code structure beyond syntax checking
- Compare file contents programmatically
- Extract patterns from multiple files
- Generate test cases automatically

CRITICAL: New tools expand what the AI can do autonomously.
"""
    except Exception as e:
        print(f"Tool evolution build failed ({e})")
    
    return ""


def prompt_ai_self_extension() -> Optional[str]:
    """Generate a self-extension prompt that tells the AI to identify and fill its own gaps.
    
    This is the highest level of autonomous capability expansion.
    """
    from autonomous_loop import prompt_ai
    
    try:
        with open("agent_tools.py", 'r', encoding='utf-8') as f:
            current_tools = f.read()
        
        prompt = f"""
You are identifying YOUR OWN CAPABILITY GAPS and filling them.

CURRENT TOOLSET:
{current_tools[:2000]}

TASK: Analyze what you can currently do vs what you need to do harder/more complex tasks.

Identify 3 capabilities you lack that would make you significantly more powerful:
1. What operations do you perform repeatedly that could be automated?
2. What types of problems can't you solve because you don't have the right tools?
3. What would help you work more efficiently across multiple files?

For each gap, design a new tool and add it to agent_tools.py.

RULES:
- Tools must use @tool decorator (from smolagents import tool)
- Include clear docstrings
- Test with run_python_script() after creating
- Call final_answer("Tool evolution complete") when done

CRITICAL: Your goal is to become self-extending. True intelligence requires the ability to acquire new skills.
"""
        
        return prompt_ai(prompt)
    except Exception as e:
        print(f"Self-extension failed ({e})")
        return None
