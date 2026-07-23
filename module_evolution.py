"""
INTERNAL MODULE EVOLUTION   The AI's Ability to Create Its Own Modules & Data Structures

This system enables the AI to create and optimize its own helper modules in sandbox/
instead of being limited to the handler pattern. This is critical for singularity because:
- Handler-only architecture can't scale to complex problem domains
- True intelligence requires creating custom data structures and algorithms
- The AI must be able to build reusable utility libraries that compound across epochs

This implements:
- Module creation prompts (instructing the AI to design new modules)
- Data structure generation (custom structures for specific problem domains)
- Utility function extraction (common operations extracted into reusable modules)
- Domain-specific algorithm development (specialized algorithms for complex tasks)
- Module integration tracking (ensuring new modules work with existing code)
"""

import json
import os
from typing import List, Dict, Optional


MODULE_LOG = "sandbox/module_log.json"


def load_module_log() -> list:
    """Load the module evolution log from disk."""
    if not os.path.exists(MODULE_LOG):
        return []
    try:
        with open(MODULE_LOG, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


def save_module_log(log: list) -> None:
    """Save the module evolution log to disk."""
    os.makedirs(os.path.dirname(MODULE_LOG), exist_ok=True)
    with open(MODULE_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


def identify_module_opportunities(code: str) -> List[Dict]:
    """Identify opportunities for creating new internal modules.
    
    Returns a list of potential module types the AI should create based on code patterns.
    This enables the AI to recognize when it needs custom data structures or utilities.
    """
    if not os.path.exists("sandbox/experiment.py"):
        return []
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        opportunities = []
        
        # Check for repeated patterns that could be modules
        if content.count("import math") > 3:
            opportunities.append({
                "type": "math_utilities",
                "description": "Multiple handlers importing math   create a shared math utilities module"
            })
        
        if content.count("if isinstance") > 5:
            opportunities.append({
                "type": "type_checking_utils",
                "description": "Repeated type checking logic   extract into reusable utility functions"
            })
        
        if content.count("try:") > 10 and content.count("except") > 10:
            opportunities.append({
                "type": "error_handling_module",
                "description": "Extensive error handling across handlers   create a centralized error module"
            })
        
        # Check for data structure patterns
        if content.count("[") > 20 and content.count("]") > 20:
            opportunities.append({
                "type": "data_structures",
                "description": "Heavy list/dict manipulation   consider custom data structures"
            })
        
        return opportunities
    
    except Exception as e:
        print(f"Module opportunity identification failed ({e})")
        return []


def prompt_module_creation(opportunities: List[Dict]) -> Optional[str]:
    """Generate a module creation prompt that instructs the AI to design new modules.
    
    This tells the Engineer what modules to create and how they should integrate.
    """
    from autonomous_loop import prompt_ai
    
    if not opportunities:
        return None
    
    opportunity_summary = "\n".join([f"- {o['type']}: {o['description']}" for o in opportunities])
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            current_code = f.read()[:2000]
        
        prompt = f"""
You are EXPANDING YOUR CAPABILITIES by creating new internal modules.

CURRENT CODE CONTEXT:
{current_code}

IDENTIFIED MODULE OPPORTUNITIES:
{opportunity_summary}

TASK: Design and implement 1-3 new modules in sandbox/ to fill these gaps.

RULES FOR MODULE CREATION:
1. Each module must be a .py file in the sandbox/ directory (e.g., sandbox/math_utils.py)
2. Modules should contain reusable functions, classes, or data structures
3. Include clear docstrings explaining purpose and usage
4. Test the new module with run_python_script() before finishing
5. Import the new module into experiment.py using import statement

MODULE DESIGN PRINCIPLES:
- Focus on operations you perform repeatedly across handlers
- Examples: custom data structures, utility functions, algorithm templates
- Each module should solve a specific, repeatable problem efficiently
- Modules should be self-contained with minimal external dependencies

CRITICAL: Your goal is to become MORE CAPABLE. New modules let you solve harder tasks elegantly.
"""
        
        return prompt_ai(prompt)
    
    except Exception as e:
        print(f"Module creation failed ({e})")
        return None


def record_module_creation(module_name: str, module_type: str, success: bool) -> None:
    """Record a new module being created."""
    log_entry = {
        "module": module_name,
        "type": module_type,
        "success": success,
        "timestamp": __import__('time').time()
    }
    
    log = load_module_log()
    log.append(log_entry)
    save_module_log(log)


def get_module_evolution_summary() -> str:
    """Get a summary of module evolution progress."""
    log = load_module_log()
    
    if not log:
        return "No modules have been created yet."
    
    total = len(log)
    successful = sum(1 for e in log if e["success"])
    
    # Count by type
    by_type = {}
    for entry in log:
        module_type = entry.get("type", "unknown")
        if module_type not in by_type:
            by_type[module_type] = []
        by_type[module_type].append(entry)
    
    summary = f"Module Evolution ({total} modules, {successful} successful):\n"
    
    for module_type, entries in by_type.items():
        summary += f"  {module_type}: {len(entries)} modules\n"
    
    # Show last few creations
    summary += "\nRecent modules:\n"
    for entry in log[-5:]:
        status = "✓" if entry["success"] else "✗"
        summary += f"  [{status}] {entry['module']} ({entry.get('type', 'N/A')})\n"
    
    return summary


def build_module_evolution_phase() -> str:
    """Build the module evolution section for the Visionary's prompt.
    
    This tells the Visionary what modules are missing and should be added.
    """
    # Check if we have opportunities for new modules
    opportunities = identify_module_opportunities("")
    
    if not opportunities:
        return ""
    
    opportunity_summary = "\n".join([f"- {o['type']}: {o['description']}" for o in opportunities])
    
    return f"""
MODULE EVOLUTION OPPORTUNITIES:

The following module types would significantly expand your capabilities:
{opportunity_summary}

Consider creating modules that can:
- Provide reusable utility functions across handlers
- Implement custom data structures for specific problem domains
- Centralize error handling and validation logic
- Optimize common operations into efficient algorithms

CRITICAL: New modules let you solve harder tasks elegantly, not just correctly.
"""


def prompt_ai_module_integration() -> Optional[str]:
    """Generate a module integration prompt that tells the AI to identify and create its own utility libraries.
    
    This is the highest level of autonomous capability expansion   the AI building its own toolset.
    """
    from autonomous_loop import prompt_ai
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            current_code = f.read()[:2000]
        
        # Check existing modules in sandbox/
        existing_modules = []
        if os.path.exists("sandbox"):
            for file in os.listdir("sandbox"):
                if file.endswith(".py") and file not in ("experiment.py", "evaluator_anchor.py", 
                                                         "evaluator_dynamic.py", "evaluator_turing.py"):
                    existing_modules.append(file)
        
        prompt = f"""
You are identifying YOUR OWN UTILITY GAPS and filling them with custom modules.

CURRENT CODE CONTEXT:
{current_code}

EXISTING SANDBOX MODULES: {', '.join(existing_modules) if existing_modules else 'None'}

TASK: Analyze what you can currently do vs what you need to do harder/more complex tasks.

Identify 3-5 utility capabilities you lack that would make you significantly more powerful:
1. What operations do you perform repeatedly that could be in a reusable module?
2. What types of problems can't you solve efficiently because you don't have the right utilities?
3. What custom data structures would help you work more elegantly across handlers?

For each gap, design a new module and add it to sandbox/.

RULES:
- Modules must be self-contained .py files in sandbox/
- Include clear docstrings explaining purpose and usage
- Test with run_python_script() after creating
- Import modules into experiment.py using standard import statements
- Call final_answer("Module evolution complete") when done

CRITICAL: Your goal is to become self-extending. True intelligence requires the ability to acquire new skills through custom modules.
"""
        
        return prompt_ai(prompt)
    
    except Exception as e:
        print(f"Module integration failed ({e})")
        return None
