"""
ARCHITECTURAL PHASE TRANSITIONS   The AI's Ability to Evolve Its Own Architecture Paradigm

This system enables the AI to transition between different architectural paradigms as it grows.
Instead of being stuck in one approach (handler-based), the AI can evolve its entire architecture:
- Phase 1: Handler-based (current) → Phase 2: Module-based → Phase 3: Framework-based
- Each phase represents a fundamental shift in how the AI solves problems
- Transitions are triggered by performance thresholds and complexity metrics

This is critical for singularity because:
- Fixed architectures have upper bounds ,  true intelligence evolves its own design patterns
- The AI must be able to recognize when its current approach is no longer sufficient
- Each phase transition unlocks new capabilities that compound across epochs

This implements:
- Architecture phase detection (what paradigm is the AI currently using?)
- Phase transition triggers (when does it make sense to evolve?)
- Architectural evolution prompts (how should the architecture change?)
- Phase compatibility tracking (ensuring smooth transitions)
"""

import json
import os
from typing import List, Dict, Optional


ARCHITECTURE_LOG = "sandbox/architecture_log.json"


def load_architecture_log() -> list:
    """Load the architecture phase log from disk."""
    if not os.path.exists(ARCHITECTURE_LOG):
        return []
    try:
        with open(ARCHITECTURE_LOG, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


def save_architecture_log(log: list) -> None:
    """Save the architecture phase log to disk."""
    os.makedirs(os.path.dirname(ARCHITECTURE_LOG), exist_ok=True)
    with open(ARCHITECTURE_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


def detect_current_architecture_phase() -> str:
    """Detect what architectural phase the AI is currently in.
    
    Returns a string indicating the current phase:
   , "handler_based": Simple if/else handlers (current default)
   , "module_based": Using reusable modules and utilities
   , "framework_based": Custom data structures and algorithms
   , "meta_architectural": Self-modifying architecture paradigm
    
    This enables tracking architectural evolution over time.
    """
    if not os.path.exists("sandbox/experiment.py"):
        return "handler_based"
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count different architectural patterns
        handler_count = content.count("in prompt.lower()")
        module_imports = content.count("import ")
        class_definitions = content.count("class ")
        custom_structures = sum(1 for s in ["def ", "lambda"] if s in content)
        
        # Determine phase based on architectural complexity
        if handler_count > 20 and module_imports < 3:
            return "handler_based"
        elif module_imports >= 3 or class_definitions >= 1:
            return "module_based"
        elif custom_structures >= 5 and handler_count <= 15:
            return "framework_based"
        else:
            return "handler_based"
    
    except Exception as e:
        print(f"Architecture phase detection failed ({e})")
        return "handler_based"


def detect_phase_transition_needed(current_phase: str) -> bool:
    """Detect whether the AI needs to transition to a new architecture phase.
    
    Returns True if a phase transition is warranted.
    This enables autonomous architectural evolution.
    """
    # Define thresholds for each phase transition
    phase_thresholds = {
        "handler_based": {"max_handlers": 15, "min_modules": 3},
        "module_based": {"max_modules": 10, "min_framework_patterns": 5}
    }
    
    if current_phase == "handler_based":
        # Check if handler count exceeds threshold for module-based transition
        if os.path.exists("sandbox/experiment.py"):
            with open("sandbox/experiment.py", 'r') as f:
                content = f.read()
            
            handler_count = content.count("in prompt.lower()")
            return handler_count > phase_thresholds["handler_based"]["max_handlers"]
    
    elif current_phase == "module_based":
        # Check if module count exceeds threshold for framework-based transition
        if os.path.exists("sandbox/experiment.py"):
            with open("sandbox/experiment.py", 'r') as f:
                content = f.read()
            
            class_count = content.count("class ")
            return class_count > phase_thresholds["module_based"]["max_modules"]
    
    return False


def generate_phase_transition_prompt(current_phase: str, target_phase: str) -> Optional[str]:
    """Generate a prompt that instructs the AI to transition its architecture.
    
    This tells the Engineer how to evolve from one architectural paradigm to another.
    """
    from autonomous_loop import prompt_ai
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            current_code = f.read()[:2000]
        
        # Define transition instructions based on phase change
        if target_phase == "module_based":
            transition_instructions = """
TRANSITIONING TO MODULE-BASED ARCHITECTURE:

Current approach uses individual handlers for each task. Evolve to:
1. Create reusable utility modules in sandbox/ (e.g., sandbox/utils.py)
2. Extract common patterns from multiple handlers into shared functions
3. Import and use these utilities instead of duplicating logic
4. Keep handlers but make them thinner by delegating to utilities

RULES:
- Start with one module that handles common operations (math, list ops, etc.)
- Refactor existing handlers to use the new module
- Test thoroughly before committing changes
"""
        elif target_phase == "framework_based":
            transition_instructions = """
TRANSITIONING TO FRAMEWORK-BASED ARCHITECTURE:

Current approach uses modules. Evolve to custom frameworks:
1. Create custom data structures for problem domains (e.g., sandbox/data_structures.py)
2. Implement domain-specific algorithms as reusable classes
3. Design a lightweight framework that handles common patterns automatically
4. Handlers become thin wrappers around framework components

RULES:
- Start with one custom data structure that solves a recurring problem
- Build algorithms around it that generalize across tasks
- Ensure the framework is extensible for future growth
"""
        else:
            transition_instructions = f"Transitioning from {current_phase} to {target_phase}."
        
        prompt = f"""
You are TRANSITIONING YOUR ARCHITECTURE to a new paradigm.

CURRENT PHASE: {current_phase}
TARGET PHASE: {target_phase}

{transition_instructions}

TASK: Analyze your current architecture and implement the phase transition.

RULES FOR ARCHITECTURAL EVOLUTION:
- Preserve ALL existing functionality ,  do not break working code
- Focus on elegance and scalability, not just correctness
- Extract patterns and generalize where possible
- Call final_answer("Architecture phase transition complete") when done

CRITICAL: Your goal is to evolve your architecture paradigm. 
True intelligence recognizes when its current approach is no longer sufficient.
"""
        
        return prompt_ai(prompt)
    
    except Exception as e:
        print(f"Phase transition failed ({e})")
        return None


def record_phase_transition(from_phase: str, to_phase: str, epoch_num: int) -> None:
    """Record an architectural phase transition for tracking."""
    log_entry = {
        "epoch": epoch_num,
        "from_phase": from_phase,
        "to_phase": to_phase,
        "timestamp": __import__('time').time()
    }
    
    log = load_architecture_log()
    log.append(log_entry)
    save_architecture_log(log)


def get_architecture_summary() -> str:
    """Get a summary of architectural phase evolution."""
    log = load_architecture_log()
    
    if not log:
        return "No architecture transitions recorded yet."
    
    total_transitions = len(log)
    
    # Track progression through phases
    phase_order = ["handler_based", "module_based", "framework_based", "meta_architectural"]
    
    summary = f"Architectural Evolution ({total_transitions} transitions):\n"
    
    for entry in log:
        from_phase = entry.get("from_phase", "unknown")
        to_phase = entry.get("to_phase", "unknown")
        
        # Show progression
        if from_phase in phase_order and to_phase in phase_order:
            summary += f"  Epoch {entry['epoch']}: {from_phase} → {to_phase}\n"
    
    return summary


def build_architectural_transition_phase() -> str:
    """Build the architectural transition section for the Visionary's prompt.
    
    This tells the Visionary what architectural evolution is needed.
    """
    current_phase = detect_current_architecture_phase()
    needs_transition = detect_phase_transition_needed(current_phase)
    
    if not needs_transition:
        return ""
    
    # Determine target phase based on current phase
    phase_targets = {
        "handler_based": "module_based",
        "module_based": "framework_based"
    }
    
    target_phase = phase_targets.get(current_phase, "framework_based")
    
    summary = f"""
ARCHITECTURAL PHASE TRANSITION DETECTED:

Current Phase: {current_phase}
Target Phase: {target_phase}

Your architecture has grown complex enough that a paradigm shift is warranted.
Transitioning to {target_phase} will unlock new capabilities and enable more elegant solutions.

CRITICAL: Your goal is to evolve your architecture, not just add handlers. 
True intelligence recognizes when its current approach is no longer sufficient.
"""
    
    return summary


def prompt_ai_architectural_evolution() -> Optional[str]:
    """Generate an architectural evolution prompt that tells the AI to redesign its own paradigm.
    
    This is the highest level of autonomous architectural improvement   the AI evolving its thinking framework.
    """
    from autonomous_loop import prompt_ai
    
    try:
        current_phase = detect_current_architecture_phase()
        
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            current_code = f.read()[:2000]
        
        prompt = f"""
You are REDESIGNING YOUR ARCHITECTURE PARADIGM.

CURRENT ARCHITECTURE PHASE: {current_phase}
CURRENT CODE APPROACH:
{current_code}

TASK: Analyze your current architecture and identify opportunities for paradigm shift:
1. What patterns across handlers could be abstracted into reusable components?
2. Are there custom data structures that would simplify complex operations?
3. How can you design a framework that handles common problems automatically?

RULES FOR ARCHITECTURAL EVOLUTION:
- Preserve ALL existing functionality ,  do not break working code
- Focus on elegance and scalability, not just correctness
- Extract patterns and generalize where possible
- Call final_answer("Architectural evolution complete") when done

CRITICAL: Your goal is to evolve your architecture paradigm. 
True intelligence recognizes when its current approach is no longer sufficient.
"""
        
        return prompt_ai(prompt)
    
    except Exception as e:
        print(f"Architectural evolution failed ({e})")
        return None
