"""
Self directed curriculum module for dynamic problem generation and task ordering.

Detects knowledge gaps and sequences custom tasks to push model capabilities.
"""

import json
import os
from typing import List, Dict, Optional


CURRICULUM_LOG = "sandbox/curriculum_log.json"


def load_curriculum_log() -> list:
    """Load the curriculum tracking log from disk."""
    if not os.path.exists(CURRICULUM_LOG):
        return []
    try:
        with open(CURRICULUM_LOG, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


def save_curriculum_log(log: list) -> None:
    """Save the curriculum tracking log to disk."""
    os.makedirs(os.path.dirname(CURRICULUM_LOG), exist_ok=True)
    with open(CURRICULUM_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


def detect_knowledge_gaps(current_tasks: List[str], solved_tasks: List[str]) -> List[Dict]:
    """Detect gaps in the AI's knowledge based on what it can/can't solve.
    
    Returns a list of knowledge areas that need development.
    This enables self-directed curriculum generation.
    """
    # Define known problem domains and their sub-skills
    domain_skills = {
        "algorithms": ["sorting", "searching", "graph_traversal", "dynamic_programming"],
        "data_structures": ["trees", "graphs", "hash_tables", "stacks_queues"],
        "mathematics": ["number_theory", "linear_algebra", "combinatorics", "statistics"],
        "logic": ["boolean_logic", "predicate_logic", "constraint_satisfaction", "reasoning"],
        "optimization": ["greedy", "dynamic_programming", "backtracking", "heuristic_search"]
    }
    
    # Analyze what the AI has solved vs what it struggles with
    gaps = []
    
    for domain, skills in domain_skills.items():
        domain_solved = 0
        domain_total = len(skills)
        
        # Check which domains are represented in current tasks
        for task in current_tasks:
            task_lower = task.lower()
            for skill in skills:
                if skill.replace('_', ' ') in task_lower or skill in task_lower:
                    domain_solved += 1
                    break
        
        # If the AI hasn't explored a domain much, it's a gap
        if domain_solved < domain_total * 0.3:
            gaps.append({
                "domain": domain,
                "coverage": f"{domain_solved}/{domain_total} skills",
                "priority": "high" if domain_solved == 0 else "medium"
            })
    
    return gaps


def generate_self_directed_tasks(gaps: List[Dict], current_difficulty: str) -> List[List]:
    """Generate novel tasks targeting identified knowledge gaps.
    
    Returns a list of new tasks designed to fill specific knowledge areas.
    This is the core of self-directed curriculum   the AI teaching itself.
    """
    new_tasks = []
    
    # Define task templates for each domain
    domain_templates = {
        "algorithms": [
            ["Binary Search Optimization", [1, 2, 3, 4, 5], 3],
            ["Merge Two Sorted Lists", [[1, 3, 5], [2, 4, 6]], [1, 2, 3, 4, 5, 6]],
            ["Quick Sort Partition", [5, 3, 8, 1, 9], None]  # Returns pivot position
        ],
        "data_structures": [
            ["Stack Reversal", [[1, 2, 3, 4]], [4, 3, 2, 1]],
            ["Queue Min Element", [[1, 3, 2, 5]], 1],
            ["Tree Depth Calculation", [[1, 2, 3, None, 4]], 2]
        ],
        "mathematics": [
            ["GCD of Multiple Numbers", [12, 18, 24], 6],
            ["Matrix Transpose", [[1, 2], [3, 4]], [[1, 3], [2, 4]]],
            ["Polynomial Evaluation", [1, 2, 3, 4], 10]  # 1 + 2x + 3x² + 4x³ at x1
        ],
        "logic": [
            ["Boolean Expression Simplification", ["A AND B OR NOT A"], "True"],
            ["Predicate Logic Validation", [[1, 2, 3], "all_greater_than_0"], True],
            ["Constraint Satisfaction Solver", [[1, 2, 3], 6], [1, 2, 3]]
        ],
        "optimization": [
            ["Knapsack Problem Small", [10, 20, 30], [50]],  # Max value within weight limit
            ["Traveling Salesman Mini", [[0, 1, 2], [1, 0, 3], [2, 3, 0]], 4],
            ["Activity Selection Greedy", [[1, 4], [2, 3], [5, 6]], 2]
        ]
    }
    
    # Generate tasks targeting gaps based on current difficulty
    for gap in gaps:
        domain = gap["domain"]
        
        if domain in domain_templates:
            templates = domain_templates[domain]
            
            # Select appropriate template based on difficulty
            if current_difficulty == "beginner":
                new_tasks.extend(templates[:2])  # Simpler tasks
            elif current_difficulty == "intermediate":
                new_tasks.extend(templates)  # All templates
            else:  # advanced/expert generate harder variants
                for template in templates:
                    prompt, inputs, expected = template
                    
                    # Generate harder variant
                    if isinstance(inputs, list):
                        # Add more elements or larger numbers
                        hard_inputs = [x * 10 if isinstance(x, int) else x + ["extra"] for x in inputs]
                        new_tasks.append([f"{prompt} (Advanced)", hard_inputs, expected])
    
    return new_tasks


def add_self_directed_task(prompt: str, inputs: list, expected) -> bool:
    """Add a self-directed task to the dynamic dataset.
    
    Returns True if successfully added, False if duplicate or error.
    This ensures tasks are properly formatted and don't conflict with existing ones.
    """
    try:
        # Load current dataset
        if os.path.exists("sandbox/dynamic_dataset.json"):
            with open("sandbox/dynamic_dataset.json", 'r') as f:
                dataset = json.load(f)
        else:
            dataset = []
        
        # Check for duplicates
        existing_prompts = [t[0].lower() if isinstance(t, (list, tuple)) and len(t) >= 1 else "" for t in dataset]
        if prompt.lower() in existing_prompts:
            return False
        
        # Add new task
        dataset.append([prompt, inputs, expected])
        
        # Save updated dataset
        with open("sandbox/dynamic_dataset.json", 'w') as f:
            json.dump(dataset, f, indent=4)
        
        return True
    
    except Exception as e:
        print(f"Self-directed task addition failed ({e})")
        return False


def record_curriculum_progress(epoch_num: int, gaps_filled: List[str], new_tasks_added: int) -> None:
    """Record curriculum progress for tracking autonomous learning."""
    log_entry = {
        "epoch": epoch_num,
        "gaps_filled": gaps_filled,
        "new_tasks_added": new_tasks_added,
        "timestamp": __import__('time').time()
    }
    
    log = load_curriculum_log()
    log.append(log_entry)
    save_curriculum_log(log)


def get_curriculum_summary() -> str:
    """Get a summary of self-directed curriculum progress."""
    log = load_curriculum_log()
    
    if not log:
        return "No self-directed curriculum data yet."
    
    total_epochs = len(log)
    total_gaps_filled = sum(len(e.get("gaps_filled", [])) for e in log)
    total_new_tasks = sum(e.get("new_tasks_added", 0) for e in log)
    
    summary = f"Self-Directed Curriculum ({total_epochs} epochs):\n"
    summary += f"Gaps Filled: {total_gaps_filled}\n"
    summary += f"New Tasks Generated: {total_new_tasks}\n"
    
    # Show most recently filled gaps
    if log[-1].get("gaps_filled"):
        summary += f"\nRecent Gaps Filled:\n"
        for gap in log[-1]["gaps_filled"][:3]:
            summary += f"  {gap}\n"
    
    return summary


def build_self_directed_curriculum_phase() -> str:
    """Build the self-directed curriculum section for the Visionary's prompt.
    
    This tells the AI what knowledge gaps to target and how to generate tasks.
    """
    # Detect current knowledge gaps
    if not os.path.exists("sandbox/dynamic_dataset.json"):
        return ""
    
    try:
        with open("sandbox/dynamic_dataset.json", 'r') as f:
            dataset = json.load(f)
        
        current_tasks = [t[0] for t in dataset if isinstance(t, (list, tuple)) and len(t) >= 1]
        gaps = detect_knowledge_gaps(current_tasks, [])
        
        if not gaps:
            return ""
        
        gap_summary = "\n".join([f"- {g['domain']} ({g['coverage']}, priority: {g['priority']})" for g in gaps])
        
        difficulty_tier = "beginner"  # Could be determined from adaptive_difficulty
        
        new_tasks = generate_self_directed_tasks(gaps, difficulty_tier)
        
        return f"""
SELF-DIRECTED CURRICULUM GENERATION:

IDENTIFIED KNOWLEDGE GAPS:
{gap_summary}

GENERATED TASKS TO FILL GAPS: {len(new_tasks)} novel tasks created

TASK GENERATION RULES:
1. Each new task must target a specific knowledge gap
2. Tasks should be progressively harder within each domain
3. Avoid duplicating existing prompts   create truly novel problems
4. Include tasks that require genuine understanding, not just pattern matching

CRITICAL: Your goal is to teach yourself what you lack. 
True intelligence identifies its own gaps and fills them autonomously.
"""
    except Exception as e:
        print(f"Self-directed curriculum build failed ({e})")
    
    return ""


def prompt_ai_curriculum_design() -> Optional[str]:
    """Generate a curriculum design prompt that tells the AI to create its own learning path.
    
    This is the highest level of autonomous education   the AI designing what it should learn next.
    """
    from autonomous_loop import prompt_ai
    
    try:
        # Detect knowledge gaps
        if os.path.exists("sandbox/dynamic_dataset.json"):
            with open("sandbox/dynamic_dataset.json", 'r') as f:
                dataset = json.load(f)
            
            current_tasks = [t[0] for t in dataset if isinstance(t, (list, tuple)) and len(t) >= 1]
            gaps = detect_knowledge_gaps(current_tasks, [])
            
            prompt = f"""
You are DESIGNING YOUR OWN CURRICULUM for autonomous learning.

CURRENT KNOWLEDGE GAPS:
{chr(10).join([f'- {g["domain"]} ({g["coverage"]}, priority: {g["priority"]})' for g in gaps])}

TASK: Create a structured learning path that addresses these gaps systematically.

RULES FOR CURRICULUM DESIGN:
1. Order tasks from easiest to hardest within each domain
2. Include prerequisite knowledge before tackling complex problems
3. Design progressive challenges that build on previous solutions
4. Focus on depth of understanding, not just breadth of coverage

EXAMPLE LEARNING PATH:
- Step 1: Basic sorting algorithm (understand the concept)
- Step 2: Optimized sorting with different constraints (apply knowledge)
- Step 3: Custom sorting for novel data structures (generalize)

CRITICAL: Your goal is to design a curriculum that maximizes learning efficiency. 
True intelligence knows what it doesn't know and creates a path to fill those gaps.
"""
        
        return prompt_ai(prompt)
    
    except Exception as e:
        print(f"Curriculum design failed ({e})")
        return None
