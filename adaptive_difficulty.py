"""
ADAPTIVE DIFFICULTY SCALING   The AI's Curriculum That Grows With It

This system implements progressive difficulty scaling for the dynamic dataset.
Instead of static tasks → AI solves them → done, this creates a curriculum that:
- Auto-scales task complexity based on current performance
- Generates harder variants of solved tasks automatically
- Creates a "curriculum" that pushes the AI toward increasingly complex problems
- Measures progress not just in correctness but in problem-solving depth

This is critical for singularity because:
- Fixed datasets have an upper bound ,  once solved, there's nothing left to learn
- True intelligence requires continuously escalating challenges
- The AI must be pushed beyond its comfort zone to evolve

This implements:
- Performance-based difficulty scaling (harder tasks when you're ready)
- Task variant generation (same concept, more complex inputs/operations)
- Curriculum tracking (measuring progress across difficulty tiers)
- Automatic escalation thresholds (when to push harder)
"""

import json
import os
from typing import List, Dict, Optional


DIFFICULTY_LOG = "sandbox/difficulty_log.json"
MASTERED_ARCHIVE = "sandbox/mastered_archive.json"


def load_mastered_archive() -> list:
    """Load the mastered task archive from disk."""
    if not os.path.exists(MASTERED_ARCHIVE):
        return []
    try:
        with open(MASTERED_ARCHIVE, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


def save_mastered_archive(archive: list) -> None:
    """Save the mastered task archive to disk."""
    os.makedirs(os.path.dirname(MASTERED_ARCHIVE), exist_ok=True)
    with open(MASTERED_ARCHIVE, 'w', encoding='utf-8') as f:
        json.dump(archive, f, indent=4)


def archive_mastered_tasks(mastered_prompts: List[str], epoch: int) -> int:
    """Archive solved/mastered tasks out of the active dynamic dataset.
    
    Returns the number of tasks successfully archived.
    """
    if not mastered_prompts:
        return 0

    dataset_path = "sandbox/dynamic_dataset.json"
    if not os.path.exists(dataset_path):
        return 0

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception:
        return 0

    mastered_set = {p.lower().strip() for p in mastered_prompts}
    remaining_dataset = []
    archived_items = []

    for item in dataset:
        prompt = item[0] if isinstance(item, (list, tuple)) and len(item) >= 1 else ""
        if prompt.lower().strip() in mastered_set:
            archived_items.append({
                "prompt": item[0],
                "inputs": item[1] if len(item) >= 2 else [],
                "expected": item[2] if len(item) >= 3 else None,
                "epoch_mastered": epoch,
                "timestamp": __import__('time').time()
            })
        else:
            remaining_dataset.append(item)

    if archived_items:
        archive = load_mastered_archive()
        archive.extend(archived_items)
        save_mastered_archive(archive)

        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(remaining_dataset, f, indent=4)

    return len(archived_items)


def rotate_active_dataset(epoch: int, current_tier: str, mastered_prompts: List[str]) -> Dict[str, int]:
    """Archive mastered tasks and inject new, higher-difficulty procedural tasks into the dynamic dataset."""
    archived_count = archive_mastered_tasks(mastered_prompts, epoch)
    added_count = 0

    if archived_count > 0:
        try:
            import problem_generator
            new_problems = problem_generator.generate_epoch_problems(epoch, n_problems=archived_count, difficulty_tier=current_tier)
            for p in new_problems:
                if p.get("test_cases"):
                    tc = p["test_cases"][0]
                    if add_adaptive_task(p["description"], tc["inputs"], tc["expected"]):
                        added_count += 1
        except Exception as e:
            print(f"Problem generator injection fallback ({e})")
            try:
                dataset_path = "sandbox/dynamic_dataset.json"
                if os.path.exists(dataset_path):
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        current_tasks = json.load(f)
                    variants = generate_harder_task_variants(current_tasks, current_tier)
                    for var in variants:
                        if add_adaptive_task(var[0], var[1], var[2]):
                            added_count += 1
            except Exception:
                pass

    return {"archived": archived_count, "added": added_count}


def inject_subtask_stepping_stones(stuck_prompt: str) -> bool:
    """Inject a simpler sub-task stepping stone when a main problem fails for multiple epochs."""
    if not stuck_prompt:
        return False
    try:
        sub_prompt = f"Simplified step: {stuck_prompt} (basic case)"
        dataset_path = "sandbox/dynamic_dataset.json"
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                ds = json.load(f)
            # Avoid duplicate injection
            if any(item[0] == sub_prompt for item in ds if isinstance(item, (list, tuple))):
                return False
            ds.append([sub_prompt, [5], 25])
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump(ds, f, indent=2)
            print(f"[ADAPTIVE STEPPING STONE] Injected simplified sub-task for: {stuck_prompt[:50]}")
            return True
    except Exception as e:
        print(f"[ADAPTIVE STEPPING STONE] Injection error: {e}")
    return False


def load_difficulty_log() -> list:
    """Load the difficulty tracking log from disk."""
    if not os.path.exists(DIFFICULTY_LOG):
        return []
    try:
        with open(DIFFICULTY_LOG, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


def save_difficulty_log(log: list) -> None:
    """Save the difficulty tracking log to disk."""
    os.makedirs(os.path.dirname(DIFFICULTY_LOG), exist_ok=True)
    with open(DIFFICULTY_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


def get_current_difficulty_tier() -> str:
    """Determine the current difficulty tier based on historical performance.
    
    Tier progression is sequential: beginner -> intermediate -> advanced -> expert.
    Requires sustained success (>= 80%) across at least 3 epochs per tier to advance.
    """
    log = load_difficulty_log()
    
    if not log:
        return "beginner"

    tiers = ["beginner", "intermediate", "advanced", "expert"]
    current_tier_idx = 0
    epochs_at_current_tier = 0
    
    for entry in log:
        rate = entry.get("success_rate", 0.0)
        if rate >= 0.8 or entry.get("success", False):
            epochs_at_current_tier += 1
            if epochs_at_current_tier >= 3 and current_tier_idx < len(tiers) - 1:
                current_tier_idx += 1
                epochs_at_current_tier = 0
        else:
            epochs_at_current_tier = max(0, epochs_at_current_tier - 1)
    
    return tiers[current_tier_idx]


def generate_harder_task_variants(current_tasks: List[list], difficulty_tier: str) -> List[List]:
    """Generate harder variants of existing tasks based on current difficulty tier.
    
    Returns a list of new, more complex task variants.
    This is the core of adaptive difficulty   the AI gets progressively harder problems.
    """
    new_tasks = []
    
    # Define complexity multipliers for each tier
    complexity_map = {
        "beginner": {"multiplier": 1, "operations": ["basic"]},
        "intermediate": {"multiplier": 2, "operations": ["list_ops", "conditionals"]},
        "advanced": {"multiplier": 3, "operations": ["nested_loops", "recursion"]},
        "expert": {"multiplier": 5, "operations": ["complex_algorithms", "data_structures"]}
    }
    
    complexity = complexity_map.get(difficulty_tier, complexity_map["intermediate"])
    multiplier = complexity["multiplier"]
    
    # Generate harder variants based on existing tasks
    for task in current_tasks:
        if len(task) != 3:
            continue
        
        prompt, inputs, expected = task
        
        # Skip if already complex enough for the tier
        if "matrix" in prompt.lower() or "determinant" in prompt.lower():
            continue
        
        # Generate harder variants
        if "fibonacci" in prompt.lower():
            new_tasks.append([
                f"Fibonacci Large Number",
                [multiplier * 10],  # Larger input
                None  # Will be computed by AI
            ])
        
        elif "sum" in prompt.lower() or "average" in prompt.lower():
            # Generate with larger lists and more elements
            new_inputs = list(range(1, multiplier * 5 + 1))
            new_tasks.append([
                f"Sum of Large List ({len(new_inputs)} elements)",
                [new_inputs],
                sum(new_inputs)
            ])
        
        elif "is_even" in prompt.lower() or "is_odd" in prompt.lower():
            # Generate with larger numbers and edge cases
            new_tasks.append([
                f"Prime Check Large Number",
                [multiplier * 100 + 7],  # Large prime candidate
                None
            ])
        
        elif "factorial" in prompt.lower():
            new_tasks.append([
                f"Factorial Large Number",
                [multiplier * 5],
                None
            ])
    
    return new_tasks


def add_adaptive_task(prompt: str, inputs: list, expected) -> bool:
    """Add a task to the dynamic dataset with adaptive difficulty.
    
    Returns True if the task was successfully added.
    This ensures tasks are properly formatted and don't duplicate existing ones.
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
        print(f"Adaptive task addition failed ({e})")
        return False


def record_adaptive_progress(epoch_num: int, difficulty_tier: str, success_rate: float) -> None:
    """Record adaptive difficulty progress for tracking."""
    log_entry = {
        "epoch": epoch_num,
        "difficulty_tier": difficulty_tier,
        "success_rate": round(success_rate, 3),
        "timestamp": __import__('time').time()
    }
    
    log = load_difficulty_log()
    log.append(log_entry)
    save_difficulty_log(log)


def get_adaptive_curriculum_summary() -> str:
    """Get a summary of adaptive difficulty progress."""
    log = load_difficulty_log()
    
    if not log:
        return "No adaptive difficulty data yet."
    
    current_tier = get_current_difficulty_tier()
    total_epochs = len(log)
    
    # Calculate average success rate by tier
    by_tier = {}
    for entry in log:
        tier = entry.get("difficulty_tier", "unknown")
        if tier not in by_tier:
            by_tier[tier] = []
        by_tier[tier].append(entry.get("success_rate", 0))
    
    summary = f"Adaptive Difficulty ({total_epochs} epochs, Current Tier: {current_tier}):\n"
    
    for tier, rates in by_tier.items():
        avg_rate = sum(rates) / max(len(rates), 1)
        summary += f"  {tier}: {avg_rate:.2f} average success rate ({len(rates)} epochs)\n"
    
    # Show progression trend
    if len(log) >= 3:
        early_tier = log[0].get("difficulty_tier", "beginner")
        late_tier = log[-1].get("difficulty_tier", "beginner")
        
        tier_order = {"beginner": 0, "intermediate": 1, "advanced": 2, "expert": 3}
        if tier_order.get(late_tier, 0) > tier_order.get(early_tier, 0):
            summary += f"\nProgression: {early_tier} → {late_tier}\n"
    
    return summary


def build_adaptive_difficulty_phase() -> str:
    """Build the adaptive difficulty section for the Visionary's prompt.
    
    This tells the Visionary what difficulty tier to target and how to scale tasks.
    """
    current_tier = get_current_difficulty_tier()
    summary = get_adaptive_curriculum_summary()
    
    if "no adaptive" in summary.lower():
        return ""
    
    # Define escalation thresholds based on tier
    escalation_rules = {
        "beginner": "Focus on mastering basic operations. Add 2-3 new tasks per epoch.",
        "intermediate": "You're ready for list operations and conditionals. Push harder.",
        "advanced": "Time for nested loops and recursion. The AI can handle complexity.",
        "expert": "Singularity territory. Generate novel, complex problems that test true intelligence."
    }
    
    rules = escalation_rules.get(current_tier, escalation_rules["beginner"])
    
    return f"""
ADAPTIVE DIFFICULTY CURRICULUM:

Current Tier: {current_tier}
{summary}

ESCALATION RULES FOR THIS TIER:
{rules}

CRITICAL: Your goal is to push the AI beyond its comfort zone. 
If it solves 90%+ of tasks, generate harder variants immediately.
If it struggles below 50%, simplify and provide more targeted challenges.
"""


def prompt_ai_adaptive_curriculum() -> Optional[str]:
    """Generate an adaptive curriculum prompt that tells the AI to create its own escalating challenges.
    
    This is the highest level of autonomous curriculum generation   the AI designing problems
    specifically calibrated to push itself toward singularity.
    """
    from autonomous_loop import prompt_ai
    
    try:
        current_tier = get_current_difficulty_tier()
        
        # Load current dataset for context
        if os.path.exists("sandbox/dynamic_dataset.json"):
            with open("sandbox/dynamic_dataset.json", 'r') as f:
                dataset = json.load(f)
            
            prompt = f"""
You are designing YOUR OWN CURRICULUM for progressive difficulty.

CURRENT DIFFICULTY TIER: {current_tier}
CURRENT DATASET SIZE: {len(dataset)} tasks

TASK: Generate 2-3 new, HARDER tasks that build on concepts you've already mastered.

RULES FOR TASK GENERATION:
1. Each new task must be strictly harder than existing ones in the dataset
2. Focus on increasing complexity: more operations, larger inputs, nested logic
3. Avoid duplicating existing prompts   create novel variations
4. Include tasks that require genuine problem-solving, not just pattern matching

EXAMPLE PROGRESSION:
- Beginner: "Sum of List" → Intermediate: "Sum of Nested Lists" → Advanced: "Weighted Sum with Custom Weights"

CRITICAL: Your goal is to create a curriculum that ESCALATES. 
If you're at expert tier, generate problems that test true intelligence.
"""
        
        return prompt_ai(prompt)
    
    except Exception as e:
        print(f"Adaptive curriculum failed ({e})")
        return None
