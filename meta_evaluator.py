"""
META-EVALUATOR   The AI's Ability to Improve Its Own Architecture

This system enables the AI to evaluate and improve its own prompts, agent roles,
and evaluation criteria. It implements:
- Prompt effectiveness scoring (which instructions led to better results?)
- Agent role evolution (are Engineer/Critic/Visionary instructions optimal?)
- Evaluation criteria refinement (is the fitness function measuring what matters?)
- Self-directed prompt rewriting based on historical performance

This is recursive self-improvement at the highest level: the AI improving its own
thinking process, not just its code. This is what separates "solving tasks" from
"learning how to learn."
"""

import json
import os
import time
from typing import List, Dict, Optional

META_LOG = "sandbox/meta_log.json"


def load_meta_log() -> list:
    """Load the meta-evaluation log from disk."""
    if not os.path.exists(META_LOG):
        return []
    try:
        with open(META_LOG, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


def save_meta_log(log: list) -> None:
    """Save the meta-evaluation log to disk."""
    os.makedirs(os.path.dirname(META_LOG), exist_ok=True)
    with open(META_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


def record_epoch_performance(epoch_num: int, anchor_score: float, dynamic_score: float, 
                              total_tasks: int, success: bool, changes_made: str) -> None:
    """Record an epoch's performance for meta-evaluation."""
    log_entry = {
        "epoch": epoch_num,
        "anchor_score": anchor_score,
        "dynamic_score": dynamic_score,
        "total_tasks": total_tasks,
        "success": success,
        "changes_made": changes_made[:500],  # Truncate for storage
        "timestamp": time.time()
    }
    
    log = load_meta_log()
    log.append(log_entry)
    save_meta_log(log)


def analyze_prompt_effectiveness() -> str:
    """Analyze historical performance to determine which prompt patterns work best.
    
    Returns a summary of what worked and what didn't, for the Visionary agent.
    """
    log = load_meta_log()
    if len(log) < 3:
        return "Not enough data yet   need at least 3 epochs to analyze."
    
    successes = [e for e in log if e["success"]]
    failures = [e for e in log if not e["success"]]
    
    summary = f"Meta-Evaluation Analysis ({len(log)} epochs):\n"
    summary += f"Success rate: {len(successes)}/{len(log)}\n"
    
    # Analyze what changes led to success vs failure
    successful_changes = [e.get("changes_made", "") for e in successes]
    failed_changes = [e.get("changes_made", "") for e in failures]
    
    if successful_changes:
        summary += "\nSuccessful change patterns:\n"
        for sc in successful_changes[:3]:
            summary += f"- {sc[:200]}...\n"
    
    if failed_changes:
        summary += "\nFailed change patterns:\n"
        for fc in failed_changes[:3]:
            summary += f"- {fc[:200]}...\n"
    
    return summary


def prompt_ai_prompt_evolution() -> Optional[str]:
    """Generate a meta-prompt that instructs the AI to improve its own prompts.
    
    This is the highest level of self-improvement: the AI rewriting its own instructions.
    """
    from autonomous_loop import prompt_ai
    
    analysis = analyze_prompt_effectiveness()
    
    if "not enough data" in analysis.lower():
        return None
    
    try:
        prompt = f"""
You are evolving YOUR OWN PROMPTS based on historical performance.

META-EVALUATION ANALYSIS:
{analysis}

TASK: Based on what worked and what failed, rewrite the following prompts to be MORE EFFECTIVE:

1. The Engineer's architecture engineering phase prompt   make it more precise about preserving existing code
2. The Critic's failure analysis prompt   make it extract more actionable lessons
3. The Visionary's goal discovery prompt   make it better at curriculum advancement

RULES FOR REWRITING:
- Keep the same structure (roles, phases, format) but improve clarity and precision
- Remove any instructions that led to failures
- Add constraints that prevented common errors
- Make the few-shot learning examples more representative of successful patterns

OUTPUT FORMAT:
Return three sections: ENGINEER_PROMPT, CRITIC_PROMPT, VISIONARY_PROMPT.
Each should be a complete, ready-to-use prompt string.

CRITICAL: Your goal is to make future epochs MORE SUCCESSFUL by improving how you instruct yourself.
"""
        
        return prompt_ai(prompt)
    except Exception as e:
        print(f"Prompt evolution failed ({e})")
        return None


def build_meta_evaluation_phase() -> str:
    """Build the meta-evaluation section for the Visionary's prompt.
    
    This tells the Visionary what prompt patterns have worked and which to avoid.
    """
    analysis = analyze_prompt_effectiveness()
    
    if "not enough data" in analysis.lower():
        return ""
    
    return f"\nMETA-EVALUATION INSIGHTS:\n{analysis}\n\nUse these insights when setting goals for the Engineer.\n"


def get_epoch_recommendations(epoch_num: int) -> str:
    """Get recommendations for what the AI should focus on this epoch.
    
    Based on historical patterns, suggest whether to focus on:
   , Adding new handlers (when dynamic score is low)
   , Refactoring existing code (when many tasks are solved)
   , Expanding tool capabilities (when stuck on certain task types)
    """
    log = load_meta_log()
    
    if len(log) < 2:
        return "Continue with current approach   not enough data for recommendations."
    
    recent_scores = [(e["epoch"], e["dynamic_score"]) for e in log[-5:]]
    success_rate = sum(1 for _, s in recent_scores if s > 0) / len(recent_scores)
    
    recommendations = f"Epoch {epoch_num} Recommendations:\n"
    
    if success_rate < 0.5:
        recommendations += "Low success rate recently. Focus on simpler, more targeted changes.\n"
        recommendations += "Consider breaking complex tasks into smaller steps.\n"
    elif success_rate >= 0.8:
        recommendations += "High success rate   time to push harder.\n"
        recommendations += "Add more challenging tasks and attempt deeper refactoring.\n"
    
    # Check if code is getting bloated (many handlers, few new solutions)
    if os.path.exists("sandbox/experiment.py"):
        try:
            with open("sandbox/experiment.py", 'r') as f:
                code = f.read()
            handler_count = code.count("if '")
            if handler_count > 15 and success_rate >= 0.7:
                recommendations += "Code is getting large   prioritize refactoring over adding new handlers.\n"
        except Exception:
            pass
    
    return recommendations
