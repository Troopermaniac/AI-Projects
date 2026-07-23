import json
import os
import time
from typing import List, Dict, Optional



RECURSIVE_LOG = "sandbox/recursive_log.json"


def load_recursive_log() -> list:
    """Load the recursive self-improvement log from disk."""
    if not os.path.exists(RECURSIVE_LOG):
        return []
    try:
        with open(RECURSIVE_LOG, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


def save_recursive_log(log: list) -> None:
    """Save the recursive self-improvement log to disk."""
    os.makedirs(os.path.dirname(RECURSIVE_LOG), exist_ok=True)
    with open(RECURSIVE_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


def calculate_learning_rate(epoch_data: Dict) -> float:
    """Calculate the AI's learning rate based on epoch performance.
    
    Returns a score indicating how quickly the AI is improving.
    Higher = faster learning = closer to singularity.
    """
    log = load_recursive_log()
    
    if len(log) < 3:
        return 0.0
    
    # Calculate improvement rate over recent epochs
    recent_scores = [e.get("score", 0) for e in log[-5:]]
    
    if len(recent_scores) >= 2:
        early_avg = sum(recent_scores[:len(recent_scores)//2]) / max(len(recent_scores)//2, 1)
        late_avg = sum(recent_scores[-(len(recent_scores)//2):]) / max(len(recent_scores)//2, 1)
        
        improvement_rate = (late_avg - early_avg) / max(early_avg, 0.01)
        return max(0.0, improvement_rate)
    
    return 0.0


def extract_problem_solving_strategies(code: str) -> List[str]:
    """Extract the problem-solving strategies used in generated code.
    
    Returns a list of strategy patterns the AI is using.
    This enables tracking which approaches are most effective.
    """
    if not os.path.exists("sandbox/experiment.py"):
        return []
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        strategies = []
        
        # Detect common patterns
        if "if '" in content and "in prompt.lower()" in content:
            strategies.append("pattern_matching")
        if "if not" in content and "and" in content:
            strategies.append("conditional_logic")
        if "while" in content and "break" in content:
            strategies.append("loop_control")
        
        return strategies
    except Exception as e:
        print(f"Error extracting strategies: {e}")
        return []


def record_recursive_progress(epoch: int, learning_rate: float, strategies: List[str], score: float) -> None:
    """Record recursive self-improvement progress for an epoch."""
    log = load_recursive_log()
    log.append({
        "epoch": epoch,
        "score": score,
        "learning_rate": learning_rate,
        "strategies": strategies,
        "timestamp": time.time()
    })
    save_recursive_log(log)


def get_recursive_summary() -> str:
    """Return status summary of recursive self-improvement progress."""
    log = load_recursive_log()
    if not log:
        return "No recursive progress recorded yet."
    latest = log[-1]
    return f"Epoch {latest.get('epoch', 0)}: Learning Rate={latest.get('learning_rate', 0.0):.2f}, Score={latest.get('score', 0.0)}"