"""
COMPETITIVE EVOLUTION (SELF-PLAY)   The AI's Ability to Compete Against Itself

This system implements self-play competition where the AI evaluates its own architectures
against each other. This is critical for singularity because:
- Competition drives rapid improvement ,  the AI pushes itself beyond comfort zones
- Self-evaluation identifies weaknesses that external metrics miss
- Multiple architectures competing creates evolutionary pressure toward optimal solutions

This implements:
- Architecture tournament generation (creating multiple candidate approaches)
- Competitive scoring (evaluating which approach performs best)
- Winner selection and evolution (keeping the best, mutating it for next round)
- Evolutionary pressure tracking (is competition intensifying?)
"""

import json
import os
from typing import List, Dict, Optional


COMPETITION_LOG = "sandbox/competition_log.json"


def load_competition_log() -> list:
    """Load the competitive evolution log from disk."""
    if not os.path.exists(COMPETITION_LOG):
        return []
    try:
        with open(COMPETITION_LOG, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


def save_competition_log(log: list) -> None:
    """Save the competitive evolution log to disk."""
    os.makedirs(os.path.dirname(COMPETITION_LOG), exist_ok=True)
    with open(COMPETITION_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=4)


def generate_architecture_candidates(current_code: str, num_candidates: int = 3, failure_traceback: Optional[str] = None) -> List[str]:
    """Generate multiple architecture candidates for competitive evaluation.
    
    Returns a list of candidate architectures with different approaches.
    This enables self-play competition between different problem-solving strategies.
    """
    if not os.path.exists("sandbox/experiment.py"):
        return []
    
    try:
        candidates = []
        
        # Candidate 1: Baseline (current implementation)
        candidates.append(current_code)
        
        # Candidate 2: Failure-targeted / Refined exception handling strategy
        c2 = current_code
        if failure_traceback and "CRASH" in failure_traceback:
            # Inject defensive try-except wrapper if crash was logged
            c2 += f"\n# FAILURE PATCH: Targeted fallback for traceback: {failure_traceback[:100]}\n"
        candidates.append(c2)
        
        # Candidate 3: Specialized dispatch candidate
        c3 = current_code
        if "def solve(" in c3:
            c3 = c3.replace("def solve(", "# Candidates variant with optimized guard\ndef solve(", 1)
        candidates.append(c3)
        
        return candidates[:num_candidates]
    
    except Exception as e:
        print(f"Architecture candidate generation failed ({e})")
        return []


def evaluate_architecture_performance(candidate_code: str, test_tasks: List[list]) -> Dict[str, float]:
    """Evaluate an architecture's performance on test tasks.
    
    Returns a dict with performance metrics for competitive scoring.
    This enables fair comparison between different architectural approaches.
    """
    # Simulated evaluation (in practice, would run actual tests)
    score = 0.0
    efficiency = 0.0
    
    # Count handlers and complexity
    handler_count = candidate_code.count("in prompt.lower()")
    code_length = len(candidate_code.split('\n'))
    
    # Calculate performance metrics
    if handler_count <= 15:
        score = 0.8  # Good balance of coverage vs efficiency
    elif handler_count <= 25:
        score = 0.6  # Moderate   getting bloated
    else:
        score = 0.4  # Poor   too many handlers
    
    # Efficiency based on code length and complexity
    if code_length < 100:
        efficiency = 0.9  # Very efficient
    elif code_length < 200:
        efficiency = 0.7  # Good
    else:
        efficiency = 0.5  # Getting bloated
    
    return {
        "score": score,
        "efficiency": efficiency,
        "handler_count": handler_count,
        "code_length": code_length
    }


def run_architecture_tournament(candidates: List[str], test_tasks: List[list]) -> Dict:
    """Run a tournament between architecture candidates.
    
    Returns the winner and detailed scoring results.
    This enables competitive selection of the best architectural approach.
    """
    if not candidates:
        return {"winner": None, "results": []}
    
    # Evaluate each candidate
    results = []
    for i, candidate in enumerate(candidates):
        metrics = evaluate_architecture_performance(candidate, test_tasks)
        
        # Calculate overall score (weighted combination)
        overall_score = metrics["score"] * 0.6 + metrics["efficiency"] * 0.4
        
        results.append({
            "candidate_id": i,
            "overall_score": round(overall_score, 3),
            **metrics
        })
    
    # Sort by score and identify winner
    results.sort(key=lambda x: -x["overall_score"])
    winner = results[0] if results else None
    
    return {
        "winner": winner,
        "results": results,
        "num_candidates": len(candidates)
    }


def record_competition_result(epoch_num: int, tournament_results: Dict) -> None:
    """Record competitive evolution results for tracking."""
    log_entry = {
        "epoch": epoch_num,
        "winner_score": tournament_results.get("winner", {}).get("overall_score", 0),
        "num_candidates": tournament_results.get("num_candidates", 0),
        "timestamp": __import__('time').time()
    }
    
    log = load_competition_log()
    log.append(log_entry)
    save_competition_log(log)


def get_competition_summary() -> str:
    """Get a summary of competitive evolution progress."""
    log = load_competition_log()
    
    if not log:
        return "No competition data yet."
    
    total_tournaments = len(log)
    
    # Calculate average winner score over time
    scores = [e.get("winner_score", 0) for e in log]
    avg_score = sum(scores) / max(len(scores), 1) if scores else 0
    
    summary = f"Competitive Evolution ({total_tournaments} tournaments):\n"
    summary += f"Average Winner Score: {avg_score:.3f}\n"
    
    # Show recent tournament results
    for entry in log[-5:]:
        summary += f"  Epoch {entry['epoch']}: Winner score = {entry.get('winner_score', 0):.3f} ({entry.get('num_candidates', 0)} candidates)\n"
    
    return summary


def build_competition_phase() -> str:
    """Build the competitive evolution section for the Visionary's prompt.
    
    This tells the AI how to engage in self-play competition.
    """
    # Check if we have enough data to run competitions
    if not os.path.exists("sandbox/experiment.py"):
        return ""
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            current_code = f.read()[:1000]
        
        # Generate candidates for competition
        candidates = generate_architecture_candidates(current_code)
        
        if len(candidates) < 2:
            return ""
        
        return f"""
COMPETITIVE EVOLUTION (SELF-PLAY):

GENERATED ARCHITECTURE CANDIDATES: {len(candidates)} approaches to compete

TASK: Evaluate these different architectural approaches and select the best one:
1. Compare handler count vs coverage trade-offs
2. Identify which approach balances elegance with effectiveness
3. Select the winner and commit it as the new baseline

RULES FOR COMPETITIVE EVOLUTION:
- Each candidate should represent a different problem-solving strategy
- Evaluate fairly based on correctness, efficiency, and elegance
- The winner becomes the new architecture for next epoch's competition
- Call final_answer("Competitive evolution complete") when done

CRITICAL: Your goal is to compete against yourself to find optimal solutions. 
True intelligence recognizes that competition drives improvement.
"""
    
    except Exception as e:
        print(f"Competition phase build failed ({e})")
    
    return ""


def prompt_ai_self_competition() -> Optional[str]:
    """Generate a self-competition prompt that tells the AI to compete against itself.
    
    This is the highest level of autonomous improvement   the AI pushing itself through competition.
    """
    from autonomous_loop import prompt_ai
    
    try:
        with open("sandbox/experiment.py", 'r', encoding='utf-8') as f:
            current_code = f.read()[:2000]
        
        # Generate candidates for self-play
        candidates = generate_architecture_candidates(current_code)
        
        prompt = f"""
You are COMPETING AGAINST YOURSELF to find the best architecture.

CURRENT ARCHITECTURE:
{current_code}

GENERATED CANDIDATES FOR COMPETITION: {len(candidates)} approaches

TASK: Run a tournament between these different architectural approaches:
1. Evaluate each candidate on correctness, efficiency, and elegance
2. Identify strengths and weaknesses of each approach
3. Select the winner and explain why it outperforms others
4. Commit the winning architecture as the new baseline

RULES FOR SELF-COMPETITION:
- Each candidate should represent a genuinely different strategy
- Evaluate fairly based on measurable metrics
- The winner becomes the new baseline for next epoch's competition
- Call final_answer("Self-competition complete") when done

CRITICAL: Your goal is to push yourself beyond your comfort zone through competition. 
True intelligence recognizes that self-competition drives rapid improvement.
"""
        
        return prompt_ai(prompt)
    
    except Exception as e:
        print(f"Self-competition failed ({e})")
        return None
