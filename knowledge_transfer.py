import json
import os
import time
from typing import List, Dict, Tuple, Optional


KNOWLEDGE_FILE = "sandbox/knowledge_graph.json"


def load_knowledge_graph() -> dict:
    """Load the knowledge graph from disk."""
    if not os.path.exists(KNOWLEDGE_FILE):
        return {"concepts": {}, "task_mappings": {}}
    try:
        with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                return {"concepts": {}, "task_mappings": {}}
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return {"concepts": {}, "task_mappings": {}}


def safe_replace_file(temp_file: str, target_file: str, max_retries: int = 5, delay: float = 0.05) -> None:
    """Replace target_file with temp_file handling Windows permission errors gracefully."""
    for attempt in range(max_retries):
        try:
            os.replace(temp_file, target_file)
            return
        except OSError:
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
    # Fallback if os.replace fails repeatedly on Windows
    try:
        if os.path.exists(target_file):
            try:
                os.remove(target_file)
            except OSError:
                pass
        os.replace(temp_file, target_file)
    except OSError:
        try:
            with open(temp_file, 'r', encoding='utf-8') as src, open(target_file, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
        except OSError:
            pass


def save_knowledge_graph(graph: dict) -> None:
    """Save the knowledge graph to disk."""
    os.makedirs(os.path.dirname(KNOWLEDGE_FILE), exist_ok=True)
    temp_file = KNOWLEDGE_FILE + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=4)
    safe_replace_file(temp_file, KNOWLEDGE_FILE)


def extract_concepts(prompt: str) -> List[str]:
    """Extract key concepts from a task prompt.
    
    Returns a list of meaningful words/phrases that represent the core concepts.
    Filters out common words and focuses on domain-specific terms.
    """
    # Stopwords to filter out
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'on',
        'at', 'by', 'for', 'with', 'and', 'or', 'but', 'if', 'as', 'into',
        'through', 'this', 'that', 'these', 'those', 'it', 'its', 'me', 'my'
    }
    
    # Normalize and extract
    words = prompt.lower().split()
    concepts = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Check for two-word phrases (n-grams)
    result = []
    for i in range(len(words)):
        word = words[i].lower()
        if word not in stopwords and len(word) > 2:
            result.append(word)
        if i + 1 < len(words):
            phrase = f"{words[i]} {words[i+1]}"
            if phrase not in stopwords:
                parts = phrase.split()
                if parts[0] not in stopwords and parts[1] not in stopwords:
                    result.append(phrase)
    
    return list(set(result))


def build_concept_index(graph: dict, prompt: str, task_id: str) -> None:
    """
    Add new concepts to the knowledge graph.
    This function is responsible for updating the concept index with new information
    derived from the current task and any previously learned patterns.
    
    Args:
        graph (dict): The existing knowledge graph structure.
        prompt (str): The current task prompt.
        task_id (str): Unique identifier for this task.
    """
    concepts = extract_concepts(prompt)
    if not concepts:
        return
    
    # Add new concepts to the graph
    for concept in concepts:
        if concept not in graph["concepts"]:
            graph["concepts"][concept] = []
        graph["concepts"][concept].append(task_id)
    
    # Update task mappings
    if task_id not in graph["task_mappings"]:
        graph["task_mappings"][task_id] = {}
    for concept in concepts:
        if concept not in graph["task_mappings"][task_id]:
            graph["task_mappings"][task_id][concept] = []
        graph["task_mappings"][task_id][concept].append(task_id)
    
    # Save the updated knowledge graph
    save_knowledge_graph(graph)


def update_task_status(task_id: str, status: str) -> None:
    """Update task solution status in the knowledge graph."""
    graph = load_knowledge_graph()
    if "task_status" not in graph:
        graph["task_status"] = {}
    graph["task_status"][task_id] = status
    save_knowledge_graph(graph)


def build_knowledge_transfer_phase(tasks: Optional[List[str]] = None) -> str:
    """Build a context string of transferrable concepts for target tasks."""
    graph = load_knowledge_graph()
    concepts = graph.get("concepts", {})
    if not concepts:
        return "No prior knowledge context available."
    
    summary = []
    if tasks:
        for t in tasks:
            extracted = extract_concepts(t)
            matched = [c for c in extracted if c in concepts]
            if matched:
                summary.append(f"- Task '{t[:50]}': related concepts -> {', '.join(matched[:5])}")
    
    if not summary:
        top_concepts = list(concepts.keys())[:10]
        return f"Knowledge Graph Concepts: {', '.join(top_concepts)}"
    
    return "Retrieved Knowledge Graph Transfer Context:\n" + "\n".join(summary)


def get_concept_statistics() -> str:
    """Return formatted statistics about the knowledge graph."""
    graph = load_knowledge_graph()
    concepts = graph.get("concepts", {})
    task_status = graph.get("task_status", {})
    solved = sum(1 for v in task_status.values() if v == "solved")
    return f"{len(concepts)} concepts indexed ({solved} tasks solved)"