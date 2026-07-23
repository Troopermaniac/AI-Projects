"""
Structured memory module for categorized long term learning.

Provides categorized memory storage, domain tagging, principle extraction,
and semantic memory retrieval for task context.
"""

import json
import os
import time
from typing import List, Dict, Optional


MEMORY_FILE = "sandbox/memories.json"
MAX_MEMORY_ENTRIES = 50
MAX_MEMORY_CHARS = 30000


def load_memories() -> list:
    """Load structured memories from disk."""
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            # Handle empty file
            if not content.strip():
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return []


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


def save_memories(memories: list) -> None:
    """Save structured memories to disk with adaptive retention.
    
    Improvement #5: Adaptive memory retention based on confidence scores.
    High-confidence memories are preserved longer; low-confidence ones are trimmed first.
    """
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    
    # Trim old entries if over limit, but prioritize by confidence
    while len(json.dumps(memories)) > MAX_MEMORY_CHARS and len(memories) > 1:
        # Find lowest-confidence entry to remove (prefer keeping important memories)
        candidates = [(i, m) for i, m in enumerate(memories)]
        candidates.sort(key=lambda x: (x[1].get("confidence", 0.5), x[1].get("timestamp", 0)))
        # Remove the lowest confidence oldest entry
        if candidates:
            idx_to_remove = candidates[0][0]
            memories.pop(idx_to_remove)
    
    temp_file = MEMORY_FILE + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(memories, f, indent=4)
    safe_replace_file(temp_file, MEMORY_FILE)


def add_memory(entry: dict) -> None:
    """Add a structured memory entry with deduplication."""
    memories = load_memories()
    
    new_title = entry.get("title", "").strip().lower()
    new_content = entry.get("content", "").strip().lower()
    
    # Deduplicate memories by title and content
    for m in memories:
        if m.get("title", "").strip().lower() == new_title and m.get("content", "").strip().lower() == new_content:
            return  # Skip duplicate entry

    # Add timestamp if not present
    if "timestamp" not in entry:
        entry["timestamp"] = time.time()
    
    memories.append(entry)
    save_memories(memories)


def retrieve_by_domain(domain: str, limit: int = 10) -> list:
    """Retrieve memories from a specific domain."""
    memories = load_memories()
    results = [m for m in memories if m.get("domain") == domain]
    return results[-limit:] if len(results) > 0 else []


def retrieve_by_tags(tags: List[str], limit: int = 10) -> list:
    """Retrieve memories matching any of the given tags (semantic retrieval)."""
    memories = load_memories()
    
    # Score entries based on tag relevance
    scored_entries = []
    for m in memories:
        score = 0.0
        for tag in tags:
            if tag in m.get("tags", []):
                score += 1.0 / len(tags)
        m["score"] = score
        scored_entries.append(m)
    
    # Sort by relevance (descending) and timestamp (ascending)
    scored_entries.sort(key=lambda x: (-x["score"], x["timestamp"]))
    
    return scored_entries[:limit]


def get_memory_summary() -> str:
    """Return a formatted status summary of stored memories."""
    memories = load_memories()
    if not memories:
        return "0 entries in memory"
    counts = {}
    for m in memories:
        m_type = m.get("type", "general") if isinstance(m, dict) else "general"
        counts[m_type] = counts.get(m_type, 0) + 1
    details = ", ".join(f"{v} {k}s" for k, v in counts.items())
    return f"{len(memories)} total entries ({details})"


def consolidate_memories() -> int:
    """Consolidate redundant memory entries to optimize retention space.
    
    Returns the number of consolidated entries.
    """
    memories = load_memories()
    if len(memories) <= 1:
        return 0
    
    initial_count = len(memories)
    seen_contents = set()
    deduped = []
    
    for m in memories:
        if isinstance(m, dict):
            key = (m.get("title", ""), m.get("content", "")[:100])
        else:
            key = str(m)[:100]
            
        if key not in seen_contents:
            seen_contents.add(key)
            deduped.append(m)
            
    consolidated_count = initial_count - len(deduped)
    if consolidated_count > 0:
        save_memories(deduped)
        
    return consolidated_count