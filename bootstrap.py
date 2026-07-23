"""Bootstrap the singularity system with initial data.

This script populates dynamic_dataset.json, seeds memories.json, and 
initializes knowledge_graph.json so the AI has something to learn from.
The autonomous loop can then start bootstrapping itself from zero.
"""

import json

def bootstrap():
    #  1. Populate dynamic dataset with initial tasks 
    # These are tasks beyond the anchor baseline   the AI must evolve handlers for them
    dynamic_tasks = [
        ["Prime Number Check", [7], True],
        ["Prime Number Check", [4], False],
        ["Prime Number Check", [2], True],
        ["GCD Computation", [48, 18], 6],
        ["GCD Computation", [1071, 462], 21],
        ["LCM Computation", [4, 6], 12],
        ["LCM Computation", [3, 5], 15],
        ["Power of Number", [2, 10], 1024],
        ["Power of Number", [3, 4], 81],
        ["Is Palindrome", [121], True],
        ["Is Palindrome", [123], False],
        ["Fibonacci Sum", 5, 15],
        ["Fibonacci Sum", 10, 129],
        ["Sum of Squares", [1, 2, 3], 14],
        ["Sum of Squares", [3, 4, 5], 50],
        ["Product of List", [1, 2, 3, 4], 24],
        ["Product of List", [5, 6, 7], 210],
        ["Second Largest", [1, 2, 3, 4, 5], 4],
        ["Second Largest", [10, 5, 8, 3, 7], 8],
        ["Remove Duplicates", [1, 2, 2, 3, 3, 3], [1, 2, 3]],
        ["Remove Duplicates", [5, 5, 5, 1, 1], [5, 1]],
        ["Merge Sorted Lists", [[1, 3, 5], [2, 4, 6]], [1, 2, 3, 4, 5, 6]],
        ["Merge Sorted Lists", [[0, 2, 4], [1, 3, 5]], [0, 1, 2, 3, 4, 5]],
        ["Rotate List Left", [[1, 2, 3, 4, 5], 2], [3, 4, 5, 1, 2]],
        ["Rotate List Right", [[1, 2, 3, 4, 5], 1], [5, 1, 2, 3, 4]],
        ["Count Vowels", "hello", 2],
        ["Count Vowels", "aeiou", 5],
        ["Count Vowels", "xyz", 0],
        ["Sum of Primes Up To N", 10, 17],
        ["Sum of Primes Up To N", 20, 77],
        ["Find Missing Number", [[1, 2, 4, 5], 6], 3],
        ["Find Missing Number", [[1, 2, 3, 5], 6], 4],
        ["Count Digits", [12345], 5],
        ["Count Digits", [0], 1],
        ["Is Prime (Harder)", [97], True],
        ["Is Prime (Harder)", [91], False],
    ]

    # Write initial seed dataset (3 starting tasks for Epoch 1)
    seed_tasks = dynamic_tasks[:3]
    with open("sandbox/dynamic_dataset.json", "w") as f:
        json.dump(seed_tasks, f, indent=4)
    print(f"Populated {len(seed_tasks)} initial tasks in dynamic_dataset.json")

    #  2. Seed memories with initial learning entries 
    memories = []

    with open("sandbox/memories.json", "w") as f:
        json.dump(memories, f, indent=4)
    print("Initialized empty memories.json")

    #  3. Initialize knowledge graph 
    kg = {"concepts": {}, "task_mappings": {}}
    
    with open("sandbox/knowledge_graph.json", "w") as f:
        json.dump(kg, f, indent=4)
    print("Initialized empty knowledge graph")

if __name__ == "__main__":
    bootstrap()
