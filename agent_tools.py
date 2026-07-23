import os
import subprocess
import time
import sys
from smolagents import tool

# Import the filesystem sandbox   resolves symlinks, blocks access outside project dir.
from filesystem_sandbox import enforce_read, enforce_write, PROJECT_ROOT

@tool
def read_file(path: str) -> str:
    """
    Reads the content of a file within the project directory.
    
    Args:
        path: The path to the file to read (relative or absolute).
    """
    try:
        # Resolve and enforce   blocks access outside project dir.
        enforce_read(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            MAX_LENGTH = 8000
            if len(content) > MAX_LENGTH:
                return content[:MAX_LENGTH] + f"\n\n... [TRUNCATED] ...\nWARNING: File is too large ({len(content)} characters). It was truncated to {MAX_LENGTH} characters to protect your context limit. Write a python script to process it instead of reading it entirely."
            return content
    except Exception as e:
        if isinstance(e, RuntimeError):
            # Propagate filesystem violation errors
            raise
        return f"Error reading file {path}: {str(e)}"

@tool
def write_file(path: str, content: str) -> str:
    """
    Writes or overwrites content to a file within the project directory.
    
    Args:
        path: The path to the file to write to (relative or absolute).
        content: The text content to write.
    """
    try:
        # Resolve and enforce   blocks access outside project dir.
        enforce_write(path)
        
        # Ensure parent directory exists if requested
        parent_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent_dir, exist_ok=True)
            
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        if isinstance(e, RuntimeError):
            # Propagate filesystem violation errors
            raise
        return f"Error writing file {path}: {str(e)}"

@tool
def run_git_command(command: str) -> str:
    """
    Executes a git command (e.g., 'status', 'add .', 'commit -m "message"').
    Do NOT include the 'git ' prefix, only the arguments.
    
    SECURITY RESTRICTION: Only safe read-only commands are allowed: status, log, diff, checkout, add, commit, push, pull, branch, clone, init, remote, show, blame, tag, describe, verify, ls-files, archive, fetch.
    DANGEROUS COMMANDS BLOCKED: rm, clean, prune, reset --hard, force, delete, branch -d, branch -D, tag -d, push --force, etc.
    
    Args:
        command: The git command arguments to run.
    """
    # SECURITY CAGE: Block destructive git commands   comprehensive list
    dangerous_patterns = [
        "rm", "clean", "prune", "reset", "--hard", "--soft", "--mixed",
        "force", "-f", "--force", "delete", "branch -d", "branch -D",
        "tag -d", "push --force", "push -f", "remote delete",
        "filter-branch", "reflog expire", "gc", "prune",
        "checkout --orphan", "stash pop", "rebase --abort",
        "--delete", "-D", "--no-tags", "--all", "--tags",
        "submodule", "worktree", "replace", "bisect",
        "merge", "revert", "am", "format-patch"
    ]
    
    # Check if command contains dangerous patterns (case-insensitive)
    cmd_lower = command.lower()
    for pattern in dangerous_patterns:
        if pattern.strip().lower() in cmd_lower:
            return f"ERROR: Security Sandbox Violation. Git command '{pattern}' is blocked for safety."
    
    # Only allow safe git commands   whitelist approach
    allowed_commands = [
        "status", "log", "diff", "checkout", "add", "commit", "push", 
        "pull", "branch", "clone", "init", "remote", "show", "blame",
        "tag", "describe", "verify", "ls-files", "archive", "fetch"
    ]
    
    # If command doesn't start with an allowed command, block it
    if not any(cmd_lower.startswith(c + " ") or cmd_lower == c for c in allowed_commands):
        return f"ERROR: Security Sandbox Violation. Only safe git commands are allowed: {', '.join(allowed_commands)}"
    
    try:
        full_command = f"git {command}"
        result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
        
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
            
        return output if output.strip() else "Command executed successfully (no output)."
    except Exception as e:
        return f"Error executing git command: {str(e)}"

@tool
def run_python_script(script_path: str, timeout_seconds: int = 60) -> str:
    """
    Executes a python script within the project directory.
    Use this to test your code before finishing your turn.
    
    Args:
        script_path: The path to the python script (e.g., 'sandbox/experiment.py').
        timeout_seconds: Optional. The maximum time in seconds to allow the script to run (default 60s). Max is 300s.
    """
    try:
        # Filesystem sandbox   blocks scripts outside project dir.
        enforce_read(script_path)
            
        if not os.path.exists(script_path):
            return f"ERROR: File {script_path} does not exist."
            
        print(f"AI is testing script: {script_path}...")
        
        # Enforce limits
        timeout_seconds = min(max(timeout_seconds, 1), 300)
        
        result = subprocess.run(
            [sys.executable, "sandbox_guard.py", script_path], 
            capture_output=True, 
            text=True, 
            timeout=timeout_seconds
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
            
        return output if output.strip() else "Script executed successfully (no output)."
    except subprocess.TimeoutExpired:
        return f"ERROR: Script timed out after {timeout_seconds} seconds. You probably wrote an infinite loop."
    except Exception as e:
        return f"Error executing script: {str(e)}"

@tool
def list_sandbox_files() -> str:
    """
    Lists all files and directories currently inside the sandbox/ directory.
    """
    try:
        if not os.path.exists("sandbox"):
            return "Sandbox directory does not exist yet."
        enforce_read("sandbox")
        files = os.listdir("sandbox")
        return f"Contents of sandbox/:\n" + "\n".join(files) if files else "Sandbox is empty."
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        return f"Error listing sandbox: {str(e)}"

@tool
def list_project_files() -> str:
    """
    Lists all files and directories in the current project directory.
    Use this to see what files exist outside of sandbox/.
    """
    try:
        enforce_read(PROJECT_ROOT)
        files = os.listdir(PROJECT_ROOT)
        return f"Contents of project ({PROJECT_ROOT}):\n" + "\n".join(files) if files else "Project is empty."
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        return f"Error listing project files: {str(e)}"

import ast

@tool
def check_syntax(script_path: str) -> str:
    """
    Checks a python script for syntax errors without executing it.
    
    Args:
        script_path: The path to the python script (e.g., 'sandbox/experiment.py').
    """
    try:
        enforce_read(script_path)
        if not os.path.exists(script_path):
            return f"ERROR: File {script_path} does not exist."
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
        ast.parse(code)
        return "Syntax is valid! No SyntaxErrors detected."
    except SyntaxError as e:
        return f"SyntaxError in {script_path}:\nLine {e.lineno}, Offset {e.offset}: {e.msg}\nCode: {e.text}"
    except Exception as e:
        return f"Error checking syntax: {str(e)}"

@tool
def profile_python_script(script_path: str, timeout_seconds: int = 60) -> str:
    """
    Runs a python script using cProfile to find out which functions take the most execution time.
    Use this to optimize your algorithm for the fitness equation.
    
    Args:
        script_path: The path to the python script (e.g., 'sandbox/experiment.py').
        timeout_seconds: Optional. The maximum time in seconds to allow the script to run (default 60s). Max is 300s.
    """
    try:
        # Filesystem sandbox   blocks scripts outside project dir.
        enforce_read(script_path)
            
        if not os.path.exists(script_path):
            return f"ERROR: File {script_path} does not exist."
            
        print(f"AI is profiling script: {script_path}...")
        
        # Enforce limits
        timeout_seconds = min(max(timeout_seconds, 1), 300)
        
        result = subprocess.run(
            [sys.executable, "-m", "cProfile", "-s", "time", "sandbox_guard.py", script_path], 
            capture_output=True, 
            text=True, 
            timeout=timeout_seconds
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
            
        # Truncate output in case profiling results are huge
        MAX_LEN = 2000
        if len(output) > MAX_LEN:
            return output[:MAX_LEN] + "\n... [TRUNCATED] ..."
        return output if output.strip() else "Script executed successfully (no profiling output)."
    except subprocess.TimeoutExpired:
        return f"ERROR: Script timed out after {timeout_seconds} seconds during profiling. Infinite loop detected."
    except Exception as e:
        return f"Error profiling script: {str(e)}"



import urllib.request
import urllib.parse
import json

@tool
def search_arxiv(query: str, max_results: int = 3) -> str:
    """
    Searches the ArXiv database for academic papers (AI research).
    
    Args:
        query: The search query (e.g., 'liquid neural networks').
        max_results: Number of results to return (default 3).
    """
    try:
        # Remove the hardcoded 'all:' prefix so the AI can use advanced queries (e.g., OR, AND)
        url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&start=0&max_results={max_results}"
        req = urllib.request.urlopen(url)
        res = req.read().decode('utf-8')
        return res[:2000] # Truncate to save context window
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"

@tool
def update_memory(entry: str) -> str:
    """
    Writes a finding to the long-term memories.json file.
    Use this to record failed hypotheses or successful architectural discoveries.
    
    Args:
        entry: The text to add to the memory bank.
    """
    try:
        memories = []
        if os.path.exists("sandbox/memories.json"):
            with open("sandbox/memories.json", 'r', encoding='utf-8') as f:
                try:
                    memories = json.load(f)
                except:
                    pass
        memories.append({"timestamp": time.time(), "memory": entry})
        
        # MEMORY LIMITER: Prevent context window crashes using a character limit,
        # so it scales perfectly depending on whether memories are short or long.
        # Set to ~15,000 characters (approx 4,000 tokens) for a standard local 7B model.
        # If you upgrade to a larger model, you can increase this!
        MAX_MEMORY_CHARS = 15000
        while len(json.dumps(memories)) > MAX_MEMORY_CHARS and len(memories) > 1:
            memories.pop(0) # Remove the oldest memory until we fit in the context window
            
        # Ensure sandbox directory exists
        os.makedirs("sandbox", exist_ok=True)
        with open("sandbox/memories.json", 'w', encoding='utf-8') as f:
            json.dump(memories, f, indent=4)
        return "Memory successfully stored."
    except Exception as e:
        return f"Error writing memory: {str(e)}"

