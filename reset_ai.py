"""
reset_ai.py — Hard reset the AI back to its committed baseline state.

This script:
  1. Kills any running AI processes
  2. Uses git to hard-reset ALL tracked source files to HEAD (the last committed state)
  3. Deletes all untracked runtime files (json logs, model artifacts, pycache, etc.)
  4. Re-copies experiment.py -> experiment_protected_baseline.py
  5. Runs bootstrap.py to initialize a clean dynamic dataset
  6. Makes a git commit recording the reset

This is IMMUTABLE — the AI cannot modify it.
"""

import os
import subprocess
import shutil
import sys


SANDBOX_DIR = "sandbox"

# Files to keep in sandbox/ (all other .py files get deleted)
SANDBOX_KEEP_PY = {"experiment.py", "evaluator_anchor.py", "experiment_protected_baseline.py", "initial_clean_baseline.py"}

# Directories inside sandbox/ to never delete (large binaries / dependencies)
SANDBOX_KEEP_DIRS = {"llama_bin_cpu", "llama_cpp_repo"}


def run(cmd, **kwargs):
    """Run a shell command and return the result."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=15, **kwargs)


def reset_ai():
    print("=" * 58)
    print("         RESETTING AI TO CLEAN BASELINE STATE")
    print("=" * 58)

    # ------------------------------------------------------------------ #
    # STEP 1: Kill any running AI processes on port 8000
    # ------------------------------------------------------------------ #
    print("\n[1/7] Killing any running AI processes...")
    try:
        for line in run(["netstat", "-ano"]).stdout.splitlines():
            if ":8000" in line and "LISTENING" in line:
                parts = line.strip().split()
                pid = parts[-1]
                run(["taskkill", "/PID", pid, "/F"])
                print(f"[OK] Killed PID {pid} on port 8000")
    except Exception as e:
        print(f"[WARN] Process kill step skipped: {e}")

    # ------------------------------------------------------------------ #
    # STEP 2: Wipe .git repository history for a completely clean start
    # ------------------------------------------------------------------ #
    print("\n[2/7] Wiping Git repository history for a clean restart...")
    if os.path.exists(".git"):
        try:
            shutil.rmtree(".git", ignore_errors=True)
            print("[OK] Removed .git repository directory")
        except Exception as e:
            print(f"[WARN] Could not remove .git directory: {e}")
    else:
        print("[OK] No .git repository found — already clean.")

    # ------------------------------------------------------------------ #
    # STEP 3: Restore experiment.py from initial clean baseline template
    # ------------------------------------------------------------------ #
    print("\n[3/7] Restoring experiment.py from initial clean baseline...")
    clean_backup = os.path.join(SANDBOX_DIR, "initial_clean_baseline.py")
    protected_backup = os.path.join(SANDBOX_DIR, "experiment_protected_baseline.py")
    target = os.path.join(SANDBOX_DIR, "experiment.py")
    
    if os.path.exists(clean_backup):
        try:
            shutil.copy(clean_backup, target)
            shutil.copy(clean_backup, protected_backup)
            print("[OK] Restored experiment.py & experiment_protected_baseline.py from clean baseline")
        except Exception as e:
            print(f"[FAIL] Could not restore clean baseline: {e}")
    else:
        print("[FAIL] initial_clean_baseline.py not found.")

    # ------------------------------------------------------------------ #
    # STEP 5: Delete all untracked runtime state from sandbox/
    # ------------------------------------------------------------------ #
    print("\n[5/7] Clearing runtime state files from sandbox/...")
    os.makedirs(SANDBOX_DIR, exist_ok=True)

    for filename in os.listdir(SANDBOX_DIR):
        filepath = os.path.join(SANDBOX_DIR, filename)

        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1]
            if ext in {".json", ".log", ".flag", ".txt", ".md"}:
                try:
                    os.remove(filepath)
                    print(f"[OK] Deleted: sandbox/{filename}")
                except Exception as e:
                    print(f"[FAIL] Could not delete sandbox/{filename}: {e}")
            elif filename.endswith(".py") and filename not in SANDBOX_KEEP_PY:
                try:
                    os.remove(filepath)
                    print(f"[OK] Deleted AI-generated: sandbox/{filename}")
                except Exception as e:
                    print(f"[FAIL] Could not delete sandbox/{filename}: {e}")

        elif os.path.isdir(filepath) and filename not in SANDBOX_KEEP_DIRS:
            try:
                shutil.rmtree(filepath)
                print(f"[OK] Deleted directory: sandbox/{filename}/")
            except Exception as e:
                print(f"[FAIL] Could not delete sandbox/{filename}/: {e}")

    # Clean other runtime dirs
    for extra_dir in ["tmp_trainer", "dist", "logs"]:
        if os.path.exists(extra_dir):
            try:
                shutil.rmtree(extra_dir)
                print(f"[OK] Deleted {extra_dir}/")
            except Exception as e:
                print(f"[FAIL] Could not delete {extra_dir}/: {e}")

    # Clear pycache
    for pdir in ["__pycache__", os.path.join(SANDBOX_DIR, "__pycache__")]:
        if os.path.exists(pdir):
            try:
                shutil.rmtree(pdir)
                print(f"[OK] Cleared {pdir}/")
            except Exception as e:
                print(f"[FAIL] Could not clear {pdir}: {e}")

    # ------------------------------------------------------------------ #
    # STEP 6: Re-run bootstrap.py to initialize a clean dynamic dataset
    # ------------------------------------------------------------------ #
    print("\n[6/7] Initializing clean dataset via bootstrap.py...")
    try:
        result = subprocess.run(
            [sys.executable, "bootstrap.py"],
            capture_output=True, text=True, timeout=20
        )
        if result.returncode == 0:
            print("[OK] bootstrap.py succeeded")
            if result.stdout.strip():
                print(result.stdout.strip())
        else:
            print(f"[FAIL] bootstrap.py failed:\n{result.stderr.strip()}")
    except Exception as e:
        print(f"[FAIL] Error running bootstrap.py: {e}")

    # ------------------------------------------------------------------ #
    # STEP 7: Re-initialize fresh Git repository
    # ------------------------------------------------------------------ #
    print("\n[7/7] Re-initializing fresh Git repository...")
    try:
        run(["git", "init"])
        run(["git", "config", "user.name", "Autonomous AI Engine"])
        run(["git", "config", "user.email", "engine@autonomous.local"])
        if os.path.exists("sandbox/experiment.py"):
            run(["git", "add", "sandbox/experiment.py"])
        res = run(["git", "commit", "-m", "Initial baseline commit after reset"])
        if res.returncode == 0:
            print("[OK] Fresh Git repository created with initial baseline commit")
        else:
            print(f"[FAIL] Git commit failed: {res.stderr.strip()}")
    except Exception as e:
        print(f"[FAIL] Git re-initialization failed: {e}")

    print("\n" + "=" * 58)
    print("  Reset complete. AI is back to clean baseline state.")
    print("  Run start_ai.bat to begin a fresh evolution run.")
    print("=" * 58)


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass
    print("--- Resetting AI State ---")
    reset_ai()
