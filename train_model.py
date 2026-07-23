import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import json
import time
import datetime
import subprocess
import sys
import argparse
import shutil

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Single source of truth: read everything from config.json
_cfg = {}
try:
    with open("config.json", "r", encoding="utf-8") as _f:
        _cfg = json.load(_f)
except Exception as e:
    print(f"[TRAIN] Warning: Could not read config.json ({e}). Using defaults.")

_train = _cfg.get("training", {})

LARGE_MODEL          = _cfg.get("hf_model") or _cfg.get("model_id", "")
TRAJECTORIES_PATH    = "sandbox/training_trajectories.json"
ADAPTER_OUTPUT_DIR   = "sandbox/lora_adapter"
MERGED_MODEL_DIR     = "sandbox/evolved_model"
RESTART_FLAG_PATH    = "sandbox/restart_with_evolved.flag"
LOGS_DIR             = "logs"

# Training hyperparameters (all editable in config.json → "training" section)
LORA_RANK                  = _train.get("lora_rank", 16)
LORA_ALPHA                 = _train.get("lora_alpha", 32)
LORA_DROPOUT               = _train.get("lora_dropout", 0.05)
LEARNING_RATE              = _train.get("learning_rate", 0.0002)
NUM_EPOCHS                 = _train.get("num_epochs", 3)
PER_DEVICE_BATCH_SIZE      = _train.get("per_device_batch_size", 1)
GRADIENT_ACCUMULATION_STEPS = _train.get("gradient_accumulation_steps", 4)
TRIGGER_THRESHOLD          = _train.get("trigger_threshold", 5)


def clean_disk_cache_if_needed():
    """Ensure at least 35GB of free disk space is available for model merging and GGUF quantization."""
    try:
        import shutil as _sh
        lm_root = os.path.dirname(os.path.abspath(__file__))
        free_bytes = _sh.disk_usage(lm_root).free
        free_gb = free_bytes / (1024 ** 3)
        if free_gb < 35.0:
            print(f"[TRAIN] Disk space low ({free_gb:.1f} GB free). Purging temp trainer directories to release space...")
            for target in ["tmp_trainer", "sandbox/temp_f16.gguf", "sandbox/evolved_model"]:
                if os.path.exists(target):
                    if os.path.isdir(target):
                        _sh.rmtree(target, ignore_errors=True)
                    else:
                        os.remove(target)
            new_free = _sh.disk_usage(lm_root).free / (1024 ** 3)
            print(f"[TRAIN] Disk cleanup complete. Free space: {new_free:.1f} GB.")
    except Exception as e:
        print(f"[TRAIN] Disk cleanup note: {e}")

def get_free_vram_gb() -> float:
    """Get available VRAM in GB. Returns 0.0 if no GPU or detection fails."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        # torch.cuda.mem_get_info returns (free, total) in bytes
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return free_bytes / (1024 ** 3)
    except Exception:
        return 0.0


def is_model_cached_locally(model_name: str) -> bool:
    """Check if a HuggingFace model is fully cached locally with weights."""
    snap = get_local_snapshot_dir(model_name)
    return snap != model_name and os.path.isdir(snap)


def get_local_snapshot_dir(model_name: str) -> str:
    """Find local snapshot directory for a model if valid weight files are present."""
    try:
        clean_name = model_name.replace("/", "--")
        cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{clean_name}/snapshots")
        if os.path.exists(cache_dir):
            snaps = [os.path.join(cache_dir, d) for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))]
            for snap in snaps:
                files = os.listdir(snap)
                if any(f.endswith(".safetensors") or f.endswith(".bin") or f == "model.safetensors.index.json" for f in files):
                    return snap
    except Exception:
        pass
    return model_name


def select_base_model() -> str:
    """
    Select the base model for fine-tuning.
    Always matches the main LM Studio model (from config.json / train_config.json)
    so the evolved GGUF directly replaces the model in use.
    """
    # 1. Explicit override
    override = os.environ.get("MODEL_OVERRIDE", "").strip()
    if override:
        print(f"[TRAIN] Using MODEL_OVERRIDE: {override}")
        return override

    free_vram = get_free_vram_gb()
    print(f"[TRAIN] Detected {free_vram:.1f} GB free VRAM after unloading LM Studio")

    # 2. Prefer exact local snapshot path for 100% offline loading from local disk
    local_snap = get_local_snapshot_dir(LARGE_MODEL)
    if os.path.isdir(local_snap):
        print(f"[TRAIN] Main model cached locally at snapshot path: {local_snap}")
        return local_snap

    print(f"[TRAIN] Using main model: {LARGE_MODEL}")
    return LARGE_MODEL



def get_lora_rank(free_vram_gb: float) -> int:
    """Choose LoRA rank based on available VRAM to avoid OOM errors."""
    if free_vram_gb >= 20.0:
        return 16   # Full rank for large VRAM
    elif free_vram_gb >= 12.0:
        return 8    # Half rank for medium VRAM
    else:
        return 4    # Minimal rank for low VRAM


def load_trajectories():
    if not os.path.exists(TRAJECTORIES_PATH):
        print(f"[TRAIN] No trajectories file found at {TRAJECTORIES_PATH}. Skipping.")
        return []
    try:
        with open(TRAJECTORIES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[TRAIN] Failed to load trajectories: {e}")
        return []


def format_dataset(trajectories):
    """Format SFT trajectories as chat messages for SFTTrainer."""
    formatted = []
    for item in trajectories:
        prompt = item.get("prompt", "")
        solution = item.get("solution", "")
        if not prompt or not solution:
            continue
        messages = [
            {"role": "system", "content": "You are the Engineer of an Autonomous AI system. Write correct and optimized Python code to solve the user's algorithmic requests."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": solution}
        ]
        formatted.append({"messages": messages})
    return formatted


def load_dpo_pairs():
    """Load DPO preference pairs from sandbox/dpo_pairs.json."""
    dpo_path = "sandbox/dpo_pairs.json"
    if not os.path.exists(dpo_path):
        print(f"[TRAIN] No DPO pairs file at {dpo_path}. Skipping DPO training.")
        return []
    try:
        with open(dpo_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[TRAIN] Failed to load DPO pairs: {e}")
        return []


def format_dpo_dataset(pairs):
    """
    Format DPO pairs for DPOTrainer.
    DPOTrainer expects a dataset with columns: prompt, chosen, rejected.
    Each is a formatted string (not a messages list) for efficiency.
    """
    system_msg = ("You are the Engineer of an Autonomous AI system. "
                  "Write correct and optimized Python code to solve the user's algorithmic requests.")
    formatted = []
    for item in pairs:
        prompt_text  = item.get("prompt", "")
        chosen_code  = item.get("chosen", "")
        rejected_code = item.get("rejected", "")
        if not prompt_text or not chosen_code or not rejected_code:
            continue
        # Format as instruction-response strings
        formatted.append({
            "prompt":   f"### System:\n{system_msg}\n\n### Task:\n{prompt_text}\n\n### Response:\n",
            "chosen":   chosen_code,
            "rejected": rejected_code,
        })
    return formatted


def merge_and_save_full_model(trainer, tokenizer, base_model_name: str) -> bool:
    """
    Merge LoRA adapter weights into the base model and save as a standalone model.
    
    This is the key step that enables Fix 1: the merged model is saved to
    MERGED_MODEL_DIR, and a flag file is written so main.py can load it on
    next startup, completing the recursive self-improvement loop.
    
    Returns True on success, False on failure.
    """
    print(f"\n[TRAIN] Merging LoRA adapter into base model...")
    try:
        from peft import PeftModel
        import torch
        from transformers import AutoModelForCausalLM

        # Save the adapter first
        trainer.model.save_pretrained(ADAPTER_OUTPUT_DIR)
        tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)
        print(f"[TRAIN] Adapter saved to {ADAPTER_OUTPUT_DIR}")

        # Free trainer model memory before reloading base for merge
        del trainer.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[TRAIN] Loading clean base model ({base_model_name}) in bfloat16 for GGUF merge...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Load on CPU for clean merge without VRAM conflict
            local_files_only=is_model_cached_locally(base_model_name),
        )
        peft_model = PeftModel.from_pretrained(base_model, ADAPTER_OUTPUT_DIR)
        print(f"[TRAIN] Merging LoRA adapter into unquantized base model weights...")
        merged_model = peft_model.merge_and_unload()

        # Force garbage collection to release open file handles
        import gc
        gc.collect()

        # Save to a fresh timestamped folder to prevent Windows OS Error 5 file locks
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target_merged_dir = os.path.abspath(f"sandbox/evolved_model_{timestamp_str}")
        os.makedirs(target_merged_dir, exist_ok=True)

        merged_dir = f"{MERGED_MODEL_DIR}_{timestamp_str}"
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        
        # Keep sandbox/evolved_model as clean alias for API hot-swap
        try:
            if os.path.exists(MERGED_MODEL_DIR):
                shutil.rmtree(MERGED_MODEL_DIR)
            shutil.copytree(merged_dir, MERGED_MODEL_DIR)
        except Exception as _alias_err:
            print(f"[TRAIN] Warning creating model alias: {_alias_err}")
            
        print(f"[TRAIN] Merged model saved to {os.path.abspath(merged_dir)}")

        # Check if GGUF target file path is available
        target_file, _ = find_lm_studio_model_dir()
        gguf_model_path = target_file if target_file else target_merged_dir

        #  FIX 1: Write the restart flag 
        # main.py reads this on startup and loads the evolved model instead of
        # the original base model, completing the recursive self-improvement loop.
        flag_data = {
            "evolved_model_path": gguf_model_path,
            "merged_hf_dir": os.path.abspath(MERGED_MODEL_DIR),
            "base_model": base_model_name,
            "adapter_path": os.path.abspath(ADAPTER_OUTPUT_DIR),
            "timestamp": time.time(),
            "note": "Load this model on next startup to apply evolved weights."
        }
        with open(RESTART_FLAG_PATH, "w", encoding="utf-8") as f:
            json.dump(flag_data, f, indent=4)
        print(f"[TRAIN] Restart flag written to {RESTART_FLAG_PATH}")
        print("[TRAIN] IMPORTANT: Restart main.py to load the evolved model weights.")
        return target_merged_dir

    except Exception as e:
        print(f"[TRAIN] WARNING: Merge failed ({e}). Adapter saved but not merged.")
        print("[TRAIN] The adapter is in sandbox/lora_adapter/ but requires manual merge.")
        return None


def unload_lm_studio_models() -> bool:
    """Unload ALL models from LM Studio to free VRAM before training."""
    print("[TRAIN] Unloading all LM Studio models to free VRAM for training...")

    # Use correct flag: -a (not --all)
    try:
        res = subprocess.run(["lms", "unload", "-a"], capture_output=True, text=True,
                             timeout=30, encoding="utf-8", errors="replace")
        if res.returncode == 0:
            print(f"[TRAIN] LM Studio unload result: {res.stdout.strip() or 'All models unloaded.'}")
            time.sleep(3)  # Give LM Studio time to free GPU memory
            freed = get_free_vram_gb()
            print(f"[TRAIN] VRAM available after unload: {freed:.1f} GB")
            return True
        else:
            print(f"[TRAIN] lms unload -a error: {res.stderr.strip()}")
    except Exception as e:
        print(f"[TRAIN] lms unload failed: {e}")

    # Fallback: LM Studio REST API v0
    try:
        import requests as _req
        _resp = _req.delete("http://127.0.0.1:1234/api/v0/models", timeout=10)
        if _resp.status_code in (200, 204):
            print("[TRAIN] Unloaded via LM Studio REST API.")
            time.sleep(3)
            return True
    except Exception:
        pass

    print("[TRAIN] WARNING: Could not unload LM Studio models. Training may fail due to insufficient VRAM.")
    return False


def find_lm_studio_model_dir() -> tuple[str | None, str | None]:
    """
    Dynamically discover the GGUF file path of the MAIN MODEL (configured in config.json)
    used for LM Studio inference.
    Returns (exact_gguf_file_path, model_key_or_path).
    """
    main_model_id = _cfg.get("model_id", "").lower()
    try:
        if os.path.exists("config.json"):
            with open("config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
                main_model_id = cfg.get("model_id", main_model_id).lower()
    except Exception:
        pass

    lm_studio_root = os.path.expanduser("~/.lmstudio/models")
    target_name = main_model_id.split("/")[-1] if "/" in main_model_id else main_model_id
    search_tokens = [t for t in target_name.replace("-", " ").replace("_", " ").split() if t]

    # First check lms ls --json if it points directly to an existing .gguf file
    try:
        res = subprocess.run(["lms", "ls", "--json"], capture_output=True, text=True, timeout=15, encoding="utf-8", errors="replace")
        if res.returncode == 0:
            models_data = json.loads(res.stdout)
            for item in models_data:
                if item.get("type") == "llm" and item.get("format") == "gguf":
                    rel_path = item.get("path", "")
                    model_key = item.get("modelKey") or item.get("indexedModelIdentifier") or ""
                    if rel_path:
                        full_model_file = os.path.join(lm_studio_root, os.path.normpath(rel_path))
                        if os.path.isfile(full_model_file) and full_model_file.lower().endswith(".gguf"):
                            if target_name in rel_path.lower() or target_name in model_key.lower():
                                return full_model_file, model_key
    except Exception as e:
        print(f"[TRAIN] Dynamic LM Studio discovery info ({e})")

    # Search inside ~/.lmstudio/models matching main model identifier tokens
    best_match = None
    best_score = -1
    best_key = main_model_id

    if os.path.exists(lm_studio_root):
        for root, dirs, files in os.walk(lm_studio_root):
            for file in files:
                if not file.lower().endswith(".gguf"):
                    continue
                if file.lower().startswith("mmproj"):
                    continue  # skip vision/multimodal projectors
                if file.lower().endswith(".part"):
                    continue  # skip incomplete downloads

                full_path = os.path.join(root, file)
                full_path_lower = full_path.lower()

                score = 0
                if target_name in full_path_lower:
                    score += 10
                for token in search_tokens:
                    if token in full_path_lower:
                        score += 2

                if score > best_score and score > 0:
                    best_score = score
                    best_match = full_path

    if best_match:
        print(f"[TRAIN] Discovered active GGUF file for '{main_model_id}': {best_match}")
        return best_match, best_key

    return None, None


def export_gguf_to_lm_studio(merged_hf_model_dir: str) -> str | None:
    """
    Convert merged HuggingFace model to GGUF format and overwrite the existing active GGUF model file.
    Ensures all loaded LLMs (in LM Studio and in Python RAM/VRAM) are fully unloaded before writing to disk.
    """
    print("\n[TRAIN] === AUTOMATIC GGUF OVERWRITE IN LM STUDIO ===")
    
    # 1. Force unload all models in LM Studio so file handles are unlocked
    print("[TRAIN] Unloading all models in LM Studio to release GGUF file locks...")
    unload_lm_studio_models()
    
    # 2. Force garbage collection in Python to release any held PyTorch model objects in memory
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[TRAIN] Cleared PyTorch memory and references.")
    except Exception:
        pass
        
    target_file, model_identifier = find_lm_studio_model_dir()
    
    if not target_file:
        print("[TRAIN] Warning: Active GGUF model file not found in LM Studio. Skipping overwrite.")
        return None

    # Always overwrite the primary active GGUF file directly in-place
    output_gguf_path = target_file
    print(f"[TRAIN] Overwriting active LM Studio model GGUF file in-place: {output_gguf_path}")

    # Remove any extra split chunks (e.g. -00002-of-00002.gguf) to ensure single-file integrity
    dir_name = os.path.dirname(target_file)
    for fname in os.listdir(dir_name):
        if fname.endswith(".gguf") and fname != os.path.basename(target_file):
            extra_path = os.path.join(dir_name, fname)
            try:
                os.remove(extra_path)
                print(f"[TRAIN] Cleaned up secondary split chunk: {fname}")
            except Exception:
                pass

    # Two-step GGUF pipeline: (1) convert HF→f16 GGUF, (2) quantize f16→Q5_K_M to match LM Studio format
    try:
        print(f"[TRAIN] Converting PyTorch safetensors ({merged_hf_model_dir}) to GGUF (Q5_K_M)...")

        # Locate convert_hf_to_gguf.py
        conv_script = None
        for path in [
            os.path.join("sandbox", "llama_cpp_repo", "convert_hf_to_gguf.py"),
            os.path.join(os.path.dirname(sys.executable), "convert_hf_to_gguf.py"),
            os.path.join(os.path.dirname(sys.executable), "Scripts", "convert_hf_to_gguf.py"),
            "convert_hf_to_gguf.py",
        ]:
            if os.path.exists(path):
                conv_script = path
                break

        # Locate llama-quantize binary (llama-cpp-python ships it as llama_cpp/llama-quantize)
        quantize_bin = None
        for path in [
            os.path.abspath("sandbox/llama_bin_cpu/llama-quantize.exe"),
            os.path.join(os.path.dirname(sys.executable), "llama-quantize.exe"),
            os.path.join(os.path.dirname(sys.executable), "llama-quantize"),
            os.path.join("sandbox", "llama_cpp_repo", "build", "bin", "Release", "llama-quantize.exe"),
            os.path.join("sandbox", "llama_cpp_repo", "build", "bin", "llama-quantize"),
        ]:
            if os.path.exists(path):
                quantize_bin = path
                break
        # Also search inside llama_cpp Python package
        if not quantize_bin:
            try:
                import llama_cpp
                pkg_dir = os.path.dirname(llama_cpp.__file__)
                for candidate in ["llama-quantize.exe", "llama-quantize"]:
                    p = os.path.join(pkg_dir, candidate)
                    if os.path.exists(p):
                        quantize_bin = p
                        break
            except ImportError:
                pass

        if not conv_script:
            print("[TRAIN] WARNING: convert_hf_to_gguf.py not found cannot convert to GGUF.")
            print("[TRAIN] Run: pip install llama-cpp-python and ensure sandbox/llama_cpp_repo exists.")
            return None

        # Step 1: Convert merged HF model -> temporary f16 GGUF inside sandbox/
        f16_temp = os.path.abspath("sandbox/temp_f16.gguf")
        cmd_convert = [sys.executable, conv_script, merged_hf_model_dir, "--outfile", f16_temp, "--outtype", "f16"]
        print(f"[TRAIN] Step 1 HF to f16 GGUF: {' '.join(cmd_convert)}")
        res = subprocess.run(cmd_convert, capture_output=True, text=True, timeout=1800, encoding="utf-8", errors="replace")
        if res.returncode != 0:
            safe_err = (res.stderr or "").encode('ascii', errors='ignore').decode('ascii').strip()
            print(f"[TRAIN] Conversion error: {safe_err}")
            return None
        print(f"[TRAIN] Step 1 complete f16 GGUF saved: {f16_temp}")

        # Step 2: Quantize f16 -> Q5_K_M to match LM Studio's original quantization
        if quantize_bin and os.path.exists(f16_temp):
            cmd_quantize = [quantize_bin, f16_temp, output_gguf_path, "Q5_K_M"]
            print(f"[TRAIN] Step 2 Quantizing f16 to Q5_K_M: {' '.join(cmd_quantize)}")
            res2 = subprocess.run(cmd_quantize, capture_output=True, text=True, timeout=1800, encoding="utf-8", errors="replace")
            if res2.returncode == 0:
                print(f"[TRAIN] Step 2 complete Q5_K_M GGUF written to: {output_gguf_path}")
                os.remove(f16_temp)  # Clean up temp file
            else:
                safe_err2 = (res2.stderr or res2.stdout or "").encode('ascii', errors='ignore').decode('ascii').strip()
                print(f"[TRAIN] Quantize warning: {safe_err2[:300]}")
                print(f"[TRAIN] Falling back to f16 GGUF (larger but usable).")
                shutil.move(f16_temp, output_gguf_path)
        else:
            if not quantize_bin:
                print("[TRAIN] llama-quantize not found using f16 GGUF (larger file, still usable).")
            shutil.move(f16_temp, output_gguf_path)

        print(f"[TRAIN] Exporting GGUF directly to LM Studio: {output_gguf_path}")
        # Clean up temporary unquantized model directories to free disk space
        if os.path.exists(f16_temp):
            try:
                os.remove(f16_temp)
            except Exception:
                pass
        if os.path.exists(merged_hf_model_dir):
            try:
                import shutil as _sh
                _sh.rmtree(merged_hf_model_dir)
                print(f"[TRAIN] Cleaned up temporary PyTorch merged directory: {merged_hf_model_dir}")
            except Exception:
                pass

        # Write metadata flag for LM Studio auto-reloader
        target_dir = os.path.dirname(target_file)
        gguf_flag = os.path.join(target_dir, "evolved_latest.json")
        with open(gguf_flag, "w", encoding="utf-8") as f:
            json.dump({
                "model_path": output_gguf_path,
                "timestamp": time.time(),
                "base_identifier": model_identifier
            }, f, indent=2)
            
        print("[TRAIN] [SUCCESS] Evolved GGUF successfully synced to LM Studio!")
        return output_gguf_path

    except Exception as e:
        print(f"[TRAIN] GGUF conversion/export note ({e}).")
        return None
    finally:
        # Mandatory cleanup of all temporary PyTorch merge artifacts to save ~250GB SSD space
        try:
            import shutil as _sh
            sandbox_dir = "sandbox"
            if os.path.exists(sandbox_dir):
                for item in os.listdir(sandbox_dir):
                    if item.startswith("evolved_model") or item.startswith("temp_f16"):
                        item_path = os.path.join(sandbox_dir, item)
                        if os.path.isdir(item_path):
                            _sh.rmtree(item_path, ignore_errors=True)
                        else:
                            os.remove(item_path)
            for temp_target in ["tmp_trainer", "sandbox/lora_adapter"]:
                if os.path.exists(temp_target):
                    if os.path.isdir(temp_target):
                        _sh.rmtree(temp_target, ignore_errors=True)
                    else:
                        os.remove(temp_target)
            print("[TRAIN] Cleaned up all temporary build artifacts.")
        except Exception as e:
            print(f"[TRAIN] Cleanup note: {e}")


def load_lm_studio_model(model_identifier_or_path: str = "") -> bool:
    """Load the evolved model into LM Studio using the model key from lms ls or config.json."""
    main_model_key = _cfg.get("model_id", "")
    
    # 1. Discover exact model key for the evolved GGUF file from lms ls
    discovered_key = None
    try:
        res_ls = subprocess.run(["lms", "ls", "--json"], capture_output=True, text=True, timeout=15, encoding="utf-8", errors="replace")
        if res_ls.returncode == 0:
            models_data = json.loads(res_ls.stdout)
            for item in models_data:
                if item.get("type") == "llm":
                    path = item.get("path", "")
                    key = item.get("modelKey") or item.get("indexedModelIdentifier")
                    if "evolved" in path.lower() or "evolved" in (key or "").lower():
                        discovered_key = key
                        break
    except Exception as e:
        print(f"[TRAIN] Dynamic model key discovery info: {e}")

    if discovered_key:
        main_model_key = discovered_key
        print(f"[TRAIN] Discovered evolved model key in LM Studio: {main_model_key}")
    else:
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r", encoding="utf-8") as f:
                    main_model_key = json.load(f).get("model_id", main_model_key)
        except Exception:
            pass

    print(f"[TRAIN] Reloading evolved model into LM Studio: {main_model_key}")
    try:
        res = subprocess.run(
            ["lms", "load", main_model_key],
            capture_output=True, text=True,
            timeout=300,
            encoding="utf-8", errors="replace"
        )
        if res.returncode == 0:
            print("[TRAIN] [SUCCESS] Evolved model reloaded into LM Studio!")
            safe_out = res.stdout.encode('ascii', 'ignore').decode('ascii').strip()
            print(f"[TRAIN] LM Studio output: {safe_out}")
            
            # Update config.json with the active evolved model key
            try:
                if os.path.exists("config.json"):
                    with open("config.json", "r", encoding="utf-8") as f:
                        cfg_data = json.load(f)
                    cfg_data["model_id"] = main_model_key
                    with open("config.json", "w", encoding="utf-8") as f:
                        json.dump(cfg_data, f, indent=4)
                    print(f"[TRAIN] Updated config.json model_id -> {main_model_key}")
            except Exception as _e:
                print(f"[TRAIN] config.json update note: {_e}")

            return True
        else:
            safe_err = (res.stderr or res.stdout or "").encode('ascii', 'ignore').decode('ascii').strip()
            print(f"[TRAIN] lms load info: {safe_err}")
    except Exception as e:
        print(f"[TRAIN] Failed to reload model in LM Studio: {e}")
    return False


def run_training():
    print("=" * 60)
    print("SINGULARITY LOCAL QLORA FINE-TUNING")
    print("=" * 60)

    # Pause server inference calls so LM Studio is never auto-loaded during export
    try:
        import requests as _req
        _req.post("http://127.0.0.1:8000/pause_inference", timeout=5)
    except Exception:
        pass

    trajectories = load_trajectories()
    if not trajectories:
        print("[TRAIN] No trajectories   skipping training run.")
        return

    print(f"[TRAIN] Loaded {len(trajectories)} training trajectories.")

    # Auto-unload LM Studio model before loading heavy PyTorch fine-tuning models
    unload_lm_studio_models()

    # Import heavy dependencies only when actually needed
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        print(f"[TRAIN] ERROR: Missing dependency: {e}")
        print("[TRAIN] Run: pip install transformers peft trl datasets bitsandbytes")
        return

    # Select model and VRAM-appropriate LoRA rank
    free_vram = get_free_vram_gb()
    base_model_name = select_base_model()
    lora_rank = get_lora_rank(free_vram)
    print(f"[TRAIN] Base model: {base_model_name}")
    print(f"[TRAIN] LoRA rank: {lora_rank} (VRAM: {free_vram:.1f} GB)")

    formatted_data = format_dataset(trajectories)
    if not formatted_data:
        print("[TRAIN] No valid formatted samples   skipping.")
        return

    dataset = Dataset.from_list(formatted_data)

    print(f"[TRAIN] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            local_files_only=is_model_cached_locally(base_model_name),
        )
    except Exception as e:
        print(f"[TRAIN] Tokenizer load failed: {e}")
        print("[TRAIN] Make sure the model is downloaded or set MODEL_OVERRIDE env variable.")
        return

    tokenizer.pad_token = tokenizer.eos_token

    # Dynamic precision detection for Windows PyTorch compatibility
    has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not has_bf16
    compute_dtype = torch.bfloat16 if has_bf16 else torch.float16

    # 4-bit QLoRA   works on RTX 4090 (24GB) for both 7B and 32B
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )

    print(f"[TRAIN] Loading model in 4-bit quantization (bf16={has_bf16}, fp16={use_fp16})...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
            local_files_only=is_model_cached_locally(base_model_name),
        )
    except Exception as e:
        print(f"[TRAIN] Model load failed: {e}")
        return

    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = SFTConfig(
        output_dir="./tmp_trainer",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        logging_steps=1,
        learning_rate=2e-4,
        fp16=use_fp16,
        bf16=has_bf16,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        max_length=1024
    )

    print("[TRAIN] Starting Supervised Fine-Tuning...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args
    )

    start_time = time.time()
    train_result = trainer.train()
    elapsed_time = time.time() - start_time
    print(f"[TRAIN] Fine-tuning completed in {elapsed_time:.2f}s")

    #  FIX 1 + FIX 5: Merge adapter and write restart flag 
    clean_disk_cache_if_needed()
    merged_dir_res = merge_and_save_full_model(trainer, tokenizer, base_model_name)

    if merged_dir_res:
        # Automatic GGUF conversion and export directly to detected LM Studio directory
        gguf_path = export_gguf_to_lm_studio(merged_dir_res)

        # Attempt to auto-load the evolved model into LM Studio
        target_model = gguf_path if gguf_path else merged_dir_res
        load_lm_studio_model(target_model)

        try:
            import requests as _req
            _server_url = "http://127.0.0.1:8000"
            print(f"\n[TRAIN] Notifying AI server to hot-swap to evolved model...")
            _resp = _req.post(f"{_server_url}/reload_evolved_model", timeout=300)
            _result_data = _resp.json()
            _status = _result_data.get("status", "unknown")
            _message = _result_data.get("message", "")
            print(f"[TRAIN] Hot-swap result: {_status} {_message}")
            if _status == "success":
                print("[TRAIN] [OK] Recursive loop complete   server is now running evolved weights!")
        except Exception as _e:
            print(f"[TRAIN] Hot-swap notification failed ({_e}).")

    # Resume server inference calls
    try:
        import requests as _req
        _req.post("http://127.0.0.1:8000/resume_inference", timeout=5)
    except Exception:
        pass

    # Write training log
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOGS_DIR, f"fine_tuning_event_{timestamp}.md")
    avg_loss = train_result.training_loss

    log_content = f"""# Evolved LLM Weight Fine-Tuning Event
* **Date/Time**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
* **Base Model**: `{base_model_name}`
* **LoRA Rank**: `{lora_rank}`
* **Free VRAM at start**: `{free_vram:.1f} GB`
* **Training Duration**: `{elapsed_time:.2f} seconds`
* **Training Samples**: `{len(trajectories)}`
* **Average Training Loss**: `{avg_loss:.4f}`
* **Adapter Output**: `{ADAPTER_OUTPUT_DIR}`
* **Merged Model Output**: `{merged_dir_res if merged_dir_res else "MERGE FAILED ,  adapter only"}`
* **Restart Flag Written**: `{'YES ,  restart main.py to load evolved weights' if merged_dir_res else 'NO ,  merge failed'}`

## Trained Prompts:
{chr(10).join([f"- {item.get('prompt')}" for item in trajectories])}

## Training Parameters:
* Quantization: `4-bit NF4`
* LoRA rank `r`: `{lora_rank}` (VRAM-adaptive)
* Learning Rate: `2e-4`
* Epochs: `3`
"""
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(log_content)

    print(f"[TRAIN] Training log saved to {log_filename}")
    print("=" * 60)


def run_dpo_training():
    """
    DPO (Direct Preference Optimization) training path.

    Uses preference pairs (chosen=passed all tests, rejected=failed at least one)
    instead of successful trajectories alone. This teaches the model WHY one
    solution is better than another, not just what a correct solution looks like.

    The existing merge + Q5_K_M GGUF pipeline is reused unchanged after training.
    """
    print("=" * 60)
    print("SINGULARITY DPO FINE-TUNING (Direct Preference Optimization)")
    print("=" * 60)

    # Pause inference
    try:
        import requests as _req
        _req.post("http://127.0.0.1:8000/pause_inference", timeout=5)
    except Exception:
        pass

    pairs = load_dpo_pairs()
    if not pairs:
        print("[TRAIN] No DPO pairs available   falling back to SFT.")
        run_training()
        return

    print(f"[TRAIN] Loaded {len(pairs)} DPO preference pairs.")
    print(f"[TRAIN] Avg chosen/rejected pass rate contrast:")
    chosen_rates   = [p.get("chosen_pass_rate", 1.0) for p in pairs]
    rejected_rates = [p.get("rejected_pass_rate", 0.0) for p in pairs]
    print(f"  chosen:   {sum(chosen_rates)/len(chosen_rates):.1%}")
    print(f"  rejected: {sum(rejected_rates)/len(rejected_rates):.1%}")

    unload_lm_studio_models()

    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import DPOTrainer, DPOConfig
    except ImportError as e:
        print(f"[TRAIN] ERROR: Missing dependency: {e}")
        print("[TRAIN] Run: pip install transformers peft trl datasets bitsandbytes")
        return

    free_vram = get_free_vram_gb()
    base_model_name = select_base_model()
    lora_rank = get_lora_rank(free_vram)
    print(f"[TRAIN] Base model:  {base_model_name}")
    print(f"[TRAIN] LoRA rank:   {lora_rank} (VRAM: {free_vram:.1f} GB)")
    print(f"[TRAIN] Training mode: DPO")

    formatted = format_dpo_dataset(pairs)
    if not formatted:
        print("[TRAIN] No valid DPO samples after formatting   skipping.")
        return

    dataset = Dataset.from_list(formatted)
    print(f"[TRAIN] DPO dataset: {len(dataset)} pairs")

    # Tokenizer
    print("[TRAIN] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            local_files_only=is_model_cached_locally(base_model_name),
        )
    except Exception as e:
        print(f"[TRAIN] Tokenizer load failed: {e}")
        return

    tokenizer.pad_token = tokenizer.eos_token

    # Dynamic precision detection for DPO
    has_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not has_bf16
    compute_dtype = torch.bfloat16 if has_bf16 else torch.float16

    # 4-bit QLoRA   same config as SFT path
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )

    print(f"[TRAIN] Loading model in 4-bit for DPO (bf16={has_bf16}, fp16={use_fp16})...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
            local_files_only=not is_model_cached_locally(base_model_name),
        )
    except Exception as e:
        print(f"[TRAIN] Model load failed: {e}")
        return

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dpo_args = DPOConfig(
        output_dir="./tmp_trainer",
        num_train_epochs=1,          # DPO is sample-efficient; 1 epoch is typical
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        logging_steps=1,
        learning_rate=5e-5,          # Lower LR for DPO than SFT
        fp16=use_fp16,
        bf16=has_bf16,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        beta=0.1,                    # KL penalty coefficient (DPO-specific)
        max_length=1024,
        max_prompt_length=256,
        gradient_checkpointing=True,
    )

    print("[TRAIN] Starting DPO training...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,        # None  use implicit reference (saves VRAM on 4090)
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    start_time = time.time()
    train_result = trainer.train()
    elapsed_time = time.time() - start_time
    print(f"[TRAIN] DPO training completed in {elapsed_time:.2f}s")
    print(f"[TRAIN] Final DPO loss: {train_result.training_loss:.4f}")

    # Merge + GGUF export   reuse identical pipeline as SFT
    clean_disk_cache_if_needed()
    merged_dir_res = merge_and_save_full_model(trainer, tokenizer, base_model_name)

    if merged_dir_res:
        gguf_path = export_gguf_to_lm_studio(merged_dir_res)
        target_model = gguf_path if gguf_path else merged_dir_res
        load_lm_studio_model(target_model)

        try:
            import requests as _req
            _server_url = "http://127.0.0.1:8000"
            print("\n[TRAIN] Notifying AI server to hot-swap to DPO-evolved model...")
            _resp = _req.post(f"{_server_url}/reload_evolved_model", timeout=300)
            _status = _resp.json().get("status", "unknown")
            print(f"[TRAIN] Hot-swap result: {_status}")
        except Exception as _e:
            print(f"[TRAIN] Hot-swap notification failed ({_e}).")

    # Resume inference
    try:
        import requests as _req
        _req.post("http://127.0.0.1:8000/resume_inference", timeout=5)
    except Exception:
        pass

    # Write training log
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOGS_DIR, f"dpo_training_event_{timestamp}.md")
    log_content = f"""# DPO Fine-Tuning Event
* **Date/Time**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
* **Training Mode**: DPO (Direct Preference Optimization)
* **Base Model**: `{base_model_name}`
* **LoRA Rank**: `{lora_rank}`
* **Free VRAM at start**: `{free_vram:.1f} GB`
* **Training Duration**: `{elapsed_time:.2f} seconds`
* **DPO Pairs**: `{len(pairs)}`
* **DPO Loss**: `{train_result.training_loss:.4f}`
* **Beta (KL penalty)**: `0.1`
* **Merged Model**: `{merged_dir_res if merged_dir_res else "MERGE FAILED"}`

## Trained Problems:
{chr(10).join([f"- {p.get('prompt', 'unknown')}" for p in pairs])}

## Training Parameters:
* Quantization: `4-bit NF4`
* LoRA rank `r`: `{lora_rank}` (VRAM-adaptive)
* Learning Rate: `5e-5` (DPO-appropriate)
* Epochs: `1`
* ref_model: `None` (implicit reference, saves VRAM)
"""
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(log_content)

    print(f"[TRAIN] DPO training log saved to {log_filename}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Singularity Fine-Tuning")
    parser.add_argument(
        "--mode",
        choices=["sft", "dpo"],
        default="sft",
        help="Training mode: 'sft' (supervised) or 'dpo' (preference optimization)"
    )
    args = parser.parse_args()

    if args.mode == "dpo":
        print("[TRAIN] Mode: DPO (Direct Preference Optimization)")
        run_dpo_training()
    else:
        print("[TRAIN] Mode: SFT (Supervised Fine-Tuning)")
        run_training()
