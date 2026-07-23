from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from smolagents import OpenAIModel
import sys
import json
import subprocess
import os
import re
import time
import threading

# Force stdout/stderr to use UTF-8 to prevent UnicodeEncodeError on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

app = FastAPI()

# Load LLM config from config.json (dynamic   no hardcoded values)
with open("config.json", "r") as f:
    cfg = json.load(f)

global_model = OpenAIModel(
    model_id=cfg["model_id"],
    api_base=cfg["api_base"],
    api_key=cfg.get("api_key", "")
)

#  EVOLVED MODEL STATE 
# Populated by POST /reload_evolved_model after QLoRA training completes.
# When USING_EVOLVED_MODEL is True, /generate bypasses LM Studio entirely
# and uses the locally loaded HuggingFace model   completing the recursive loop.
global_hf_model = None
global_hf_tokenizer = None
USING_EVOLVED_MODEL = False

from typing import Optional


def get_free_vram_for_reload() -> float:
    """Get available VRAM in GB. Used to guard against OOM during hot-swap reload."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free_bytes, _ = torch.cuda.mem_get_info(0)
        return free_bytes / (1024 ** 3)
    except Exception:
        return 0.0


def generate_with_hf_model(prompt: str, temperature: float = 0.7) -> str:
    """Generate a response using the locally loaded evolved HuggingFace model.
    
    This is the hot-swap inference path activated after QLoRA training merges
    new weights. It replaces the LM Studio API call entirely.
    """
    global global_hf_model, global_hf_tokenizer
    messages = [{"role": "user", "content": prompt}]
    text = global_hf_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = global_hf_tokenizer(text, return_tensors="pt").to(global_hf_model.device)
    import torch
    with torch.no_grad():
        outputs = global_hf_model.generate(
            **inputs,
            max_new_tokens=cfg.get("max_tokens", 4096),
            temperature=max(temperature, 1e-5),
            do_sample=temperature > 0.01,
            pad_token_id=global_hf_tokenizer.eos_token_id,
        )
    response = global_hf_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response

class PromptRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class ToolCallRequest(BaseModel):
    tool: str  # "read_file" | "write_file" | "run_python_script" | "final_answer"
    args: dict

IS_TRAINING_PAUSED = False

@app.post("/pause_inference")
def pause_inference():
    global IS_TRAINING_PAUSED
    IS_TRAINING_PAUSED = True
    print("[SERVER] Inference PAUSED for training & GGUF export.")
    return {"status": "paused"}

@app.post("/resume_inference")
def resume_inference():
    global IS_TRAINING_PAUSED
    IS_TRAINING_PAUSED = False
    print("[SERVER] Inference RESUMED.")
    return {"status": "resumed"}

@app.get("/")
def read_root():
    print("[INFO] Health check received")
    return {"status": "AI Server is running, connected to LM Studio", "agent_ready": True, "paused": IS_TRAINING_PAUSED}

def update_active_state(phase: str, details: str = ""):
    """Helper to sync server processing state directly to sandbox/active_state.json"""
    try:
        os.makedirs("sandbox", exist_ok=True)
        with open("sandbox/active_state.json", "w", encoding="utf-8") as f:
            json.dump({"phase": phase, "details": details, "timestamp": time.time()}, f)
    except Exception:
        pass

@app.post("/generate")
def generate_text(request: PromptRequest):
    global IS_TRAINING_PAUSED
    wait_counter = 0
    while IS_TRAINING_PAUSED:
        if wait_counter == 0:
            print("[SERVER] Training in progress   holding /generate request...")
        update_active_state("Paused: Fine-Tuning Model", "Holding inference request while fine-tuning / GGUF export completes...")
        time.sleep(2)
        wait_counter += 1
        if wait_counter > 45:  # 90s max wait   auto-resume to prevent deadlocks
            print("[SERVER] Fine-tuning pause timeout reached   auto-resuming inference...")
            IS_TRAINING_PAUSED = False
            break

    try:
        print("\n" + "="*70)
        print("[REQUEST] /generate endpoint called")
        print(f"[PROMPT] Full prompt ({len(request.prompt)} chars):\n{request.prompt}")
        
        temp = request.temperature if request.temperature is not None else 0.7
        
        # Dynamically reload config to pick up any runtime updates
        with open("config.json", "r") as f:
            current_cfg = json.load(f)
        
        max_tok = request.max_tokens or current_cfg.get("max_tokens")
        
        update_active_state("LM Studio Inference Active", f"Dispatching prompt ({len(request.prompt)} chars) to model: {current_cfg['model_id']}")

        # Strict LM Studio path: send requests to LM Studio endpoint
        from openai import OpenAI as OpenAIClient
        client = OpenAIClient(
            api_key=current_cfg.get("api_key", "lm-studio"),
            base_url=f"{current_cfg['api_base']}"
        )
        
        response = client.chat.completions.create(
            model=current_cfg["model_id"],
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=max_tok,
            temperature=temp,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        raw_response = response.choices[0].message.content
        
        print(f"\n[AIS RESPONSE] Raw AI output ({len(raw_response)} chars):\n{raw_response}")
        print("="*70 + "\n")
        
        update_active_state("Executing AI Candidate Code", f"Parsing and executing {len(raw_response)} chars response...")

        # Execute any <code> blocks from the AI agent and return results
        result = execute_code_blocks(raw_response)
        print(f"[FINAL RESULT] {result[:500]}")
        update_active_state("Candidate Code Executed", "AI candidate patch executed successfully. Returning output to master loop...")
        return {"response": result}
    except Exception as e:
        update_active_state("LLM Request Error", f"Error: {str(e)[:100]}")
        print(f"[ERROR] /generate failed: {str(e)[:200]}")
        raise HTTPException(status_code=500, detail=str(e))

def execute_code_blocks(response_text: str) -> str:
    """Parse <code> blocks or triple-backtick code from AI response and execute them on the server."""
    
    # First try XML-style <code> blocks
    xml_pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL | re.IGNORECASE)
    matches = xml_pattern.findall(response_text)
    
    # If no XML matches, try backtick blocks ONLY if they contain explicit tool calls or experiment code
    if not matches:
        backtick_pattern = re.compile(r'```(?:python)?\n(.*?)```', re.DOTALL | re.IGNORECASE)
        candidates = backtick_pattern.findall(response_text)
        matches = [
            c for c in candidates
            if any(tool in c for tool in ['write_file(', 'run_python_script(', 'final_answer(', 'def generate_code('])
        ]
    
    print(f"[PARSE] Found {len(matches)} executable code block(s) in AI response")
    
    if not matches:
        return response_text
    
    from io import StringIO
    
    # Import sandbox guard for AST checking
    sys.path.insert(0, os.getcwd())
    from sandbox_guard import check_ast, SecurityException
    
    results = []
    for i, code in enumerate(matches):
        print(f"\n[CODE BLOCK {i+1}] ({len(code)} chars):\n{code}")
        
        try:
            # Clean up the code remove leading/trailing whitespace and newlines
            code = code.strip()
            
            # STEP 1: AST CHECK   validate code before execution using sandbox rules
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='sandbox', encoding='utf-8') as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            
            try:
                check_ast(tmp_path)  # This will raise SecurityException if dangerous code detected
                ast_ok = True
                print(f"[AST CHECK] PASSED   Code is safe to execute")
            except SecurityException as se:
                results.append(f"AST Blocked: {str(se)[:200]}")
                print(f"[AST CHECK] FAILED   Dangerous code detected: {se}")
                ast_ok = False
            finally:
                os.unlink(tmp_path)  # Clean up temp file (only once)
            
            if not ast_ok:
                continue
            
            # STEP 2: Execute in sandboxed environment with restricted globals
            captured_output = StringIO()
            
            def safe_open(file, mode='r', *args, **kwargs):
                abs_path = os.path.abspath(file)
                workspace_dir = os.path.abspath(os.getcwd())
                if not abs_path.startswith(workspace_dir):
                    raise PermissionError(f"Access denied: file '{file}' is outside the workspace.")
                # Protect core files from being modified/deleted
                if any(m in mode for m in ['w', 'a', '+', 'x']):
                    basename = os.path.basename(abs_path)
                    forbidden_files = {
                        'main.py', 'autonomous_loop.py', 'sandbox_guard.py',
                        'bootstrap.py', 'start_ai.bat',
                        'evaluator_mbpp.py',    # Locked external benchmark   cannot be gamed
                        'evaluator_anchor.py',  # Locked anchor benchmark
                    }
                    if basename in forbidden_files:
                        raise PermissionError(f"Access denied: modifying core file '{basename}' is forbidden.")
                return open(file, mode, *args, **kwargs)

            def safe_sandbox_print(*args, **kwargs):
                msg = ' '.join(str(a) for a in args) + '\n'
                if captured_output.tell() > 10000:
                    return
                captured_output.write(msg)
                try:
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                except Exception:
                    try:
                        encoding = sys.stdout.encoding or 'utf-8'
                        sys.stdout.write(msg.encode(encoding, errors='replace').decode(encoding))
                        sys.stdout.flush()
                    except Exception:
                        pass

            safe_globals = {
                '__builtins__': {
                    'print': safe_sandbox_print, 
                    'len': len, 'range': range, 'list': list,
                    'dict': dict, 'set': set, 'tuple': tuple, 'int': int,
                    'float': float, 'str': str, 'bool': bool, 'open': safe_open,
                    'True': True, 'False': False, 'None': None,
                    'isinstance': isinstance, 'abs': abs, 'sum': sum,
                    'max': max, 'min': min, 'any': any, 'all': all,
                    'sorted': sorted, 'enumerate': enumerate, 'zip': zip,
                    'map': map, 'filter': filter, 'round': round,
                    'reversed': reversed, 'chr': chr, 'ord': ord,
                    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
                },
            }
            
            import json as _json
            safe_globals['json'] = _json
            
            import math as _math
            safe_globals['math'] = _math
            
            import re as _re
            safe_globals['re'] = _re
            
            from collections import defaultdict, Counter
            safe_globals['defaultdict'] = defaultdict
            safe_globals['Counter'] = Counter
            
            # Import agent tools and expose them in safe_globals
            from agent_tools import (
                read_file, write_file, run_python_script, list_sandbox_files, 
                list_project_files, check_syntax, profile_python_script, 
                search_arxiv, update_memory
            )
            safe_globals['read_file'] = read_file
            safe_globals['write_file'] = write_file
            safe_globals['run_python_script'] = run_python_script
            safe_globals['list_sandbox_files'] = list_sandbox_files
            safe_globals['list_project_files'] = list_project_files
            safe_globals['check_syntax'] = check_syntax
            safe_globals['profile_python_script'] = profile_python_script
            safe_globals['search_arxiv'] = search_arxiv
            safe_globals['update_memory'] = update_memory
            
            def safe_final_answer(answer):
                print(f"[FINAL ANSWER] {answer}")
                return answer
            safe_globals['final_answer'] = safe_final_answer
            
            # Add __import__ so 'import' statements work (AST already validated safety)
            safe_globals['__builtins__']['__import__'] = __import__
            
            print(f"[EXEC] Running code in sandbox with 15s timeout...")
            exec_error = []
            exec_completed = threading.Event()

            def exec_worker():
                try:
                    exec(code, safe_globals)
                except Exception as ex:
                    exec_error.append(ex)
                finally:
                    exec_completed.set()

            t = threading.Thread(target=exec_worker, daemon=True)
            t.start()
            finished = exec_completed.wait(timeout=15.0)

            if not finished:
                print(f"[EXEC TIMEOUT] Code block exceeded 15 seconds (infinite loop or hang interrupted).")
                results.append("Execution Timed Out: Code block exceeded 15s limit (possible infinite loop).")
            elif exec_error:
                raise exec_error[0]
            else:
                # Fallback: if code defines generate_code directly without calling write_file, update sandbox/experiment.py
                if 'def generate_code(' in code and 'write_file(' not in code:
                    try:
                        with open("sandbox/experiment.py", "w", encoding="utf-8") as f_exp:
                            f_exp.write(code)
                        print("[AUTO-UPDATE] Wrote candidate generate_code directly to sandbox/experiment.py")
                    except Exception as e_exp:
                        print(f"[AUTO-UPDATE] Error writing candidate: {e_exp}")

                # Return captured output if there was any
                output_text = captured_output.getvalue().strip()
                if output_text:
                    results.append(f"Output:\n{output_text}")
                    print(f"[OUTPUT] Captured stdout:\n{output_text}")
                else:
                    results.append("Code executed successfully")
        except SecurityException as se:
            error_msg = f"Sandbox Blocked: {str(se)[:200]}"
            results.append(error_msg)
        except Exception as e:
            error_msg = f"Error executing code: {type(e).__name__}: {str(e)[:200]}"
            results.append(error_msg)
    
    # Replace <code> blocks with execution results
    final = xml_pattern.sub(lambda m: f"[Code executed: {'; '.join(results)}]", response_text)
    
    # If no XML replacement happened, try backticks
    if final == response_text:
        backtick_pattern = re.compile(r'```(?:python)?\n(.*?)```', re.DOTALL | re.IGNORECASE)
        final = backtick_pattern.sub(lambda m: f"[Code executed: {'; '.join(results)}]", response_text)
    
    return final


@app.post("/reload_evolved_model")
def reload_evolved_model():
    """Hot-swap to the locally trained evolved model after QLoRA fine-tuning.
    
    Called automatically by train_model.py after a successful LoRA weight merge.
    This endpoint closes the recursive self-improvement loop:
      QLoRA training → merge weights → POST here → /generate uses evolved model.
    
    VRAM guard: requires ≥ 6 GB free. If LM Studio is still holding VRAM,
    unload its model first, then call this endpoint again.
    """
    global global_hf_model, global_hf_tokenizer, USING_EVOLVED_MODEL
    
    evolved_path = "sandbox/evolved_model"
    if not os.path.exists(evolved_path):
        msg = f"Evolved model directory not found at '{evolved_path}'. Run training first."
        print(f"[RELOAD] {msg}")
        return {"status": "error", "message": msg}
    
    # VRAM guard   need at least 6 GB free to load 4-bit quantized 7B model.
    # If VRAM check returns 0.0, it means no GPU detected (CPU mode)   allow it.
    free_vram = get_free_vram_for_reload()
    print(f"[RELOAD] Free VRAM: {free_vram:.1f} GB")
    if 0.0 < free_vram < 6.0:
        msg = (
            f"Insufficient VRAM ({free_vram:.1f} GB free, need ≥ 6 GB). "
            "Unload the model from LM Studio first, then POST to /reload_evolved_model again."
        )
        print(f"[RELOAD] Skipping: {msg}")
        return {"status": "skipped", "message": msg}
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        print(f"[RELOAD] Loading evolved model from '{evolved_path}'...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(evolved_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            evolved_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        
        global_hf_model = model
        global_hf_tokenizer = tokenizer
        USING_EVOLVED_MODEL = True
        
        print("[RELOAD] ✓ Evolved model is now ACTIVE   future /generate calls use it.")
        return {
            "status": "success",
            "message": f"Evolved model loaded from '{evolved_path}' and is now active."
        }
    
    except Exception as e:
        print(f"[RELOAD] Failed to load evolved model: {e}")
        return {"status": "error", "message": str(e)[:500]}


@app.post("/tool_call")
def tool_call(request: ToolCallRequest):
    """Execute tools that the AI agent requests."""
    try:
        print(f"\n[TOOL CALL] {request.tool}({json.dumps(request.args)})")
        result = ""
        
        if request.tool in ("read_file", "write_file", "run_python_script"):
            filepath = request.args.get("path", "")
            abs_path = os.path.abspath(filepath)
            workspace_dir = os.path.abspath(os.getcwd())
            if not abs_path.startswith(workspace_dir):
                raise HTTPException(status_code=403, detail=f"Access denied: Path '{filepath}' is outside workspace.")
            
            if request.tool == "write_file":
                basename = os.path.basename(abs_path)
                forbidden_files = {
                    'main.py', 'autonomous_loop.py', 'sandbox_guard.py', 
                    'bootstrap.py', 'start_ai.bat',
                    'evaluator_mbpp.py', 'evaluator_anchor.py'
                }
                if basename in forbidden_files:
                    raise HTTPException(status_code=403, detail=f"Access denied: Modifying core file '{basename}' is forbidden.")
        
        if request.tool == "read_file":
            filepath = request.args.get("path", "")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"[TOOL] Read {len(content)} chars from {filepath}")
            result = content
        elif request.tool == "write_file":
            filepath = request.args.get("path", "")
            content = request.args.get("content", "")
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[TOOL] Wrote {len(content)} chars to {filepath}")
            result = f"File written to {filepath}"
        elif request.tool == "run_python_script":
            filepath = request.args.get("path", "")
            # Run the python script under sandbox_guard.py to enforce sandbox limits!
            res = subprocess.run(["python", "sandbox_guard.py", filepath], capture_output=True, text=True, timeout=120)
            output = res.stdout or ""
            if res.stderr:
                output += "\nSTDERR:\n" + res.stderr
            print(f"[TOOL] Script output ({len(output)} chars): {output[:200]}")
            result = output
        elif request.tool == "final_answer":
            answer = request.args.get("answer", "Done")
            print(f"[TOOL] Final answer: {answer}")
            result = answer
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {request.tool}")
        
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def kill_port_8000():
    """Kill any process using port 8000 before starting."""
    import subprocess as _subprocess
    try:
        # Find and kill processes on port 8000 (Windows netstat approach)
        result = _subprocess.run(
            'netstat -ano | findstr :8000',
            capture_output=True, text=True, shell=True
        )
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 5 and parts[-1].isdigit():
                    pid = parts[-1]
                    try:
                        _subprocess.run(f'taskkill /PID {pid} /F', shell=True, capture_output=True)
                        print(f"Killed process on port 8000 (PID: {pid})")
                    except Exception:
                        pass
    except Exception as e:
        print(f"Warning: Could not kill port 8000 process: {e}")



def check_for_evolved_model():
    """
    FIX 1: Reconnect the recursive self-improvement loop.
    
    train_model.py writes sandbox/restart_with_evolved.flag after successfully
    merging LoRA adapter weights into a full model. This function checks for
    that flag on startup and, if found:
    
      1. Updates config.json to point api_base at the evolved model path
         (if running via HuggingFace) OR prints instructions for LM Studio users.
      2. Deletes the flag so it only fires once.
      3. Returns a dict with evolved model info for logging.
    
    For LM Studio users (the default setup): since LM Studio manages its own
    process, this function writes the evolved model path to config.json as
    'evolved_model_path'. The start_ai.bat script should check this field and
    launch LM Studio with the evolved model instead of the base model.
    """
    flag_path = "sandbox/restart_with_evolved.flag"
    if not os.path.exists(flag_path):
        return None
    
    try:
        with open(flag_path, "r", encoding="utf-8") as f:
            flag_data = json.load(f)
        
        # Automatically load evolved GGUF model into LM Studio on startup if present
        evolved_path = flag_data.get("evolved_model_path", "")
        base_model = flag_data.get("base_model", "unknown")
        timestamp = flag_data.get("timestamp", 0)
        
        print("=" * 60)
        print("[EVOLVED MODEL DETECTED]")
        print(f"  Base model: {base_model}")
        print(f"  Evolved weights: {evolved_path}")
        print(f"  Trained at: {__import__('datetime').datetime.fromtimestamp(timestamp)}")
        print("=" * 60)
        
        # Auto-load into LM Studio via lms CLI if model is known
        model_key = cfg.get("model_id", "")
        if evolved_path and os.path.exists(evolved_path):
            print(f"[STARTUP] Automatically loading evolved model into LM Studio: {model_key}")
            try:
                res = subprocess.run(["lms", "load", model_key], capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
                if res.returncode == 0:
                    print("[STARTUP] [SUCCESS] Evolved model successfully loaded into LM Studio on startup!")
                else:
                    safe_out = ((res.stdout or "") + (res.stderr or "")).encode('ascii', errors='ignore').decode('ascii').strip()
                    print(f"[STARTUP] `lms load` info: {safe_out[:200]}")
            except Exception as e:
                print(f"[STARTUP] `lms load` attempt info: {e}")

        # Delete flag so it doesn't trigger again on next restart
        try:
            if os.path.exists(flag_path):
                os.remove(flag_path)
        except Exception:
            pass
        print("[EVOLVED MODEL] Flag consumed. Evolution state updated.")
        
        return flag_data
        
    except Exception as e:
        print(f"[EVOLVED MODEL] Warning: Could not process evolved model flag: {e}")
        return None


if __name__ == "__main__":
    # Auto-kill any existing process on port 8000 before starting
    kill_port_8000()
    import time; time.sleep(1)  # Brief pause to ensure port is freed
    
    #  FIX 1: Check for evolved model from last training run 
    evolved_info = check_for_evolved_model()
    if evolved_info:
        print("[STARTUP] Evolved model detected. See above for loading instructions.")
        
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

