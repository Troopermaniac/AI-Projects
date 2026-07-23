# Ouroboros Engine — Singularity Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AutoML & Code Evolution](https://img.shields.io/badge/Architecture-AutoML_%26_Genetic_Evolution-brightgreen.svg)]()

The **Ouroboros Engine** is an **Automated Machine Learning (AutoML) & Genetic Code Evolution Harness**. It provides an automated software engineering pipeline that systematically refines prompts, explores candidate solution spaces through multi-candidate tournaments, collects execution trajectories (SFT/DPO pairs), and executes local QLoRA fine-tuning for continuous model alignment.

---

## Technical Overview & Scope

The system operates as a closed-loop evolutionary harness:
- **Execution Trajectory Collection**: Captures SFT & DPO training pairs from successful/failed task attempts.
- **Local Fine-Tuning Pipeline**: Triggers QLoRA fine-tuning on local LLMs to improve task domain familiarity and syntax alignment.
- **Parallel Candidate Tournaments**: Executes candidate implementations head-to-head to select optimal baseline solutions based on weighted quality and performance metrics.
- **Heuristic & Prompt Evolution**: Dynamically rewrites agent system prompts and operational heuristics based on empirical success rates.

> **Note on Technical Boundaries:**
> The term **"Singularity"** refers strictly to a project benchmark milestone ($\ge 95\%$ dynamic dataset pass rate and $\ge 80\%$ locked MBPP pass rate) where prompt orchestration yields execution directly to the fine-tuned Child Model. It operates under standard transformer limits and closed-loop synthetic data bounds.

---

## Architecture & Autonomous Loop

Each epoch in `autonomous_loop.py` executes a multi-stage Darwinian evolution cycle:

```
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 0: META-EVALUATION & RESEARCH PLANNING                           │
│ Evaluate performance trends & capability gaps → Formulate research plan│
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: CURRICULUM & ADAPTIVE PROBLEM GENERATION                      │
│ Rotate datasets, archive mastered tasks, generate synthetic challenges │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: COMPETITIVE CANDIDATE GENERATION & CORE MODIFICATION          │
│ Generate N parallel candidates (CONFIG_NUM_CANDIDATES) in sandbox      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: EVALUATION & FITNESS TOURNAMENT                               │
│ Benchmark candidates (Anchor, Dynamic, Turing, MBPP) & score fitness   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: DARWINIAN SELECTION & TRAJECTORY COLLECTION                   │
│ Commit winner, collect SFT/DPO pairs, update structured memory         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: RECURSIVE MODEL FINE-TUNING & HOT-SWAP (Triggered)            │
│ Train QLoRA weights (`train_model.py`), export GGUF, hot-swap model    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 6: STATUS REPORTING & ARCHITECTURAL CHECK                        │
│ Log metrics to monitor (`show_progress.py`), verify phase transitions  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Key Subsystems

- **Structured Memory (`structured_memory.py`)**: Categorized memory indexing, principle extraction, and semantic retrieval.
- **Knowledge Transfer Engine (`knowledge_transfer.py`)**: Concept mapping and cross-task knowledge graphs.
- **Competitive Evolution (`competitive_evolution.py`)**: Multi-candidate generation and tournament evaluation.
- **Adaptive Difficulty (`adaptive_difficulty.py`)**: Dynamic task rotation, tier scaling, and problem archiving.
- **Quality & Fitness Evaluator (`quality_fitness.py`)**: Weighted scoring combining correctness, execution time, and maintainability.
- **Fine-Tuning Engine (`train_model.py` / `model_selector.py`)**: QLoRA model fine-tuning, holdout benchmarking, and weight hot-swapping.
- **Safety Guards (`sandbox_guard.py` / `self_evolution_guard.py`)**: Isolated execution sandboxing, file permission checks, and safe modification rollbacks.

---

## Repository Structure & Index

| File | Category | Purpose |
|------|----------|---------|
| [autonomous_loop.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/autonomous_loop.py) | Core | Master Darwinian execution loop |
| [main.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/main.py) | Core | Brain Router & API gateway |
| [config.json](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/config.json) | Config | Runtime parameter single source of truth |
| [structured_memory.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/structured_memory.py) | Memory | Categorized memory & rule extraction |
| [knowledge_transfer.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/knowledge_transfer.py) | Learning | Cross-task concept mapping & knowledge graph |
| [competitive_evolution.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/competitive_evolution.py) | Candidate | Parallel implementation tournaments |
| [quality_fitness.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/quality_fitness.py) | Evaluation | Multi-metric code quality & fitness scoring |
| [problem_generator.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/problem_generator.py) | Synthesis | Multi-tier synthetic problem generation |
| [trajectory_collector.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/trajectory_collector.py) | Dataset | SFT & DPO execution pair collector |
| [train_model.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/train_model.py) | Fine-Tuning | QLoRA model fine-tuning engine |
| [convert_hf_to_gguf.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/convert_hf_to_gguf.py) | Export | PyTorch/HuggingFace to GGUF format converter |
| [sandbox_guard.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/sandbox_guard.py) | Safety | Execution sandboxing & permission enforcement |
| [self_evolution_guard.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/self_evolution_guard.py) | Safety | Security cage & rollback guard for core edits |
| [show_progress.py](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/show_progress.py) | Monitor | Real-time CLI progress & metric visualizer |

---

## Quick Start

### 1. Prerequisites & Virtual Environment

Ensure you have Python 3.10+ installed along with PyTorch and CUDA drivers configured for GPU acceleration.

```bash
# Setup virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Setting Up & Configuring LM Studio

The Ouroboros Engine connects to a local LLM inference server via an OpenAI-compatible API. **LM Studio** is the recommended provider for hosting local models (e.g. `Qwen/Qwen2.5-Coder-14B-Instruct` or similar GGUF models).

#### A. Download & Launch LM Studio
1. Download and install [LM Studio](https://lmstudio.ai/).
2. Search for and download your preferred code model (e.g., `Qwen2.5-Coder-14B-Instruct-GGUF`).

#### B. Start the Local Server in LM Studio
1. Open the **Local Server** tab (`<->` icon) on the left sidebar in LM Studio.
2. Select your loaded model from the top dropdown menu.
3. Verify or set the **Port** to `1234` (or update `api_base` in [config.json](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/config.json) to match).
4. Click **Start Server**. The endpoint will be available at `http://localhost:1234/v1`.

#### C. Configure `config.json`
Verify or update [config.json](file:///c:/Users/Troop/OneDrive/Desktop/AI%20Project/config.json) in your project root:

```json
{
    "llm_provider": "lm_studio",
    "api_base": "http://localhost:1234/v1",
    "model_id": "qwen2.5-coder-14b-instruct",
    "api_key": "lm-studio",
    "context_length": 32768
}
```

*Note: `model_id` can match your loaded model identifier or custom alias set in LM Studio.*

> **Important Warning — Model Backup & Disk Space Required:**
> - **Model Backup:** During Phase 5 (Recursive Model Fine-Tuning), the training pipeline fine-tunes the model via QLoRA, exports new weights, and converts/overwrites the target GGUF file (`convert_hf_to_gguf.py`). Make a backup copy of your initial `.gguf` file before starting the loop.
> - **HuggingFace Cache & Disk Space:** Fine-tuning downloads full PyTorch/Safetensors base weights from HuggingFace (`hf_model` in `config.json`), storing them in your default HuggingFace cache directory `C:\Users\<User>\.cache\huggingface\hub\`). Depending on the model size (e.g., 14B+ parameters), downloading and merging base weights requires **tens to hundreds of gigabytes** of free disk space. Ensure your system drive has sufficient storage.

---

### 3. Launching the System

You can control and run the system using the provided scripts or direct CLI commands:

```bash
# Launch full system (Brain Router + Autonomous Loop + Live Monitor):
start_ai.bat

# Stop all running system processes:
stop_ai.bat
```

### 4. Direct Execution Commands

```bash
# Run master autonomous loop directly:
python autonomous_loop.py

# Launch real-time visual progress monitor:
python show_progress.py

# Reset AI state back to baseline:
python reset_ai.py
```
