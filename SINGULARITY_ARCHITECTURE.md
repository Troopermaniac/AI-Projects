# SINGULARITY ARCHITECTURE — Self-Improving AI System

## Technical Overview

This system operates as an **Automated Machine Learning (AutoML) & Genetic Code Evolution Harness**. Rather than being a theoretical Artificial General Intelligence (AGI), it is an automated software engineering pipeline that systematically:
- **Refines execution heuristics & prompts** based on structured failure/success analysis.
- **Collects execution trajectories** (SFT & DPO pairs) during task runs.
- **Triggers QLoRA fine-tuning** on local LLMs to improve task domain familiarity.
- **Executes parallel candidate tournaments** to select high-fitness Python code solutions.

The term **"Singularity"** within this project is used as a project-specific benchmark milestone metric—representing the target operational threshold where the local fine-tuned Child Model solves 95%+ of dynamic dataset tasks and 80%+ of MBPP benchmark tasks autonomously without Meta-Brain prompt orchestration.

---

## Realistic Technical Scope & Boundaries

To maintain technical accuracy, the system's architecture operates under several real-world engineering constraints:

### 1. Base LLM Reasoning Ceiling
Fine-tuning local models (e.g. QLoRA SFT/DPO via `train_model.py`) improves syntax alignment, domain familiarity, and prompt compliance. However, self-fine-tuning on open-weights models (7B–14B parameters) does not expand fundamental mathematical reasoning or overcome hard architectural limits inherent to transformer base models.

### 2. Synthetic Data Feedback & Overfitting
Generating synthetic challenges (`problem_generator.py`) and fine-tuning on self-generated solutions creates a closed-loop feedback mechanism. Without continuous ingestion of novel, verified external knowledge, the model risks domain overfitting and data collapse on narrow algorithmic patterns.

### 3. Fixed Evaluation Boundaries
System evaluations (`evaluator_mbpp.py`, `evaluator_anchor.py`, `evaluator_dynamic.py`) measure performance on structured Python algorithmic puzzles. Achieving high pass rates demonstrates strong automated code synthesis within specific test suites, but does not represent general open-domain reasoning.

### 4. Heuristic Search vs. Genuine Self-Re-architecting
The system evolves solutions by exploring prompt variations, code refactoring templates, and candidate solution selection (`competitive_evolution.py`). The core Python runtime and framework rules remain governed by the surrounding Python codebase and sandbox guards.

---

## Configuration & Hardware Execution

### Dynamic Configuration (`config.json`)
The system utilizes `config.json` as a single source of truth for runtime parameters:
- `max_tokens`: Maximum generation token limit per candidate step (default: 4096).
- `context_length`: Context window allocation for prompts and history (default: 40960).
- `num_candidates`: Number of architectural/code candidates generated in parallel per evolution step (default: 5).
- `max_epochs`: Target iteration epoch limit (0 for continuous operation).

### GPU Hardware Acceleration (`venv`)
To ensure high-performance PyTorch execution using local CUDA acceleration (e.g., NVIDIA RTX 4090), `autonomous_loop.py` dynamically resolves the Python executable (`PYTHON_EXE`):
- Probes `venv/Scripts/python.exe` relative to project root.
- Uses dedicated virtual environment dependencies if available, falling back to system Python if unconfigured.

---

## Core Evolutionary Subsystems

### 1. Structured Memory (`structured_memory.py`)
**What it does:** Replaces flat append-only memory with categorized, searchable memory.
- **Domain categorization**: Memories tagged by domain (`algorithms`, `data_structures`, `optimization`, `patterns`).
- **Principle extraction**: Extracts generalizable rules post-attempt.
- **Semantic retrieval**: Retrieves relevant past experiences for new problems.
- **Memory consolidation**: Merges related memories into higher-level abstractions.

### 2. Knowledge Transfer Engine (`knowledge_transfer.py`)
**What it does:** Cross-task pattern recognition and concept abstraction.
- **Concept extraction**: Parses problem prompts into key algorithmic concepts.
- **Knowledge graph**: Tracks concept relationships across solved tasks.
- **Adaptation suggestions**: Recommends adapting solutions from prior related tasks.

### 3. Code Refactorer (`code_refactorer.py`)
**What it does:** Consolidates evolved codebase components to prevent handler bloat and duplication.
- **Pattern analysis**: Detects redundant handlers across evolved files.
- **Consolidation prompts**: Directs the Engineer agent to merge similar handlers into modular abstractions.

### 4. Meta-Evaluator (`meta_evaluator.py`)
**What it does:** Evaluates and evolves system prompts and operational strategies.
- **Prompt scoring**: Analyzes instruction pattern effectiveness.
- **Self-directed prompt evolution**: Rewrites internal agent prompts based on empirical success metrics.

### 5. Performance Optimizer (`performance_optimizer.py`)
**What it does:** Tracks time/space complexity and execution performance.
- **AST complexity estimation**: Estimates time complexity ($O(1)$, $O(N)$, $O(N^2)$).
- **Execution profiling**: Logs wall-clock time and resource metrics across epochs.

### 6. Adaptive Difficulty Engine (`adaptive_difficulty.py`)
**What it does:** Manages dynamic problem scaling and curriculum progression.
- **Tier progression**: Dynamically scales problem difficulty based on system fitness.
- **Task archival & rotation**: Archives mastered tasks and rotates active problem sets to prevent overfitting.

### 7. Quality & Code Fitness Evaluator (`quality_fitness.py`)
**What it does:** Multi-metric fitness scoring evaluating code quality beyond simple pass/fail.
- **Maintainability & style analysis**: Evaluates code cleanliness, modularity, and error handling.
- **Weighted fitness metric**: Combines correctness, performance, and code quality into unified fitness scores.

### 8. Module Evolution (`module_evolution.py`)
**What it does:** Identifies external module integration opportunities.
- **Capability discovery**: Detects missing python packages/modules that can optimize execution.
- **Integration prompts**: Guides code synthesis to incorporate high-performance libraries.

### 9. Recursive Self-Improvement (`recursive_self_improvement.py`)
**What it does:** Adjusts systemic learning parameters and extracts meta-solving strategies.
- **Adaptive learning rate**: Scales self-modification scope based on performance trends.
- **Strategy extraction**: Distills successful problem-solving workflows into procedural heuristics.

### 10. Self-Directed Curriculum (`self_directed_curriculum.py`)
**What it does:** Detects capability gaps and autonomously generates targeted learning tasks.
- **Gap analysis**: Identifies weak algorithm domains.
- **Synthetic task creation**: Generates custom challenges to strengthen deficient skill areas.

### 11. Architectural Transitions (`architectural_transitions.py`)
**What it does:** Manages macro-level architectural phase shifts across system evolution.
- **Phase transition detection**: Recognizes when local optimizations plateau and structural architecture updates are needed.

### 12. Competitive Evolution (`competitive_evolution.py`)
**What it does:** Generates competing candidate implementations and executes tournament selection.
- **Multi-candidate generation**: Spawns parallel solution candidates (`CONFIG_NUM_CANDIDATES`).
- **Tournament evaluation**: Scores candidates head-to-head to select the optimal baseline.

### 13. Tool Evolution (`tool_evolution.py`)
**What it does:** Identifies capability gaps and constructs new utility scripts/tools autonomously.

### 14. Neural Self-Evolution & Fine-Tuning Pipeline
**What it does:** Gathers execution trajectories for neural self-tuning.
- **Problem Generator (`problem_generator.py`)**: Synthesizes custom algorithmic challenges across domain tiers.
- **Trajectory Collector (`trajectory_collector.py`)**: Captures SFT/DPO execution pairs upon successful task runs.
- **Train Model Engine (`train_model.py`)**: Executes QLoRA fine-tuning and updates local models.
- **Model Selector (`model_selector.py`)**: Benchmarks newly trained models against holdout evaluation datasets before hot-swapping.
- **HF to GGUF Converter (`convert_hf_to_gguf.py`)**: Converts trained PyTorch/HuggingFace checkpoints to GGUF format for local LLM inference engines.
- **Capability Frontier (`capability_frontier.py`)**: Tracks domain mastery percentages across all algorithm categories.
- **Research Planner (`research_planner.py`)**: Formulates multi-epoch research agendas targeted at capability frontiers.
- **Core Modification Proposer (`core_modification_proposer.py`) & Guard (`self_evolution_guard.py`, `sandbox_guard.py`, `filesystem_sandbox.py`)**: Gated proposal system allowing controlled, sandboxed edits to core system files.
- **Locked Benchmark (`evaluator_mbpp.py`)**: External, unmodifiable benchmark for objective validation.

---

## The Enhanced Autonomous Loop Cycle

Each epoch in `autonomous_loop.py` executes the following multi-stage cycle:

```
PHASE 0: META-EVALUATION & RESEARCH PLANNING
  → Evaluate historical performance trends & capability frontier gaps
  → Formulate targeted research plans & auto-generate synthetic problem sets

PHASE 1: CURRICULUM & ADAPTIVE PROBLEM GENERATION
  → Rotate active problem datasets & archive mastered challenge sets
  → Generate adaptive difficulty task variants & self-directed challenges

PHASE 2: COMPETITIVE CANDIDATE GENERATION & CORE MODIFICATION
  → Generate N parallel candidates (CONFIG_NUM_CANDIDATES) via Engineer/Visionary prompts
  → Propose candidate core modifications (gated & isolated by self_evolution_guard)

PHASE 3: EVALUATION & FITNESS TOURNAMENT
  → Execute code candidates in sandboxed environment across benchmark suite:
      • Anchor Evaluator (evaluator_anchor.py)
      • Dynamic Evaluator (evaluator_dynamic.py)
      • Turing Evaluator (evaluator_turing.py)
      • Locked MBPP Benchmark (evaluator_mbpp.py)
  → Calculate combined Quality & Performance Fitness scores (quality_fitness.py)
  → Select winning candidate via tournament selection (competitive_evolution.py)

PHASE 4: DARWINIAN SELECTION & TRAJECTORY COLLECTION
  → SUCCESS: Commit winning changes, record SFT/DPO trajectories, update structured memory
  → FAILURE: Revert changes, analyze root causes, update knowledge graph

PHASE 5: RECURSIVE MODEL FINE-TUNING & HOT-SWAP (Triggered)
  → Trigger QLoRA fine-tuning when trajectory buffers reach threshold (`train_model.py`)
  → Benchmark candidate model using model_selector.py on holdout sets
  → Convert weights to GGUF format (`convert_hf_to_gguf.py`) and hot-swap active model

PHASE 6: STATUS REPORTING & ARCHITECTURAL CHECK
  → Log progress to live monitor dashboard (`show_progress.py`)
  → Check architectural phase transition criteria (`architectural_transitions.py`)
```

---

## Benchmark Threshold Milestone ("Singularity" Swap)

Within this AutoML pipeline, the **"Singularity Swap"** represents an automated milestone trigger condition when the system passes both benchmark targets simultaneously:
1. **$\ge$ 95% Pass Rate on Dynamic Dataset** (`SINGULARITY_DYNAMIC_THRESHOLD = 0.95`)
2. **$\ge$ 80% Pass Rate on Locked MBPP Benchmark** (`SINGULARITY_MBPP_THRESHOLD = 0.80`)

Upon meeting these targets:
- The system logs the benchmark milestone in structured memory.
- Meta-Brain prompt orchestration shuts down, yielding execution directly to the fine-tuned Child Model.

---

## Workspace File Index

| File | Category | Purpose |
|------|----------|---------|
| `autonomous_loop.py` | Core | Master Darwinian execution loop |
| `main.py` | Core | Brain Router & API gateway |
| `config.json` | Configuration | Single source of truth for runtime parameters |
| `structured_memory.py` | Learning | Categorized memory & rule extraction |
| `knowledge_transfer.py` | Learning | Cross-task concept mapping & knowledge graph |
| `code_refactorer.py` | Refactoring | Code deduplication & structural consolidation |
| `meta_evaluator.py` | Meta-Learning | Prompt effectiveness scoring & auto-evolution |
| `performance_optimizer.py` | Optimization | AST time/space complexity & execution profiling |
| `adaptive_difficulty.py` | Curriculum | Dynamic problem scaling, task rotation & archiving |
| `quality_fitness.py` | Evaluation | Multi-metric code quality & maintainability scoring |
| `module_evolution.py` | Capabilities | Third-party module capability integration |
| `recursive_self_improvement.py` | Meta-Learning | Adaptive learning rate & strategic heuristics |
| `self_directed_curriculum.py` | Curriculum | Autonomous knowledge gap detection & task creation |
| `architectural_transitions.py` | Architecture | Macro architectural phase detection & transition prompts |
| `competitive_evolution.py` | Candidate Selection | Parallel implementation candidate tournaments |
| `tool_evolution.py` | Capabilities | Autonomous creation of custom utility scripts |
| `problem_generator.py` | Synthesis | Multi-tier synthetic problem creation |
| `trajectory_collector.py` | Dataset | SFT & DPO execution pair gathering |
| `train_model.py` | Fine-Tuning | QLoRA model fine-tuning engine |
| `convert_hf_to_gguf.py` | Model Export | PyTorch/HF to GGUF format converter |
| `model_selector.py` | Model Evaluation | Model evaluation & holdout benchmarking |
| `capability_frontier.py` | Tracking | Domain mastery percentage tracking |
| `research_planner.py` | Strategy | Multi-epoch research agenda planning |
| `core_modification_proposer.py` | Safety / System | Core system edit proposal generator |
| `self_evolution_guard.py` | Safety / System | Security cage & safe rollback guard for core edits |
| `sandbox_guard.py` | Safety / System | Execution sandboxing & permission checks |
| `filesystem_sandbox.py` | Safety / System | File access scope enforcement |
| `evaluator_anchor.py` | Benchmarking | Anchor test suite evaluator |
| `evaluator_dynamic.py` | Benchmarking | Dynamic adaptive test evaluator |
| `evaluator_turing.py` | Benchmarking | Turing test suite evaluator |
| `evaluator_mbpp.py` | Benchmarking | Locked external MBPP benchmark evaluator |
| `show_progress.py` | Visualization | Real-time CLI progress & metric visualizer |
| `reset_ai.py` | Utility | State & memory reset utility |
| `bootstrap.py` | System | Environment setup & initialization |
| `start_ai.bat` | Control | System launch script |
| `stop_ai.bat` | Control | Process termination script |

---

## How to Run & Manage

```bash
# Launch full system (Brain Router + Autonomous Loop + Live Monitor):
start_ai.bat

# Stop all running system processes:
stop_ai.bat

# Run master autonomous loop directly:
python autonomous_loop.py

# Launch real-time visual progress monitor:
python show_progress.py

# Reset AI state back to baseline:
python reset_ai.py
```
