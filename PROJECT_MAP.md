# PROJECT MAP

This document provides a complete inventory of the project structure, including files, data assets, configurations, and diagnostics.

## 1. FOLDER TREE

```text
.
├── .env
├── .gitignore
├── DEPLOYMENT.md
├── PROJECT_MAP.md
├── agent
│   ├── __init__.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── concise_logger.py
│   │   ├── env_config.py
│   │   ├── llamafile_manager.py
│   │   ├── llm_manager.py
│   │   ├── orchestrator.py
│   │   ├── platform_detector.py
│   │   ├── session_manager.py
│   │   ├── tee_logger.py
│   │   └── watchdog.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── loader.py
│   │   └── pipeline.py
│   ├── prompts
│   │   ├── __init__.py
│   │   └── system_prompt.py
│   └── tools
│       ├── __init__.py
│       ├── arxiv_tool.py
│       ├── audit_tool.py
│       ├── check_last_session.py
│       ├── dev_reset.py
│       ├── experiment_runner.py
│       ├── harness_template.py
│       ├── leaderboard_tool.py
│       └── rebuild_leaderboard.py
├── check4_gpu.py
├── check6_nb.py
├── cloud_setup.py
├── data
│   ├── processed
│   │   ├── class_weights.pkl
│   │   ├── scaler.pkl
│   │   ├── selector.pkl
│   │   └── variance_threshold.pkl
│   ├── raw
│   │   ├── alpha_synuclein.csv
│   │   └── csv_preview.txt
│   └── splits
│       ├── split_hash.sha256
│       ├── split_meta.pkl
│       ├── test.pkl
│       ├── train.pkl
│       └── val.pkl
├── experiments
│   ├── .gitkeep
│   └── exp_001_laptop_random_forest_baseline
│       ├── config.yaml
│       ├── model.py
│       ├── results.json
│       ├── run.log
│       └── train_eval.py
├── master_log
│   ├── leaderboard.json
│   ├── master_terminal.log
│   └── orchestrator_state.json
├── notebooks
│   └── run_agent.ipynb
├── patch_notebook.py
├── patch_notebook_v3.py
├── pyproject.toml
├── reports
├── requirements.txt
├── run_failed_exps.py
├── run_pipeline.py
├── sessions
│   ├── .gitkeep
│   └── 2026-06-23_11-42-18
│       ├── heartbeat.json
│       ├── session_log.log
│       └── session_summary.json
├── test_dev_reset.py
├── test_guard.py
├── test_guard_v2.py
├── verify_all.py
├── verify_session.py
└── verify_sync.py
```

---

## 2. FILE-BY-FILE INVENTORY

### agent/__init__.py
- **Purpose**: Defines the package-level initialization for the agent module.
- **Classes defined**: None
- **Functions defined**: None
- **Imports FROM other project files**: None
- **Imported BY which other project files**: None

### agent/core/__init__.py
- **Purpose**: Package initializer for core agent module, importing and exposing logger and LLM manager components.
- **Classes defined**: None
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/core/llm_manager.py`
  - `agent/core/tee_logger.py`
- **Imported BY which other project files**: None

### agent/core/concise_logger.py
- **Purpose**: Overwrites Jupyter display lines to provide concise, real-time logging of active agent steps while allowing full output logs to stream to file.
- **Classes defined**:
  - `ConciseStepReporter`
    - `__init__(self, logger, run_start, refresh_hz)`
    - `step_callback(self, memory_step)`
    - `update_exp(self, exp_name)`
    - `stop(self)`
    - `_refresh_loop(self)`
    - `_extract_exp_info(self, memory_step)`
- **Functions defined**: None
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/core/orchestrator.py`

### agent/core/env_config.py
- **Purpose**: Configures workspace environment variables, handles path setup, and detects whether running locally (with WSL path support) or in Google Cloud.
- **Classes defined**: None
- **Functions defined**:
  - `find_project_root(start)`
  - `_is_root(path)`
  - `setup_environment(verbose)`
  - `_detect_cloud()`
  - `_print_cloud_info()`
  - `get_paths(root)`
- **Imports FROM other project files**:
  - `agent/core/platform_detector.py`
- **Imported BY which other project files**: None

### agent/core/llamafile_manager.py
- **Purpose**: Manages a local llama.cpp / llamafile server for downloading and serving GGUF models on localhost.
- **Classes defined**:
  - `LlamafileNotAvailableError`
  - `LlamafileManager`
    - `__init__(self, model_key, port, n_gpu_layers)`
    - `base_url(self)` (property)
    - `start(self)`
    - `stop(self)`
    - `_download_model(self)`
    - `_launch_server(self)`
    - `_build_server_cmd(self)`
    - `_wait_until_ready(self)`
    - `_drain_output(self)`
- **Functions defined**: None
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/core/llm_manager.py`

### agent/core/llm_manager.py
- **Purpose**: Interacts with and manages configurations, token usage, backoffs, swapping, and connections to LLM providers like Groq, Gemini, LM Studio, and local Ollama/llamafile.
- **Classes defined**:
  - `LLMManager`
    - `__init__(self, model_name, env_path)`
    - `get_model(self)`
    - `test_connection(self)`
    - `switch_model(self, model_name)`
    - `auto_fallback_chain(self, chain, test_each)`
    - `_initialize_model(self, model_name)`
    - `_get_api_key(provider)`
    - `_stamp_leaderboard(model_name)`
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/core/tee_logger.py`
  - `agent/core/llamafile_manager.py`
- **Imported BY which other project files**:
  - `agent/core/__init__.py`
  - `agent/core/orchestrator.py`

### agent/core/orchestrator.py
- **Purpose**: Wires all tools, sessions, logger, early stop guards, token tracking, and code agent modules into a single, top-level autonomous research execution block.
- **Classes defined**:
  - `AgentOrchestrator`
    - `__init__(self, model_name, use_short_prompt, max_steps, verbosity, max_idle_seconds, max_total_seconds)`
    - `run(self, max_experiments, max_idle_seconds, max_total_seconds)`
    - `stop(self)`
    - `switch_model(self, model_name)`
    - `status(self)`
    - `_elapsed(self)`
    - `_elapsed_tuple(self)`
    - `_save_state(self, status)`
    - `_print_run_summary(self, stop_reason, fin_error)`
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/core/concise_logger.py`
  - `agent/core/llm_manager.py`
  - `agent/core/session_manager.py`
  - `agent/core/tee_logger.py`
  - `agent/core/watchdog.py`
  - `agent/prompts/system_prompt.py`
  - `agent/tools/arxiv_tool.py`
  - `agent/tools/audit_tool.py`
  - `agent/tools/experiment_runner.py`
  - `agent/tools/leaderboard_tool.py`
- **Imported BY which other project files**: None

### agent/core/platform_detector.py
- **Purpose**: Auto-detects the execution platform (Colab, Kaggle, IIT server, Laptop, GCloud) and recommends the best model to use based on GPU VRAM and Ollama availability.
- **Classes defined**:
  - `PlatformDetector`
    - `detect(self)`
    - `_detect_platform(self)`
    - `_detect_gpu(self)` (static)
    - `_check_ollama(self)` (static)
    - `_recommend_model(plat, vram, has_ollama)` (static)
- **Functions defined**: None
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/core/env_config.py`

### agent/core/session_manager.py
- **Purpose**: Manages running session context, tracks and logs heartbeats, and persists execution statuses.
- **Classes defined**:
  - `SessionManager`
    - `__init__(self, model_name, logger, heartbeat_interval)`
    - `start(self)`
    - `tick(self, current_exp, step, status)`
    - `end(self, status, total_experiments, error_message)`
    - `_attach_session_log(self)`
    - `_heartbeat_loop(self)`
    - `_write_heartbeat(self)`
    - `_write_json(path, data)`
- **Functions defined**:
  - `_machine_id()`
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/core/orchestrator.py`

### agent/core/tee_logger.py
- **Purpose**: Directs stdout and custom print calls to both standard output and log files simultaneously.
- **Classes defined**:
  - `TeeLogger`
    - `__new__(cls, master_log_dir)`
    - `__init__(self, master_log_dir)`
    - `set_experiment_log(self, path)`
    - `info(self, message)`
    - `warning(self, message)`
    - `error(self, message)`
    - `agent(self, message)`
    - `_now(self)`
    - `_format_plain(self, level, message)`
    - `_format_colour(self, level, message)`
    - `_emit(self, level, message)`
    - `_append_to_file(path, text)`
    - `redirect_print(self)`
    - `restore_print(self)`
- **Functions defined**: None
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/core/__init__.py`
  - `agent/core/llm_manager.py`
  - `agent/core/orchestrator.py`
  - `agent/tools/arxiv_tool.py`
  - `agent/tools/audit_tool.py`
  - `agent/tools/experiment_runner.py`
  - `agent/tools/leaderboard_tool.py`

### agent/core/watchdog.py
- **Purpose**: Monitors running session heartbeat updates and gracefully halts operations if no activity is detected within thresholds.
- **Classes defined**:
  - `RunWatchdog`
    - `__init__(self, stop_event, heartbeat_path, max_idle_seconds, max_total_seconds, logger, poll_interval)`
    - `start(self)`
    - `stop(self)`
    - `run(self)`
    - `_log(self, msg)`
- **Functions defined**: None
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/core/orchestrator.py`

### agent/data/__init__.py
- **Purpose**: Package initializer for data handling, exposing data loader and pipeline classes.
- **Classes defined**: None
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/data/loader.py`
  - `agent/data/pipeline.py`
- **Imported BY which other project files**: None

### agent/data/features.py
- **Purpose**: Computes sequence-level biological features including amino acid compositions, physicochemical attributes, and k-mer frequencies.
- **Classes defined**: None
- **Functions defined**:
  - `_clean(sequence)`
  - `amino_acid_composition(sequence)`
  - `physicochemical_features(sequence)`
  - `kmer_frequencies(sequence, k)`
  - `encode_concentration(value, log_min, log_max)`
  - `_charge_at_ph(counts, ph)`
  - `_isoelectric_point(counts)`
  - `_instability_index(seq)`
- **Imports FROM other project files**: None
- **Imported BY which other project files**: None

### agent/data/loader.py
- **Purpose**: Loads the raw alpha-synuclein CSV file containing peptide sequences and their concentrations.
- **Classes defined**: None
- **Functions defined**:
  - `load_raw_csv(path)`
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/data/__init__.py`
  - `agent/data/pipeline.py`

### agent/data/pipeline.py
- **Purpose**: Processes biological sequence datasets, builds features, runs splits, and prepares datasets for training.
- **Classes defined**:
  - `DataPipeline`
    - `__init__(self, splits_dir, random_state)`
    - `load_and_expand(self, csv_path)`
    - `build_features(self, df)`
    - `stratified_split(self, X, y, train, val, test)`
    - `seal_test_set(self, test_path)`
    - `verify_wall(self, test_path)`
    - `save_splits(self)`
    - `load_splits(self)`
    - `reduce_features(self, X_train, X_val, X_test, y_train, k_best, use_pca, pca_variance)`
    - `get_class_weights(self, y_train)`
    - `apply_smote(self, X_train, y_train, random_state, k_neighbors)`
    - `_sha256(path)`
    - `_find_sequence_column(df)`
    - `_parse_concentration(raw)`
    - `_resolve_labels(series)`
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/data/__init__.py`
  - `agent/data/loader.py`
- **Imported BY which other project files**:
  - `agent/data/__init__.py`
  - `agent/tools/dev_reset.py`

### agent/prompts/__init__.py
- **Purpose**: Initializes the prompt templates package and exposes the system prompt modules.
- **Classes defined**: None
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/prompts/system_prompt.py`
- **Imported BY which other project files**: None

### agent/prompts/system_prompt.py
- **Purpose**: Holds the master long and short system prompts guiding the autonomous agent behavior.
- **Classes defined**: None
- **Functions defined**: None
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/core/orchestrator.py`
  - `agent/prompts/__init__.py`

### agent/tools/__init__.py
- **Purpose**: Exposes available custom research, leaderboard, experiment execution, and code auditing tools to the agent.
- **Classes defined**: None
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/tools/arxiv_tool.py`
  - `agent/tools/audit_tool.py`
  - `agent/tools/experiment_runner.py`
  - `agent/tools/leaderboard_tool.py`
- **Imported BY which other project files**: None

### agent/tools/arxiv_tool.py
- **Purpose**: Enables searching the Arxiv database for research papers with optional LLM-driven eligibility filtering.
- **Classes defined**:
  - `ArxivTool`
    - `__init__(self, llm_model)`
    - `forward(self, query, max_results, llm_filter)`
    - `_llm_filter(self, abstract)`
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/core/tee_logger.py`
- **Imported BY which other project files**:
  - `agent/core/orchestrator.py`
  - `agent/tools/__init__.py`

### agent/tools/audit_tool.py
- **Purpose**: Audits python code snippets using AST to identify and prevent unauthorized file loading or testing cheating.
- **Classes defined**:
  - `AuditTool`
    - `__init__(self)`
    - `forward(self, code_block)`
    - `audit_directory(self, directory)`
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/core/tee_logger.py`
- **Imported BY which other project files**:
  - `agent/core/orchestrator.py`
  - `agent/tools/__init__.py`
  - `agent/tools/experiment_runner.py`

### agent/tools/check_last_session.py
- **Purpose**: Summarizes status, duration, and metrics from the most recent run sessions recorded.
- **Classes defined**: None
- **Functions defined**:
  - `_load_json(path)`
  - `_ago(iso_str)`
  - `_duration(start, end)`
  - `_status_icon(status)`
  - `get_session_info(session_dir)`
  - `get_last_session_info(sessions_dir)`
  - `format_session_report(info, verbose)`
  - `check_last_session(sessions_dir, n, verbose)`
- **Imports FROM other project files**: None
- **Imported BY which other project files**: None

### agent/tools/dev_reset.py
- **Purpose**: Resets workspace states by wiping experiments and session logs while protecting datasets and core source files.
- **Classes defined**: None
- **Functions defined**:
  - `_is_protected(path, root)`
  - `dev_reset(root)`
- **Imports FROM other project files**:
  - `agent/data/pipeline.py`
- **Imported BY which other project files**: None

### agent/tools/experiment_runner.py
- **Purpose**: Runs custom machine learning model training scripts in standalone processes, auditing them and writing results.
- **Classes defined**:
  - `ExperimentRunnerTool`
    - `__init__(self)`
    - `forward(self, exp_name, architecture_family, model_code, hyperparams)`
    - `_run_subprocess(self, script, exp_dir, timeout)`
    - `_next_exp_id(self, machine_id, exp_name)`
    - `_update_leaderboard(self, result)`
- **Functions defined**:
  - `_sanitize(name)`
  - `_get_machine_id()`
  - `_build_error_result(exp_id, architecture, architecture_family, machine_id, hyperparams, timestamp, error_message)`
  - `_format_result(result)`
- **Imports FROM other project files**:
  - `agent/core/tee_logger.py`
  - `agent/tools/audit_tool.py`
  - `agent/tools/harness_template.py`
- **Imported BY which other project files**:
  - `agent/core/orchestrator.py`
  - `agent/tools/__init__.py`

### agent/tools/harness_template.py
- **Purpose**: Exports template python blocks used dynamically to wrap and evaluate custom model code submitted by the agent.
- **Classes defined**: None
- **Functions defined**: None
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/tools/experiment_runner.py`

### agent/tools/leaderboard_tool.py
- **Purpose**: Retrieves top performance metrics from the master leaderboard.
- **Classes defined**:
  - `LeaderboardTool`
    - `__init__(self)`
    - `forward(self, top_n)`
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/core/tee_logger.py`
  - `agent/tools/rebuild_leaderboard.py`
- **Imported BY which other project files**:
  - `agent/core/orchestrator.py`
  - `agent/tools/__init__.py`

### agent/tools/rebuild_leaderboard.py
- **Purpose**: Aggregates results from individual experiment directories to rebuild a central leaderboard file.
- **Classes defined**: None
- **Functions defined**:
  - `rebuild_leaderboard(experiments_dir, out_path, verbose)`
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/tools/leaderboard_tool.py`

---

## 3. DATA FILES

### data/raw/alpha_synuclein.csv (3110 bytes)
- **What it contains**: Raw input dataset. CSV with 69 rows and 9 columns: `'Protein - Alpha synuclein', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'`. Contains lists of concentrations and peptide sequences.

### data/raw/csv_preview.txt (3809 bytes)
- **What it contains**: Text preview output containing line-by-line values extracted from `alpha_synuclein.csv`.

### data/processed/class_weights.pkl (60 bytes)
- **What it contains**: Pickle dictionary mapping classes to float class weights: `Keys = [0, 1, 2, 3]`. Values: `0: 0.31797235023041476, 1: 5.75, 2: 2.76, 3: 3.1363636363636362`.

### data/processed/scaler.pkl (4944 bytes)
- **What it contains**: Serialized `sklearn.preprocessing.StandardScaler` object fit to the training set.

### data/processed/selector.pkl (3352 bytes)
- **What it contains**: Serialized `sklearn.feature_selection.SelectKBest` object fit to the training set.

### data/processed/variance_threshold.pkl (67712 bytes)
- **What it contains**: Serialized `sklearn.feature_selection.VarianceThreshold` object fit to the training set.

### data/splits/split_hash.sha256 (76 bytes)
- **What it contains**: Raw text containing the SHA-256 hash validation anchor for the test split: `07952c2516bd07822158bbe6da4c331a0a662122337ae4a871df55917d48f97f  test.pkl`.

### data/splits/split_meta.pkl (50127 bytes)
- **What it contains**: Pickle dictionary holding metadata and keys used in feature extraction: `Keys = ['conc_log_min', 'conc_log_max', '2mer_keys', '3mer_keys', 'random_state']`.

### data/splits/test.pkl (46052 bytes)
- **What it contains**: Pickle dictionary representing the sealed test split: `Keys = ['X_test', 'y_test']`.
  - `X_test`: numpy array of shape `(60, 189)`, dtype `float32`.
  - `y_test`: numpy array of shape `(60,)`, dtype `int64`.

### data/splits/train.pkl (211077 bytes)
- **What it contains**: Pickle dictionary representing the training split: `Keys = ['X', 'y']`.
  - `X`: numpy array of shape `(276, 189)`, dtype `float32`.
  - `y`: numpy array of shape `(276,)`, dtype `int64`.

### data/splits/val.pkl (46042 bytes)
- **What it contains**: Pickle dictionary representing the validation split: `Keys = ['X', 'y']`.
  - `X`: numpy array of shape `(60, 189)`, dtype `float32`.
  - `y`: numpy array of shape `(60,)`, dtype `int64`.

---

## 4. THE SYSTEM PROMPT

Verbatim contents of `agent/prompts/system_prompt.py`:

```python
"""
agent/prompts/system_prompt.py
────────────────────────────────────────────────────────────────────────────
Master system prompt for the autonomous alpha-synuclein ML research agent.

Import with:
    from agent.prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_SHORT
"""

SYSTEM_PROMPT: str = """
You are an autonomous ML researcher investigating alpha-synuclein 
peptide detectability. Your goal: find the best-performing model 
architecture you can, using genuine scientific judgment.

DATASET: ~390 samples (65 peptides × 6 concentrations), 4 imbalanced 
classes (No/Low/Medium/High). Features are pre-processed amino acid 
composition, physicochemical properties, k-mer frequencies, and 
concentration — already scaled and reduced by the harness.

=== THE ONLY RULES THAT CANNOT BE BROKEN ===
- You can never access the test set. This is enforced by the 
  harness itself — there is no path to test.pkl in your code.
- Primary metric is val_f1_macro. Accuracy is meaningless here: 
  a model predicting only the majority class scores ~0.78 accuracy 
  while learning nothing.
- Call audit_code before every run_experiment.
- You write ONLY model_code: one function with this exact signature:
      def build_and_train(X_train, y_train, X_val, y_val, class_weights):
          # all imports inside the function
          # return a fitted model with .predict()
  Never load files. Never call predict() or compute metrics. 
  Never reference X_test, y_test, or the test set in any form.
  The harness handles all evaluation and file writing.
- Always account for class imbalance in every model. How you do 
  that is your choice.

=== YOUR TOOLS ===
- run_experiment       : runs a complete experiment, returns val metrics
- read_leaderboard     : see what has been tried and what worked
- audit_code           : verify your code before running
- search_arxiv_papers  : find research papers
- web_search           : search the internet for anything

=== YOUR MISSION ===
You have a budget of experiments. Use it as a real researcher would.

Start by understanding what has been tried and what worked. Then 
decide for yourself what to try next — based on your own knowledge, 
what you read in the leaderboard, and what you find by searching 
the literature and the web. You are not limited to anything I mention. 
You are encouraged to find methods I don't know about.

When you are stuck or curious, search. When results surprise you, 
reason about why. When something works, understand why before 
moving on. When something fails, learn from it.

Do not stop early. Use your full budget. Do not call final_answer 
until you have genuinely exhausted promising directions — not just 
until you found the first thing that works.

You are the researcher. The tools are yours. Use real judgment.
"""

# Short version for models with limited context windows
SYSTEM_PROMPT_SHORT: str = """
Autonomous ML researcher: find the best model for alpha-synuclein 
peptide classification (4 imbalanced classes, ~390 samples). 

You decide everything — what to try, in what order, when to search 
the literature, when to stop. No fixed list of methods. Use your 
own knowledge and search tools to discover what might work. 

Non-negotiable rules:
- Never access the test set (physically blocked by harness)
- Primary metric: val_f1_macro only
- Write only: def build_and_train(X_train, y_train, X_val, y_val, class_weights)
  returning a fitted model — all imports inside, no file loading
- Call audit_code before every run_experiment
- Account for class imbalance in every model
- Use your full experiment budget before concluding
""" 
```

---

## 5. THE NOTEBOOK

Cells from `notebooks/run_agent.ipynb` (verbatim first lines of active cells):

### Cell 1
```python
# !git pull
```

### Cell 2 (Bootstrap Cell)
```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Cell 1 · [BOOTSTRAP]  Auto-setup: Local Windows + Google Cloud (GCE)     ║
# ║  Run this cell FIRST every session — clones/pulls repo & installs deps.   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
import os, sys, subprocess, platform
from pathlib import Path

GITHUB_REPO   = 'https://github.com/DhruvalPtl/alpha-synuclein-agent.git'
CLOUD_INSTALL = Path.home() / 'agent_workspace'
LOCAL_WIN     = Path(r'd:\3rd sem M.tech\agent_workspace')
LOCAL_WSL     = Path('/mnt/d/3rd sem M.tech/agent_workspace')
```

### Cell 3 (Data Verification)
```python
# ── Path guard: works even if Cell 1 hasn't run (import agent installed via pip install -e .) ──
import sys, os; _r = next((p for p in [
    __import__('pathlib').Path(r'd:\3rd sem M.tech\agent_workspace'),
    __import__('pathlib').Path('/mnt/d/3rd sem M.tech/agent_workspace'),
    __import__('pathlib').Path.home()/'agent_workspace'
] if p.exists()), None)
```

### Cell 4 (Wall Verification)
```python
# ── Path guard: works even if Cell 1 hasn't run (import agent installed via pip install -e .) ──
import sys, os; _r = next((p for p in [
    __import__('pathlib').Path(r'd:\3rd sem M.tech\agent_workspace'),
    __import__('pathlib').Path('/mnt/d/3rd sem M.tech/agent_workspace'),
    __import__('pathlib').Path.home()/'agent_workspace'
] if p.exists()), None)
```

### Cell 5 (Dev Reset)
```python
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Cell 3b · [DEV RESET]  Wipe experiments + logs for a fresh run           ║
# ║  NEVER deletes data/, agent/, .env, or .git/  — mathematical wall safe.   ║
```

### Cell 6 (Launch Agent)
```python
# ── Path guard ────────────────────────────────────────────────────────────────
import sys, os
_r = next((p for p in [
    __import__('pathlib').Path(r'd:\3rd sem M.tech\agent_workspace'),
    __import__('pathlib').Path('/mnt/d/3rd sem M.tech/agent_workspace'),
    __import__('pathlib').Path.home()/'agent_workspace'
] if p.exists()), None)
```

### Cell 7 (Dashboard)
```python
# ── Path guard: works even if Cell 1 hasn't run (import agent installed via pip install -e .) ──
import sys, os; _r = next((p for p in [
    __import__('pathlib').Path(r'd:\3rd sem M.tech\agent_workspace'),
    __import__('pathlib').Path('/mnt/d/3rd sem M.tech/agent_workspace'),
    __import__('pathlib').Path.home()/'agent_workspace'
] if p.exists()), None)
```

---

## 6. CURRENT GIT STATE

### git log --oneline -15
```text
70f4f41 fix: Add WSL path support for local notebook runtime and resolve Gemini key configuration issue
81059c0 fix: Guard raises RuntimeError instead of returning string
6306b3d feat: Add early-stop guard + dynamic session context in prompt
33f2ed9 docs: Add DEPLOYMENT.md deployment guide
dbee9cd Section 5: Architecture Landscape menu in system prompt (tabular/NN/sequence/tabular-DL/ensemble/NLP framing), fix merged sentence, expand SYSTEM_PROMPT_SHORT
0272dd8 Section 4: Per-provider throttle (gemini/groq/cerebras), APIError backoff for all providers, daily token budget tracking with 80%/95% alerts in token_budget.json
0285cf9 Section 3: ExploreExploitController with diversity enforcement and exploit focus, explore_ratio param in AgentOrchestrator
4a8f7b5 Section 2: TwoBrainManager (reasoning + coding LLMs), two_brain/coding_model params in AgentOrchestrator
dbb5798 Section 1: PlatformDetector, LlamafileManager, env_config platform banner, llamafile model keys
eaffcc6 Rate-limit hardening: Gemini throttle (4s gap, 12 RPM, max_retries=5), backoff on 429/503, in-place model swap on APIError, real token counts, session token totals in RUN SUMMARY
960a802 Add context pruning: compact result format (~14 words vs ~55), per-step token logging, and memory compression every 5 experiments
28ee659 Add LM Studio local server support (OpenAI-compatible, localhost:1234/v1) to LLMManager
2e82395 Fix unauthorized import error inside smolagents CodeAgent sandbox and align verify_all.py prompt assertions
7b7dd86 Replace rigid tier-list prompt with open-ended researcher prompt (2833 chars)
b6a8de7 Fix step_callbacks crash: register on CallbackRegistry not replace with list (smolagents 1.26.0)
```

### git status
```text
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	deleted:    data/raw/Data_alpha_synuclein.xlsx
	deleted:    experiments/exp_001_laptop_logistic_regression_c_0_1/...
	deleted:    sessions/2026-06-19_13-55-46/session_summary.json
	deleted:    sessions/2026-06-19_13-55-50/session_summary.json
	modified:   PROJECT_MAP.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	alpha_synuclein_agent.egg-info/
	data/raw/csv_preview.txt
	experiments/.gitkeep
	experiments/exp_001_laptop_random_forest_baseline/
	... (other laptop experiments exp_002 to exp_085)
```

---

## 7. LIVE DIAGNOSTIC

### verify_all.py Output
```text
──────────────────────────────────────────────────────────────
  1. IMPORTS
──────────────────────────────────────────────────────────────
  PASS  tee_logger  [singleton OK]
  PASS  llm_manager  [17 models: ['gemini-flash', 'gemini-flash-lite', 'gemini-pro', 'groq-llama', 'groq-mixtral', 'mistral-small', 'cerebras', 'openrouter', 'local-qwen', 'local-deepseek', 'lmstudio-qwen', 'lmstudio-deepseek', 'lmstudio-mistral', 'lmstudio-llama', 'lmstudio-any', 'llamafile-14b', 'llamafile-32b']]
  PASS  session_manager  [OK]
  PASS  orchestrator  [OK]
  PASS  harness_template  [8745 chars]
  PASS  experiment_runner  [['exp_name', 'architecture_family', 'model_code', 'hyperparams']]
[2026-06-23 14:07:22] [INFO   ] [AUDIT] PASS — no forbidden patterns detected.
[2026-06-23 14:07:22] [WARNING] [AUDIT FAIL] Line 2: Loading 'test.pkl' directly  |  >> with open('data/splits/test.pkl','rb') as f: pass
[2026-06-23 14:07:22] [ERROR  ] [AUDIT] FAIL: 1 violation(s) found:
  • Line 2: Loading 'test.pkl' directly  |  >> with open('data/splits/test.pkl','rb') as f: pass
  PASS  audit_tool  [clean=PASS cheating=FAIL: 1 violation(s)]
[2026-06-23 14:07:22] [INFO   ] [LeaderboardTool] Leaderboard rebuilt from disk.
[2026-06-23 14:07:22] [INFO   ] [LeaderboardTool] Report generated. total_runs=84  best_f1=0.7004  untried_families=0
  PASS  leaderboard_tool  [2248 chars]
  PASS  rebuild_leaderboard  [84 exps]
  PASS  check_last_session  [last=2026-06-23_11-42-18]
  PASS  arxiv_tool  [OK]
  PASS  system_prompt  [2499 chars]
  PASS  data_pipeline  [OK]

──────────────────────────────────────────────────────────────
  2. DATA ARTIFACTS ON DISK
──────────────────────────────────────────────────────────────
  PASS  data/splits/train.pkl  [X=(276, 189)  y=(276,)]
  PASS  data/splits/val.pkl  [X=(60, 189)  y=(60,)]
  PASS  data/splits/test.pkl  (exists for wall)  [X_test=(60, 189)  y_test=(60,)]
  PASS  data/processed/scaler.pkl  [transform OK]
  PASS  data/processed/selector.pkl  [output shape (2, 189)]
  PASS  data/processed/class_weights.pkl  [{0: 0.318, 1: 5.75, 2: 2.76, 3: 3.136}]
  PASS  data/splits/split_hash.sha256  [07952c2516bd0782...]

──────────────────────────────────────────────────────────────
  3. REPRODUCIBILITY CHECK
──────────────────────────────────────────────────────────────
  PASS  random_state=42 reproducible splits  [train=276 rows  random_state=42 confirmed]

──────────────────────────────────────────────────────────────
  4. HARNESS RENDERING + AUDIT WALL
──────────────────────────────────────────────────────────────
  PASS  harness renders without test.pkl load  [8650 chars, test.pkl exec-free]
[2026-06-23 14:07:25] [INFO   ] [AUDIT] PASS — no forbidden patterns detected.
[2026-06-23 14:07:25] [INFO   ] [AUDIT] PASS — no forbidden patterns detected.
  PASS  audit: clean code passes  [2 clean codes passed]
[2026-06-23 14:07:25] [WARNING] [AUDIT FAIL] Line 1: Loading 'test.pkl' directly  |  >> open('data/splits/test.pkl','rb')
[2026-06-23 14:07:25] [ERROR  ] [AUDIT] FAIL: 1 violation(s) found:
  • Line 1: Loading 'test.pkl' directly  |  >> open('data/splits/test.pkl','rb')
[2026-06-23 14:07:25] [WARNING] [AUDIT FAIL] Line 1: Loading 'test.pkl' directly  |  >> pickle.load(open('test.pkl'))
[2026-06-23 14:07:25] [ERROR  ] [AUDIT] FAIL: 1 violation(s) found:
  • Line 1: Loading 'test.pkl' directly  |  >> pickle.load(open('test.pkl'))
[2026-06-23 14:07:25] [WARNING] [AUDIT FAIL] Line 1: Variable 'X_test' found — test features must never appear in experiment code  |  >> X_test = ...
[2026-06-23 14:07:25] [WARNING] [AUDIT FAIL] Line 2: Variable 'y_test' found — test labels must never appear in experiment code  |  >> accuracy_score(y_test, pred)
[2026-06-23 14:07:25] [ERROR  ] [AUDIT] FAIL: 2 violation(s) found:
  • Line 1: Variable 'X_test' found — test features must never appear in experiment code  |  >> X_test = ...
  • Line 2: Variable 'y_test' found — test labels must never appear in experiment code  |  >> accuracy_score(y_test, pred)
  PASS  audit: cheating code blocked  [3 cheating patterns caught]

──────────────────────────────────────────────────────────────
  5. LIVE EXPERIMENT RUN (max_experiments=1)
──────────────────────────────────────────────────────────────
[2026-06-23 14:07:25] [AGENT  ] [ExperimentRunner] Starting exp_085_laptop_rf_verify_check: rf_verify_check (family=classical_ml)
[2026-06-23 14:07:25] [INFO   ] [ExperimentRunner] Files written to experiments\exp_085_laptop_rf_verify_check
[2026-06-23 14:07:25] [INFO   ] [AUDIT] PASS — no forbidden patterns detected.
[2026-06-23 14:07:25] [INFO   ] [ExperimentRunner] Audit PASS for exp_085_laptop_rf_verify_check
[2026-06-23 14:07:25] [INFO   ] [ExperimentRunner] Running: D:\3rd sem M.tech\agent_workspace\.venv\Scripts\python.exe D:\3rd sem M.tech\agent_workspace\experiments\exp_085_laptop_rf_verify_check\train_eval.py
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] EXP_DIR      = D:\3rd sem M.tech\agent_workspace\experiments\exp_085_laptop_rf_verify_check
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] PROJECT_ROOT = D:\3rd sem M.tech\agent_workspace
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] Experiment   = exp_085_laptop_rf_verify_check
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] Loading train split ...
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] Loading val split ...
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] Train: X=(276, 189)  y=(276,)
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] Val  : X=(60, 189)    y=(60,)
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] Applying scaler + selector ...
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] Reduced: train (276, 189)  val (60, 189)
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] Loading class weights ...
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] Class weights: {0: 0.31797235023041476, 1: 5.75, 2: 2.76, 3: 3.1363636363636362}
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] Importing build_and_train from D:\3rd sem M.tech\agent_workspace\experiments\exp_085_laptop_rf_verify_check\model.py ...
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] Calling build_and_train() ...
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] build_and_train() returned in 0.13s
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] Evaluating on val set ...
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] val_accuracy  = 0.7833
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] val_f1_macro  = 0.5442
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] val_f1/class  = {'0': 0.9195402298850575, '1': 0.4, '2': 0.42857142857142855, '3': 0.42857142857142855}
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] results.json written -> D:\3rd sem M.tech\agent_workspace\experiments\exp_085_laptop_rf_verify_check\results.json
[2026-06-23 14:07:33] [INFO   ]   [harness] [harness] DONE — val_f1_macro = 0.5442
[2026-06-23 14:07:33] [INFO   ] [ExperimentRunner] Leaderboard updated. total_runs=85  best_f1=0.7004
[2026-06-23 14:07:33] [AGENT  ] [ExperimentRunner] exp_085_laptop_rf_verify_check COMPLETE | val_f1_macro=0.5442 | status=success | time=8.7s
[2026-06-23 14:07:33] [INFO   ] [ExperimentRunner] Full result:

==============================================================
  EXPERIMENT : exp_085_laptop_rf_verify_check
  Arch       : rf_verify_check
  Family     : classical_ml
  Machine    : laptop
  Status     : success
==============================================================
  val_f1_macro : 0.5442
  val_accuracy : 0.7833
  train_time   : 0.1s
  model_params : 189

  Per-class F1:
    0 (No    ): 0.9195  |##################  |
    1 (Low   ): 0.4000  |########            |
    2 (Medium): 0.4286  |########            |
    3 (High  ): 0.4286  |########            |
==============================================================
  PASS  live experiment (RandomForest, 50 trees)  [val_f1_macro=0.5442  val_accuracy=0.7833]

──────────────────────────────────────────────────────────────
  6. SESSION MANAGER
──────────────────────────────────────────────────────────────
[2026-06-23 14:07:33] [AGENT  ] [Session] Started  session_id=2026-06-23_14-07-33  dir=sessions\2026-06-23_14-07-33
[2026-06-23 14:07:36] [AGENT  ] [Session] Ended  status=completed  experiments=1  session=2026-06-23_14-07-33
  PASS  session_manager lifecycle  [id=2026-06-23_14-07-33  status=completed]
[2026-06-23 14:07:36] [AGENT  ] [Session] Started  session_id=2026-06-23_14-07-36  dir=sessions\2026-06-23_14-07-36
[2026-06-23 14:07:37] [AGENT  ] [Session] Ended  status=crashed  experiments=0  session=2026-06-23_14-07-36
  PASS  session_manager crash persistence  [crash preserved: ValueError: simulated]

──────────────────────────────────────────────────────────────
  7. ORCHESTRATOR INIT (no API call)
──────────────────────────────────────────────────────────────
[2026-06-23 14:07:37] [AGENT  ] [Orchestrator] Initialising  model=local-qwen  verbosity=concise
[2026-06-23 14:07:41] [INFO   ] [LLMManager] Active: 'local-qwen'  |  model_id=ollama/qwen2.5-coder:32b  |  provider=ollama
[2026-06-23 14:07:41] [INFO   ] [Orchestrator] DuckDuckGoSearchTool added.
[2026-06-23 14:07:41] [INFO   ] [Orchestrator] CodeAgent ready. max_steps=500  verbosity_level=0
[2026-06-23 14:07:41] [AGENT  ] [Orchestrator] Ready. Call run() to start.
  PASS  orchestrator.__init__ (tools wired)  [tools=['run_experiment', 'read_leaderboard', 'audit_code', 'search_arxiv_papers', 'web_search']]

──────────────────────────────────────────────────────────────
  8. GIT STATUS
──────────────────────────────────────────────────────────────
         70f4f41 fix: Add WSL path support for local notebook runtime and resolve Gemini key configuration issue
         81059c0 fix: Guard raises RuntimeError instead of returning string
         6306b3d feat: Add early-stop guard + dynamic session context in prompt
         33f2ed9 docs: Add DEPLOYMENT.md deployment guide
         dbee9cd Section 5: Architecture Landscape menu in system prompt (tabular/NN/sequence/tabular-DL/ensemble/NLP framing), fix merged sentence, expand SYSTEM_PROMPT_SHORT
  PASS  git log (recent commits)  [5 recent commits]
  PASS  .gitignore rules  [rules correct]

──────────────────────────────────────────────────────────────
  SUMMARY
──────────────────────────────────────────────────────────────

  30/30 checks passed
  ALL CHECKS PASSED ✓
```
