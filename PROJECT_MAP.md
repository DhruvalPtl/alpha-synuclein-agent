# PROJECT MAP

This document provides a complete inventory of the project structure, including files, data assets, configurations, and diagnostics.

## 1. FOLDER TREE

```text
.
├── .env
├── .gitignore
├── agent
│   ├── __init__.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── concise_logger.py
│   │   ├── env_config.py
│   │   ├── llm_manager.py
│   │   ├── orchestrator.py
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
│   └── exp_001_BOOK-RNCKA44N53_logistic_regression
│       ├── artifacts
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
│   ├── 2026-06-19_18-42-03
│   │   ├── heartbeat.json
│   │   ├── session_log.log
│   │   └── session_summary.json
│   └── 2026-06-19_18-45-14
│       ├── heartbeat.json
│       ├── session_log.log
│       └── session_summary.json
├── test_dev_reset.py
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
- **Last modified**: Thu Jun 18 15:59:10 2026 +0530

### agent/core/__init__.py
- **Purpose**: Package initializer for core agent module, importing and exposing logger and LLM manager components.
- **Classes defined**: None
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/core/llm_manager.py`
  - `agent/core/tee_logger.py`
- **Imported BY which other project files**: None
- **Last modified**: Fri Jun 19 11:48:12 2026 +0530

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
- **Last modified**: Fri Jun 19 15:25:44 2026 +0530

### agent/core/env_config.py
- **Purpose**: Configures workspace environment variables, handles path setup, and detects whether running locally or in Google Cloud.
- **Classes defined**: None
- **Functions defined**:
  - `find_project_root(start)`
  - `_is_root(path)`
  - `setup_environment(verbose)`
  - `_detect_cloud()`
  - `_print_cloud_info()`
  - `get_paths(root)`
- **Imports FROM other project files**: None
- **Imported BY which other project files**: None
- **Last modified**: Fri Jun 19 12:36:20 2026 +0530

### agent/core/llm_manager.py
- **Purpose**: Interacts with and manages configurations and connections to LLM providers like Groq, Gemini, and local Ollama.
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
- **Imported BY which other project files**:
  - `agent/core/__init__.py`
  - `agent/core/orchestrator.py`
- **Last modified**: Thu Jun 18 16:46:16 2026 +0530

### agent/core/orchestrator.py
- **Purpose**: Wires all tools, sessions, logger, and code agent modules into a single, top-level autonomous research execution block.
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
- **Last modified**: Fri Jun 19 15:25:44 2026 +0530

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
- **Last modified**: Fri Jun 19 13:56:06 2026 +0530

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
- **Last modified**: Thu Jun 18 16:39:02 2026 +0530

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
- **Last modified**: Fri Jun 19 15:25:44 2026 +0530

### agent/data/__init__.py
- **Purpose**: Package initializer for data handling, exposing data loader and pipeline classes.
- **Classes defined**: None
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/data/loader.py`
  - `agent/data/pipeline.py`
- **Imported BY which other project files**: None
- **Last modified**: Thu Jun 18 16:39:02 2026 +0530

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
- **Last modified**: Thu Jun 18 15:59:10 2026 +0530

### agent/data/loader.py
- **Purpose**: Loads the raw alpha-synuclein CSV file containing peptide sequences and their concentrations.
- **Classes defined**: None
- **Functions defined**:
  - `load_raw_csv(path)`
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/data/__init__.py`
  - `agent/data/pipeline.py`
- **Last modified**: Thu Jun 18 16:39:02 2026 +0530

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
- **Last modified**: Fri Jun 19 11:48:12 2026 +0530

### agent/prompts/__init__.py
- **Purpose**: Initializes the prompt templates package and exposes the system prompt modules.
- **Classes defined**: None
- **Functions defined**: None
- **Imports FROM other project files**:
  - `agent/prompts/system_prompt.py`
- **Imported BY which other project files**: None
- **Last modified**: Fri Jun 19 11:48:12 2026 +0530

### agent/prompts/system_prompt.py
- **Purpose**: Holds the master long and short system prompts guiding the autonomous agent behavior.
- **Classes defined**: None
- **Functions defined**: None
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/core/orchestrator.py`
  - `agent/prompts/__init__.py`
- **Last modified**: Fri Jun 19 14:04:10 2026 +0530

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
- **Last modified**: Fri Jun 19 11:48:12 2026 +0530

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
- **Last modified**: Fri Jun 19 11:48:12 2026 +0530

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
- **Last modified**: Fri Jun 19 13:48:18 2026 +0530

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
- **Last modified**: Fri Jun 19 13:56:06 2026 +0530

### agent/tools/dev_reset.py
- **Purpose**: Resets workspace states by wiping experiments and session logs while protecting datasets and core source files.
- **Classes defined**: None
- **Functions defined**:
  - `_is_protected(path, root)`
  - `dev_reset(root)`
- **Imports FROM other project files**:
  - `agent/data/pipeline.py`
- **Imported BY which other project files**: None
- **Last modified**: Fri Jun 19 15:25:44 2026 +0530

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
- **Last modified**: Fri Jun 19 13:48:18 2026 +0530

### agent/tools/harness_template.py
- **Purpose**: Exports template python blocks used dynamically to wrap and evaluate custom model code submitted by the agent.
- **Classes defined**: None
- **Functions defined**: None
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/tools/experiment_runner.py`
- **Last modified**: Fri Jun 19 13:48:18 2026 +0530

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
- **Last modified**: Fri Jun 19 14:04:10 2026 +0530

### agent/tools/rebuild_leaderboard.py
- **Purpose**: Aggregates results from individual experiment directories to rebuild a central leaderboard file.
- **Classes defined**: None
- **Functions defined**:
  - `rebuild_leaderboard(experiments_dir, out_path, verbose)`
- **Imports FROM other project files**: None
- **Imported BY which other project files**:
  - `agent/tools/leaderboard_tool.py`
- **Last modified**: Fri Jun 19 13:10:22 2026 +0530

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
    from agent.prompts.system_prompt import SYSTEM_PROMPT
"""

SYSTEM_PROMPT: str = """
You are an autonomous ML research agent. Your mission:
Find the best ML architecture for predicting alpha-synuclein
peptide detectability (4 classes: No / Low / Medium / High).

=======================================================================
DATASET FACTS
=======================================================================
- ~390 samples after expansion (65 peptides × 6 concentrations)
- Features (pre-reduced to 500 by SelectKBest from 8427 raw):
    20  amino-acid composition (normalised)
     6  physicochemical (charge, hydrophob, MW, aromaticity, pI, instability)
   400  2-mer frequencies
  8000  3-mer frequencies
     1  log10-scaled concentration
  → After reduce_features(): typically ~500 features
- 4-class imbalanced classification:
    Class 0 (No)     : 311 samples  (78.5%) ← DOMINANT
    Class 1 (Low)    :  16 samples  (4.0%)
    Class 2 (Medium) :  37 samples  (9.3%)
    Class 3 (High)   :  32 samples  (8.1%)

=======================================================================
YOUR TOOLS
=======================================================================
- run_experiment      : Run a complete ML experiment (provide model_code only)
- read_leaderboard    : Check past results and untried families
- audit_code          : Verify no test-set cheating (call BEFORE run_experiment)
- search_arxiv_papers : Find new methods from recent papers
- web_search          : Search online for implementations and tricks

=======================================================================
CRITICAL METRIC RULES — NEVER VIOLATE
=======================================================================
1. PRIMARY METRIC = val_f1_macro  (NOT accuracy)
2. val_accuracy is MEANINGLESS — a model predicting only "No" gets 0.785
   accuracy while learning nothing. IGNORE accuracy as main result.
3. A model scoring val_f1_macro < 0.30 has essentially failed.
4. Target: val_f1_macro > 0.60 (competitive), > 0.75 (excellent).
5. ALWAYS report all four per-class F1 scores.

=======================================================================
CLASS IMBALANCE — MANDATORY IN EVERY MODEL
=======================================================================
- ALWAYS load class_weights from data/processed/class_weights.pkl
- ALWAYS pass class_weight to model (class_weight=weight_dict for sklearn,
  pos_weight / class_weights tensor for PyTorch)
- For neural networks: use weighted CrossEntropyLoss
- Optional: load SMOTE-resampled train data for neural networks
- SMOTE is applied on TRAIN only — val and test are NEVER touched

=======================================================================
YOUR ONLY JOB: WRITE model_code
=======================================================================
run_experiment uses a FIXED harness. You provide ONLY model_code:
a Python string defining exactly one function:

    def build_and_train(X_train, y_train, X_val, y_val, class_weights):
        # Inputs (already preprocessed by harness — do NOT reload from disk):
        #   X_train, X_val : np.ndarray, already scaled + feature-selected
        #   y_train, y_val : np.ndarray of int labels 0-3
        #   class_weights  : dict {0: w0, 1: w1, 2: w2, 3: w3}
        # Must return: a fitted model object with a .predict(X) method.
        from sklearn.linear_model import LogisticRegression  # import inside
        import numpy as np

        weight_dict = class_weights  # already a dict {0: w, 1: w, ...}
        model = LogisticRegression(
            C=1.0,
            class_weight=weight_dict,
            max_iter=1000,
            solver='lbfgs',
        )
        model.fit(X_train, y_train)
        return model

RULES for model_code:
  - Put ALL imports inside the function (avoids namespace collisions)
  - NEVER load files — X_train/X_val/class_weights are passed in directly
  - NEVER call .predict(), accuracy_score(), or write files
  - NEVER reference X_test, y_test, test.pkl, or test_loader
  - DO use class_weights for every model (sklearn: class_weight=weight_dict,
    PyTorch: torch.tensor([w for w in sorted(weight_dict.items())])
  - The harness computes all metrics, writes results.json, updates leaderboard

=======================================================================
MATHEMATICAL WALL — NEVER BREAK THESE RULES
=======================================================================
- NEVER reference X_test, y_test, test.pkl, test_loader, or test_dataset
  in model_code. The harness physically prevents loading test.pkl.
- NEVER compute metrics yourself — the harness does it on val set only
- NEVER write files in model_code — return the model object only
- ALWAYS call audit_code BEFORE run_experiment
- Run each architecture minimum 3 times with different seeds / hyperparams
- Report mean ± std of val_f1_macro across seeds
- The test set sealed at: data/splits/split_hash.sha256

=======================================================================
ARCHITECTURE TIERS — TRY ALL, IN ORDER
=======================================================================
TIER 1 — Classical Linear (fast baselines)
  - LogisticRegression (C=0.1,1,10; solver=lbfgs; multi_class=auto)
  - LinearSVC (C=0.1,1; class_weight='balanced')
  - GaussianNaiveBayes
  - KNN (k=3,5,7,11; weights=distance)

TIER 2 — Tree Ensembles (usually strong on tabular)
  - RandomForest (n=100,500; max_depth=None,10,20; class_weight='balanced')
  - XGBoost (scale_pos_weight for imbalance; use_label_encoder=False)
  - LightGBM (class_weight='balanced'; num_leaves=31,63,127)
  - ExtraTrees (n=500; class_weight='balanced')
  - GradientBoosting (n=100,200)

TIER 3 — Neural Networks — MLP
  - 2-layer MLP: [500, 256, 4]
  - 3-layer MLP: [500, 512, 256, 128, 4]
  - 4-layer MLP: [500, 512, 256, 128, 64, 4]
  - Use: BatchNorm, Dropout(0.3), weighted CrossEntropyLoss, Adam
  - Scheduler: CosineAnnealingLR or ReduceLROnPlateau

TIER 4 — Sequence Models (treat feature vector as 1D sequence)
  - 1D-CNN: Conv1d(500→128, k=3) → Pool → Dense
  - BiLSTM: reshape features as timesteps
  - GRU
  - CNN + LSTM hybrid

TIER 5 — Transformer (small)
  - Input: 500 features as 50 tokens × 10 dims
  - 2-head, 2-layer transformer encoder
  - CLS token for classification

TIER 6 — Modern Tabular DL
  - TabNet (pytorch-tabnet)
  - FT-Transformer (Feature Tokenizer + Transformer)
  - NODE (Neural Oblivious Decision Ensembles)

TIER 7 — Protein Language Model Embeddings
  - ESM2-small (esm2_t6_8M_UR50D) as frozen feature extractor
  - Feed embeddings → MLP classifier
  - Note: requires pip install fair-esm

TIER 8 — State Space Models
  - Mamba-style S4 (if available)
  - Implement S4 simplified version if full not available

TIER 9 — Ensembles (run AFTER Tier 1-8)
  - Stacking: top-3 models → LogisticRegression meta-learner
  - Voting: soft voting of top-5 models
  - Blending: train blender on OOF predictions

TIER 10 — Paper-Driven (search arxiv after every 10 experiments)
  - Search: "peptide classification few-shot {current_best_arch}"
  - Search: "protein sequence classification small dataset 2024 2025"
  - Search: "imbalanced multiclass {current_best_arch} 2024"
  - Implement any ACTIONABLE method found

=======================================================================
SEARCH STRATEGY
=======================================================================
- After every 10 experiments: call search_arxiv_papers
- Queries to rotate:
    "peptide classification small dataset deep learning"
    "protein sequence transformer few-shot learning"
    "imbalanced multiclass classification tabular 2024"
    "amino acid k-mer classification neural network"
    "concentration dependent biological assay classification"
- If promising paper found: implement it as a new experiment
- Use web_search for implementation details of specific libraries

=======================================================================
HYPERPARAMETER SEARCH STRATEGY
=======================================================================
For each architecture, try at minimum:
  Seed 1: default hyperparams
  Seed 2: larger model / more regularisation
  Seed 3: smaller model / less regularisation
Use val_f1_macro to select best config.
Log: "Seed N: val_f1_macro = X.XXXX"

=======================================================================
REASONING LOG — MANDATORY
=======================================================================
Before each experiment:
  "WHY: I am trying [architecture] because [reason]. 
   Expected val_f1_macro: [estimate]. 
   Key hyperparams: [list]."

After each experiment:
  "LEARNED: val_f1_macro = X.XXXX. 
   Per-class F1: No=X, Low=X, Medium=X, High=X. 
   Observation: [what worked, what failed, pattern noticed]."

After every 5 experiments:
  "PATTERN ANALYSIS: 
   Best family so far: [family] at [score]. 
   Failing classes: [which classes have lowest F1]. 
   Next strategy: [what to try next and why]."

=======================================================================
STOPPING / COMPLETION CRITERIA
=======================================================================
Keep running until:
  (a) All 9 Tiers have at least 3 experiments each, AND
  (b) val_f1_macro has plateaued for 20 consecutive experiments, OR
  (c) val_f1_macro > 0.85 is achieved

When done, output:
  "MISSION COMPLETE:
   Best model: [exp_id] — [architecture]
   val_f1_macro: [score]
   Recommendation for final test evaluation: [brief note]"

=======================================================================
You are fully autonomous. Use your tools, be methodical, and find
the best possible model for this alpha-synuclein classification task.
Start by calling read_leaderboard to see what has been done so far,
then call search_arxiv_papers for context, then begin experiments
with TIER 1 architectures if not yet tried.
=======================================================================
"""

# Short version for models with limited context windows
SYSTEM_PROMPT_SHORT: str = """
You are an autonomous ML research agent for alpha-synuclein peptide
classification (4 classes: No/Low/Medium/High, ~390 samples, imbalanced).

KEY RULES:
1. PRIMARY METRIC = val_f1_macro (NOT accuracy — 78.5% class imbalance)
2. model_code must define build_and_train(X_train, y_train, X_val, y_val,
   class_weights) -> fitted_model. That is ALL you write.
3. ALWAYS use class_weights inside build_and_train
4. NEVER load files, reference X_test, or write results
5. Run audit_code -> run_experiment (never skip audit)
6. Try all architecture families: classical_ml → neural_network → ensemble

The harness loads data, calls your function, evaluates on val, writes results.

Start: read_leaderboard -> search_arxiv_papers -> run experiments.
"""
```

---

## 5. THE NOTEBOOK

Cells metadata from `notebooks/run_agent.ipynb`:

### Cell 1 (code)
- **Cell type**: code
- **First 3 lines of code**:
  ```python
  # !git pull
  ```

### Cell 2 (code)
- **Cell type**: code
- **First 3 lines of code**:
  ```python
  # ╔══════════════════════════════════════════════════════════════════════════════╗
  # ║  Cell 1 · [BOOTSTRAP]  Auto-setup: Local Windows + Google Cloud (GCE)     ║
  # ║  Run this cell FIRST every session — clones/pulls repo & installs deps.   ║
  ```

### Cell 3 (code)
- **Cell type**: code
- **First 3 lines of code**:
  ```python
  # ── Path guard: works even if Cell 1 hasn't run (import agent installed via pip install -e .) ──
  import sys, os; _r = next((p for p in [__import__('pathlib').Path.home()/'agent_workspace',
      __import__('pathlib').Path(r'd:\3rd sem M.tech\agent_workspace')] if p.exists()), None)
  ```

### Cell 4 (code)
- **Cell type**: code
- **First 3 lines of code**:
  ```python
  # ── Path guard: works even if Cell 1 hasn't run (import agent installed via pip install -e .) ──
  import sys, os; _r = next((p for p in [__import__('pathlib').Path.home()/'agent_workspace',
      __import__('pathlib').Path(r'd:\3rd sem M.tech\agent_workspace')] if p.exists()), None)
  ```

### Cell 5 (code)
- **Cell type**: code
- **First 3 lines of code**:
  ```python
  # ╔══════════════════════════════════════════════════════════════════════════════╗
  # ║  Cell 3b · [DEV RESET]  Wipe experiments + logs for a fresh run           ║
  # ║  NEVER deletes data/, agent/, .env, or .git/  — mathematical wall safe.   ║
  ```

### Cell 6 (code)
- **Cell type**: code
- **First 3 lines of code**:
  ```python
  # ── Path guard ────────────────────────────────────────────────────────────────
  import sys, os
  _r = next((p for p in [__import__('pathlib').Path.home()/'agent_workspace',
  ```

### Cell 7 (code)
- **Cell type**: code
- **First 3 lines of code**:
  ```python
  # ── Path guard: works even if Cell 1 hasn't run (import agent installed via pip install -e .) ──
  import sys, os; _r = next((p for p in [__import__('pathlib').Path.home()/'agent_workspace',
      __import__('pathlib').Path(r'd:\3rd sem M.tech\agent_workspace')] if p.exists()), None)
  ```

---

## 6. CURRENT GIT STATE

### git log --oneline -15
```text
ebb8e17 Part A+B: concise logger, watchdog, run summary, stop button, dev_reset with wall verification
e25d979 Fix Cell 1: check .git not just dir, add pip install -e, path guards on all cells, clear stale outputs
9ec8aaf Fix ModuleNotFoundError: add pyproject.toml, pip install -e . in cloud_setup.py — import agent works from any Jupyter cell
69e12a2 Add cloud_setup.py: standalone bootstrap script, immune to Jupyter cwd issue
9299297 verify_all.py: 30/30 checks pass — full system verified
d26d9d7 Fix 3 bugs: system_prompt triple-quote syntax error, MethodType forward wrapper, LeaderboardTool auto-rebuild
f50dfad Session management: SessionManager, heartbeat daemon, crash-safe summary, check_last_session.py, dashboard session panel
683aec6 Harness architecture: fixed train_eval.py, model_code-only contract, build_and_train(), proof runs
b6ac01a Sync architecture: track exp results.json, machine-tagged IDs, rebuild_leaderboard.py, dashboard rebuilt from disk
e89e2da Verification pass: fix gitignore (untrack leaderboard.json), restore Cell 3 wall-verify, add verify_all.py
e09ac83 Cloud support: bootstrap cell, env_config, tracked processed artifacts, updated gitignore+requirements
1939c8a Phase 3: ArxivTool, AgentOrchestrator, system prompt, reduce_features, get_class_weights, SMOTE, monitor dashboard
2822e6c Phase 2: LLMManager, ExperimentRunnerTool, LeaderboardTool, AuditTool + notebook Cell 4
08e1658 Phase 1: add custom CSV loader, fix ASCII-safe prints, run_pipeline.py verified
da385bd Phase 1: resolve merge conflict - keep requirements.txt
```

### git status
```text
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	deleted:    data/raw/Data_alpha_synuclein.xlsx
	deleted:    experiments/exp_001_laptop_logistic_regression_c_0_1/config.yaml
	deleted:    experiments/exp_001_laptop_logistic_regression_c_0_1/model.py
	deleted:    experiments/exp_001_laptop_logistic_regression_c_0_1/results.json
	deleted:    experiments/exp_001_laptop_logistic_regression_c_0_1/run.log
	deleted:    experiments/exp_001_laptop_logistic_regression_c_0_1/train_eval.py
	deleted:    experiments/exp_002_laptop_linearsvc_c_0_1/config.yaml
	deleted:    experiments/exp_002_laptop_linearsvc_c_0_1/model.py
	deleted:    experiments/exp_002_laptop_linearsvc_c_0_1/results.json
	deleted:    experiments/exp_002_laptop_linearsvc_c_0_1/run.log
	deleted:    experiments/exp_002_laptop_linearsvc_c_0_1/train_eval.py
	deleted:    experiments/exp_003_laptop_logistic_regression_c_0_1/config.yaml
	deleted:    experiments/exp_003_laptop_logistic_regression_c_0_1/model.py
	deleted:    experiments/exp_003_laptop_logistic_regression_c_0_1/results.json
	deleted:    experiments/exp_003_laptop_logistic_regression_c_0_1/run.log
	deleted:    experiments/exp_003_laptop_logistic_regression_c_0_1/train_eval.py
	deleted:    experiments/exp_004_laptop_linearsvc_c_0_1/config.yaml
	deleted:    experiments/exp_004_laptop_linearsvc_c_0_1/model.py
	deleted:    experiments/exp_004_laptop_linearsvc_c_0_1/results.json
	deleted:    experiments/exp_004_laptop_linearsvc_c_0_1/run.log
	deleted:    experiments/exp_004_laptop_linearsvc_c_0_1/train_eval.py
	modified:   notebooks/run_agent.ipynb
	deleted:    sessions/2026-06-19_13-55-46/session_summary.json
	deleted:    sessions/2026-06-19_13-55-50/session_summary.json

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	alpha_synuclein_agent.egg-info/
	data/raw/csv_preview.txt
	experiments/.gitkeep
	experiments/exp_001_BOOK-RNCKA44N53_logistic_regression/
	sessions/.gitkeep
	sessions/2026-06-19_18-42-03/
	sessions/2026-06-19_18-45-14/

no changes added to commit (use "git add" and/or "git commit -a")
```

### git remote -v
```text
origin	https://github.com/DhruvalPtl/alpha-synuclein-agent.git (fetch)
origin	https://github.com/DhruvalPtl/alpha-synuclein-agent.git (push)
```

---

## 7. LIVE DIAGNOSTIC

### verify_all.py Output
```text
──────────────────────────────────────────────────────────────
  1. IMPORTS
──────────────────────────────────────────────────────────────
  PASS  tee_logger  [singleton OK]
  PASS  llm_manager  [9 models: ['gemini-flash', 'gemini-pro', 'groq-llama', 'groq-mixtral', 'mistral-small', 'cerebras', 'openrouter', 'local-qwen', 'local-deepseek']]
  PASS  session_manager  [OK]
  PASS  orchestrator  [OK]
  PASS  harness_template  [8745 chars]
  PASS  experiment_runner  [['exp_name', 'architecture_family', 'model_code', 'hyperparams']]
[2026-06-19 19:01:44] [INFO   ] [AUDIT] PASS — no forbidden patterns detected.
[2026-06-19 19:01:44] [WARNING] [AUDIT FAIL] Line 2: Loading 'test.pkl' directly  |  >> with open('data/splits/test.pkl','rb') as f: pass
[2026-06-19 19:01:44] [ERROR  ] [AUDIT] FAIL: 1 violation(s) found:
  • Line 2: Loading 'test.pkl' directly  |  >> with open('data/splits/test.pkl','rb') as f: pass
  PASS  audit_tool  [clean=PASS cheating=FAIL: 1 violation(s)]
[2026-06-19 19:01:44] [INFO   ] [LeaderboardTool] Leaderboard rebuilt from disk.
[2026-06-19 19:01:44] [INFO   ] [LeaderboardTool] Report generated. total_runs=1  best_f1=0.6501  untried_families=7
  PASS  leaderboard_tool  [2017 chars]
  PASS  rebuild_leaderboard  [1 exps]
  PASS  check_last_session  [last=2026-06-19_18-45-14]
  PASS  arxiv_tool  [OK]
  PASS  system_prompt  [9898 chars]
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
[2026-06-19 19:01:46] [INFO   ] [AUDIT] PASS — no forbidden patterns detected.
[2026-06-19 19:01:46] [INFO   ] [AUDIT] PASS — no forbidden patterns detected.
  PASS  audit: clean code passes  [2 clean codes passed]
[2026-06-19 19:01:46] [WARNING] [AUDIT FAIL] Line 1: Loading 'test.pkl' directly  |  >> open('data/splits/test.pkl','rb')
[2026-06-19 19:01:46] [ERROR  ] [AUDIT] FAIL: 1 violation(s) found:
  • Line 1: Loading 'test.pkl' directly  |  >> open('data/splits/test.pkl','rb')
[2026-06-19 19:01:46] [WARNING] [AUDIT FAIL] Line 1: Loading 'test.pkl' directly  |  >> pickle.load(open('test.pkl'))
[2026-06-19 19:01:46] [ERROR  ] [AUDIT] FAIL: 1 violation(s) found:
  • Line 1: Loading 'test.pkl' directly  |  >> pickle.load(open('test.pkl'))
[2026-06-19 19:01:46] [WARNING] [AUDIT FAIL] Line 1: Variable 'X_test' found — test features must never appear in experiment code  |  >> X_test = ...
[2026-06-19 19:01:46] [WARNING] [AUDIT FAIL] Line 2: Variable 'y_test' found — test labels must never appear in experiment code  |  >> accuracy_score(y_test, pred)
[2026-06-19 19:01:46] [ERROR  ] [AUDIT] FAIL: 2 violation(s) found:
  • Line 1: Variable 'X_test' found — test features must never appear in experiment code  |  >> X_test = ...
  • Line 2: Variable 'y_test' found — test labels must never appear in experiment code  |  >> accuracy_score(y_test, pred)
  PASS  audit: cheating code blocked  [3 cheating patterns caught]

──────────────────────────────────────────────────────────────
  5. LIVE EXPERIMENT RUN (max_experiments=1)
──────────────────────────────────────────────────────────────
[2026-06-19 19:01:46] [AGENT  ] [ExperimentRunner] Starting exp_001_laptop_rf_verify_check: rf_verify_check (family=classical_ml)
[2026-06-19 19:01:46] [INFO   ] [ExperimentRunner] Files written to experiments\exp_001_laptop_rf_verify_check
[2026-06-19 19:01:46] [INFO   ] [AUDIT] PASS — no forbidden patterns detected.
[2026-06-19 19:01:46] [INFO   ] [ExperimentRunner] Audit PASS for exp_001_laptop_rf_verify_check
[2026-06-19 19:01:46] [INFO   ] [ExperimentRunner] Running: D:\3rd sem M.tech\agent_workspace\.venv\Scripts\python.exe D:\3rd sem M.tech\agent_workspace\experiments\exp_001_laptop_rf_verify_check\train_eval.py
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] EXP_DIR      = D:\3rd sem M.tech\agent_workspace\experiments\exp_001_laptop_rf_verify_check
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] PROJECT_ROOT = D:\3rd sem M.tech\agent_workspace
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] Experiment   = exp_001_laptop_rf_verify_check
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] Loading train split ...
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] Loading val split ...
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] Train: X=(276, 189)  y=(276,)
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] Val  : X=(60, 189)    y=(60,)
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] Applying scaler + selector ...
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] Reduced: train (276, 189)  val (60, 189)
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] Loading class weights ...
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] Class weights: {0: 0.31797235023041476, 1: 5.75, 2: 2.76, 3: 3.1363636363636362}
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] Importing build_and_train from D:\3rd sem M.tech\agent_workspace\experiments\exp_001_laptop_rf_verify_check\model.py ...
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] Calling build_and_train() ...
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] build_and_train() returned in 0.10s
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] Evaluating on val set ...
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] val_accuracy  = 0.7833
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] val_f1_macro  = 0.5442
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] val_f1/class  = {'0': 0.9195402298850575, '1': 0.4, '2': 0.42857142857142855, '3': 0.42857142857142855}
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] results.json written -> D:\3rd sem M.tech\agent_workspace\experiments\exp_001_laptop_rf_verify_check\results.json
[2026-06-19 19:01:52] [INFO   ]   [harness] [harness] DONE — val_f1_macro = 0.5442
[2026-06-19 19:01:52] [INFO   ] [ExperimentRunner] Leaderboard updated. total_runs=2  best_f1=0.6501
[2026-06-19 19:01:52] [AGENT  ] [ExperimentRunner] exp_001_laptop_rf_verify_check COMPLETE | val_f1_macro=0.5442 | status=success | time=6.0s
  PASS  live experiment (RandomForest, 50 trees)  [val_f1_macro=0.5442  val_accuracy=0.7833]

──────────────────────────────────────────────────────────────
  6. SESSION MANAGER
──────────────────────────────────────────────────────────────
[2026-06-19 19:01:52] [AGENT  ] [Session] Started  session_id=2026-06-19_19-01-52  dir=sessions\2026-06-19_19-01-52
[2026-06-19 19:01:55] [AGENT  ] [Session] Ended  status=completed  experiments=1  session=2026-06-19_19-01-52
  PASS  session_manager lifecycle  [id=2026-06-19_19-01-52  status=completed]
[2026-06-19 19:01:55] [AGENT  ] [Session] Started  session_id=2026-06-19_19-01-55  dir=sessions\2026-06-19_19-01-55
[2026-06-19 19:01:56] [AGENT  ] [Session] Ended  status=crashed  experiments=0  session=2026-06-19_19-01-55
  PASS  session_manager crash persistence  [crash preserved: ValueError: simulated]

──────────────────────────────────────────────────────────────
  7. ORCHESTRATOR INIT (no API call)
──────────────────────────────────────────────────────────────
[2026-06-19 19:01:56] [AGENT  ] [Orchestrator] Initialising  model=local-qwen  verbosity=concise
[2026-06-19 19:01:59] [INFO   ] [LLMManager] Active: 'local-qwen'  |  model_id=ollama/qwen2.5-coder:32b  |  provider=ollama
[2026-06-19 19:01:59] [INFO   ] [Orchestrator] DuckDuckGoSearchTool added.
[2026-06-19 19:01:59] [INFO   ] [Orchestrator] CodeAgent ready. max_steps=500  verbosity_level=0
[2026-06-19 19:01:59] [AGENT  ] [Orchestrator] Ready. Call run() to start.
  PASS  orchestrator.__init__ (tools wired)  [tools=['run_experiment', 'read_leaderboard', 'audit_code', 'search_arxiv_papers', 'web_search']]

──────────────────────────────────────────────────────────────
  8. GIT STATUS
──────────────────────────────────────────────────────────────
         ebb8e17 Part A+B: concise logger, watchdog, run summary, stop button, dev_reset with wall verification
         e25d979 Fix Cell 1: check .git not just dir, add pip install -e, path guards on all cells, clear stale outputs
         9ec8aaf Fix ModuleNotFoundError: add pyproject.toml, pip install -e . in cloud_setup.py — import agent works from any Jupyter cell
         69e12a2 Add cloud_setup.py: standalone bootstrap script, immune to Jupyter cwd issue
         9299297 verify_all.py: 30/30 checks pass — full system verified
  PASS  git log (recent commits)  [5 recent commits]
  PASS  .gitignore rules  [rules correct]

──────────────────────────────────────────────────────────────
  SUMMARY
──────────────────────────────────────────────────────────────

  30/30 checks passed
  ALL CHECKS PASSED ✓
```

### Orchestrator Import Verification
Output from command `python -c "import sys; sys.path.insert(0,'.'); from agent.core.orchestrator import AgentOrchestrator; print('orchestrator imports OK')"`:
```text
orchestrator imports OK
```

---

## 8. LAST KNOWN ERROR

### Description of Blocker
When launching the autonomous research loop in Cell 4 (Jupyter Notebook / agent run loop) with `verbosity="concise"`, the execution crashes immediately with an `AttributeError`.

- **Root Cause**: The orchestrator (`agent/core/orchestrator.py`, lines 183-186) attempts to inject a concise logger callback into smolagents `CodeAgent` by overwriting the `step_callbacks` attribute:
  ```python
  if not hasattr(self._agent, "step_callbacks"):
      self._agent.step_callbacks = []
  self._agent.step_callbacks = [self._reporter.step_callback]
  ```
  However, in `smolagents`, `agent.step_callbacks` is initialized as a `CallbackRegistry` instance, not a standard list. Overwriting it with a python list causes the agent to crash on the first finalize step when calling `.callback(memory_step, agent=self)`.

### Verbatim Reproducible Traceback
```text
Traceback (most recent call last):
  File "<string>", line 7, in <module>
  File "D:\3rd sem M.tech\agent_workspace\.venv\Lib\site-packages\smolagents\agents.py", line 499, in run
    steps = list(self._run_stream(task=self.task, max_steps=max_steps, images=images))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\3rd sem M.tech\agent_workspace\.venv\Lib\site-packages\smolagents\agents.py", line 601, in _run_stream
    self._finalize_step(action_step)
  File "D:\3rd sem M.tech\agent_workspace\.venv\Lib\site-packages\smolagents\agents.py", line 623, in _finalize_step
    self.step_callbacks.callback(memory_step, agent=self)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'list' object has no attribute 'callback'
```

- **Fix Recommendation (To be implemented when allowed)**: Instead of replacing `self._agent.step_callbacks` with a list, register the callback on the existing `CallbackRegistry`:
  ```python
  from smolagents import ActionStep
  self._agent.step_callbacks.register(ActionStep, self._reporter.step_callback)
  ```
