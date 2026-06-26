"""
agent/tools/experiment_runner.py
────────────────────────────────────────────────────────────────────────────
ExperimentRunnerTool — the agent's primary action tool.

Runs agent-generated model code in a secure sandboxed subprocess, evaluates
the trained model via the EvaluationOracle, persists results, and optionally
writes a vector-DB memory entry for semantic recall.

Contract (what the LLM must implement in model_code)
─────────────────────────────────────────────────────
    def build_and_train(df_train, df_val, class_weights):
        \"\"\"
        df_train, df_val : pandas DataFrames with columns:
            ['sequence', 'concentration', 'label_int', 'label_str']
        class_weights    : dict {0: w0, 1: w1, 2: w2, 3: w3}

        Returns a fitted model object that has a .predict(df) -> np.ndarray
        method.  All imports must be inside this function.
        \"\"\"
        ...
        return fitted_model
"""

import os
import json
import uuid
import datetime
import re
from pathlib import Path
from typing import Optional

try:
    from smolagents import Tool as _SmolTool
    _HAS_SMOLAGENTS = True
except ImportError:
    _HAS_SMOLAGENTS = False
    class _SmolTool:          # minimal shim so the class body still works
        def __init__(self): pass

from agent.core.tee_logger import TeeLogger
from agent.tools.audit_tool import AuditTool
from agent.core.sandbox import SandboxedExecutor
from agent.core.oracle import EvaluationOracle

# ── Module-level singletons (instantiated once at import time) ────────────────
_PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent.parent)
).resolve()
_sandbox = SandboxedExecutor(_PROJECT_ROOT)
_oracle  = EvaluationOracle(_PROJECT_ROOT)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:40]

def _get_machine_id() -> str:
    return os.environ.get("MACHINE_ID", "unknown")

def _next_exp_id(exp_name: str) -> str:
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    uid  = uuid.uuid4().hex[:6]
    safe = _sanitize(exp_name)
    mid  = _get_machine_id()
    return f"exp_{ts}_{mid}_{safe}_{uid}"

def _format_result(result: dict) -> str:
    f1  = result.get("val_f1_macro", 0.0)
    acc = result.get("val_accuracy",  0.0)
    mcc = result.get("val_mcc",       0.0)
    f1c = result.get("val_f1_per_class", {})
    lines = [
        "EXPERIMENT SUCCESS",
        f"Exp ID    : {result.get('exp_id')}",
        f"F1 Macro  : {f1:.4f}  |  MCC: {mcc:.4f}  |  Accuracy: {acc:.4f}",
    ]
    if f1c:
        pc_str = "  ".join(f"class{k}={v:.3f}" for k, v in sorted(f1c.items()))
        lines.append(f"Per-class : {pc_str}")
    lines.append(f"Train time: {result.get('train_time_s', 0):.1f}s")
    return "\n".join(lines)

def _build_error_result(
    exp_id: str, exp_name: str, machine_id: str, hp: dict,
    timestamp: str, error_message: str
) -> dict:
    return {
        "exp_id":      exp_id,
        "architecture": exp_name,
        "hypothesis":   "",
        "timestamp":    timestamp,
        "machine_id":   machine_id,
        "hyperparams":  hp,
        "train_time_s": 0.0,
        "val_f1_macro": 0.0,
        "val_mcc":      0.0,
        "val_accuracy": 0.0,
        "val_f1_per_class": {},
        "val_confusion_matrix": [],
        "model_params": 0,
        "status":       "failed",
        "error_message": error_message,
    }


# ── The Tool ──────────────────────────────────────────────────────────────────

class ExperimentRunnerTool(_SmolTool):
    """
    Smolagents Tool that sandboxes model training and evaluates via Oracle.
    Compatible with AgentOrchestrator and smolagents CodeAgent.
    """

    name = "run_experiment"
    description = (
        "Run a machine learning experiment on the alpha-synuclein dataset.\n"
        "Provide model_code that defines:\n"
        "    def build_and_train(df_train, df_val, class_weights):\n"
        "df_train and df_val are DataFrames with columns:\n"
        "    ['sequence', 'concentration', 'label_int', 'label_str']\n"
        "class_weights is a dict {0: w0, 1: w1, 2: w2, 3: w3}.\n"
        "Returns a fitted model with .predict(df) -> np.ndarray.\n"
        "All imports must be inside build_and_train. Never load files.\n"
        "The harness evaluates on the val set and writes results.json."
    )
    inputs = {
        "exp_name": {
            "type":        "string",
            "description": "Short descriptive name (e.g. 'xgboost_focal_smote')",
        },
        "model_code": {
            "type":        "string",
            "description": "Python code defining build_and_train(df_train, df_val, class_weights)",
        },
        "hyperparams": {
            "type":        "string",
            "description": "JSON of key hyperparams for logging (e.g. '{\"lr\": 0.01}'). Use '{}' if none.",
            "nullable":    True,
        },
    }
    output_type = "string"

    def __init__(self) -> None:
        if _HAS_SMOLAGENTS:
            super().__init__()
        self._logger  = TeeLogger(master_log_dir="master_log")
        self._auditor = AuditTool()

    def forward(self, exp_name: str, model_code: str, hyperparams: str = "{}") -> str:
        # ── 1. Audit ─────────────────────────────────────────────────────────
        audit_result = self._auditor.forward(model_code)
        if not audit_result.startswith("PASS"):
            return f"AUDIT FAILED — fix your code before running:\n{audit_result}"

        # ── 2. Setup ──────────────────────────────────────────────────────────
        exp_id    = _next_exp_id(exp_name)
        exp_dir   = _PROJECT_ROOT / "experiments" / exp_id
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        machine   = _get_machine_id()

        try:
            hp = json.loads(hyperparams)
        except Exception:
            hp = {}

        self._logger.agent(f"[ExperimentRunner] Starting {exp_id}")

        # ── 3. Sandboxed training ─────────────────────────────────────────────
        train_result = _sandbox.execute_training(
            exp_dir, model_code, timeout_seconds=600
        )

        if train_result.get("status") != "success":
            err = train_result.get("error", "unknown error")
            self._logger.warning(
                f"[ExperimentRunner] FAILED training {exp_id}: {err[:200]}"
            )
            err_result = _build_error_result(exp_id, exp_name, machine, hp, timestamp, err)
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "results.json").write_text(
                json.dumps(err_result, indent=2), encoding="utf-8"
            )
            return f"EXPERIMENT FAILED (training)\nExp: {exp_id}\nError:\n{err}"

        # ── 4. Oracle evaluation ──────────────────────────────────────────────
        model_pkl   = exp_dir / "artifacts" / "model.pkl"
        eval_result = _oracle.evaluate_model(model_pkl)

        if eval_result.get("status") != "success":
            err = eval_result.get("error_message", "unknown error")
            self._logger.warning(
                f"[ExperimentRunner] FAILED eval {exp_id}: {err}"
            )
            err_result = _build_error_result(exp_id, exp_name, machine, hp, timestamp, err)
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "results.json").write_text(
                json.dumps(err_result, indent=2), encoding="utf-8"
            )
            return f"EXPERIMENT FAILED (evaluation)\nExp: {exp_id}\nError:\n{err}"

        # ── 5. Save results.json ──────────────────────────────────────────────
        result = {
            "exp_id":               exp_id,
            "architecture":         exp_name,
            "hypothesis":           f"Experiment: {exp_name}",
            "timestamp":            timestamp,
            "machine_id":           machine,
            "hyperparams":          hp,
            "train_time_s":         train_result.get("train_time", 0.0),
            "val_f1_macro":         eval_result.get("val_f1_macro",   0.0),
            "val_mcc":              eval_result.get("val_mcc",         0.0),
            "val_accuracy":         eval_result.get("val_accuracy",    0.0),
            "val_f1_per_class":     eval_result.get("val_f1_per_class",   {}),
            "val_confusion_matrix": eval_result.get("val_confusion_matrix", []),
            "model_params":         eval_result.get("model_params_count", 0),
            "status":               "success",
            "error_message":        None,
        }
        (exp_dir / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        self._logger.agent(
            f"[ExperimentRunner] SUCCESS {exp_id}  "
            f"f1={result['val_f1_macro']:.4f}  mcc={result['val_mcc']:.4f}"
        )

        # ── 6. Write to persistent memory (best-effort) ───────────────────────
        try:
            from agent.core.memory import PersistentMemory
            mem = PersistentMemory(_PROJECT_ROOT)
            mem.add_experiment(
                exp_id       = exp_id,
                hypothesis   = result["hypothesis"],
                architecture = exp_name,
                f1_macro     = result["val_f1_macro"],
                mcc          = result["val_mcc"],
            )
        except Exception as _mem_exc:
            self._logger.info(
                f"[ExperimentRunner] ChromaDB memory write skipped: {_mem_exc}"
            )

        return _format_result(result)

    # ── Legacy compat shims (called by orchestrator wrappers) ─────────────────
    def _run_subprocess(self, *args, **kwargs):
        pass

    def _next_exp_id(self, machine_id: str, exp_name: str) -> str:
        return _next_exp_id(exp_name)

    def _update_leaderboard(self, result: dict) -> None:
        """No-op: rebuild_leaderboard reads from experiments/ directory."""
        pass
