"""
agent/tools/experiment_runner.py
────────────────────────────────────────────────────────────────────────────
Smolagents Tool: end-to-end experiment orchestration.

Experiment ID scheme
────────────────────
  Format : exp_{NNN}_{machine_id}_{arch_name}
  Example: exp_003_gcloud_logistic_regression
           exp_003_laptop_logistic_regression

  NNN        — zero-padded counter from disk scan (works on fresh clone)
  machine_id — MACHINE_ID env var (.env) → platform.node() fallback
  arch_name  — sanitised exp_name, max 30 chars

LLM contract (NEW — simpler and more reliable)
──────────────────────────────────────────────
  The LLM provides ONLY model_code: a Python string with exactly one function:

      def build_and_train(X_train, y_train, X_val, y_val, class_weights):
          # Train the model. Return a fitted object with .predict().
          ...
          return fitted_model

  Everything else is handled by the FIXED harness (harness_template.py):
  - Loading train.pkl / val.pkl (never test.pkl)
  - Applying scaler + selector
  - Calling build_and_train()
  - Computing ALL metrics on val set
  - Writing results.json (always, even on failure)

  This eliminates the entire class of bug where the LLM defines functions
  without calling them, or skips writing results.json.

Workflow inside forward()
─────────────────────────
  1.  Resolve experiment ID (disk scan + MACHINE_ID)
  2.  Create experiments/exp_NNN_<machine>_<name>/ directory
  3.  Write model.py (LLM-provided) + train_eval.py (fixed harness)
  4.  Write config.yaml with hyperparams
  5.  Anti-cheat audit on model_code only
  6.  Execute train_eval.py in subprocess, stream output via TeeLogger
  7.  Read results.json written by harness (guaranteed to exist)
  8.  Update master_log/leaderboard.json (local cache — gitignored)
  9.  Return human-readable results string

Results schema (VAL ONLY — never test metrics)
──────────────────────────────────────────────
  {
    "exp_id":               "exp_003_gcloud_logistic_regression",
    "machine_id":           "gcloud",
    "timestamp":            "2024-01-01T12:00:00",
    "architecture":         "LogisticRegression",
    "architecture_family":  "linear",
    "hyperparams":          {...},
    "val_accuracy":         0.87,
    "val_f1_macro":         0.72,
    "val_f1_per_class":     {"0": 0.95, "1": 0.30, "2": 0.65, "3": 0.72},
    "val_confusion_matrix": [[...], ...],
    "train_time_seconds":   12.4,
    "model_params_count":   0,
    "status":               "success",
    "error_message":        null
  }
"""

import datetime
import json
import os
import platform
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

try:
    from smolagents import Tool
    _SMOLAGENTS_AVAILABLE = True
except ImportError:
    _SMOLAGENTS_AVAILABLE = False
    class Tool:  # type: ignore[no-redef]
        pass

from agent.core.tee_logger import TeeLogger
from agent.tools.audit_tool import AuditTool
from agent.tools.harness_template import HARNESS_CODE

_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
_LEADERBOARD_PATH = _PROJECT_ROOT / "master_log" / "leaderboard.json"
_EXPERIMENTS_DIR  = _PROJECT_ROOT / "experiments"
_SPLITS_DIR       = _PROJECT_ROOT / "data" / "splits"

_HARNESS_TIMEOUT = 3600   # 1 hour total (train + eval combined)


class ExperimentRunnerTool(Tool if _SMOLAGENTS_AVAILABLE else object):  # type: ignore[misc]
    """
    Run a complete ML experiment end-to-end and record results.

    The agent provides ONLY model_code — a Python string defining:

        def build_and_train(X_train, y_train, X_val, y_val, class_weights):
            ...
            return fitted_model

    The harness handles all file I/O, metric computation, and results.json.
    """

    name        = "run_experiment"
    description = textwrap.dedent("""\
        Run a complete ML experiment using a fixed harness.

        You must provide:
          exp_name            : short snake_case name (e.g. "logistic_regression_c1")
          model_code          : Python string defining EXACTLY ONE function:

              def build_and_train(X_train, y_train, X_val, y_val, class_weights):
                  \"\"\"
                  X_train, X_val : np.ndarray, already scaled + feature-selected
                  y_train, y_val : np.ndarray of int labels (0-3)
                  class_weights  : dict {0: w0, 1: w1, 2: w2, 3: w3}
                  Must return a fitted model with a .predict(X) method.
                  \"\"\"
                  # ... your code here ...
                  return fitted_model

          hyperparams         : JSON string of hyperparameter dict (for logging)

        DO NOT include train/eval loops, file I/O, or metric computation —
        the fixed harness handles all of that. NEVER load test.pkl.
        Returns a formatted string with val_f1_macro and full results.
    """)
    inputs = {
        "exp_name": {
            "type": "string",
            "description": "Short snake_case experiment name.",
        },
        "model_code": {
            "type": "string",
            "description": (
                "Complete Python source defining build_and_train(X_train, y_train, "
                "X_val, y_val, class_weights) -> fitted_model."
            ),
        },
        "hyperparams": {
            "type": "string",
            "description": "JSON string of hyperparameter dict (for logging only).",
        },
    }
    output_type = "string"

    def __init__(self) -> None:
        if _SMOLAGENTS_AVAILABLE:
            super().__init__()
        self.logger  = TeeLogger()
        self.auditor = AuditTool()

    # ── Main entry point ───────────────────────────────────────────────────────

    def forward(
        self,
        exp_name:            str,
        model_code:          str,
        hyperparams:         str,
    ) -> str:
        """Execute the experiment and return a formatted results string."""

        architecture_family = _infer_family(model_code)

        # ── 1. Resolve experiment ID ──────────────────────────────────────────
        machine_id = _get_machine_id()
        exp_id     = self._next_exp_id(machine_id, exp_name)
        exp_dir    = _EXPERIMENTS_DIR / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "artifacts").mkdir(exist_ok=True)

        self.logger.agent(
            f"[ExperimentRunner] Starting {exp_id}: {exp_name}"
            f" (family={architecture_family})"
        )
        self.logger.set_experiment_log(str(exp_dir / "run.log"))

        # ── 2. Parse hyperparams ──────────────────────────────────────────────
        try:
            hp_dict: Dict[str, Any] = json.loads(hyperparams)
        except (json.JSONDecodeError, TypeError):
            hp_dict = {"raw": str(hyperparams)}

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        # ── 3. Write model.py (LLM-provided) ─────────────────────────────────
        (exp_dir / "model.py").write_text(model_code, encoding="utf-8")

        # ── 4. Write train_eval.py (FIXED harness — never LLM-generated) ─────
        harness_src = HARNESS_CODE.format(
            exp_id              = exp_id,
            architecture        = exp_name,
            architecture_family = architecture_family,
            timestamp           = timestamp,
            hyperparams_json    = json.dumps(hp_dict),
        )
        (exp_dir / "train_eval.py").write_text(harness_src, encoding="utf-8")

        # ── 5. Write config.yaml ──────────────────────────────────────────────
        config = {
            "exp_id":               exp_id,
            "exp_name":             exp_name,
            "machine_id":           machine_id,
            "architecture_family":  architecture_family,
            "hyperparams":          hp_dict,
            "timestamp":            timestamp,
            "splits_dir":           str(_SPLITS_DIR.resolve()),
        }
        with open(exp_dir / "config.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"[ExperimentRunner] Files written to {exp_dir}")

        # ── 6. Anti-cheat audit on model_code only ────────────────────────────
        audit_result = self.auditor.forward(model_code)
        if not audit_result.startswith("PASS"):
            self.logger.error(
                f"[ExperimentRunner] AUDIT FAILED for {exp_id}:\n{audit_result}"
            )
            error_result = _build_error_result(
                exp_id, exp_name, architecture_family, machine_id,
                hp_dict, timestamp,
                f"AUDIT FAILED: {audit_result}",
            )
            self._update_leaderboard(error_result)
            return _format_result(error_result)

        self.logger.info(f"[ExperimentRunner] Audit PASS for {exp_id}")

        # ── 7. Run train_eval.py (fixed harness) ──────────────────────────────
        t_start = time.perf_counter()
        ok, output = self._run_subprocess(
            script   = exp_dir / "train_eval.py",
            exp_dir  = exp_dir,
            timeout  = _HARNESS_TIMEOUT,
        )
        elapsed = time.perf_counter() - t_start

        # ── 8. Read results.json — harness ALWAYS writes it ───────────────────
        results_path = exp_dir / "results.json"
        if not results_path.exists():
            # Should never happen — harness writes results.json even on failure
            self.logger.error(
                f"[ExperimentRunner] results.json missing after harness run for {exp_id}"
            )
            error_result = _build_error_result(
                exp_id, exp_name, architecture_family, machine_id,
                hp_dict, timestamp,
                f"results.json not written — harness exit_ok={ok}\n{output[-2000:]}",
            )
            error_result["train_time_seconds"] = round(elapsed, 2)
            self._update_leaderboard(error_result)
            return _format_result(error_result)

        with open(results_path, "r", encoding="utf-8") as fh:
            result: Dict[str, Any] = json.load(fh)

        # Ensure all required fields are present (harness should set them all)
        result.setdefault("exp_id",               exp_id)
        result.setdefault("machine_id",           machine_id)
        result.setdefault("timestamp",            timestamp)
        result.setdefault("architecture",         exp_name)
        result.setdefault("architecture_family",  architecture_family)
        result.setdefault("hyperparams",          hp_dict)
        result.setdefault("train_time_seconds",   round(elapsed, 2))
        result.setdefault("model_params_count",   0)
        result.setdefault("status",               "success")
        result.setdefault("error_message",        None)
        result.setdefault("val_accuracy",         0.0)
        result.setdefault("val_f1_macro",         0.0)
        result.setdefault("val_f1_per_class",     {})
        result.setdefault("val_confusion_matrix", [])

        # Re-write with any defaults filled in
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

        # ── 9. Update leaderboard ─────────────────────────────────────────────
        self._update_leaderboard(result)

        f1 = result.get("val_f1_macro", 0.0)
        self.logger.agent(
            f"[ExperimentRunner] {exp_id} COMPLETE"
            f" | val_f1_macro={f1:.4f}"
            f" | status={result.get('status')}"
            f" | time={elapsed:.1f}s"
        )
        # Log verbose version to file for human inspection, but return the
        # compact version to the agent to keep its context window small.
        self.logger.info(
            f"[ExperimentRunner] Full result:\n{_format_result_verbose(result)}"
        )
        return _format_result_compact(result)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _run_subprocess(
        self,
        script:  Path,
        exp_dir: Path,
        timeout: int,
    ) -> tuple[bool, str]:
        """
        Run *script* in a subprocess, streaming every output line through
        TeeLogger in real time.  Returns (success: bool, combined_output: str).
        """
        cmd = [sys.executable, str(script.resolve())]
        env = os.environ.copy()
        env["PYTHONPATH"]    = str(Path(".").resolve())
        env["EXP_DIR"]       = str(exp_dir.resolve())
        env["PROJECT_ROOT"]  = str(Path(".").resolve())

        self.logger.info(f"[ExperimentRunner] Running: {' '.join(cmd)}")

        output_lines: List[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(Path(".").resolve()),
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                output_lines.append(line)
                self.logger.info(f"  [harness] {line}")
            proc.wait(timeout=timeout)
            success = proc.returncode == 0
            if not success:
                self.logger.error(
                    f"[ExperimentRunner] harness exit code={proc.returncode}"
                )
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.logger.error(f"[ExperimentRunner] TIMEOUT after {timeout}s")
            output_lines.append(f"TIMEOUT after {timeout}s")
            success = False
        except Exception as exc:
            self.logger.error(f"[ExperimentRunner] Subprocess error: {exc}")
            output_lines.append(str(exc))
            success = False

        return success, "\n".join(output_lines)

    def _next_exp_id(self, machine_id: str, exp_name: str) -> str:
        """
        Return the next collision-safe experiment ID.
        Format: exp_{NNN}_{machine_id}_{arch_name}
        NNN is from disk scan — works on fresh clones with no leaderboard.json.
        """
        existing = []
        if _EXPERIMENTS_DIR.exists():
            existing = [
                d for d in _EXPERIMENTS_DIR.iterdir()
                if d.is_dir()
                and d.name.startswith("exp_")
                and f"_{machine_id}_" in d.name
            ]
        n = len(existing) + 1
        arch_slug = _sanitize(exp_name)[:30]
        return f"exp_{n:03d}_{machine_id}_{arch_slug}"

    def _update_leaderboard(self, result: Dict[str, Any]) -> None:
        """Append result to leaderboard.json and update summary fields."""
        _LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not _LEADERBOARD_PATH.exists():
            self.logger.info(
                "[ExperimentRunner] leaderboard.json not found — creating fresh leaderboard."
            )
            lb = {
                "experiments": [],
                "total_runs": 0,
                "best_val_f1_macro": 0.0,
                "best_experiment": None,
                "architectures_tried": [],
                "families_completed": [],
                "agent_model_used": "",
                "last_updated": "",
            }
        else:
            with open(_LEADERBOARD_PATH, "r", encoding="utf-8") as fh:
                lb = json.load(fh)


        summary = {
            "exp_id":               result.get("exp_id"),
            "machine_id":           result.get("machine_id", "unknown"),
            "architecture":         result.get("architecture"),
            "architecture_family":  result.get("architecture_family"),
            "val_f1_macro":         result.get("val_f1_macro", 0.0),
            "val_accuracy":         result.get("val_accuracy", 0.0),
            "val_f1_per_class":     result.get("val_f1_per_class", {}),
            "train_time_seconds":   result.get("train_time_seconds", 0.0),
            "model_params_count":   result.get("model_params_count", 0),
            "status":               result.get("status", "failed"),
            "timestamp":            result.get("timestamp", ""),
            "hyperparams":          result.get("hyperparams", {}),
        }
        lb.setdefault("experiments", []).append(summary)
        lb["total_runs"] = len(lb["experiments"])

        f1 = result.get("val_f1_macro", 0.0)
        if f1 > lb.get("best_val_f1_macro", 0.0) and result.get("status") == "success":
            lb["best_val_f1_macro"] = f1
            lb["best_experiment"]   = result.get("exp_id")

        arch = result.get("architecture", "")
        if arch and arch not in lb.get("architectures_tried", []):
            lb.setdefault("architectures_tried", []).append(arch)

        fam = result.get("architecture_family", "")
        if fam and fam not in lb.get("families_completed", []):
            lb.setdefault("families_completed", []).append(fam)

        lb["last_updated"] = datetime.datetime.now().isoformat(timespec="seconds")

        with open(_LEADERBOARD_PATH, "w", encoding="utf-8") as fh:
            json.dump(lb, fh, indent=2)

        self.logger.info(
            f"[ExperimentRunner] Leaderboard updated. "
            f"total_runs={lb['total_runs']}  "
            f"best_f1={lb.get('best_val_f1_macro', 0.0):.4f}"
        )


# ── Module-level helpers ───────────────────────────────────────────────────────

def _infer_family(model_code: str) -> str:
    code = model_code.lower()
    if any(k in code for k in ["transformer", "attention", "rope"]):
        return "attention"
    if any(k in code for k in ["lstm", "gru", "rnn", "conv1d"]):
        return "sequence"
    if any(k in code for k in ["embedding", "char", "token"]):
        return "embedding"
    if any(k in code for k in ["xgb", "lgbm", "lightgbm", 
                                 "catboost", "gradient"]):
        return "boosting"
    if any(k in code for k in ["torch", "nn.module", "neural", 
                                 "mlp", "dense"]):
        return "neural"
    if any(k in code for k in ["forest", "tree", "bagging", 
                                 "extra"]):
        return "tree_ensemble"
    if any(k in code for k in ["logistic", "svm", "svc", 
                                 "linear", "ridge"]):
        return "linear"
    return "other"

def _sanitize(name: str) -> str:
    """Convert to safe folder-name characters."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")


def _get_machine_id() -> str:
    """
    Return a short, stable machine identifier (max 16 chars).
    Priority: MACHINE_ID env var → platform.node() hostname.
    """
    raw = os.environ.get("MACHINE_ID", "").strip()
    if not raw:
        raw = platform.node()
    if not raw:
        raw = "unknown"
    return _sanitize(raw)[:16]


def _build_error_result(
    exp_id:              str,
    architecture:        str,
    architecture_family: str,
    machine_id:          str,
    hyperparams:         Dict,
    timestamp:           str,
    error_message:       str,
) -> Dict[str, Any]:
    return {
        "exp_id":               exp_id,
        "machine_id":           machine_id,
        "timestamp":            timestamp,
        "architecture":         architecture,
        "architecture_family":  architecture_family,
        "hyperparams":          hyperparams,
        "val_accuracy":         0.0,
        "val_f1_macro":         0.0,
        "val_f1_per_class":     {},
        "val_confusion_matrix": [],
        "train_time_seconds":   0.0,
        "model_params_count":   0,
        "status":               "failed",
        "error_message":        error_message,
    }


def _format_result_compact(result: Dict[str, Any]) -> str:
    """
    Compact one-liner returned TO THE AGENT (~80 tokens).
    Keeps context window small — no ASCII bars, no decorations.
    Format:
      exp_NNN | arch_name | family | val_f1_macro=X.XXXX | No=X.XXX Low=X.XXX Med=X.XXX High=X.XXX | Xs
    On failure:
      exp_NNN | arch_name | FAILED | <first 200 chars of error>
    """
    exp_id  = result.get("exp_id", "?")
    arch    = result.get("architecture", "?")
    family  = result.get("architecture_family", "?")
    status  = result.get("status", "?")

    if status == "success":
        f1     = result.get("val_f1_macro", 0.0)
        t      = result.get("train_time_seconds", 0.0)
        pc     = result.get("val_f1_per_class", {})
        label_map = {"0": "No", "1": "Low", "2": "Med", "3": "High",
                     0: "No",   1: "Low",   2: "Med",   3: "High"}
        per_cls = " ".join(
            f"{label_map.get(k, str(k))}={float(v):.3f}"
            for k, v in sorted(pc.items(), key=lambda x: str(x[0]))
        )
        return (
            f"{exp_id} | {arch} | {family} | "
            f"val_f1_macro={f1:.4f} | {per_cls} | {t:.1f}s"
        )
    else:
        err = (result.get("error_message") or "unknown error")[:200]
        return f"{exp_id} | {arch} | FAILED | {err}"


def _format_result_verbose(result: Dict[str, Any]) -> str:
    """
    Verbose human-readable block written to log files only (not sent to agent).
    Keeps the original ASCII bar chart format for easy human inspection.
    """
    lines = [
        "",
        "=" * 62,
        f"  EXPERIMENT : {result.get('exp_id')}",
        f"  Arch       : {result.get('architecture')}",
        f"  Family     : {result.get('architecture_family')}",
        f"  Machine    : {result.get('machine_id', '?')}",
        f"  Status     : {result.get('status')}",
        "=" * 62,
    ]
    if result.get("status") == "success":
        lines += [
            f"  val_f1_macro : {result.get('val_f1_macro', 0.0):.4f}",
            f"  val_accuracy : {result.get('val_accuracy', 0.0):.4f}",
            f"  train_time   : {result.get('train_time_seconds', 0.0):.1f}s",
            f"  model_params : {result.get('model_params_count', 0):,}",
            "",
            "  Per-class F1:",
        ]
        label_names = {
            "0": "No", "1": "Low", "2": "Medium", "3": "High",
            0: "No",   1: "Low",   2: "Medium",   3: "High",
        }
        for cls, f1 in sorted(
            result.get("val_f1_per_class", {}).items(),
            key=lambda x: str(x[0])
        ):
            bar = "#" * int(f1 * 20)
            lines.append(
                f"    {cls} ({label_names.get(cls, '?'):<6}): "
                f"{f1:.4f}  |{bar:<20}|"
            )
    else:
        err = result.get("error_message", "unknown error")
        lines.append(f"  ERROR: {err[:500]}")
    lines.append("=" * 62)
    return "\n".join(lines)


# keep old name as alias so any external code importing _format_result still works
_format_result = _format_result_verbose
