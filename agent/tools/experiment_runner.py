"""
agent/tools/experiment_runner.py
────────────────────────────────────────────────────────────────────────────
Smolagents Tool: end-to-end experiment orchestration.

Experiment ID scheme
────────────────────
  Format : exp_{NNN}_{machine_id}_{arch_name}
  Example: exp_003_gcloud_random_forest_baseline
           exp_003_laptop_random_forest_baseline

  NNN        — zero-padded counter, derived by scanning experiments/ on disk.
               Scanning disk (not leaderboard.json) ensures it works even
               when leaderboard.json is gitignored / absent on a fresh clone.
  machine_id — read from MACHINE_ID env var in .env
               (e.g. MACHINE_ID=laptop  or  MACHINE_ID=gcloud)
               Falls back to platform.node() if not set — never empty.
  arch_name  — sanitised exp_name, max 30 chars.

  Because machine_id is part of the folder name, exp_003_laptop and
  exp_003_gcloud are guaranteed to be DIFFERENT folders.  Git can track both
  without conflict.  Two machines running simultaneously will never silently
  overwrite each other's experiments.

Workflow inside forward()
─────────────────────────
  1.  Resolve experiment ID (disk scan + MACHINE_ID)
  2.  Create experiments/exp_NNN_<machine>_<name>/ directory
  3.  Write model.py, train.py, eval.py, config.yaml
  4.  Anti-cheat audit (AuditTool) — abort if FAIL
  5.  Execute train.py in subprocess, stream all output via TeeLogger
  6.  Execute eval.py in subprocess, stream output
  7.  Read results.json written by eval.py
  8.  Update master_log/leaderboard.json (local cache — gitignored)
  9.  Return human-readable results string

Results schema (VAL ONLY — never test metrics)
──────────────────────────────────────────────
  {
    "exp_id":               "exp_003_gcloud_random_forest_baseline",
    "machine_id":           "gcloud",
    "timestamp":            "2024-01-01T12:00:00",
    "architecture":         "RandomForestClassifier",
    "architecture_family":  "classical_ml",
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

import json
import os
import platform
import re
import subprocess
import sys
import time
import textwrap
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # populate MACHINE_ID from .env if present
except ImportError:
    pass

import yaml

try:
    from smolagents import Tool
    _SMOLAGENTS_AVAILABLE = True
except ImportError:
    _SMOLAGENTS_AVAILABLE = False
    class Tool:  # type: ignore[no-redef]
        pass

from agent.core.tee_logger import TeeLogger
from agent.tools.audit_tool import AuditTool

_LEADERBOARD_PATH = Path("master_log/leaderboard.json")
_EXPERIMENTS_DIR  = Path("experiments")
_SPLITS_DIR       = Path("data/splits")

# Timeout for each subprocess (seconds)
_TRAIN_TIMEOUT = 3600   # 1 hour
_EVAL_TIMEOUT  = 300    # 5 minutes


class ExperimentRunnerTool(Tool if _SMOLAGENTS_AVAILABLE else object):  # type: ignore[misc]
    """
    Run a complete ML experiment end-to-end and record results.

    The agent provides the full source code for three files:
      • model.py  — architecture definition
      • train.py  — training loop; must save model to artifacts/
      • eval.py   — evaluation on val set; must write results.json

    The tool handles everything else:
    folder creation, audit, execution, logging, leaderboard update.
    """

    name        = "run_experiment"
    description = textwrap.dedent("""\
        Run a complete ML experiment.

        You must provide:
          exp_name            : short snake_case name (e.g. "random_forest_baseline")
          architecture_family : one of classical_ml / linear / neural_network /
                                deep_residual / ensemble_stack / attention_based /
                                graph_neural / automl
          model_code          : full contents of model.py
          train_code          : full contents of train.py
                                - Must load train split from data/splits/train.pkl
                                - Must load val split from data/splits/val.pkl
                                - Must save model to artifacts/model.*
                                - NEVER load test.pkl
          eval_code           : full contents of eval.py
                                - Must evaluate on VAL ONLY (not test)
                                - Must write results.json to experiment folder
          hyperparams         : JSON string of hyperparameter dict

        Returns a formatted string with val_f1_macro and full results.
        NEVER include test set in any code.
    """)
    inputs = {
        "exp_name": {
            "type": "string",
            "description": "Short snake_case experiment name.",
        },
        "architecture_family": {
            "type": "string",
            "description": "Architecture family (e.g. classical_ml, neural_network).",
        },
        "model_code": {
            "type": "string",
            "description": "Complete Python source for model.py.",
        },
        "train_code": {
            "type": "string",
            "description": "Complete Python source for train.py.",
        },
        "eval_code": {
            "type": "string",
            "description": "Complete Python source for eval.py.",
        },
        "hyperparams": {
            "type": "string",
            "description": "JSON string of hyperparameter dict.",
        },
    }
    output_type = "string"

    def __init__(self) -> None:
        if _SMOLAGENTS_AVAILABLE:
            super().__init__()
        self.logger = TeeLogger()
        self.auditor = AuditTool()

    # ── Main entry point ───────────────────────────────────────────────────────

    def forward(
        self,
        exp_name: str,
        architecture_family: str,
        model_code: str,
        train_code: str,
        eval_code: str,
        hyperparams: str,
    ) -> str:
        """
        Execute the experiment and return a formatted results string.
        """
        # ── 1. Resolve experiment ID (collision-safe, cross-machine) ─────────
        machine_id  = _get_machine_id()
        exp_id      = self._next_exp_id(machine_id, exp_name)
        exp_dir     = _EXPERIMENTS_DIR / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = exp_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

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

        # ── 3. Write source files ─────────────────────────────────────────────
        (exp_dir / "model.py").write_text(model_code, encoding="utf-8")
        (exp_dir / "train.py").write_text(train_code, encoding="utf-8")
        (exp_dir / "eval.py" ).write_text(eval_code,  encoding="utf-8")

        config = {
            "exp_id":               exp_id,
            "exp_name":             exp_name,
            "machine_id":           machine_id,
            "architecture_family":  architecture_family,
            "hyperparams":          hp_dict,
            "timestamp":            datetime.datetime.now().isoformat(timespec="seconds"),
            "splits_dir":           str(_SPLITS_DIR.resolve()),
            "artifacts_dir":        str(artifacts_dir.resolve()),
        }
        with open(exp_dir / "config.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(config, fh, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"[ExperimentRunner] Files written to {exp_dir}")

        # ── 4. Anti-cheat audit ───────────────────────────────────────────────
        combined_code = f"# model.py\n{model_code}\n# train.py\n{train_code}\n# eval.py\n{eval_code}"
        audit_result = self.auditor.forward(combined_code)
        if not audit_result.startswith("PASS"):
            self.logger.error(
                f"[ExperimentRunner] AUDIT FAILED for {exp_id}:\n{audit_result}"
            )
            error_result = self._build_error_result(
                exp_id, exp_name, architecture_family, hp_dict,
                f"AUDIT FAILED: {audit_result}"
            )
            self._update_leaderboard(error_result)
            return self._format_result(error_result)

        self.logger.info(f"[ExperimentRunner] Audit PASS for {exp_id}")

        # ── 5. Run train.py ───────────────────────────────────────────────────
        train_start = time.perf_counter()
        train_ok, train_output = self._run_subprocess(
            script=exp_dir / "train.py",
            exp_dir=exp_dir,
            label="TRAIN",
            timeout=_TRAIN_TIMEOUT,
        )
        train_elapsed = time.perf_counter() - train_start

        if not train_ok:
            self.logger.error(
                f"[ExperimentRunner] train.py FAILED for {exp_id}"
            )
            error_result = self._build_error_result(
                exp_id, exp_name, architecture_family, hp_dict,
                f"train.py failed:\n{train_output[-2000:]}"
            )
            error_result["train_time_seconds"] = round(train_elapsed, 2)
            self._update_leaderboard(error_result)
            return self._format_result(error_result)

        # ── 6. Run eval.py ────────────────────────────────────────────────────
        eval_ok, eval_output = self._run_subprocess(
            script=exp_dir / "eval.py",
            exp_dir=exp_dir,
            label="EVAL",
            timeout=_EVAL_TIMEOUT,
        )

        if not eval_ok:
            self.logger.error(
                f"[ExperimentRunner] eval.py FAILED for {exp_id}"
            )
            error_result = self._build_error_result(
                exp_id, exp_name, architecture_family, hp_dict,
                f"eval.py failed:\n{eval_output[-2000:]}"
            )
            error_result["train_time_seconds"] = round(train_elapsed, 2)
            self._update_leaderboard(error_result)
            return self._format_result(error_result)

        # ── 7. Read results.json ──────────────────────────────────────────────
        results_path = exp_dir / "results.json"
        if not results_path.exists():
            error_result = self._build_error_result(
                exp_id, exp_name, architecture_family, hp_dict,
                "eval.py ran successfully but did not write results.json"
            )
            error_result["train_time_seconds"] = round(train_elapsed, 2)
            self._update_leaderboard(error_result)
            return self._format_result(error_result)

        with open(results_path, "r", encoding="utf-8") as fh:
            result: Dict[str, Any] = json.load(fh)

        # Ensure required fields are present
        result.setdefault("exp_id",               exp_id)
        result.setdefault("machine_id",           machine_id)
        result.setdefault("timestamp",            config["timestamp"])
        result.setdefault("architecture",         exp_name)
        result.setdefault("architecture_family",  architecture_family)
        result.setdefault("hyperparams",          hp_dict)
        result.setdefault("train_time_seconds",   round(train_elapsed, 2))
        result.setdefault("model_params_count",   0)
        result.setdefault("status",               "success")
        result.setdefault("error_message",        None)
        result.setdefault("val_accuracy",         0.0)
        result.setdefault("val_f1_macro",         0.0)
        result.setdefault("val_f1_per_class",     {})
        result.setdefault("val_confusion_matrix", [])

        # Overwrite results.json with the enriched version
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)

        # ── 8. Update leaderboard ─────────────────────────────────────────────
        self._update_leaderboard(result)

        f1 = result.get("val_f1_macro", 0.0)
        self.logger.agent(
            f"[ExperimentRunner] {exp_id} COMPLETE"
            f" | val_f1_macro={f1:.4f}"
            f" | train_time={train_elapsed:.1f}s"
        )

        return self._format_result(result)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _run_subprocess(
        self,
        script: Path,
        exp_dir: Path,
        label: str,
        timeout: int,
    ) -> tuple[bool, str]:
        """
        Run *script* in a subprocess, streaming every output line through
        TeeLogger in real time.

        Returns (success: bool, combined_output: str)
        """
        cmd = [sys.executable, str(script.resolve())]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(".").resolve())
        env["EXP_DIR"]    = str(exp_dir.resolve())

        self.logger.info(
            f"[ExperimentRunner] [{label}] Running: {' '.join(cmd)}"
        )

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

            # Stream line by line
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                output_lines.append(line)
                self.logger.info(f"  [{label}] {line}")

            proc.wait(timeout=timeout)
            success = proc.returncode == 0

            if not success:
                self.logger.error(
                    f"[ExperimentRunner] [{label}] exit code={proc.returncode}"
                )
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.logger.error(
                f"[ExperimentRunner] [{label}] TIMEOUT after {timeout}s"
            )
            output_lines.append(f"TIMEOUT after {timeout}s")
            success = False
        except Exception as exc:
            self.logger.error(
                f"[ExperimentRunner] [{label}] Subprocess error: {exc}"
            )
            output_lines.append(str(exc))
            success = False

        return success, "\n".join(output_lines)

    def _next_exp_id(self, machine_id: str, exp_name: str) -> str:
        """
        Return the next collision-safe experiment ID.

        Format: exp_{NNN}_{machine_id}_{arch_name}

        NNN is derived by scanning experiments/ on disk — this works
        on a fresh clone where leaderboard.json does not exist yet.
        """
        # Count existing folders on disk that belong to THIS machine
        # (so NNN is per-machine sequential, always incrementing)
        prefix = f"exp_"
        existing = [
            d for d in _EXPERIMENTS_DIR.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
               and f"_{machine_id}_" in d.name
        ] if _EXPERIMENTS_DIR.exists() else []
        n = len(existing) + 1
        arch_slug = _sanitize(exp_name)[:30]
        return f"exp_{n:03d}_{machine_id}_{arch_slug}"

    def _update_leaderboard(self, result: Dict[str, Any]) -> None:
        """Append result to leaderboard.json and update summary fields."""
        if not _LEADERBOARD_PATH.exists():
            self.logger.warning(
                "[ExperimentRunner] leaderboard.json not found; skipping update."
            )
            return

        with open(_LEADERBOARD_PATH, "r", encoding="utf-8") as fh:
            lb = json.load(fh)

        # Append experiment (store a compact summary, not the full result)
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

        # Update totals
        lb["total_runs"] = len(lb["experiments"])

        # Update best
        f1 = result.get("val_f1_macro", 0.0)
        if f1 > lb.get("best_val_f1_macro", 0.0) and result.get("status") == "success":
            lb["best_val_f1_macro"] = f1
            lb["best_experiment"]   = result.get("exp_id")

        # Track architectures and families
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
            f"best_f1={lb['best_val_f1_macro']:.4f}"
        )

    @staticmethod
    def _build_error_result(
        exp_id: str,
        architecture: str,
        architecture_family: str,
        hyperparams: Dict,
        error_message: str,
    ) -> Dict[str, Any]:
        return {
            "exp_id":               exp_id,
            "timestamp":            datetime.datetime.now().isoformat(timespec="seconds"),
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

    @staticmethod
    def _format_result(result: Dict[str, Any]) -> str:
        """Return a readable summary string of the experiment result."""
        lines = [
            "",
            "=" * 60,
            f"  EXPERIMENT: {result.get('exp_id')}  |  {result.get('architecture')}",
            f"  Family    : {result.get('architecture_family')}",
            f"  Status    : {result.get('status')}",
            "=" * 60,
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
            label_names = {0: "No", 1: "Low", 2: "Medium", 3: "High",
                           "0": "No", "1": "Low", "2": "Medium", "3": "High"}
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

        lines.append("=" * 60)
        return "\n".join(lines)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    """Convert to safe folder name characters."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")


def _get_machine_id() -> str:
    """
    Return a short, stable machine identifier.

    Priority:
      1. MACHINE_ID env var (set in .env, e.g. MACHINE_ID=gcloud)
      2. platform.node() hostname (always available, never empty)

    The value is sanitised to [a-zA-Z0-9_-] and capped at 16 chars
    so folder names stay readable.
    """
    raw = os.environ.get("MACHINE_ID", "").strip()
    if not raw:
        raw = platform.node()  # hostname — e.g. 'quantkit', 'DESKTOP-XYZ'
    if not raw:
        raw = "unknown"
    return _sanitize(raw)[:16]
