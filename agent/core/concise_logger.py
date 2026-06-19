"""
agent/core/concise_logger.py
────────────────────────────────────────────────────────────────────────────
ConciseStepReporter: Jupyter-friendly step display for the autonomous loop.

Full detail (Thought / code / execution traces) still flows to:
  - master_log/master_terminal.log
  - sessions/{id}/session_log.log

For the Jupyter cell, only prints:
  \r-overwriting line while a step is running   (elapsed + step# + exp_name)
  One final non-overwriting DONE line per step  (includes val_f1_macro if found)

Usage (wired automatically by AgentOrchestrator when verbosity="concise"):
    reporter = ConciseStepReporter(logger=tee_logger, run_start=time.time())
    # Pass reporter.step_callback to CodeAgent's step_callbacks list
    agent = CodeAgent(..., step_callbacks=[reporter.step_callback])
"""

import sys
import time
import threading
from typing import Optional, Any


class ConciseStepReporter:
    """
    Smolagents step callback that outputs a single overwriting line per step
    to Jupyter cell output, while suppressing the full Thought/code/exec dumps.

    Parameters
    ----------
    logger      : TeeLogger instance — for writing full detail to log files
    run_start   : float — time.time() at the start of the run
    refresh_hz  : float — how often (seconds) to refresh the elapsed counter
    """

    def __init__(self, logger, run_start: float, refresh_hz: float = 1.0) -> None:
        self.logger       = logger
        self.run_start    = run_start
        self.refresh_hz   = refresh_hz

        self._step_num    = 0
        self._step_start  = run_start
        self._current_exp = "init"
        self._done        = threading.Event()

        # Background thread: refreshes the \r line every second while running
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, daemon=True
        )
        self._refresh_thread.start()

    # ── Public API ────────────────────────────────────────────────────────────

    def step_callback(self, memory_step: Any) -> None:
        """
        Called by Smolagents after EVERY completed step.
        Prints the DONE line for this step (non-overwriting, \n at end).
        """
        step_time  = time.time() - self._step_start
        elapsed    = int(time.time() - self.run_start)
        mins, secs = divmod(elapsed, 60)

        # Extract experiment info from the step if available
        exp_info   = self._extract_exp_info(memory_step)
        if exp_info != self._current_exp and exp_info != "unknown":
            self._current_exp = exp_info

        done_line = (
            f"\r[Step {self._step_num:3d}] "
            f"{mins:02d}:{secs:02d} elapsed | "
            f"DONE | {self._current_exp} | "
            f"{step_time:.1f}s"
        )
        sys.stdout.write(f"{done_line:<88}\n")
        sys.stdout.flush()

        # Prepare counters for next step
        self._step_num   += 1
        self._step_start  = time.time()

    def update_exp(self, exp_name: str) -> None:
        """Call when the orchestrator knows which experiment is starting."""
        self._current_exp = exp_name

    def stop(self) -> None:
        """Signal the refresh thread to exit."""
        self._done.set()

    # ── Private ───────────────────────────────────────────────────────────────

    def _refresh_loop(self) -> None:
        """Daemon: continuously overwrites the current line with elapsed time."""
        while not self._done.wait(self.refresh_hz):
            elapsed    = int(time.time() - self.run_start)
            mins, secs = divmod(elapsed, 60)
            step_elapsed = int(time.time() - self._step_start)
            sm, ss = divmod(step_elapsed, 60)
            line = (
                f"[Step {self._step_num:3d}] "
                f"{mins:02d}:{secs:02d} elapsed | "
                f"{self._current_exp} | "
                f"running ... ({sm:02d}:{ss:02d} this step)"
            )
            sys.stdout.write(f"\r{line:<88}")
            sys.stdout.flush()

    def _extract_exp_info(self, memory_step: Any) -> str:
        """
        Try to pull experiment name and val_f1_macro from a Smolagents step.
        Returns a descriptive string, falling back to self._current_exp.
        """
        try:
            # ActionStep from smolagents has tool_calls + observations
            if hasattr(memory_step, "tool_calls") and memory_step.tool_calls:
                for call in memory_step.tool_calls:
                    name = getattr(call, "name", "") or ""
                    if name == "run_experiment":
                        args = getattr(call, "arguments", {}) or {}
                        if isinstance(args, dict):
                            return args.get("exp_name", self._current_exp)
                        if isinstance(args, str):
                            import json
                            try:
                                d = json.loads(args)
                                return d.get("exp_name", self._current_exp)
                            except Exception:
                                pass

            # Observations may contain val_f1_macro=X.XXXX
            obs = ""
            if hasattr(memory_step, "observations"):
                obs = str(memory_step.observations or "")
            elif hasattr(memory_step, "observation"):
                obs = str(memory_step.observation or "")

            if "val_f1_macro=" in obs:
                f1_part = obs.split("val_f1_macro=")[1].split()[0].rstrip("|, \n")
                return f"{self._current_exp} | f1={f1_part}"

        except Exception:
            pass

        return self._current_exp
