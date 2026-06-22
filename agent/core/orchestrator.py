"""
agent/core/orchestrator.py
────────────────────────────────────────────────────────────────────────────
AgentOrchestrator — top-level controller that wires LLMManager, all tools,
and the Smolagents CodeAgent into a single runnable unit.

Usage
-----
    from agent.core.orchestrator import AgentOrchestrator

    agent = AgentOrchestrator(model_name="groq-llama", verbosity="concise")
    agent.run(max_experiments=200)

    # Graceful stop (e.g. from a Jupyter button)
    agent.stop()
"""

import json
import threading
import time
import datetime
from pathlib import Path
from typing import Optional

from agent.core.tee_logger import TeeLogger
from agent.core.llm_manager import LLMManager
from agent.core.session_manager import SessionManager
from agent.core.concise_logger import ConciseStepReporter
from agent.core.watchdog import RunWatchdog
from agent.prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_SHORT
from agent.tools.experiment_runner import ExperimentRunnerTool
from agent.tools.leaderboard_tool import LeaderboardTool
from agent.tools.audit_tool import AuditTool
from agent.tools.arxiv_tool import ArxivTool

_LEADERBOARD_PATH = Path("master_log/leaderboard.json")
_STATE_PATH       = Path("master_log/orchestrator_state.json")

_SHORT_PROMPT_MODELS = {"groq-mixtral", "mistral-small"}

_AUTHORIZED_IMPORTS = [
    "numpy",
    "numpy.*",
    "pandas",
    "pandas.*",
    "scipy",
    "scipy.*",
    "sklearn",
    "sklearn.*",
    "xgboost",
    "xgboost.*",
    "lightgbm",
    "lightgbm.*",
    "imblearn",
    "imblearn.*",
    "torch",
    "torch.*",
    "matplotlib",
    "matplotlib.*",
    "seaborn",
    "seaborn.*",
    "json",
    "os",
    "sys",
    "pathlib",
    "pickle",
    "yaml",
    "joblib",
]


class AgentOrchestrator:
    """
    Wires all components into a single autonomous research loop.

    Parameters
    ----------
    model_name       : str   — LLM key from llm_manager.MODELS
    use_short_prompt : bool  — force the short system prompt
    max_steps        : int   — max Smolagents inner steps per run()
    verbosity        : str   — "concise" (default) | "full"
                               concise = single overwriting line per step in
                               Jupyter; full = all Smolagents output
    max_idle_seconds  : int  — watchdog: stop if no progress for this long
    max_total_seconds : int  — watchdog: stop after this many total seconds
    """

    def __init__(
        self,
        model_name:         str  = "local-qwen",
        use_short_prompt:   bool = False,
        max_steps:          int  = 500,
        verbosity:          str  = "concise",
        max_idle_seconds:   int  = 300,
        max_total_seconds:  int  = 3600,
    ) -> None:
        self.logger = TeeLogger(master_log_dir="master_log")
        self.logger.agent(
            f"[Orchestrator] Initialising  model={model_name}  "
            f"verbosity={verbosity}"
        )

        self.verbosity          = verbosity.lower()
        self.max_idle_seconds   = max_idle_seconds
        self.max_total_seconds  = max_total_seconds

        # ── LLM ──────────────────────────────────────────────────────────────
        self.llm = LLMManager(model_name=model_name)

        # ── Tools ─────────────────────────────────────────────────────────────
        _arxiv = ArxivTool(llm_model=self.llm.get_model())
        self.tools = [
            ExperimentRunnerTool(),
            LeaderboardTool(),
            AuditTool(),
            _arxiv,
        ]

        try:
            from smolagents import DuckDuckGoSearchTool
            self.tools.append(DuckDuckGoSearchTool())
            self.logger.info("[Orchestrator] DuckDuckGoSearchTool added.")
        except (ImportError, Exception) as exc:
            self.logger.warning(
                f"[Orchestrator] DuckDuckGoSearchTool unavailable: {exc}"
            )

        # ── Smolagents CodeAgent ──────────────────────────────────────────────
        # verbosity_level: 0=silent, 1=minimal, 2=verbose
        # In concise mode we suppress smolagents' built-in output (level 0)
        # and drive display ourselves via ConciseStepReporter.
        _vlevel = 0 if self.verbosity == "concise" else 2
        try:
            from smolagents import CodeAgent
            self._agent = CodeAgent(
                tools                         = self.tools,
                model                         = self.llm.get_model(),
                max_steps                     = max_steps,
                verbosity_level               = _vlevel,
                additional_authorized_imports = _AUTHORIZED_IMPORTS,
            )
            self.logger.info(
                f"[Orchestrator] CodeAgent ready. "
                f"max_steps={max_steps}  verbosity_level={_vlevel}"
            )
        except Exception as exc:
            self.logger.error(
                f"[Orchestrator] CodeAgent init failed: {exc}. "
                "Install smolagents[all] and retry."
            )
            self._agent = None

        # ── Prompt selection ──────────────────────────────────────────────────
        if use_short_prompt or model_name in _SHORT_PROMPT_MODELS:
            self._prompt = SYSTEM_PROMPT_SHORT
            self.logger.info("[Orchestrator] Using SHORT system prompt.")
        else:
            self._prompt = SYSTEM_PROMPT

        self.model_name   = model_name
        self._stop_event  = threading.Event()
        self._running     = False
        self._run_start:  Optional[float] = None
        self._session:    Optional[SessionManager] = None
        self._watchdog:   Optional[RunWatchdog] = None
        self._reporter:   Optional[ConciseStepReporter] = None
        self._exp_count   = 0

        self.logger.agent("[Orchestrator] Ready. Call run() to start.")

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        max_experiments:   int = 200,
        max_idle_seconds:  Optional[int] = None,
        max_total_seconds: Optional[int] = None,
    ) -> Optional[str]:
        """
        Start the autonomous research loop.

        Parameters
        ----------
        max_experiments  : soft cap injected into the prompt
        max_idle_seconds : override watchdog idle limit for this run
        max_total_seconds: override watchdog total limit for this run
        """
        if self._agent is None:
            self.logger.error("[Orchestrator] Agent not initialised. Cannot run.")
            return None

        idle_limit  = max_idle_seconds  or self.max_idle_seconds
        total_limit = max_total_seconds or self.max_total_seconds

        self._stop_event.clear()
        self._running   = True
        self._run_start = time.time()
        self._exp_count = 0

        # ── Session ───────────────────────────────────────────────────────────
        self._session = SessionManager(
            model_name = self.model_name,
            logger     = self.logger,
        )
        self._session.start()
        self._session.tick(current_exp="none", step=0, status="running")

        # ── Concise reporter ──────────────────────────────────────────────────
        if self.verbosity == "concise":
            self._reporter = ConciseStepReporter(
                logger    = self.logger,
                run_start = self._run_start,
            )
            # Register the concise callback on the existing CallbackRegistry.
            # smolagents 1.26.0+ stores step_callbacks as a CallbackRegistry
            # instance — never replace it with a plain list or it breaks
            # the internal `self.step_callbacks.callback(...)` call in
            # _finalize_step().
            from smolagents.memory import ActionStep
            self._agent.step_callbacks.register(
                ActionStep, self._reporter.step_callback
            )
            print(
                f"\n[Agent] Running in CONCISE mode — "
                f"step summary only; full log → "
                f"sessions/{self._session.session_id}/session_log.log\n"
            )

        # ── Watchdog ──────────────────────────────────────────────────────────
        heartbeat_path = Path("sessions") / self._session.session_id / "heartbeat.json"
        self._watchdog = RunWatchdog(
            stop_event        = self._stop_event,
            heartbeat_path    = heartbeat_path,
            max_idle_seconds  = idle_limit,
            max_total_seconds = total_limit,
            logger            = self.logger,
        )
        self._watchdog.start()

        # ── Track experiments + context pruning via runner wrapper ──────────────
        runner_tool = next(
            (t for t in self.tools if isinstance(t, ExperimentRunnerTool)), None
        )
        if runner_tool is not None:
            _orig_forward = runner_tool.forward
            _orch = self

            def _tracked_forward(exp_name, architecture_family, model_code, hyperparams):
                if _orch._reporter is not None:
                    _orch._reporter.update_exp(exp_name)
                _orch._session.tick(
                    current_exp = exp_name,
                    step        = _orch._exp_count,
                    status      = "running",
                )
                result = _orig_forward(exp_name, architecture_family, model_code, hyperparams)
                _orch._exp_count += 1
                _orch._session.tick(
                    current_exp = exp_name,
                    step        = _orch._exp_count,
                    status      = "running",
                )
                return result

            runner_tool.forward = _tracked_forward

        # ── Token-count logger + memory-pruning step_callback ─────────────────
        # This callback fires AFTER each agent step (ActionStep), which means
        # we can read the token usage logged by smolagents and decide whether
        # to compress memory before the NEXT step.
        _orch = self
        _token_counts: list[int] = []   # one entry per step

        from smolagents.memory import ActionStep as _ActionStep

        def _pruning_callback(step, agent=None):
            """Log input token count and trigger compression every 5 experiments."""
            if not isinstance(step, _ActionStep):
                return

            # ── Count tokens ─────────────────────────────────────────────────
            tok = 0
            if step.token_usage is not None:
                try:
                    tok = step.token_usage.input_tokens
                except AttributeError:
                    tok = getattr(step.token_usage, "prompt_tokens", 0)
            elif step.model_input_messages:
                # Rough estimate: 4 chars ≈ 1 token
                tok = sum(
                    len(str(m)) for m in step.model_input_messages
                ) // 4
            _token_counts.append(tok)
            step_n = len(_token_counts)
            print(
                f"[Token] Step {step_n:3d} | input_tokens≈{tok:6,} | "
                f"memory_steps={len(_orch._agent.memory.steps)}",
                flush=True,
            )
            _orch.logger.info(
                f"[Context] step={step_n}  input_tokens≈{tok}  "
                f"memory_steps={len(_orch._agent.memory.steps)}"
            )

            # ── Compress every 5 experiments ─────────────────────────────────
            if _orch._exp_count > 0 and _orch._exp_count % 5 == 0:
                # Only compress once per 5-exp boundary, not on every step
                # during the same boundary.  Use a marker attribute.
                marker = f"_compressed_at_{_orch._exp_count}"
                if not getattr(_orch, marker, False):
                    setattr(_orch, marker, True)
                    _orch._compress_memory()

        # Register AFTER the concise reporter (so it runs second)
        self._agent.step_callbacks.register(_ActionStep, _pruning_callback)

        # ── Build prompt ──────────────────────────────────────────────────────
        prompt = self._prompt + (
            f"\n\nADDITIONAL CONSTRAINT FOR THIS SESSION:\n"
            f"Maximum experiments allowed: {max_experiments}.\n"
            f"Session ID: {self._session.session_id}\n"
            f"Current time: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"Idle timeout: {idle_limit}s    Total timeout: {total_limit}s\n"
        )

        self.logger.agent(
            f"[Orchestrator] Launching agent. "
            f"budget={max_experiments}  model={self.model_name}  "
            f"idle={idle_limit}s  wall={total_limit}s"
        )
        self._save_state("running")

        # ── Main run — always exits through finally ────────────────────────────
        result      = None
        fin_status  = "completed"
        fin_error   = None
        stop_reason = "normal"

        try:
            result = self._agent.run(prompt)
            if self._stop_event.is_set():
                # Watchdog or user button triggered stop
                stop_reason = self._watchdog.stop_reason or "interrupted"
                fin_status  = "interrupted"
            else:
                self.logger.agent(
                    f"[Orchestrator] Agent finished normally. "
                    f"elapsed={self._elapsed()}"
                )

        except KeyboardInterrupt:
            stop_reason = "interrupted"
            fin_status  = "interrupted"
            fin_error   = "KeyboardInterrupt"
            self.logger.warning("[Orchestrator] KeyboardInterrupt — stopping.")

        except Exception as exc:
            stop_reason = "error"
            fin_status  = "crashed"
            fin_error   = f"{type(exc).__name__}: {exc}"
            self.logger.error(f"[Orchestrator] Agent crashed: {exc}")

        finally:
            # ── Teardown ─────────────────────────────────────────────────────
            self._running = False
            self._save_state("idle")

            if self._watchdog is not None:
                self._watchdog.stop()

            if self._reporter is not None:
                self._reporter.stop()

            if self._session is not None:
                self._session.end(
                    status            = fin_status,
                    total_experiments = self._exp_count,
                    error_message     = fin_error,
                )

            # ── RUN SUMMARY (always printed regardless of exit path) ──────────
            self._print_run_summary(
                stop_reason = stop_reason,
                fin_error   = fin_error,
            )

        return result

    def stop(self) -> None:
        """Graceful stop — current experiment finishes before loop exits."""
        self.logger.warning("[Orchestrator] Stop requested by user.")
        self._stop_event.set()
        self._save_state("stopping")
        if self._reporter is not None:
            self._reporter.stop()

    def switch_model(self, model_name: str) -> None:
        """Hot-swap the LLM without restarting the orchestrator."""
        self.logger.agent(
            f"[Orchestrator] Switching model: {self.model_name} -> {model_name}"
        )
        self.llm.switch_model(model_name)
        self.model_name = model_name
        _vlevel = 0 if self.verbosity == "concise" else 2
        try:
            from smolagents import CodeAgent
            self._agent = CodeAgent(
                tools                         = self.tools,
                model                         = self.llm.get_model(),
                max_steps                     = 500,
                verbosity_level               = _vlevel,
                additional_authorized_imports = _AUTHORIZED_IMPORTS,
            )
            self.logger.agent(f"[Orchestrator] Switched to {model_name}.")
        except Exception as exc:
            self.logger.error(f"[Orchestrator] Could not rebuild CodeAgent: {exc}")
        for tool in self.tools:
            if isinstance(tool, ArxivTool):
                tool._llm = self.llm.get_model()

    def status(self) -> dict:
        """Return a live snapshot of orchestrator state."""
        lb = {}
        if _LEADERBOARD_PATH.exists():
            try:
                with open(_LEADERBOARD_PATH) as fh:
                    lb = json.load(fh)
            except Exception:
                pass
        return {
            "running":         self._running,
            "model":           self.model_name,
            "verbosity":       self.verbosity,
            "total_runs":      lb.get("total_runs", 0),
            "best_val_f1":     lb.get("best_val_f1_macro", 0.0),
            "best_experiment": lb.get("best_experiment"),
            "families_done":   lb.get("families_completed", []),
            "elapsed":         self._elapsed() if self._running else None,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compress_memory(self) -> None:
        """
        Compress agent memory every 5 experiments to prevent unbounded
        context growth.

        Steps
        -----
        1. Read the last 5 experiment summaries from leaderboard.json.
        2. Ask the LLM (small direct call, not part of agent loop) to
           produce a ≤150-word digest.
        3. Reset agent memory (clears all steps, keeps system prompt).
        4. Re-inject the digest + current leaderboard snapshot as a
           synthetic assistant message so the agent remembers what happened.
        """
        self.logger.agent(
            f"[Orchestrator] Compressing memory after {self._exp_count} experiments."
        )

        # \u2500\u2500 1. Gather last 5 experiments from leaderboard \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n        recent_exps = []
        best_f1   = 0.0
        best_exp  = "none"
        try:
            if _LEADERBOARD_PATH.exists():
                lb = json.loads(_LEADERBOARD_PATH.read_text())
                all_exps  = lb.get("experiments", [])
                recent_exps = all_exps[-5:]      # last 5
                best_f1   = lb.get("best_val_f1_macro", 0.0)
                best_exp  = lb.get("best_experiment", "none")
        except Exception as exc:
            self.logger.warning(f"[Orchestrator] Could not read leaderboard for compression: {exc}")

        if not recent_exps:
            self.logger.warning("[Orchestrator] No experiments to compress yet; skipping.")
            return

        # \u2500\u2500 2. Build the summarisation prompt \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        exp_lines = []
        for e in recent_exps:
            status = e.get("status", "?")
            f1     = e.get("val_f1_macro", 0.0)
            arch   = e.get("architecture", "?")
            family = e.get("architecture_family", "?")
            pc     = e.get("val_f1_per_class", {})
            pc_str = " ".join(
                f"{k}={float(v):.3f}" for k, v in sorted(pc.items(), key=lambda x: str(x[0]))
            )
            exp_lines.append(f"  - {arch} ({family}): f1={f1:.4f} [{status}] {pc_str}")

        summarise_prompt = (
            "You are summarising ML experiment history for an autonomous research agent.\n"
            "Summarise the following experiments in UNDER 150 words:\n"
            "- What architectures were tried\n"
            "- Their val_f1_macro scores\n"
            "- Key insight learned (one sentence)\n"
            "- What the current best is\n\n"
            "Experiments:\n"
            + "\n".join(exp_lines)
            + f"\n\nCurrent best: {best_exp} (val_f1_macro={best_f1:.4f})\n"
            "Reply with ONLY the summary, nothing else."
        )

        # \u2500\u2500 3. Ask LLM directly (small call outside agent loop) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        summary = ""
        try:
            messages = [{"role": "user", "content": summarise_prompt}]
            response = self.llm.get_model()(messages)
            if hasattr(response, "content"):
                summary = str(response.content).strip()
            else:
                summary = str(response).strip()
            self.logger.agent(
                f"[Orchestrator] Compression summary ({len(summary.split())} words):\n{summary}"
            )
        except Exception as exc:
            self.logger.warning(f"[Orchestrator] LLM compression call failed: {exc}")
            # Fall back to a simple bullet list
            summary = "Previous experiments:\n" + "\n".join(exp_lines)
            summary += f"\nBest so far: {best_exp} (val_f1_macro={best_f1:.4f})"

        # \u2500\u2500 4. Reset agent memory and re-inject summary \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        try:
            from smolagents.memory import TaskStep

            # Clear all steps (keeps system_prompt inside AgentMemory)
            self._agent.memory.reset()

            # Re-inject as a synthetic task step so the agent has context
            # for the next reasoning step without the full history.
            injected_task = (
                f"[MEMORY COMPRESSION — {self._exp_count} experiments completed]\n\n"
                f"{summary}\n\n"
                f"Current leaderboard best: {best_exp}  val_f1_macro={best_f1:.4f}\n"
                "Continue your research from this point forward."
            )
            self._agent.memory.steps.append(TaskStep(task=injected_task))

            mem_after = len(self._agent.memory.steps)
            print(
                f"\n[Compress] Memory reset at exp#{self._exp_count} → "
                f"{mem_after} step(s) in memory (summary injected)\n",
                flush=True,
            )
            self.logger.agent(
                f"[Orchestrator] Memory compressed. steps_after={mem_after}"
            )
        except Exception as exc:
            self.logger.error(f"[Orchestrator] Memory reset failed: {exc}")


    def _elapsed(self) -> str:
        if self._run_start is None:
            return "n/a"
        secs = int(time.time() - self._run_start)
        h, rem = divmod(secs, 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}h{m:02d}m{s:02d}s"

    def _elapsed_tuple(self) -> tuple[int, int]:
        """Return (minutes, seconds) of total run time."""
        if self._run_start is None:
            return 0, 0
        secs = int(time.time() - self._run_start)
        return divmod(secs, 60)

    def _save_state(self, status: str) -> None:
        state = {
            "status":     status,
            "model":      self.model_name,
            "verbosity":  self.verbosity,
            "started_at": self._run_start,
            "updated_at": time.time(),
        }
        try:
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_STATE_PATH, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except Exception:
            pass

    def _print_run_summary(self, stop_reason: str, fin_error: Optional[str]) -> None:
        """Always-printed summary block regardless of exit path."""
        mins, secs = self._elapsed_tuple()

        # Best result from leaderboard
        best_f1  = 0.0
        best_exp = "none"
        try:
            if _LEADERBOARD_PATH.exists():
                lb = json.loads(_LEADERBOARD_PATH.read_text())
                best_f1  = lb.get("best_val_f1_macro", 0.0)
                best_exp = lb.get("best_experiment") or "none"
        except Exception:
            pass

        stop_label = {
            "normal":        "normal — agent finished its task",
            "idle timeout":  "idle timeout (watchdog)",
            "total timeout": "total time limit (watchdog)",
            "error":         "error / crash",
            "interrupted":   "interrupted (KeyboardInterrupt or Stop button)",
        }.get(stop_reason, stop_reason)

        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  RUN SUMMARY")
        print(sep)
        print(f"  Stopped because  : {stop_label}")
        print(f"  Experiments run  : {self._exp_count}")
        print(f"  Total time       : {mins}m {secs}s")
        print(f"  Best val_f1_macro: {best_f1:.4f}  ({best_exp})")
        if fin_error:
            print(f"  Last error       : {fin_error}")
            if self._session:
                print(
                    f"  Full traceback   : "
                    f"sessions/{self._session.session_id}/session_log.log"
                )
        print(sep + "\n")

        self.logger.agent(
            f"[Orchestrator] RUN SUMMARY: "
            f"stop={stop_reason}  exps={self._exp_count}  "
            f"time={mins}m{secs}s  best_f1={best_f1:.4f}  "
            f"best_exp={best_exp}"
        )
