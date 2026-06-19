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
                tools          = self.tools,
                model          = self.llm.get_model(),
                max_steps      = max_steps,
                verbosity_level = _vlevel,
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
            # Inject as step callback into the agent
            if not hasattr(self._agent, "step_callbacks"):
                self._agent.step_callbacks = []
            self._agent.step_callbacks = [self._reporter.step_callback]
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

        # ── Track experiments via runner wrapper ──────────────────────────────
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
                tools           = self.tools,
                model           = self.llm.get_model(),
                max_steps       = 500,
                verbosity_level = _vlevel,
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
