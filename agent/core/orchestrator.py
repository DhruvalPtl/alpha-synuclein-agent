"""
agent/core/orchestrator.py
────────────────────────────────────────────────────────────────────────────
AgentOrchestrator — top-level controller that wires LLMManager, all tools,
and the Smolagents CodeAgent into a single runnable unit.

Usage
-----
    from agent.core.orchestrator import AgentOrchestrator

    agent = AgentOrchestrator(model_name="local-qwen")
    agent.run(max_experiments=200)

    # Switch model mid-session
    agent.switch_model("groq-llama")

    # Graceful stop
    agent.stop()
"""

import json
import os
import signal
import threading
import time
import datetime
from pathlib import Path
from typing import Optional, List

from agent.core.tee_logger import TeeLogger
from agent.core.llm_manager import LLMManager
from agent.prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_SHORT
from agent.tools.experiment_runner import ExperimentRunnerTool
from agent.tools.leaderboard_tool import LeaderboardTool
from agent.tools.audit_tool import AuditTool
from agent.tools.arxiv_tool import ArxivTool

_LEADERBOARD_PATH = Path("master_log/leaderboard.json")
_STATE_PATH       = Path("master_log/orchestrator_state.json")

# Models with smaller context windows that need the short prompt
_SHORT_PROMPT_MODELS = {"groq-mixtral", "mistral-small"}


class AgentOrchestrator:
    """
    Wires all Phase 3 components into a single autonomous research loop.

    Parameters
    ----------
    model_name : str
        LLM to drive the agent (key from llm_manager.MODELS).
    use_short_prompt : bool
        Force the short system prompt (for models with limited context).
    max_steps : int
        Maximum Smolagents inner steps per run() call.
    """

    def __init__(
        self,
        model_name:       str  = "local-qwen",
        use_short_prompt: bool = False,
        max_steps:        int  = 500,
    ) -> None:
        self.logger = TeeLogger(master_log_dir="master_log")
        self.logger.agent(
            f"[Orchestrator] Initialising with model={model_name}"
        )

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

        # Add DuckDuckGoSearchTool if available
        try:
            from smolagents import DuckDuckGoSearchTool
            self.tools.append(DuckDuckGoSearchTool())
            self.logger.info("[Orchestrator] DuckDuckGoSearchTool added.")
        except (ImportError, Exception) as exc:
            self.logger.warning(
                f"[Orchestrator] DuckDuckGoSearchTool unavailable: {exc}"
            )

        # ── Smolagents CodeAgent ──────────────────────────────────────────────
        try:
            from smolagents import CodeAgent
            self._agent = CodeAgent(
                tools=self.tools,
                model=self.llm.get_model(),
                max_steps=max_steps,
                verbosity_level=2,
            )
            self.logger.info(
                f"[Orchestrator] CodeAgent ready. max_steps={max_steps}"
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
        self._run_start   = None

        self.logger.agent("[Orchestrator] Ready. Call agent.run() to start.")

    # ── Public API ──────────────────────────────────────────────────────────────

    def run(self, max_experiments: int = 200) -> Optional[str]:
        """
        Start the autonomous research loop.

        The CodeAgent runs the SYSTEM_PROMPT as its task and is free to
        call any tool any number of times until it decides it is done
        or max_steps is reached.

        Parameters
        ----------
        max_experiments : int
            Soft cap on experiments. Injected into the prompt so the
            agent knows the budget.

        Returns
        -------
        str  — agent's final answer, or None if interrupted
        """
        if self._agent is None:
            self.logger.error(
                "[Orchestrator] Agent not initialised. Cannot run."
            )
            return None

        self._stop_event.clear()
        self._running   = True
        self._run_start = time.time()

        # Inject the experiment budget into the prompt
        prompt = self._prompt + (
            f"\n\nADDITIONAL CONSTRAINT FOR THIS SESSION:\n"
            f"Maximum experiments allowed: {max_experiments}.\n"
            f"Current time: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        )

        self.logger.agent(
            f"[Orchestrator] Launching agent. "
            f"budget={max_experiments} experiments  "
            f"model={self.model_name}"
        )
        self._save_state("running")

        result = None
        try:
            result = self._agent.run(prompt)
            self.logger.agent(
                f"[Orchestrator] Agent finished. "
                f"elapsed={self._elapsed()}"
            )
        except KeyboardInterrupt:
            self.logger.warning(
                "[Orchestrator] KeyboardInterrupt — stopping gracefully."
            )
        except Exception as exc:
            self.logger.error(
                f"[Orchestrator] Agent crashed: {exc}"
            )
            raise
        finally:
            self._running = False
            self._save_state("idle")

        return result

    def stop(self) -> None:
        """
        Request a graceful stop. The current experiment will finish before
        the loop exits.
        """
        self.logger.warning("[Orchestrator] Stop requested.")
        self._stop_event.set()
        self._save_state("stopping")

    def switch_model(self, model_name: str) -> None:
        """
        Hot-swap the underlying LLM without restarting the orchestrator.
        The CodeAgent will use the new model on its next tool call.
        """
        self.logger.agent(
            f"[Orchestrator] Switching model: "
            f"{self.model_name} -> {model_name}"
        )
        self.llm.switch_model(model_name)
        self.model_name = model_name

        # Rebuild the CodeAgent with the new model
        try:
            from smolagents import CodeAgent
            self._agent = CodeAgent(
                tools=self.tools,
                model=self.llm.get_model(),
                max_steps=500,
                verbosity_level=2,
            )
            self.logger.agent(
                f"[Orchestrator] Switched to {model_name}. "
                "CodeAgent rebuilt."
            )
        except Exception as exc:
            self.logger.error(
                f"[Orchestrator] Could not rebuild CodeAgent: {exc}"
            )

        # Update ArxivTool with the new model for LLM filtering
        for tool in self.tools:
            if isinstance(tool, ArxivTool):
                tool._llm = self.llm.get_model()

    def status(self) -> dict:
        """Return a snapshot of the orchestrator status."""
        lb = {}
        if _LEADERBOARD_PATH.exists():
            try:
                with open(_LEADERBOARD_PATH) as fh:
                    lb = json.load(fh)
            except Exception:
                pass

        return {
            "running":          self._running,
            "model":            self.model_name,
            "total_runs":       lb.get("total_runs", 0),
            "best_val_f1":      lb.get("best_val_f1_macro", 0.0),
            "best_experiment":  lb.get("best_experiment"),
            "families_done":    lb.get("families_completed", []),
            "elapsed":          self._elapsed() if self._running else None,
        }

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _elapsed(self) -> str:
        if self._run_start is None:
            return "n/a"
        secs = int(time.time() - self._run_start)
        h, rem = divmod(secs, 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}h{m:02d}m{s:02d}s"

    def _save_state(self, status: str) -> None:
        """Persist minimal state so the dashboard can poll it."""
        state = {
            "status":     status,
            "model":      self.model_name,
            "started_at": self._run_start,
            "updated_at": time.time(),
        }
        try:
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_STATE_PATH, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except Exception:
            pass
