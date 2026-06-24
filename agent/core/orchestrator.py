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

try:
    from smolagents import ToolException
except ImportError:
    class ToolException(Exception):
        pass

from agent.core.tee_logger import TeeLogger
from agent.core.llm_manager import LLMManager, TwoBrainManager, MODELS, FALLBACK_CHAIN
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
    "textwrap",
]


# ── Phase instruction helper ──────────────────────────────────────────────────
# Simplified: no family tracking. Agent explores freely.

class ExploreExploitController:
    """
    Injects a simple budget-progress instruction into the agent's task prompt.
    No family tracking — the agent decides what to explore.
    """

    def __init__(self, explore_ratio: float = 0.6, total_budget: int = 200) -> None:
        self.explore_budget = max(1, int(total_budget * explore_ratio))
        self.phase: str = "explore"

    def get_phase_instruction(self, leaderboard: dict) -> str:
        total_runs = leaderboard.get("total_runs", 0)
        if total_runs < self.explore_budget:
            self.phase = "explore"
            remaining = self.explore_budget - total_runs
            return (
                f"\n\n[EXPLORE PHASE {total_runs}/{self.explore_budget}] "
                f"You have {remaining} exploration experiments left. "
                f"Try a variety of different model types and approaches. "
                f"Be creative — think beyond standard classifiers."
            )
        else:
            self.phase = "exploit"
            return (
                f"\n\n[EXPLOIT PHASE] Exploration complete ({total_runs} experiments done). "
                f"Focus on refining and ensembling the best approaches found so far. "
                f"Tune hyperparameters aggressively."
            )

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
        reasoning_model:    Optional[str] = None,   # two-brain: overrides model_name for reasoning
        coding_model:       Optional[str] = None,   # two-brain: dedicated coder model
        two_brain:          bool = False,
        use_short_prompt:   bool = False,
        max_steps:          int  = 500,   # base default; overridden per run() as max_experiments*10
        verbosity:          str  = "concise",
        max_idle_seconds:   int  = 300,
        max_total_seconds:  int  = 3600,
        explore_ratio:      float = 0.6,    # fraction of budget for exploration phase
    ) -> None:
        self.logger = TeeLogger(master_log_dir="master_log")
        self.logger.agent(
            f"[Orchestrator] Initialising  model={model_name}  "
            f"verbosity={verbosity}"
        )

        self.verbosity          = verbosity.lower()
        self.max_idle_seconds   = max_idle_seconds
        self.max_total_seconds  = max_total_seconds
        self.max_steps          = max_steps   # stored so switch_model rebuild can reuse it

        # ── LLM ───────────────────────────────────────────────────────────────
        _r_model = reasoning_model or model_name
        if two_brain and coding_model and coding_model != _r_model:
            self.two_brain = TwoBrainManager(
                reasoning_model = _r_model,
                coding_model    = coding_model,
            )
            self.llm       = self.two_brain.reasoner
            _agent_model   = self.two_brain.get_reasoning_model()
            model_name     = _r_model
            self.logger.agent(
                f"[Orchestrator] Two-brain mode: reasoning={_r_model}  coding={coding_model}"
            )
        else:
            self.two_brain = None
            self.llm       = LLMManager(model_name=_r_model)
            _agent_model   = self.llm.get_model()
            model_name     = _r_model

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
                model                         = _agent_model,
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

        # ── Minimum-experiments guard (installed in run(), stored here) ───────
        # Actual override happens in run() once max_experiments is known.
        self._min_experiments_before_stop: int = 0

        # ── Prompt selection ──────────────────────────────────────────────────
        if use_short_prompt or model_name in _SHORT_PROMPT_MODELS:
            self._prompt = SYSTEM_PROMPT_SHORT
            self.logger.info("[Orchestrator] Using SHORT system prompt.")
        else:
            self._prompt = SYSTEM_PROMPT

        # ── Ollama: remind the model of the required output format ─────────────
        # thinking tokens are disabled at API level (extra_body={"think": false}),
        # but we still reinforce the Thought:/code-block format so smolagents
        # can parse the response correctly.
        _active_provider = MODELS.get(model_name, "").split("/")[0]
        if _active_provider == "ollama":
            _ollama_reminder = (
                "\n\nIMPORTANT: Always respond using exactly this format:\n"
                "Thought: <one sentence summary of what you will do>\n"
                "<code>\n"
                "<your python code here>\n"
                "</code>"
            )
            self._prompt = self._prompt + _ollama_reminder
            self.logger.info(
                "[Orchestrator] Ollama model detected: appended no-thinking "
                "reminder to system prompt."
            )

        self.model_name   = model_name
        self._stop_event  = threading.Event()
        self._running     = False
        self._run_start:  Optional[float] = None
        self._session:    Optional[SessionManager] = None
        self._watchdog:   Optional[RunWatchdog] = None
        self._reporter:   Optional[ConciseStepReporter] = None
        self._exp_count   = 0
        self._explore_ratio = explore_ratio  # stored for run() to use

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
            # max_experiments captured in closure so _tracked_forward can enforce it
            _max_exp = max_experiments

            def _tracked_forward(exp_name, model_code, hyperparams="{}"):
                # ── HARD BUDGET CAP ─────────────────────────────────────────────
                # ToolException is shown to the LLM as an error it must respond to.
                if _orch._exp_count >= _max_exp:
                    _orch.logger.warning(
                        f"[BudgetGuard] HARD STOP: {_orch._exp_count}/{_max_exp} experiments done."
                    )
                    print(
                        f"\n[BudgetGuard] BUDGET EXHAUSTED: {_orch._exp_count}/{_max_exp} experiments done."
                        f" You MUST call final_answer() now to conclude the session.\n",
                        flush=True,
                    )
                    raise ToolException(
                        f"BUDGET EXHAUSTED: You have already run {_orch._exp_count} experiments "
                        f"out of your budget of {_max_exp}. "
                        f"You MUST call final_answer() now. Do NOT call run_experiment again."
                    )
                if _orch._reporter is not None:
                    _orch._reporter.update_exp(exp_name)
                _orch._session.tick(
                    current_exp = exp_name,
                    step        = _orch._exp_count,
                    status      = "running",
                )
                result = _orig_forward(exp_name, model_code, hyperparams)
                _orch._exp_count += 1
                _orch._session.tick(
                    current_exp = exp_name,
                    step        = _orch._exp_count,
                    status      = "running",
                )
                # Notify when budget cap is hit
                if _orch._exp_count >= _max_exp:
                    print(
                        f"\n[BudgetGuard] Budget reached: {_orch._exp_count}/{_max_exp} experiments done."
                        f" Call final_answer() now to finish.\n",
                        flush=True,
                    )
                return result

            runner_tool.forward = _tracked_forward
        else:
            _max_exp = max_experiments  # still needed for guards below

        # ── Token-count logger + memory-pruning step_callback ─────────────────
        # This callback fires AFTER each agent step (ActionStep), which means
        # we can read the token usage logged by smolagents and decide whether
        # to compress memory before the NEXT step.
        _orch = self
        _token_counts: list[int] = []   # one entry per step
        # Running total across all steps for the session-level summary
        _total_tokens: list[int] = [0]  # use list so closure can mutate it

        from smolagents.memory import ActionStep as _ActionStep

        def _pruning_callback(step, agent=None):
            """Log input token count (real first, estimated fallback) and trigger compression."""
            if not isinstance(step, _ActionStep):
                return

            # ── Read REAL token count first ─────────────────────────────────
            tok: Optional[int] = None

            # Path 1: token_usage on the step object (most reliable)
            if step.token_usage is not None:
                try:
                    tok = int(step.token_usage.input_tokens)
                except (AttributeError, TypeError):
                    try:
                        tok = int(step.token_usage.prompt_tokens)
                    except (AttributeError, TypeError):
                        pass

            # Path 2: raw usage inside the ChatMessage (litellm >=1.x)
            if tok is None and step.model_output_message is not None:
                msg = step.model_output_message
                try:
                    raw_usage = msg.raw.usage
                    tok = int(getattr(raw_usage, "prompt_tokens",
                              getattr(raw_usage, "input_tokens", None)))
                except (AttributeError, TypeError):
                    pass

            # Path 3: char-based estimate (~4 chars per token)
            if tok is None:
                if step.model_input_messages:
                    tok = sum(len(str(m)) for m in step.model_input_messages) // 4
                else:
                    tok = 0
                source = "est"
            else:
                source = "real"

            _token_counts.append(tok)
            _total_tokens[0] += tok
            step_n = len(_token_counts)
            print(
                f"[Token] Step {step_n:3d} | input_tokens={tok:6,} ({source}) | "
                f"memory_steps={len(_orch._agent.memory.steps)}",
                flush=True,
            )
            _orch.logger.info(
                f"[Context] step={step_n}  input_tokens={tok} ({source})  "
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

            # ── Budget watchdog: trigger stop if over-budget for >3 steps ─────
            # This handles the case where the LLM ignores the ToolException and
            # keeps producing output without calling run_experiment or final_answer.
            if _orch._exp_count >= _max_exp:
                _over_marker = f"_over_budget_steps"
                _over_count  = getattr(_orch, _over_marker, 0) + 1
                setattr(_orch, _over_marker, _over_count)
                if _over_count >= 3:
                    _orch.logger.warning(
                        f"[BudgetGuard] Agent has been over-budget for {_over_count} steps. "
                        f"Triggering stop_event."
                    )
                    print(
                        f"\n[BudgetGuard] Agent ignored budget for {_over_count} steps — "
                        f"forcing stop.\n",
                        flush=True,
                    )
                    _orch._stop_event.set()

        # Register AFTER the concise reporter (so it runs second)
        self._agent.step_callbacks.register(_ActionStep, _pruning_callback)

        # Expose total-token counter so _print_run_summary can access it
        self._total_tokens_ref = _total_tokens

        # ── Minimum-experiments before final_answer is allowed ────────────────
        self._min_experiments_before_stop = int(max_experiments * 0.8)

        # ── Recalculate max_steps based on max_experiments ────────────────────
        # 10 steps per experiment gives the agent enough room to plan, write,
        # and run each experiment without spinning forever.
        effective_max_steps = max_experiments * 10
        self.max_steps = effective_max_steps
        if self._agent is not None:
            try:
                self._agent.max_steps = effective_max_steps
                self.logger.info(
                    f"[Orchestrator] max_steps set to {effective_max_steps} "
                    f"(max_experiments={max_experiments} × 10)"
                )
            except Exception as _ms_exc:
                self.logger.warning(f"[Orchestrator] Could not update max_steps: {_ms_exc}")

        # ── Override final_answer to block premature stopping ─────────────────
        # HOW IT WORKS (CodeAgent-specific):
        #   The Python executor detects final_answer() by catching the internal
        #   FinalAnswerException.  If forward() raises a *different* exception,
        #   that propagates as InterpreterError -> AgentExecutionError (AgentError)
        #   which the outer step-loop catches, records as step.error, and continues.
        #   Returning a string does NOT stop the agent from stopping — the
        #   is_final_answer flag is set by FinalAnswerException, not the return value.
        if self._agent is not None:
            try:
                _original_fa = self._agent.tools["final_answer"].forward
                _orch_ref    = self
                _min_exp     = self._min_experiments_before_stop
                _logger_ref  = self.logger
                _lb_path     = _LEADERBOARD_PATH   # captured from module-level constant

                def _guarded_final_answer(answer):
                    runs = _orch_ref._exp_count

                    # Always allow if budget is exhausted (hard cap hit)
                    if runs >= _max_exp:
                        _logger_ref.agent(
                            f"[Guard] final_answer APPROVED (budget exhausted) -- "
                            f"{runs}/{_max_exp} experiments done"
                        )
                        return _original_fa(answer)

                    if runs < _min_exp:
                        _logger_ref.warning(
                            f"[Guard] final_answer BLOCKED -- "
                            f"{runs}/{_min_exp} experiments done"
                        )
                        print(
                            f"\n[Guard] BLOCKED: final_answer rejected -- "
                            f"{runs}/{_min_exp} experiments done. "
                            f"Keep exploring!\n",
                            flush=True,
                        )
                        raise ToolException(
                            f"Cannot conclude yet. You have run {runs} experiments "
                            f"this session but need at least {_min_exp}. "
                            f"Continue exploring -- do not call final_answer again "
                            f"until you have run {_min_exp - runs} more experiments."
                        )

                    _logger_ref.agent(
                        f"[Guard] final_answer APPROVED -- "
                        f"{runs}/{_min_exp} experiments done"
                    )
                    return _original_fa(answer)

                self._agent.tools["final_answer"].forward = _guarded_final_answer
                self.logger.info(
                    f"[Orchestrator] final_answer guard installed "
                    f"(min_experiments={_min_exp}, raise=ToolException)"
                )
            except (KeyError, AttributeError) as _ge:
                self.logger.warning(
                    f"[Orchestrator] Could not install final_answer guard: {_ge}"
                )

        # ── Override run_experiment to block exploitation loops ───────────────
        # If the last 3 experiments all share the same base-learner keyword,
        # raise ToolException so the agent must try something genuinely different.
        if runner_tool is not None:
            try:
                _EXPLOIT_KEYWORDS = ["xgb", "rf", "lgbm", "mlp", "svc", "gb", "lr", "knn"]
                # At this point runner_tool.forward is already _tracked_forward
                # (set earlier in run()).  We wrap that to add the repetition check
                # while keeping tracking intact.
                _tracked_fwd = runner_tool.forward
                _rep_logger  = self.logger
                _orch_ref    = self
                
                # Keep a running list of base learner keywords used
                _history: list[str] = []

                def _guarded_run_experiment(
                    exp_name, model_code, hyperparams="{}"
                ):
                    # Identify the base learner for this experiment
                    lower_name = exp_name.lower()
                    kw_found = "other"
                    for kw in _EXPLOIT_KEYWORDS:
                        if kw in lower_name:
                            kw_found = kw
                            break
                    
                    # Check the last 2 runs (if we are about to run the 3rd)
                    if kw_found != "other" and len(_history) >= 2:
                        if _history[-1] == kw_found and _history[-2] == kw_found:
                            msg = (
                                f"Repetition Guard: You have already run 3 consecutive "
                                f"experiments using the '{kw_found}' base learner. "
                                f"You MUST explore a different architecture now to find the best model. "
                                f"Try a completely different algorithm."
                            )
                            _rep_logger.warning(f"[RepetitionGuard] BLOCKED {exp_name} ({kw_found})")
                            print(f"\\n[RepetitionGuard] BLOCKED: {msg}\\n", flush=True)
                            raise ToolException(msg)
                    
                    _history.append(kw_found)

                    # All clear — delegate to the tracked forward
                    return _tracked_fwd(exp_name, model_code, hyperparams)

                runner_tool.forward = _guarded_run_experiment
                self.logger.info(
                    "[Orchestrator] run_experiment repetition guard installed "
                    "(blocks 3 consecutive same-base-learner runs, raise=ToolException)"
                )
            except Exception as _rge:
                self.logger.warning(
                    f"[Orchestrator] Could not install run_experiment guard: {_rge}"
                )

        # ── Build prompt ───────────────────────────────────────────────────────
        # Initialise explore/exploit controller for this run session
        _ee_ctrl = ExploreExploitController(
            explore_ratio = self._explore_ratio,
            total_budget  = max_experiments,
        )

        # Reload leaderboard for dynamic session context
        _lb_state: dict = {}
        try:
            from agent.tools.rebuild_leaderboard import rebuild_leaderboard
            _lb_state = rebuild_leaderboard(verbose=False)
        except Exception:
            try:
                if _LEADERBOARD_PATH.exists():
                    _lb_state = json.loads(_LEADERBOARD_PATH.read_text())
            except Exception:
                pass

        _phase_instruction = _ee_ctrl.get_phase_instruction(_lb_state)

        # Build session context from leaderboard
        import os as _os
        _best_f1         = _lb_state.get("best_val_f1_macro", 0.0)
        _best_exp        = _lb_state.get("best_experiment", "none yet")
        _total_ever      = _lb_state.get("total_runs", 0)


        _session_context = (
            f"\n\n=== THIS SESSION ==="
            f"\nBudget this session        : {max_experiments} experiments"
            f"\nTotal experiments run so far: {_total_ever} (across all sessions)"
            f"\nCurrent best F1            : {_best_f1:.4f}  ({_best_exp})"
            f"\nRead the leaderboard for full history."
            f"\n"
            f"\nINSTRUCTIONS FOR THIS SESSION:"
            f"\n1. Read the leaderboard first (call read_leaderboard)."
            f"\n2. Do not repeat experiments already in the leaderboard."
            f"\n3. You MUST run at least {self._min_experiments_before_stop} new experiments"
            f"\n   this session before you are allowed to call final_answer."
            f"\n4. If you find a good model early, do NOT stop --"
            f"\n   instead write: 'X works well. Now I will explore Y to confirm"
            f"\n   X is the true best and not just the first thing that worked.'"
            f"\n"
            f"\nSession ID : {self._session.session_id}"
            f"\nMachine    : {_os.getenv('MACHINE_ID', 'unknown')}"
            f"\nIdle timeout: {idle_limit}s    Total timeout: {total_limit}s"
        )

        task_prompt = (
            self._prompt
            + f"\n\nADDITIONAL CONSTRAINT FOR THIS SESSION:"
            f"\nMaximum experiments allowed: {max_experiments}."
            f"\nCurrent time: {datetime.datetime.now().isoformat(timespec='seconds')}"
            + _session_context
            + _phase_instruction
        )

        self.logger.agent(
            f"[Orchestrator] Launching agent. "
            f"budget={max_experiments}  model={self.model_name}  "
            f"idle={idle_limit}s  wall={total_limit}s  "
            f"min_before_stop={self._min_experiments_before_stop}"
        )
        self._save_state("running")

        # ── Main run — always exits through finally ────────────────────────────
        result      = None
        fin_status  = "completed"
        fin_error   = None
        stop_reason = "normal"

        try:
            result = self._agent.run(task_prompt)
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
            # ── Attempt in-place model swap for rate-limit / API errors ────────
            _swapped = False
            try:
                import litellm
                _api_errors = (
                    litellm.RateLimitError,
                    litellm.ServiceUnavailableError,
                    litellm.APIError,
                )
                if isinstance(exc, _api_errors):
                    # ── Use FALLBACK_CHAIN for rate-limit auto-switch ──────────
                    _next_model = FALLBACK_CHAIN.get(self.model_name)
                    if _next_model:
                        self.logger.warning(
                            f"[Fallback] {type(exc).__name__} on {self.model_name!r} — "
                            f"auto-switching to {_next_model!r} via FALLBACK_CHAIN"
                        )
                        print(
                            f"\n[Fallback] Rate-limit hit on {self.model_name!r}. "
                            f"Auto-switching to {_next_model!r}.",
                            flush=True,
                        )
                        try:
                            self.llm.switch_model(_next_model)
                            self._agent.model = self.llm.get_model()
                            self.model_name   = _next_model
                            self.logger.agent(
                                f"[Fallback] Switched to {_next_model!r} — "
                                f"continuing without restart"
                            )
                            _swapped = True
                        except Exception as _sw_exc:
                            self.logger.error(
                                f"[Fallback] Switch to {_next_model!r} failed: {_sw_exc}. "
                                f"Trying full auto_fallback_chain."
                            )

                    if not _swapped:
                        # Chain lookup failed or switch errored — try full chain
                        self.logger.warning(
                            f"[Fallback] {type(exc).__name__}: {str(exc)[:100]}"
                        )
                        fallback = self.llm.auto_fallback_chain(
                            chain=["gemini-flash", "groq-llama", "local-qwen"],
                            test_each=True,
                        )
                        if fallback:
                            # Swap model directly on the running agent — no loop restart
                            self._agent.model = self.llm.get_model()
                            self.model_name   = fallback
                            self.logger.agent(
                                f"[Fallback] Switched to {fallback} — "
                                f"continuing without restart"
                            )
                            print(
                                f"\n[Fallback] Switched to {fallback} — agent continues.",
                                flush=True,
                            )
                            _swapped = True
                        else:
                            self.logger.error(
                                "[Fallback] All fallback models failed — raising."
                            )
            except Exception as swap_exc:
                self.logger.error(
                    f"[Fallback] Swap attempt failed: {swap_exc}"
                )

            if not _swapped:
                # Genuine crash — record and exit
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
                max_steps                     = self.max_steps,
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
            pc     = e.get("val_f1_per_class", {})
            pc_str = " ".join(
                f"{k}={float(v):.3f}" for k, v in sorted(pc.items(), key=lambda x: str(x[0]))
            )
            exp_lines.append(f"  - {arch}: f1={f1:.4f} [{status}] {pc_str}")

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

        # Token stats
        total_tok = getattr(self, "_total_tokens_ref", [0])[0]
        gemini_daily_limit = 1_000_000  # Gemini free-tier tokens / day
        pct_used = (total_tok / gemini_daily_limit * 100) if gemini_daily_limit else 0.0

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
        print(f"  Total tokens used: {total_tok:,}")
        print(f"  Gemini free tier : {pct_used:.1f}% of daily 1M limit used")
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
            f"best_exp={best_exp}  total_tokens={total_tok}  "
            f"gemini_pct={pct_used:.1f}%"
        )
