"""
agent/core/llm_manager.py
────────────────────────────────────────────────────────────────────────────
Multi-provider LLM manager built on LiteLLM + Smolagents.

Supports local (Ollama) and cloud (Gemini, Groq, Mistral, Cerebras,
OpenRouter) providers with hot-swapping and auto-fallback.

Usage
-----
    from agent.core.llm_manager import LLMManager

    llm = LLMManager(model_name="local-qwen")
    llm.test_connection()

    model = llm.get_model()   # pass to CodeAgent or ToolCallingAgent

    # Switch on the fly
    llm.switch_model("groq-llama")

    # Auto-fallback if primary fails
    active = llm.auto_fallback_chain()
"""

import os
import time
import threading
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

from agent.core.tee_logger import TeeLogger

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS: Dict[str, str] = {
    # ── Cloud models ────────────────────────────────────────────────
    "gemini-flash":   "gemini/gemini-3.5-flash",
    "gemini-flash-lite": "gemini/gemini-3.1-flash-lite",
    "gemini-pro":     "gemini/gemini-3.1-pro-preview",
    "groq-llama":     "groq/llama-3.3-70b-versatile",
    "groq-mixtral":   "groq/mixtral-8x7b-32768",
    "mistral-small":  "mistral/mistral-small-latest",
    "cerebras":       "cerebras/llama-3.3-70b",
    "openrouter":     "openrouter/meta-llama/llama-3.3-70b",
    # ── Ollama (local, http://localhost:11434) ───────────────────────
    "local-qwen":         "ollama/qwen2.5-coder:32b",
    "local-deepseek":     "ollama/deepseek-r1:14b",
    "local-qwen3-coder":  "ollama/qwen3-coder:30b",
    "local-qwen3":        "ollama/qwen3.6:27b",
    # ── LM Studio (local, http://localhost:1234/v1) ──────────────────
    # In LM Studio: enable the local server, load any model, then
    # set MODEL_NAME to one of the keys below that matches what you
    # have loaded.  The model string after 'lmstudio/' is ignored by
    # LiteLLM — it always talks to whatever is loaded in LM Studio.
    "lmstudio-qwen":     "openai/lmstudio-qwen",
    "lmstudio-deepseek": "openai/lmstudio-deepseek",
    "lmstudio-mistral":  "openai/lmstudio-mistral",
    "lmstudio-llama":    "openai/lmstudio-llama",
    "lmstudio-any":      "openai/lmstudio-any",   # generic: use when unsure
    # ── Llamafile (Colab/Kaggle, http://localhost:8080/v1) ───────────
    # Start via: LlamafileManager("qwen-14b-coder").start()
    # Then set MODEL_NAME to one of these keys.
    "llamafile-14b":  "openai/llamafile-qwen-14b",
    "llamafile-32b":  "openai/llamafile-qwen-32b",
}

# ── Provider → env-var for API key ────────────────────────────────────────────
_API_KEY_MAP: Dict[str, str] = {
    "gemini":      "GEMINI_API_KEY",
    "groq":        "GROQ_API_KEY",
    "mistral":     "MISTRAL_API_KEY",
    "openrouter":  "OPENROUTER_API_KEY",
    "cerebras":    "CEREBRAS_API_KEY",
}

# ── LM Studio base URL ────────────────────────────────────────────────────────
# LM Studio's local server is OpenAI-compatible. Default port is 1234.
# Change this if you've configured a different port in LM Studio.
_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
_LMSTUDIO_PREFIXES = {"lmstudio-"}   # any model key starting with this

# ── Default fallback order (first working model wins) ─────────────────────────
DEFAULT_FALLBACK_CHAIN: List[str] = [
    "lmstudio-any",  # LM Studio — no rate limit, free, private
    "local-qwen",    # Ollama — free, fast, no rate limit
    "groq-llama",    # Groq cloud — generous free tier
    "gemini-flash",  # Google cloud — free tier
]

_MASTER_LOG_DIR = Path("master_log")
_TOKEN_BUDGET_PATH = _MASTER_LOG_DIR / "token_budget.json"

# ── Per-provider rate limits ───────────────────────────────────────────────────
# Free-tier RPM limits — we target slightly below to leave a safety margin.
PROVIDER_RPM_LIMITS: dict = {
    "gemini":    5,       # free tier 15 RPM → target 5 (very conservative)
    "groq":      25,      # free tier 30 RPM → target 25
    "cerebras":  25,
    "mistral":   30,
    "openrouter": 20,
    "ollama":    9999,    # local — unlimited
    "openai":    9999,    # LM Studio / llamafile — local, unlimited
}

# Minimum seconds between calls = 60 / RPM_LIMIT
def _min_gap(provider: str) -> float:
    rpm = PROVIDER_RPM_LIMITS.get(provider, 30)
    return 60.0 / max(rpm, 1)


# ── Daily token budgets ────────────────────────────────────────────────────────
DAILY_LIMITS: dict = {
    "gemini":    1_000_000,   # Google free tier 1M tokens/day
    "groq":        100_000,   # Groq free tier ~100K/day
    "cerebras":    100_000,
    "mistral":     100_000,
    "ollama":  999_999_999,   # local — no limit
    "openai":  999_999_999,   # local — no limit
}

_WARN_THRESHOLD  = 0.80   # warn at 80% of daily limit
_SWITCH_THRESHOLD = 0.95  # auto-switch provider at 95% of daily limit

# ── Backoff delays (shared, used for all providers) ───────────────────────────
_BACKOFF_DELAYS = [5, 10, 20, 40, 60]   # seconds before each retry

# Legacy aliases kept for backward compatibility
_GEMINI_RPM          = PROVIDER_RPM_LIMITS["gemini"]
_GEMINI_MIN_CALL_GAP = _min_gap("gemini")
_GEMINI_BACKOFF_DELAYS = _BACKOFF_DELAYS



class LLMManager:
    """
    Hot-swappable LLM manager with automatic fallback.

    Parameters
    ----------
    model_name : str
        Key from the MODELS registry.  Default "local-qwen".
    env_path : str
        Path to .env file with API keys.  Default ".env".
    """

    def __init__(
        self,
        model_name: str = "local-qwen",
        env_path: str = ".env",
    ) -> None:
        load_dotenv(env_path, override=False)
        self.logger = TeeLogger(master_log_dir=str(_MASTER_LOG_DIR))

        self.model_name: str = ""
        self._model = None            # LiteLLMModel instance

        # Per-provider throttle tracking  {provider: last_call_monotonic}
        self._provider_last_call: dict[str, float] = {}
        self._provider_lock = threading.Lock()
        # Backward-compat alias
        self._last_gemini_call: float = 0.0
        self._gemini_lock = self._provider_lock

        self._initialize_model(model_name)

        # Record in leaderboard which model is driving this session
        self._stamp_leaderboard(model_name)

    # ── Public API ──────────────────────────────────────────────────────────────

    def get_model(self):
        """Return the active LiteLLMModel for use with Smolagents agents."""
        return self._model

    def call_with_backoff(self, fn, *args, **kwargs):
        """
        Call *fn* with exponential backoff on 429 / 503 / APIError for ANY provider.

        Parameters
        ----------
        fn     : callable to invoke
        *args  : positional arguments forwarded to fn
        **kwargs: keyword arguments forwarded to fn
        """
        import litellm
        delays = _BACKOFF_DELAYS
        for i, delay in enumerate(delays):
            try:
                return fn(*args, **kwargs)
            except (
                litellm.RateLimitError,
                litellm.ServiceUnavailableError,
                litellm.APIError,
            ) as exc:
                if i == len(delays) - 1:
                    raise   # exhausted all retries
                self.logger.warning(
                    f"[Backoff] retry {i+1}/{len(delays)} "
                    f"after {delay}s — {type(exc).__name__}: {str(exc)[:80]}"
                )
                time.sleep(delay)

    def track_tokens(self, provider: str, tokens: int) -> None:
        """
        Record token usage for *provider* in token_budget.json.
        Warns at _WARN_THRESHOLD and logs alert at _SWITCH_THRESHOLD.
        """
        import datetime as _dt
        try:
            _MASTER_LOG_DIR.mkdir(parents=True, exist_ok=True)
            today = _dt.date.today().isoformat()
            budget: dict = {}
            if _TOKEN_BUDGET_PATH.exists():
                try:
                    budget = json.loads(_TOKEN_BUDGET_PATH.read_text())
                except Exception:
                    budget = {}
            # Reset if new day
            if budget.get("date") != today:
                budget = {"date": today, "providers": {}}
            pdata = budget.setdefault("providers", {}).setdefault(
                provider, {"tokens_used": 0, "warned_80": False, "warned_95": False}
            )
            pdata["tokens_used"] += tokens
            used   = pdata["tokens_used"]
            limit  = DAILY_LIMITS.get(provider, 100_000)
            frac   = used / limit if limit else 0.0

            if frac >= _SWITCH_THRESHOLD and not pdata.get("warned_95"):
                pdata["warned_95"] = True
                self.logger.warning(
                    f"[DailyBudget] {provider}: {used:,}/{limit:,} tokens "
                    f"({frac*100:.0f}%) — approaching limit! Consider switching provider."
                )
            elif frac >= _WARN_THRESHOLD and not pdata.get("warned_80"):
                pdata["warned_80"] = True
                self.logger.warning(
                    f"[DailyBudget] {provider}: {used:,}/{limit:,} tokens "
                    f"({frac*100:.0f}%) — 80% of daily limit used."
                )

            _TOKEN_BUDGET_PATH.write_text(json.dumps(budget, indent=2))
        except Exception as e:
            self.logger.warning(f"[DailyBudget] Could not update budget: {e}")

    def get_daily_usage(self, provider: str) -> tuple:
        """Return (tokens_used_today, daily_limit, fraction_used) for *provider*."""
        import datetime as _dt
        today = _dt.date.today().isoformat()
        try:
            if _TOKEN_BUDGET_PATH.exists():
                budget = json.loads(_TOKEN_BUDGET_PATH.read_text())
                if budget.get("date") == today:
                    used  = budget.get("providers", {}).get(provider, {}).get("tokens_used", 0)
                    limit = DAILY_LIMITS.get(provider, 100_000)
                    return used, limit, used / limit if limit else 0.0
        except Exception:
            pass
        return 0, DAILY_LIMITS.get(provider, 100_000), 0.0

    def test_connection(self) -> bool:
        """
        Send a minimal probe prompt and measure latency.

        Returns True if the model responds successfully, False otherwise.
        Logs result to master_terminal.log.
        """
        self.logger.info(
            f"[LLMManager] Testing connection: {self.model_name}"
            f" ({MODELS[self.model_name]}) ..."
        )
        t0 = time.perf_counter()
        try:
            # Build a minimal messages list
            messages = [{"role": "user", "content": "Reply with OK only."}]
            response = self._model(messages)
            latency = time.perf_counter() - t0

            # Extract text from response (handles ChatMessage or str)
            if hasattr(response, "content"):
                reply = str(response.content).strip()
            else:
                reply = str(response).strip()

            self.logger.info(
                f"[LLMManager] Connection OK | latency={latency:.3f}s"
                f" | reply={reply[:80]!r}"
            )
            return True

        except Exception as exc:
            latency = time.perf_counter() - t0
            self.logger.error(
                f"[LLMManager] Connection FAILED | latency={latency:.3f}s"
                f" | error={exc}"
            )
            return False

    def switch_model(self, model_name: str) -> None:
        """
        Hot-swap to a different model mid-session.

        Useful when a free-tier rate limit is hit or a local model is
        unavailable.  Completely replaces the internal LiteLLMModel
        instance so downstream agents pick up the change on the next call.

        Parameters
        ----------
        model_name : str  — key from MODELS registry
        """
        prev = self.model_name
        self.logger.info(
            f"[LLMManager] Switching: {prev} -> {model_name}"
        )
        self._initialize_model(model_name)
        self._stamp_leaderboard(model_name)
        self.logger.info(
            f"[LLMManager] Switch complete. Active: {model_name}"
        )

    def auto_fallback_chain(
        self,
        chain: Optional[List[str]] = None,
        test_each: bool = True,
    ) -> Optional[str]:
        """
        Try models in *chain* order, returning the name of the first one
        that initialises and passes test_connection().

        Parameters
        ----------
        chain     : list of model name keys  (default DEFAULT_FALLBACK_CHAIN)
        test_each : bool — call test_connection() to confirm responsiveness

        Returns
        -------
        str  — name of the winning model, or None if all fail
        """
        chain = chain or DEFAULT_FALLBACK_CHAIN
        self.logger.agent(
            f"[LLMManager] Starting fallback chain: {chain}"
        )

        for name in chain:
            try:
                self._initialize_model(name)
                if test_each and not self.test_connection():
                    self.logger.warning(
                        f"[LLMManager] {name} initialised but test failed."
                    )
                    continue
                self.logger.agent(
                    f"[LLMManager] Fallback resolved -> {name}"
                )
                self._stamp_leaderboard(name)
                return name

            except Exception as exc:
                self.logger.warning(
                    f"[LLMManager] {name} failed to init: {exc}"
                )

        self.logger.error("[LLMManager] All fallback models failed!")
        return None

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _initialize_model(self, model_name: str) -> None:
        """Instantiate a fresh LiteLLMModel for *model_name*."""
        if model_name not in MODELS:
            raise ValueError(
                f"Unknown model key '{model_name}'. "
                f"Available: {list(MODELS.keys())}"
            )

        # Import here so smolagents is not required at module import time
        from smolagents import LiteLLMModel

        model_id = MODELS[model_name]
        provider = model_id.split("/")[0]
        api_key  = self._get_api_key(provider)
        is_gemini = (provider == "gemini")

        kwargs: Dict[str, Any] = {"model_id": model_id}
        if api_key:
            kwargs["api_key"] = api_key

        # Gemini — add rate-limit parameters to stay under free tier (15 RPM)
        if is_gemini:
            kwargs["max_retries"] = 5
            kwargs["rpm"]         = _GEMINI_RPM  # 12 RPM target
            self.logger.info(
                f"[LLMManager] Gemini model: applying rpm={_GEMINI_RPM}, "
                f"max_retries=5, min_call_gap={_GEMINI_MIN_CALL_GAP}s"
            )

        # Ollama — local, no API key, custom base URL
        if provider == "ollama":
            kwargs.setdefault("api_base", "http://localhost:11434")

        # LM Studio — OpenAI-compatible local server, no API key needed.
        # We use provider=openai with a custom api_base pointing to LM Studio.
        # LiteLLM sends the model_id string to the server but LM Studio
        # ignores it and uses whatever model is currently loaded.
        if any(model_name.startswith(p) for p in _LMSTUDIO_PREFIXES):
            kwargs.setdefault("api_base", _LMSTUDIO_BASE_URL)
            kwargs.setdefault("api_key", "lm-studio")  # LM Studio ignores the key

        # Llamafile — OpenAI-compatible server started by LlamafileManager.
        # Default port 8080.  No API key needed.
        if model_name.startswith("llamafile-"):
            kwargs.setdefault("api_base", "http://localhost:8080/v1")
            kwargs.setdefault("api_key", "llamafile")  # server ignores it

        raw_model = LiteLLMModel(**kwargs)

        # Wrap with per-provider throttle for any provider that has a meaningful
        # RPM limit.  Local providers (ollama, openai/llamafile/lmstudio) have
        # rpm=9999 so _min_gap ≈ 0 and the wrapper is essentially a no-op.
        gap = _min_gap(provider)
        if gap > 0.1:  # only install wrapper if gap is non-trivial
            self._model = self._wrap_provider_throttle(raw_model, provider, gap)
        else:
            self._model = raw_model

        self.model_name = model_name

        self.logger.info(
            f"[LLMManager] Active: {model_name!r}  |  "
            f"model_id={model_id}  |  provider={provider}"
        )

    def _wrap_provider_throttle(self, raw_model, provider: str, min_gap: float):
        """
        Return a thin proxy around *raw_model* that enforces *min_gap* seconds
        between consecutive calls for this *provider*.

        The proxy is transparent: all attributes are forwarded via __getattr__.
        Backoff for 429/503/APIError is applied inside via call_with_backoff.
        """
        manager = self
        rpm_display = PROVIDER_RPM_LIMITS.get(provider, "?")

        class _ThrottledModel:
            """Per-provider inter-call gap enforcer."""

            def __call__(self_inner, *args, **kwargs):
                with manager._provider_lock:
                    now  = time.monotonic()
                    last = manager._provider_last_call.get(provider, 0.0)
                    gap  = now - last
                    needed = min_gap - gap
                    if needed > 0:
                        manager.logger.info(
                            f"[Throttle:{provider}] sleeping {needed:.2f}s "
                            f"(target {rpm_display} RPM)"
                        )
                        time.sleep(needed)
                    manager._provider_last_call[provider] = time.monotonic()
                    # keep backward-compat alias
                    if provider == "gemini":
                        manager._last_gemini_call = manager._provider_last_call[provider]

                return manager.call_with_backoff(raw_model, *args, **kwargs)

            def __getattr__(self_inner, name):
                return getattr(raw_model, name)

        return _ThrottledModel()

    # Backward-compat alias
    def _wrap_gemini_throttle(self, raw_model):
        """Deprecated — use _wrap_provider_throttle."""
        return self._wrap_provider_throttle(raw_model, "gemini", _GEMINI_MIN_CALL_GAP)

    @staticmethod
    def _get_api_key(provider: str) -> Optional[str]:
        """Look up the API key env-var for a provider; return None if missing."""
        env_var = _API_KEY_MAP.get(provider)
        if env_var:
            return os.getenv(env_var) or None
        return None

    @staticmethod
    def _stamp_leaderboard(model_name: str) -> None:
        """Write which model is driving the current session into leaderboard.json."""
        lb_path = _MASTER_LOG_DIR / "leaderboard.json"
        if not lb_path.exists():
            return
        try:
            import datetime
            with open(lb_path, "r", encoding="utf-8") as fh:
                lb = json.load(fh)
            lb["agent_model_used"] = model_name
            lb["last_updated"] = datetime.datetime.now().isoformat(
                timespec="seconds"
            )
            with open(lb_path, "w", encoding="utf-8") as fh:
                json.dump(lb, fh, indent=2)
        except Exception:
            pass   # never crash because of leaderboard write


# ── Two-Brain Manager ──────────────────────────────────────────────────────────

class TwoBrainManager:
    """
    Separates the reasoning brain from the coding brain.

    In many scenarios the best arrangement is:
      - A strong cloud reasoning model (Groq Llama, Gemini) for planning,
        analysis, and deciding what to try next  (~300 tokens/step, cheap)
      - A powerful local coder (Qwen2.5-Coder via Ollama) for writing the
        actual model_code  (free, fast, high quality)

    When coding_model is None the class runs in single-brain mode, using
    the reasoning model for everything (identical to plain LLMManager).

    Parameters
    ----------
    reasoning_model : str   — key from MODELS registry (used by agent loop)
    coding_model    : str | None  — key from MODELS (used for code generation)
    env_path        : str   — path to .env file

    Usage
    -----
        tb = TwoBrainManager("groq-llama", "local-qwen")
        code = tb.generate_code("Write a RandomForest with SMOTE...")
    """

    def __init__(
        self,
        reasoning_model: str,
        coding_model: Optional[str] = None,
        env_path: str = ".env",
    ) -> None:
        self.reasoning_model_name = reasoning_model
        self.coding_model_name    = coding_model or reasoning_model
        self._single_brain        = (coding_model is None or
                                     coding_model == reasoning_model)

        self.reasoner = LLMManager(model_name=reasoning_model, env_path=env_path)
        if self._single_brain:
            self.coder = self.reasoner   # same object — no extra cost
        else:
            self.coder = LLMManager(model_name=self.coding_model_name, env_path=env_path)

        self.reasoner.logger.agent(
            f"[TwoBrainManager] reasoning={reasoning_model}  "
            f"coding={self.coding_model_name}  "
            f"single_brain={self._single_brain}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_reasoning_model(self):
        """Return the reasoning LiteLLMModel for the main smolagents agent loop."""
        return self.reasoner.get_model()

    def get_coding_model(self):
        """Return the coding LiteLLMModel (may be same as reasoning in single-brain)."""
        return self.coder.get_model()

    def generate_code(self, prompt: str) -> str:
        """
        Call the coding model DIRECTLY (not via the agent loop) and return
        only the code string.  Strips markdown code fences if present.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Python ML engineer. "
                    "Return ONLY the Python code, no explanations, no markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = self.coder.get_model()(messages)
        code = str(response.content).strip() if hasattr(response, "content") else str(response).strip()
        # Strip markdown fences
        if code.startswith("```"):
            lines = code.splitlines()
            code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return code

    def switch_reasoning_model(self, model_name: str) -> None:
        """Hot-swap the reasoning model mid-session."""
        self.reasoner.switch_model(model_name)
        self.reasoning_model_name = model_name

    def auto_fallback_chain(self, **kwargs) -> Optional[str]:
        """Delegate fallback to the reasoning manager."""
        return self.reasoner.auto_fallback_chain(**kwargs)

    def get_model(self):
        """Compatibility shim — returns reasoning model (so TwoBrainManager
        can be passed anywhere LLMManager is expected)."""
        return self.get_reasoning_model()

