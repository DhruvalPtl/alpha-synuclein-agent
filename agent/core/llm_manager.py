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
    "local-qwen":     "ollama/qwen2.5-coder:32b",
    "local-deepseek": "ollama/deepseek-r1:14b",
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

# ── Gemini rate-limit guard ───────────────────────────────────────────────────
# Free tier: 15 RPM hard limit → we target 12 to leave a safety margin.
_GEMINI_RPM          = 5
_GEMINI_MIN_CALL_GAP = 11.0   # seconds between Gemini API calls (60/5 = 12s → use 11s to avoid rounding)
_GEMINI_BACKOFF_DELAYS = [5, 10, 20, 40, 60]  # seconds before each retry



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

        # Gemini throttle state (shared across calls within this manager)
        self._last_gemini_call: float = 0.0
        self._gemini_lock = threading.Lock()

        self._initialize_model(model_name)

        # Record in leaderboard which model is driving this session
        self._stamp_leaderboard(model_name)

    # ── Public API ──────────────────────────────────────────────────────────────

    def get_model(self):
        """Return the active LiteLLMModel for use with Smolagents agents."""
        return self._model

    def call_with_backoff(self, fn, *args, **kwargs):
        """
        Call *fn* with exponential backoff on Gemini 429 / 503 errors.

        Safe to use for any provider — non-Gemini exceptions propagate
        immediately without any delay.

        Parameters
        ----------
        fn     : callable to invoke
        *args  : positional arguments forwarded to fn
        **kwargs: keyword arguments forwarded to fn
        """
        import litellm
        delays = _GEMINI_BACKOFF_DELAYS
        for i, delay in enumerate(delays):
            try:
                return fn(*args, **kwargs)
            except (litellm.RateLimitError,
                    litellm.ServiceUnavailableError) as exc:
                if i == len(delays) - 1:
                    raise   # exhausted all retries
                self.logger.warning(
                    f"[RateLimit/503] retry {i+1}/{len(delays)} "
                    f"after {delay}s — {type(exc).__name__}: {str(exc)[:80]}"
                )
                time.sleep(delay)

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

        # For Gemini: wrap the model callable so every call is throttled to
        # at most _GEMINI_RPM per minute.  Non-Gemini models are unwrapped.
        if is_gemini:
            self._model = self._wrap_gemini_throttle(raw_model)
        else:
            self._model = raw_model

        self.model_name = model_name

        self.logger.info(
            f"[LLMManager] Active: {model_name!r}  |  "
            f"model_id={model_id}  |  provider={provider}"
        )

    def _wrap_gemini_throttle(self, raw_model):
        """
        Return a thin wrapper around *raw_model* that enforces a minimum
        gap of _GEMINI_MIN_CALL_GAP seconds between consecutive calls.
        The wrapper is transparent: it has the same interface as LiteLLMModel
        (callable + all attributes forwarded via __getattr__).

        Only Gemini calls are throttled; groq / local models bypass this entirely.
        """
        manager = self   # capture for closure

        class _ThrottledModel:
            """Proxy that enforces the 4-second Gemini inter-call gap."""

            def __call__(self_inner, *args, **kwargs):
                with manager._gemini_lock:
                    now    = time.monotonic()
                    gap    = now - manager._last_gemini_call
                    needed = _GEMINI_MIN_CALL_GAP - gap
                    if needed > 0:
                        manager.logger.info(
                            f"[Gemini throttle] sleeping {needed:.2f}s to respect "
                            f"{_GEMINI_RPM} RPM limit"
                        )
                        time.sleep(needed)
                    manager._last_gemini_call = time.monotonic()

                return manager.call_with_backoff(raw_model, *args, **kwargs)

            # Forward all attribute access to the underlying model so that
            # smolagents can read .model_id, .last_input_token_count, etc.
            def __getattr__(self_inner, name):
                return getattr(raw_model, name)

        return _ThrottledModel()

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
