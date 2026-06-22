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
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

from agent.core.tee_logger import TeeLogger

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS: Dict[str, str] = {
    # ── Cloud models ────────────────────────────────────────────────
    "gemini-flash":   "gemini/gemini-1.5-flash",
    "gemini-pro":     "gemini/gemini-1.5-pro",
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

        self._initialize_model(model_name)

        # Record in leaderboard which model is driving this session
        self._stamp_leaderboard(model_name)

    # ── Public API ──────────────────────────────────────────────────────────────

    def get_model(self):
        """Return the active LiteLLMModel for use with Smolagents agents."""
        return self._model

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

        kwargs: Dict[str, Any] = {"model_id": model_id}
        if api_key:
            kwargs["api_key"] = api_key

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

        self._model    = LiteLLMModel(**kwargs)
        self.model_name = model_name

        self.logger.info(
            f"[LLMManager] Active: {model_name!r}  |  "
            f"model_id={model_id}  |  provider={provider}"
        )

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
