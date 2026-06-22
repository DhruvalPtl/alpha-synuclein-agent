"""
agent/core/llamafile_manager.py
────────────────────────────────────────────────────────────────────────────
Manages a local llama.cpp / llamafile server for Colab and Kaggle
environments where Ollama is not available.

Downloads a GGUF model from HuggingFace, starts an OpenAI-compatible
HTTP server on localhost:8080, and returns the base_url for LLMManager.

Usage
-----
    from agent.core.llamafile_manager import LlamafileManager
    mgr = LlamafileManager("qwen-14b-coder")
    base_url = mgr.start()    # blocks until server is ready
    # Then in LLMManager: api_base=base_url

Notes
-----
- Requires: huggingface_hub, llama-cpp-python (or llamafile binary)
- If llama-cpp-python is not installed, falls back to the llamafile
  single-file executable if it exists at ~/llamafile or ./llamafile.
- If nothing is available, raises LlamafileNotAvailableError with
  a clear install message.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Optional


# ── Supported GGUF models (HuggingFace repo/filename) ─────────────────────────
GGUF_MODELS: dict[str, str] = {
    "qwen-14b-coder": (
        "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF/"
        "qwen2.5-coder-14b-instruct-q4_k_m.gguf"
    ),
    "qwen-32b-coder": (
        "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/"
        "qwen2.5-coder-32b-instruct-q4_k_m.gguf"
    ),
}

# ── Server settings ────────────────────────────────────────────────────────────
_DEFAULT_PORT    = 8080
_DEFAULT_N_GPU   = -1      # -1 = all GPU layers
_DEFAULT_N_CTX   = 8192    # context window
_SERVER_TIMEOUT  = 120     # seconds to wait for server to become ready
_CACHE_DIR       = Path.home() / ".cache" / "llamafile_models"


class LlamafileNotAvailableError(RuntimeError):
    """Raised when neither llama-cpp-python nor llamafile binary is available."""
    pass


class LlamafileManager:
    """
    Download and serve a GGUF model via an OpenAI-compatible HTTP server.

    Parameters
    ----------
    model_key : str
        Key from GGUF_MODELS dict.  e.g. "qwen-14b-coder".
    port : int
        Local port for the HTTP server (default 8080).
    n_gpu_layers : int
        GPU layers to offload (-1 = all).
    """

    def __init__(
        self,
        model_key:    str = "qwen-14b-coder",
        port:         int = _DEFAULT_PORT,
        n_gpu_layers: int = _DEFAULT_N_GPU,
    ) -> None:
        if model_key not in GGUF_MODELS:
            raise ValueError(
                f"Unknown model key {model_key!r}. "
                f"Available: {list(GGUF_MODELS.keys())}"
            )
        self.model_key    = model_key
        self.port         = port
        self.n_gpu_layers = n_gpu_layers
        self._proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._model_path: Optional[Path] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def start(self) -> str:
        """
        Download model (if needed) and start the server.

        Returns
        -------
        str  — base_url of the OpenAI-compatible server
        """
        self._model_path = self._download_model()
        self._launch_server()
        self._wait_until_ready()
        return self.base_url

    def stop(self) -> None:
        """Terminate the server process."""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=10)
            print(f"[LlamafileManager] Server stopped (port={self.port})")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _download_model(self) -> Path:
        """Download GGUF from HuggingFace if not already cached."""
        repo_and_file = GGUF_MODELS[self.model_key]
        parts     = repo_and_file.split("/")
        # Format: "Owner/Repo/filename.gguf"
        repo_id   = "/".join(parts[:2])
        filename  = parts[2]
        cache_path = _CACHE_DIR / filename

        if cache_path.exists():
            print(f"[LlamafileManager] Using cached model: {cache_path}")
            return cache_path

        print(f"[LlamafileManager] Downloading {filename} from {repo_id} …")
        print(f"  This may take 10-30 minutes depending on your connection.")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise LlamafileNotAvailableError(
                "huggingface_hub is required to download GGUF models.\n"
                "Install it with:  pip install huggingface_hub"
            )

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id   = repo_id,
            filename  = filename,
            local_dir = str(_CACHE_DIR),
        )
        print(f"[LlamafileManager] Download complete: {downloaded}")
        return Path(downloaded)

    def _launch_server(self) -> None:
        """Start the llama.cpp server in the background."""
        cmd = self._build_server_cmd()
        print(f"[LlamafileManager] Starting server: {' '.join(str(c) for c in cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        # Drain output in background thread so the pipe doesn't fill up
        threading.Thread(
            target=self._drain_output,
            daemon=True,
        ).start()

    def _build_server_cmd(self) -> list:
        """Return the command to start the server."""
        # Prefer llama-cpp-python's built-in server
        try:
            import llama_cpp  # noqa: F401
            return [
                sys.executable, "-m", "llama_cpp.server",
                "--model",         str(self._model_path),
                "--n_gpu_layers",  str(self.n_gpu_layers),
                "--n_ctx",         str(_DEFAULT_N_CTX),
                "--port",          str(self.port),
                "--host",          "127.0.0.1",
            ]
        except ImportError:
            pass

        # Fallback: llamafile binary
        for candidate in [
            Path.home() / "llamafile",
            Path("./llamafile"),
            Path("/usr/local/bin/llamafile"),
        ]:
            if candidate.exists():
                return [
                    str(candidate),
                    "--model",    str(self._model_path),
                    "--port",     str(self.port),
                    "--host",     "127.0.0.1",
                    "-ngl",       str(self.n_gpu_layers if self.n_gpu_layers > 0 else 99),
                    "--ctx-size", str(_DEFAULT_N_CTX),
                    "--server",
                ]

        raise LlamafileNotAvailableError(
            "Neither llama-cpp-python nor a llamafile binary was found.\n"
            "Install one of:\n"
            "  pip install llama-cpp-python[server]   (recommended)\n"
            "  Download llamafile from https://github.com/Mozilla-Ocho/llamafile/releases"
        )

    def _wait_until_ready(self) -> None:
        """Poll the server health endpoint until it responds or times out."""
        import urllib.request
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.time() + _SERVER_TIMEOUT
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError(
                    f"[LlamafileManager] Server process exited unexpectedly "
                    f"(code={self._proc.returncode})"
                )
            try:
                urllib.request.urlopen(url, timeout=2)
                print(f"[LlamafileManager] Server ready at {self.base_url}")
                return
            except Exception:
                time.sleep(2)
        raise TimeoutError(
            f"[LlamafileManager] Server did not start within {_SERVER_TIMEOUT}s"
        )

    def _drain_output(self) -> None:
        """Read and print server stdout/stderr in a background thread."""
        if self._proc and self._proc.stdout:
            for line in self._proc.stdout:
                print(f"  [llamafile] {line.rstrip()}", flush=True)
