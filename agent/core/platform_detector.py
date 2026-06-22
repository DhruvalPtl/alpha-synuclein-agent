"""
agent/core/platform_detector.py
────────────────────────────────────────────────────────────────────────────
Auto-detects the execution platform and recommends the best LLM to use.

Supports: Colab, Kaggle, IIT/quantkit server, laptop, gcloud.

Usage
-----
    from agent.core.platform_detector import PlatformDetector
    info = PlatformDetector().detect()
    print(info["platform"], info["recommended_model"])
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Optional


# ── Model recommendation thresholds (VRAM in GB) ──────────────────────────────
_VRAM_TIERS = [
    (8,   "groq-llama",      "VRAM < 8 GB — use Groq cloud API (free tier)"),
    (16,  "llamafile-14b",   "VRAM 8-16 GB — run Qwen2.5-Coder-14B locally via llamafile"),
    (24,  "local-qwen",      "VRAM 16-24 GB — run Qwen2.5-Coder-32B via Ollama"),
    (999, "local-qwen",      "VRAM > 24 GB — run Qwen2.5-Coder-32B at full precision via Ollama"),
]

# Kaggle 2×T4 gives ~32 GB combined, recommend 32B GGUF
_KAGGLE_VRAM_THRESHOLD = 28   # GB total


class PlatformDetector:
    """
    Detect the execution platform and recommend an appropriate LLM.

    Returns
    -------
    dict with keys:
        platform               : str  — "colab" | "kaggle" | "iit_server" | "laptop" | "gcloud"
        gpu_vram_gb            : float or None
        gpu_name               : str or None
        has_ollama             : bool
        recommended_model      : str  — key from llm_manager.MODELS
        recommended_model_reason : str
    """

    def detect(self) -> dict:
        plat      = self._detect_platform()
        gpu_name, vram = self._detect_gpu()
        has_ollama = self._check_ollama()
        model, reason = self._recommend_model(plat, vram, has_ollama)

        return {
            "platform":                 plat,
            "gpu_vram_gb":              vram,
            "gpu_name":                 gpu_name,
            "has_ollama":               has_ollama,
            "recommended_model":        model,
            "recommended_model_reason": reason,
        }

    # ── Platform detection ────────────────────────────────────────────────────

    def _detect_platform(self) -> str:
        # Colab: /content exists and google.colab is importable
        if os.path.isdir("/content"):
            try:
                import google.colab  # noqa: F401
                return "colab"
            except ImportError:
                pass

        # Kaggle: /kaggle dir exists or proxy token env var set
        if os.path.isdir("/kaggle") or os.environ.get("KAGGLE_DATA_PROXY_TOKEN"):
            return "kaggle"

        # IIT/quantkit server: env var or hostname hint
        machine_id = os.environ.get("MACHINE_ID", "").lower()
        hostname   = platform.node().lower()
        if "quantkit" in machine_id or "quantkit" in hostname:
            return "iit_server"

        # GCloud: metadata server reachable or cloud-like hostname
        if self._is_gcloud():
            return "gcloud"

        # Default: local laptop / workstation
        return "laptop"

    @staticmethod
    def _is_gcloud() -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(
                "http://metadata.google.internal/",
                headers={"Metadata-Flavor": "Google"},
            )
            urllib.request.urlopen(req, timeout=0.5)
            return True
        except Exception:
            pass
        cloud_hints = ("compute", "gce", "gcloud", "instance")
        return any(h in platform.node().lower() for h in cloud_hints)

    # ── GPU detection ─────────────────────────────────────────────────────────

    @staticmethod
    def _detect_gpu() -> tuple[Optional[str], Optional[float]]:
        """Return (gpu_name, vram_gb) or (None, None) if no GPU found."""
        # Method 1: nvidia-smi (fastest, no Python deps)
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
                if lines:
                    # Sum VRAM across all GPUs (Kaggle 2×T4 etc.)
                    total_vram = 0.0
                    first_name = ""
                    names = []
                    for line in lines:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 2:
                            name = parts[0]
                            vram = float(parts[1]) / 1024   # MiB → GB
                            names.append(name)
                            if not first_name:
                                first_name = name
                            total_vram += vram
                    gpu_name = f"{first_name}" + (
                        f" ×{len(names)}" if len(names) > 1 else ""
                    )
                    return gpu_name, round(total_vram, 1)
        except (FileNotFoundError, Exception):
            pass

        # Method 2: torch (if installed)
        try:
            import torch
            if torch.cuda.is_available():
                total = sum(
                    torch.cuda.get_device_properties(i).total_memory
                    for i in range(torch.cuda.device_count())
                )
                name = torch.cuda.get_device_name(0)
                return name, round(total / (1024 ** 3), 1)
        except Exception:
            pass

        return None, None

    # ── Ollama check ──────────────────────────────────────────────────────────

    @staticmethod
    def _check_ollama() -> bool:
        """Return True if an Ollama server is reachable at localhost:11434."""
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
            return True
        except Exception:
            pass
        # Also check if the binary exists even if server is not running
        try:
            r = subprocess.run(
                ["ollama", "--version"],
                capture_output=True, timeout=3,
            )
            return r.returncode == 0
        except (FileNotFoundError, Exception):
            return False

    # ── Model recommendation ──────────────────────────────────────────────────

    @staticmethod
    def _recommend_model(
        plat: str,
        vram: Optional[float],
        has_ollama: bool,
    ) -> tuple[str, str]:
        """Return (model_key, reason) based on platform and GPU."""

        # Colab: no Ollama, but can run llamafile
        if plat == "colab":
            if vram and vram >= 14:
                return (
                    "llamafile-14b",
                    "Colab T4 (≥14 GB) — run Qwen2.5-Coder-14B via llamafile server"
                )
            return (
                "groq-llama",
                "Colab — no local server available; use Groq cloud API"
            )

        # Kaggle: 2×T4 → 32B GGUF; 1×T4 → 14B GGUF
        if plat == "kaggle":
            if vram and vram >= _KAGGLE_VRAM_THRESHOLD:
                return (
                    "llamafile-32b",
                    f"Kaggle 2×T4 ({vram:.0f} GB combined) — run Qwen2.5-Coder-32B-Q4 via llamafile"
                )
            return (
                "llamafile-14b",
                f"Kaggle 1×T4 ({vram or 0:.0f} GB) — run Qwen2.5-Coder-14B-Q4 via llamafile"
            )

        # If Ollama is available and VRAM allows, prefer it
        if has_ollama and vram:
            if vram >= 16:
                return (
                    "local-qwen",
                    f"Ollama available, {vram:.0f} GB VRAM — Qwen2.5-Coder:32b fits comfortably"
                )

        # VRAM-based tiers for any other platform
        if vram is not None:
            for threshold, model, reason in _VRAM_TIERS:
                if vram < threshold:
                    # Adjust reason text
                    full_reason = f"{reason} (detected {vram:.1f} GB VRAM)"
                    # If suggesting llamafile but Ollama is present, prefer Ollama
                    if model.startswith("llamafile") and has_ollama:
                        if vram >= 8:
                            return (
                                "local-qwen",
                                f"Ollama available and {vram:.1f} GB VRAM — using Ollama instead of llamafile"
                            )
                    return model, full_reason

        # No GPU at all → API only
        return (
            "groq-llama",
            "No GPU detected — use Groq cloud API (generous free tier)"
        )
