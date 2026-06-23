"""
agent/core/env_config.py
────────────────────────────────────────────────────────────────────────────
Environment auto-detector.

Works whether the kernel is running:
  • Locally on Windows (d:/3rd sem M.tech/agent_workspace/)
  • Remotely on a Google Cloud / Linux instance
  • From inside the notebooks/ sub-directory
  • From the project root

Call setup_environment() at the TOP of every notebook cell before any
other import from the `agent` package.

Usage
-----
    # ONE line at the very top of every notebook cell:
    import sys; sys.path.insert(0, __import__('os').path.abspath('..'))
    # --- OR, more robustly ---
    exec(open('path/to/env_config.py').read())   # fallback

    # Preferred — if agent is importable at all:
    from agent.core.env_config import setup_environment
    PROJECT_ROOT = setup_environment()
"""

import os
import sys
import platform
from pathlib import Path
from typing import Optional

# Lazy import of PlatformDetector to avoid circular deps at module level
_platform_info: Optional[dict] = None

# ── Marker files that prove we are at the project root ────────────────────────
_ROOT_MARKERS = ("requirements.txt", "agent", "data", "notebooks")

# ── Cloud instance details (from your gcloud setup) ───────────────────────────
CLOUD_PROJECT   = "quantkit-project"
CLOUD_INSTANCE  = "quantkit"
CLOUD_ZONE      = "us-east4-b"
CLOUD_REPO_DIR  = "~/agent_workspace"   # where repo lives on the instance

# ── Local Windows project root (used as fallback anchor) ─────────────────────
_LOCAL_WIN_ROOT = r"d:\3rd sem M.tech\agent_workspace"
_LOCAL_WSL_ROOT = "/mnt/d/3rd sem M.tech/agent_workspace"


def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Walk upward from *start* (default: cwd) until all _ROOT_MARKERS exist.

    Search order
    ------------
    1. Exact match walking up from cwd
    2. Hardcoded local Windows/WSL path (dev machine only)
    3. HOME/agent_workspace (common cloud deployment)
    4. HOME/alpha-synuclein-agent (git clone default name)
    5. First ancestor that contains both 'agent/' and 'notebooks/'

    Raises
    ------
    RuntimeError if no root found.
    """
    start = Path(start or os.getcwd()).resolve()

    # ── 1. Walk upward from start ─────────────────────────────────────────────
    for candidate in [start] + list(start.parents):
        if _is_root(candidate):
            return candidate

    # ── 2. Known local Windows/WSL path ───────────────────────────────────────
    for local_path_str in (_LOCAL_WIN_ROOT, _LOCAL_WSL_ROOT):
        local_path = Path(local_path_str)
        if local_path.exists() and _is_root(local_path):
            return local_path

    # ── 3. ~/agent_workspace ─────────────────────────────────────────────────
    home = Path.home()
    for name in ("agent_workspace", "alpha-synuclein-agent", "alpha_synuclein"):
        candidate = home / name
        if candidate.exists() and _is_root(candidate):
            return candidate

    # ── 4. Any ancestor with agent/ + notebooks/ ──────────────────────────────
    for candidate in [start] + list(start.parents):
        if (candidate / "agent").is_dir() and (candidate / "notebooks").is_dir():
            return candidate

    raise RuntimeError(
        f"Cannot find project root.\n"
        f"  Searched from : {start}\n"
        f"  Looking for   : {_ROOT_MARKERS}\n"
        f"  Hint: make sure the repo is cloned at {CLOUD_REPO_DIR} "
        f"on the cloud instance, or that you run from the project directory."
    )


def _is_root(path: Path) -> bool:
    return all((path / m).exists() for m in _ROOT_MARKERS)


def setup_environment(verbose: bool = True) -> Path:
    """
    Auto-detect project root, chdir to it, and patch sys.path.

    Must be called BEFORE any `from agent.xxx import ...` statement.

    Returns
    -------
    Path  — the resolved project root
    """
    root = find_project_root()

    # Change working directory so all relative paths work
    os.chdir(root)

    # Add to Python path
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if verbose:
        # ── Basic environment info ───────────────────────────────────────────
        is_cloud = _detect_cloud()
        env_label = "CLOUD (GCE)" if is_cloud else "LOCAL"
        py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"[env_config] Environment : {env_label}")
        print(f"[env_config] Project root: {root}")
        print(f"[env_config] Python       : {py}  ({sys.executable})")
        print(f"[env_config] Platform     : {platform.system()} {platform.machine()}")
        if is_cloud:
            _print_cloud_info()

        # ── Platform + GPU + model recommendation ─────────────────────────
        global _platform_info
        try:
            from agent.core.platform_detector import PlatformDetector
            _platform_info = PlatformDetector().detect()
            gpu_name = _platform_info.get("gpu_name") or "none"
            vram     = _platform_info.get("gpu_vram_gb")
            vram_str = f"{vram:.0f} GB" if vram else "no GPU"
            model    = _platform_info.get("recommended_model", "?")
            reason   = _platform_info.get("recommended_model_reason", "")
            ollama   = "✅ ollama" if _platform_info.get("has_ollama") else "❌ no ollama"
            plat     = _platform_info.get("platform", "?")
            print(
                f"[env_config] Platform: {plat} | "
                f"GPU: {gpu_name} {vram_str} | {ollama}"
            )
            print(f"[env_config] Recommended model: {model} — {reason}")
        except Exception as pd_exc:
            print(f"[env_config] Platform detection skipped: {pd_exc}")

    return root


def _detect_cloud() -> bool:
    """Return True if running on a GCE / cloud instance."""
    # GCE metadata server is always at this IP
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

    # Fallback: check hostname
    hostname = platform.node().lower()
    cloud_hints = ("compute", "gce", "gcloud", "instance", "quantkit")
    return any(h in hostname for h in cloud_hints)


def _print_cloud_info() -> None:
    """Print GPU and system info on cloud."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            print(f"[env_config] GPU          : {r.stdout.strip()}")
    except Exception:
        pass


def get_paths(root: Optional[Path] = None) -> dict:
    """
    Return a dict of all important project paths relative to root.
    All paths are absolute Path objects, safe to use on any OS.

    Usage
    -----
        from agent.core.env_config import setup_environment, get_paths
        root  = setup_environment()
        paths = get_paths(root)
        print(paths['data_raw'])
    """
    root = root or find_project_root()
    return {
        "root":           root,
        "agent":          root / "agent",
        "data_raw":       root / "data" / "raw",
        "data_splits":    root / "data" / "splits",
        "data_processed": root / "data" / "processed",
        "experiments":    root / "experiments",
        "master_log":     root / "master_log",
        "notebooks":      root / "notebooks",
        "env_file":       root / ".env",
        "requirements":   root / "requirements.txt",
        "leaderboard":    root / "master_log" / "leaderboard.json",
        "csv_raw":        root / "data" / "raw" / "alpha_synuclein.csv",
    }
