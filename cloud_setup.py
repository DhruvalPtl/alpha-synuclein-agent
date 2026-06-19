#!/usr/bin/env python3
"""
cloud_setup.py — Run this ONCE in the cloud SSH terminal to bootstrap
the entire agent workspace from scratch.

Usage:
    python3 ~/cloud_setup.py

Or paste this URL into the cloud terminal:
    curl -sL https://raw.githubusercontent.com/DhruvalPtl/alpha-synuclein-agent/main/cloud_setup.py | python3
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

GITHUB_REPO   = "https://github.com/DhruvalPtl/alpha-synuclein-agent.git"
CLOUD_INSTALL = Path.home() / "agent_workspace"
PYTHON        = sys.executable

print("=" * 65)
print("  Alpha-Synuclein Agent — Cloud Bootstrap")
print(f"  Python  : {sys.version.split()[0]}  ({PYTHON})")
print(f"  Platform: {platform.system()} {platform.machine()}")
print(f"  Target  : {CLOUD_INSTALL}")
print("=" * 65)

# ── Step 1: Clone or pull ─────────────────────────────────────────────────────
if not (CLOUD_INSTALL / ".git").exists():
    print(f"\n[1/5] Cloning {GITHUB_REPO} ...")
    r = subprocess.run(
        ["git", "clone", GITHUB_REPO, str(CLOUD_INSTALL)],
        text=True
    )
    if r.returncode != 0:
        print(f"ERROR: git clone failed (exit {r.returncode})")
        sys.exit(1)
    print("Clone complete.")
else:
    print(f"\n[1/5] Repo exists at {CLOUD_INSTALL} — pulling latest ...")
    r = subprocess.run(
        ["git", "-C", str(CLOUD_INSTALL), "pull"],
        text=True
    )
    if r.returncode != 0:
        print(f"WARNING: git pull returned exit {r.returncode} — continuing anyway")

# ── Step 2: Install requirements ──────────────────────────────────────────────
req = CLOUD_INSTALL / "requirements.txt"
print(f"\n[2/5] Installing {req} ...")
r = subprocess.run(
    [PYTHON, "-m", "pip", "install", "-q", "-r", str(req)],
    cwd=str(CLOUD_INSTALL),
    text=True
)
if r.returncode != 0:
    print(f"ERROR: pip install failed (exit {r.returncode})")
    sys.exit(1)
print("Requirements installed.")

# ── Step 3: GPU detection ─────────────────────────────────────────────────────
def _has_nvidia():
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False

if _has_nvidia():
    print("\n[3/5] GPU detected — installing PyTorch CUDA 12.1 ...")
    r = subprocess.run([
        PYTHON, "-m", "pip", "install", "-q",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], text=True, cwd=str(CLOUD_INSTALL))
    if r.returncode == 0:
        print("PyTorch CUDA installed.")
    else:
        print("WARNING: CUDA wheel failed — CPU fallback will be used.")
else:
    print("\n[3/5] No GPU detected — skipping CUDA torch install.")

# ── Step 4: Create required directories ──────────────────────────────────────
print("\n[4/5] Creating directory structure ...")
for d in ["master_log", "data/raw", "data/splits", "data/processed",
          "experiments", "sessions"]:
    (CLOUD_INSTALL / d).mkdir(parents=True, exist_ok=True)
    print(f"  {d}/")

# ── Step 5: Create .env if missing ────────────────────────────────────────────
env_path = CLOUD_INSTALL / ".env"
print(f"\n[5/5] Checking .env ...")
if not env_path.exists():
    env_path.write_text(
        "# Fill in the API keys you want to use\n"
        "GEMINI_API_KEY=\n"
        "GROQ_API_KEY=\n"
        "MISTRAL_API_KEY=\n"
        "OPENROUTER_API_KEY=\n"
        "CEREBRAS_API_KEY=\n"
        "# Machine tag — avoids experiment ID collisions with laptop\n"
        "MACHINE_ID=gcloud\n"
    )
    print(f"  Created: {env_path}")
    print("  *** IMPORTANT: edit .env and add your API keys! ***")
else:
    print(f"  Already exists: {env_path}")

# ── Rebuild leaderboard from existing experiments ─────────────────────────────
lb_script = CLOUD_INSTALL / "agent" / "tools" / "rebuild_leaderboard.py"
if lb_script.exists():
    print("\n[extra] Rebuilding leaderboard from disk ...")
    r = subprocess.run(
        [PYTHON, str(lb_script)],
        cwd=str(CLOUD_INSTALL), text=True, capture_output=True
    )
    print("  Leaderboard rebuilt." if r.returncode == 0
          else f"  Leaderboard rebuild skipped ({r.stderr[:80]})")

# ── Show last session status ──────────────────────────────────────────────────
try:
    sys.path.insert(0, str(CLOUD_INSTALL))
    from agent.tools.check_last_session import check_last_session
    print("\n── Last session status ──")
    check_last_session(sessions_dir=CLOUD_INSTALL / "sessions")
except Exception as e:
    print(f"\n  (No session history yet: {e})")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BOOTSTRAP COMPLETE")
print(f"  Project root: {CLOUD_INSTALL}")
print()
print("  Next steps:")
print("  1. cd ~/agent_workspace && nano .env    # add API keys")
print("  2. Open JupyterLab → notebooks/run_agent.ipynb")
print("  3. Run Cell 2 (data), Cell 3 (wall), Cell 4 (launch agent)")
print()
print("  Quick verify: python3 ~/agent_workspace/verify_all.py")
print("=" * 65)
