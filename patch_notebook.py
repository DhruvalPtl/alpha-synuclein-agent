"""
patch_notebook.py
Rewrites Cell 1 of notebooks/run_agent.ipynb to fix:
  1. Uses (CLOUD_INSTALL / '.git').exists() instead of CLOUD_INSTALL.exists()
     so a broken/empty directory doesn't skip the clone step.
  2. Adds pip install -e . for kernel-persistent imports.
  3. Clears all stale cached outputs so the notebook shows clean.
Run: python patch_notebook.py
"""
import json
from pathlib import Path

NB_PATH = Path("notebooks/run_agent.ipynb")

NEW_CELL1_SOURCE = [
    "# ╔══════════════════════════════════════════════════════════════════════════════╗\n",
    "# ║  Cell 1 · [BOOTSTRAP]  Auto-setup: Local Windows + Google Cloud (GCE)     ║\n",
    "# ║  Run this cell FIRST every session — clones/pulls repo & installs deps.   ║\n",
    "# ╚══════════════════════════════════════════════════════════════════════════════╝\n",
    "import os, sys, subprocess, platform\n",
    "from pathlib import Path\n",
    "\n",
    "GITHUB_REPO   = 'https://github.com/DhruvalPtl/alpha-synuclein-agent.git'\n",
    "CLOUD_INSTALL = Path.home() / 'agent_workspace'\n",
    "LOCAL_WIN     = Path(r'd:\\3rd sem M.tech\\agent_workspace')\n",
    "\n",
    "def _on_cloud():\n",
    "    try:\n",
    "        import urllib.request\n",
    "        req = urllib.request.Request(\n",
    "            'http://metadata.google.internal/',\n",
    "            headers={'Metadata-Flavor': 'Google'}\n",
    "        )\n",
    "        urllib.request.urlopen(req, timeout=0.8)\n",
    "        return True\n",
    "    except Exception:\n",
    "        pass\n",
    "    return not LOCAL_WIN.exists()\n",
    "\n",
    "IS_CLOUD = _on_cloud()\n",
    "print(f'Environment : {\"GOOGLE CLOUD (GCE)\" if IS_CLOUD else \"LOCAL\"}')\n",
    "print(f'Platform    : {platform.system()} {platform.machine()}')\n",
    "print(f'Python      : {sys.version.split()[0]}  ({sys.executable})')\n",
    "\n",
    "if IS_CLOUD:\n",
    "    # ── KEY FIX: check for .git not just directory ─────────────────────────────\n",
    "    # A directory can exist without being a git repo (e.g. after instance reset\n",
    "    # where /home persists but the repo was deleted, or a partial previous run).\n",
    "    # Always check (CLOUD_INSTALL / '.git').exists() not CLOUD_INSTALL.exists().\n",
    "    if not (CLOUD_INSTALL / '.git').exists():\n",
    "        print(f'\\nNo git repo at {CLOUD_INSTALL} — cloning ...')\n",
    "        if CLOUD_INSTALL.exists() and any(CLOUD_INSTALL.iterdir()):\n",
    "            print(f'  (directory exists but is not a git repo — removing it first)')\n",
    "            import shutil\n",
    "            shutil.rmtree(CLOUD_INSTALL)\n",
    "        r = subprocess.run(\n",
    "            ['git', 'clone', GITHUB_REPO, str(CLOUD_INSTALL)],\n",
    "            text=True, capture_output=True\n",
    "        )\n",
    "        if r.returncode != 0:\n",
    "            raise RuntimeError(\n",
    "                f'git clone failed (exit {r.returncode}):\\n{r.stderr}\\n'\n",
    "                'Check your internet connection on the cloud instance.'\n",
    "            )\n",
    "        print('Clone complete.')\n",
    "    else:\n",
    "        print(f'\\nRepo found at {CLOUD_INSTALL} — pulling latest ...')\n",
    "        r = subprocess.run(\n",
    "            ['git', '-C', str(CLOUD_INSTALL), 'pull'],\n",
    "            text=True, capture_output=True\n",
    "        )\n",
    "        out = r.stdout.strip() or r.stderr.strip()\n",
    "        print(out)\n",
    "        if r.returncode != 0:\n",
    "            print('WARNING: git pull failed — continuing with existing local copy.')\n",
    "        print('Rebuilding leaderboard from disk after pull ...')\n",
    "        rb_r = subprocess.run(\n",
    "            [sys.executable, '-m', 'agent.tools.rebuild_leaderboard'],\n",
    "            capture_output=True, text=True, cwd=str(CLOUD_INSTALL)\n",
    "        )\n",
    "        print('Leaderboard rebuilt.' if rb_r.returncode == 0\n",
    "              else f'Leaderboard rebuild skipped: {rb_r.stderr[:100]}')\n",
    "    PROJECT_ROOT = CLOUD_INSTALL\n",
    "else:\n",
    "    PROJECT_ROOT = LOCAL_WIN\n",
    "    print(f'\\nLocal workspace: {PROJECT_ROOT}')\n",
    "\n",
    "os.chdir(PROJECT_ROOT)\n",
    "if str(PROJECT_ROOT) not in sys.path:\n",
    "    sys.path.insert(0, str(PROJECT_ROOT))\n",
    "print(f'Working dir : {os.getcwd()}')\n",
    "\n",
    "def _has_nvidia():\n",
    "    try:\n",
    "        return subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0\n",
    "    except FileNotFoundError:\n",
    "        return False\n",
    "\n",
    "if IS_CLOUD and _has_nvidia():\n",
    "    print('\\nGPU detected — installing PyTorch CUDA 12.1 ...')\n",
    "    r = subprocess.run([\n",
    "        sys.executable, '-m', 'pip', 'install', '-q',\n",
    "        'torch', 'torchvision', 'torchaudio',\n",
    "        '--index-url', 'https://download.pytorch.org/whl/cu121'\n",
    "    ], capture_output=True, text=True)\n",
    "    print('PyTorch CUDA OK.' if r.returncode == 0\n",
    "          else f'CUDA wheel failed, CPU fallback: {r.stderr[:200]}')\n",
    "elif IS_CLOUD:\n",
    "    print('\\nNo GPU — CPU PyTorch will install via requirements.')\n",
    "\n",
    "print('\\nInstalling requirements.txt ...')\n",
    "r = subprocess.run(\n",
    "    [sys.executable, '-m', 'pip', 'install', '-q', '-r',\n",
    "     str(PROJECT_ROOT / 'requirements.txt')],\n",
    "    text=True, capture_output=True\n",
    ")\n",
    "if r.returncode != 0:\n",
    "    print('[STDERR]', r.stderr[-2000:])\n",
    "    raise RuntimeError('pip install failed — see above')\n",
    "print('Requirements OK.')\n",
    "\n",
    "# ── Install package in editable mode ──────────────────────────────────────────\n",
    "# This makes 'import agent' work from ANY cell even after kernel restart.\n",
    "# It writes a .pth file to site-packages — a one-time permanent install.\n",
    "print('\\nInstalling package (editable) so import agent works everywhere ...')\n",
    "r = subprocess.run(\n",
    "    [sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT_ROOT)],\n",
    "    text=True, capture_output=True, cwd=str(PROJECT_ROOT)\n",
    ")\n",
    "if r.returncode == 0:\n",
    "    print('Package installed — import agent now works from any cell.')\n",
    "else:\n",
    "    print(f'Editable install skipped (fallback to sys.path): {r.stderr[:100]}')\n",
    "\n",
    "for d in ['master_log', 'data/raw', 'data/splits', 'data/processed', 'experiments', 'sessions']:\n",
    "    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "env_path = PROJECT_ROOT / '.env'\n",
    "if not env_path.exists():\n",
    "    env_path.write_text(\n",
    "        '# Fill in the API keys you want to use\\n'\n",
    "        'GEMINI_API_KEY=\\nGROQ_API_KEY=\\nMISTRAL_API_KEY=\\n'\n",
    "        'OPENROUTER_API_KEY=\\nCEREBRAS_API_KEY=\\n'\n",
    "        '# Machine tag (avoids experiment ID collisions across machines)\\n'\n",
    "        'MACHINE_ID=gcloud\\n'\n",
    "    )\n",
    "    print(f'\\nCreated .env at {env_path}  <-- edit MACHINE_ID and API keys!')\n",
    "else:\n",
    "    print(f'\\n.env found at {env_path}')\n",
    "\n",
    "if IS_CLOUD and _has_nvidia():\n",
    "    r = subprocess.run(\n",
    "        ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',\n",
    "         '--format=csv,noheader'],\n",
    "        capture_output=True, text=True\n",
    "    )\n",
    "    print(f'\\nGPU : {r.stdout.strip()}')\n",
    "\n",
    "import os as _os\n",
    "machine_id = _os.environ.get('MACHINE_ID', '') or platform.node()\n",
    "print(f'\\nMACHINE_ID  : {machine_id}  (experiments tagged with this)')\n",
    "\n",
    "# Show last session status immediately after pull\n",
    "try:\n",
    "    from agent.tools.check_last_session import check_last_session\n",
    "    print('\\n── Last session status ──')\n",
    "    check_last_session()\n",
    "except Exception as _e:\n",
    "    print(f'  (no session history yet: {_e})')\n",
    "\n",
    "print('\\n' + '='*60)\n",
    "print('  BOOTSTRAP COMPLETE')\n",
    "print(f'  Root    : {PROJECT_ROOT}')\n",
    "print(f'  Cloud   : {IS_CLOUD}')\n",
    "print('  --> Run Cell 2 to build / verify the data pipeline.')\n",
    "print('='*60)",
]

print(f"Reading {NB_PATH} ...")
nb = json.loads(NB_PATH.read_bytes())

fixed = 0
cleared = 0
for cell in nb["cells"]:
    # Clear ALL stale cached outputs
    if cell.get("outputs"):
        cell["outputs"] = []
        cell["execution_count"] = None
        cleared += 1

    # Rewrite Cell 1
    if cell.get("id") == "cell-01-bootstrap":
        cell["source"] = NEW_CELL1_SOURCE
        print("  Cell 1 (cell-01-bootstrap): source rewritten")
        fixed += 1

    # Add path guard to every other cell that imports agent
    if cell.get("id") in ("cell-02-verify-data",
                           "cell-03-verify-wall",
                           "cell-04-launch-agent",
                           "cell-05-dashboard"):
        src = cell.get("source", [])
        guard = (
            "# ── Path guard: works even if Cell 1 hasn't run (import agent installed via pip install -e .) ──\n"
            "import sys, os; _r = next((p for p in [__import__('pathlib').Path.home()/'agent_workspace',\n"
            "    __import__('pathlib').Path(r'd:\\3rd sem M.tech\\agent_workspace')] if p.exists()), None)\n"
            "if _r and str(_r) not in sys.path: sys.path.insert(0, str(_r))\n"
            "if _r: os.chdir(_r)\n"
            "\n"
        )
        # Only add if guard not already present
        src_joined = "".join(src)
        if "Path guard" not in src_joined:
            cell["source"] = [guard] + list(src)
            print(f"  Cell {cell['id']}: path guard added")

print(f"\nCells patched: {fixed}   Outputs cleared: {cleared}")

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved: {NB_PATH}")
print("Done. Run: git add notebooks/run_agent.ipynb && git commit -m 'Fix Cell1: .git check, editable install, path guards' && git push")
