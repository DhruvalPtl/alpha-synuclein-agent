"""
patch_notebook_v3.py
Patches notebooks/run_agent.ipynb to add:
  1. NEW cell-03b-dev-reset  (placed between Cell 3 and Cell 4)
     → ipywidgets RESET UI that runs dev_reset()
  2. UPDATE cell-04-launch
     → adds verbosity/watchdog params + "Stop Agent" ipywidgets button
  3. Clears all stale outputs

Run: python patch_notebook_v3.py
"""
import json
from pathlib import Path

NB_PATH = Path("notebooks/run_agent.ipynb")

# ─────────────────────────────────────────────────────────────────────────────
# NEW CELL 3b: Dev-mode reset with ipywidgets confirmation UI
# ─────────────────────────────────────────────────────────────────────────────
CELL_DEV_RESET = {
    "cell_type": "code",
    "execution_count": None,
    "id": "cell-03b-dev-reset",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ╔══════════════════════════════════════════════════════════════════════════════╗\n",
        "# ║  Cell 3b · [DEV RESET]  Wipe experiments + logs for a fresh run           ║\n",
        "# ║  NEVER deletes data/, agent/, .env, or .git/  — mathematical wall safe.   ║\n",
        "# ║  Skip this cell for production — only use during development.             ║\n",
        "# ╚══════════════════════════════════════════════════════════════════════════════╝\n",
        "import sys, os\n",
        "_r = next((p for p in [__import__('pathlib').Path.home()/'agent_workspace',\n",
        "    __import__('pathlib').Path(r'd:\\3rd sem M.tech\\agent_workspace')] if p.exists()), None)\n",
        "if _r and str(_r) not in sys.path: sys.path.insert(0, str(_r))\n",
        "if _r: os.chdir(_r)\n",
        "\n",
        "import ipywidgets as widgets\n",
        "from IPython.display import display, clear_output\n",
        "from agent.tools.dev_reset import dev_reset\n",
        "\n",
        "# ── Widget definitions ────────────────────────────────────────────────────────\n",
        "_confirm_text = widgets.Text(\n",
        "    placeholder='Type RESET to confirm',\n",
        "    layout=widgets.Layout(width='260px'),\n",
        ")\n",
        "_reset_btn = widgets.Button(\n",
        "    description='⚠ Wipe experiments + logs (dev mode)',\n",
        "    button_style='danger',\n",
        "    disabled=True,\n",
        "    layout=widgets.Layout(width='320px'),\n",
        "    tooltip='Only enabled after typing RESET in the text box',\n",
        ")\n",
        "_output = widgets.Output()\n",
        "\n",
        "def _on_text_change(change):\n",
        "    _reset_btn.disabled = (change['new'].strip() != 'RESET')\n",
        "\n",
        "def _on_reset_click(_btn):\n",
        "    _reset_btn.disabled = True\n",
        "    _confirm_text.value = ''\n",
        "    with _output:\n",
        "        clear_output(wait=True)\n",
        "        dev_reset()\n",
        "\n",
        "_confirm_text.observe(_on_text_change, names='value')\n",
        "_reset_btn.on_click(_on_reset_click)\n",
        "\n",
        "_label = widgets.HTML(\n",
        "    '<b style=\"color:#cc0000\">DEV RESET</b> — '\n",
        "    'type <code>RESET</code> then click the button.'\n",
        ")\n",
        "display(widgets.VBox([\n",
        "    _label,\n",
        "    widgets.HBox([_confirm_text, _reset_btn]),\n",
        "    _output,\n",
        "]))\n",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# UPDATED CELL 4: launch cell with Stop button + verbosity/watchdog params
# ─────────────────────────────────────────────────────────────────────────────
CELL_LAUNCH = [
    "# ── Path guard ────────────────────────────────────────────────────────────────\n",
    "import sys, os\n",
    "_r = next((p for p in [__import__('pathlib').Path.home()/'agent_workspace',\n",
    "    __import__('pathlib').Path(r'd:\\3rd sem M.tech\\agent_workspace')] if p.exists()), None)\n",
    "if _r and str(_r) not in sys.path: sys.path.insert(0, str(_r))\n",
    "if _r: os.chdir(_r)\n",
    "\n",
    "# ╔══════════════════════════════════════════════════════════════════════════════╗\n",
    "# ║  Cell 4 · [LAUNCH AGENT]  Start autonomous ML research loop               ║\n",
    "# ║  Concise mode: one line per step in cell; full log → session_log.log      ║\n",
    "# ║  Stop button: graceful stop — current experiment finishes first           ║\n",
    "# ╚══════════════════════════════════════════════════════════════════════════════╝\n",
    "from dotenv import load_dotenv\n",
    "load_dotenv()\n",
    "\n",
    "import ipywidgets as widgets\n",
    "from IPython.display import display\n",
    "from agent.core.orchestrator import AgentOrchestrator\n",
    "\n",
    "# ── Configuration — edit these before launching ───────────────────────────────\n",
    "MODEL_NAME        = 'groq-llama'   # or gemini-flash, local-qwen, cerebras, ...\n",
    "MAX_EXPERIMENTS   = 200            # soft cap injected into the prompt\n",
    "VERBOSITY         = 'concise'      # 'concise' (recommended) | 'full'\n",
    "MAX_IDLE_SECONDS  = 300            # watchdog: stop if stuck for 5 min\n",
    "MAX_TOTAL_SECONDS = 7200           # watchdog: stop after 2 h total\n",
    "\n",
    "# ── Build agent ───────────────────────────────────────────────────────────────\n",
    "agent = AgentOrchestrator(\n",
    "    model_name        = MODEL_NAME,\n",
    "    verbosity         = VERBOSITY,\n",
    "    max_idle_seconds  = MAX_IDLE_SECONDS,\n",
    "    max_total_seconds = MAX_TOTAL_SECONDS,\n",
    ")\n",
    "print('Agent status:', agent.status())\n",
    "\n",
    "# ── Stop button (ipywidgets) ──────────────────────────────────────────────────\n",
    "_stop_btn = widgets.Button(\n",
    "    description='⏹  Stop Agent',\n",
    "    button_style='warning',\n",
    "    layout=widgets.Layout(width='160px', height='36px'),\n",
    "    tooltip='Graceful stop — current experiment finishes first',\n",
    ")\n",
    "_status_lbl = widgets.Label('Status: idle')\n",
    "\n",
    "def _on_stop(_btn):\n",
    "    _stop_btn.disabled = True\n",
    "    _status_lbl.value = 'Status: stop requested — waiting for current step to finish ...'\n",
    "    agent.stop()\n",
    "\n",
    "_stop_btn.on_click(_on_stop)\n",
    "display(widgets.HBox([_stop_btn, _status_lbl]))\n",
    "\n",
    "# ── Launch ────────────────────────────────────────────────────────────────────\n",
    "print(f'\\nStarting autonomous research loop (max {MAX_EXPERIMENTS} experiments) ...')\n",
    "print('Click \\\"Stop Agent\\\" above for a graceful stop.')\n",
    "print('Interrupt kernel (■) for immediate stop (may leave experiment mid-run).\\n')\n",
    "\n",
    "_status_lbl.value = 'Status: running'\n",
    "try:\n",
    "    result = agent.run(\n",
    "        max_experiments   = MAX_EXPERIMENTS,\n",
    "        max_idle_seconds  = MAX_IDLE_SECONDS,\n",
    "        max_total_seconds = MAX_TOTAL_SECONDS,\n",
    "    )\n",
    "    if result:\n",
    "        print('\\nAgent final answer:', result)\n",
    "finally:\n",
    "    _stop_btn.disabled = True\n",
    "    _status_lbl.value  = 'Status: finished'\n",
]

# ─────────────────────────────────────────────────────────────────────────────
# Patch the notebook JSON
# ─────────────────────────────────────────────────────────────────────────────
print(f"Reading {NB_PATH} ...")
nb = json.loads(NB_PATH.read_bytes())

# Clear all stale outputs
for cell in nb["cells"]:
    cell["outputs"] = []
    cell["execution_count"] = None

# Update cell-04-launch source
patched_04 = False
for cell in nb["cells"]:
    if cell.get("id") == "cell-04-launch":
        cell["source"] = CELL_LAUNCH
        patched_04 = True
        print("  cell-04-launch: source updated (verbosity, watchdog, Stop button)")

if not patched_04:
    print("  WARNING: cell-04-launch not found by id — check notebook structure")

# Insert cell-03b-dev-reset AFTER cell-03-verify-wall
new_cells = []
inserted = False
for cell in nb["cells"]:
    new_cells.append(cell)
    if cell.get("id") == "cell-03-verify-wall" and not inserted:
        new_cells.append(CELL_DEV_RESET)
        inserted = True
        print("  cell-03b-dev-reset: inserted after cell-03-verify-wall")

if not inserted:
    # Fallback: insert before cell-04-launch
    final_cells = []
    for cell in new_cells:
        if cell.get("id") == "cell-04-launch" and not inserted:
            final_cells.append(CELL_DEV_RESET)
            inserted = True
            print("  cell-03b-dev-reset: inserted before cell-04-launch (fallback)")
        final_cells.append(cell)
    new_cells = final_cells

nb["cells"] = new_cells

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved: {NB_PATH}")
print("Done.")
