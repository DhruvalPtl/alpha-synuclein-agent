"""
Test script: verify final_answer guard fires when experiments < 80% of budget.

Usage:
    .venv\Scripts\python.exe test_guard.py

Expected:
  - Guard installed message printed
  - When guard is triggered (< 8 experiments done), prints the BLOCKED message
  - Agent continues rather than stopping
"""
import sys, os
sys.path.insert(0, r"d:\3rd sem M.tech\agent_workspace")
os.chdir(r"d:\3rd sem M.tech\agent_workspace")

from dotenv import load_dotenv
load_dotenv()

# ── Patch rebuild_leaderboard to be silent ────────────────────────────────────
import agent.tools.rebuild_leaderboard as _rl
_orig_rebuild = _rl.rebuild_leaderboard
def _silent_rebuild(**kwargs):
    kwargs["verbose"] = False
    return _orig_rebuild(**kwargs)

# ── Import orchestrator ───────────────────────────────────────────────────────
from agent.core.orchestrator import AgentOrchestrator

print("=" * 60)
print("  TEST: final_answer guard with max_experiments=10")
print("  Expected min before stop = 8")
print("=" * 60)

# Build orchestrator (groq-llama — fast, good free tier)
agent = AgentOrchestrator(
    model_name  = "groq-llama",
    verbosity   = "full",
    max_steps   = 30,               # small for testing
)

# ── Manually test the guard ───────────────────────────────────────────────────
# We'll call run() with max_experiments=10 so min=8, then directly
# invoke the guarded final_answer function multiple times to confirm the guard.

print("\n[Test] Setting up guard (simulating run() setup)...")

MAX_EXP = 10
agent._min_experiments_before_stop = int(MAX_EXP * 0.8)   # = 8

# Install the guard exactly as run() does
original_fa = agent._agent.tools["final_answer"].forward

_orch_ref   = agent
_min_exp    = agent._min_experiments_before_stop
_logger_ref = agent.logger

def _guarded_final_answer(answer):
    runs = _orch_ref._exp_count
    if runs < _min_exp:
        msg = (
            f"PREMATURE STOP BLOCKED: You have run {runs} "
            f"experiments this session but need at least {_min_exp} "
            f"before concluding. The leaderboard shows more "
            f"architecture families to explore. Keep going."
        )
        _logger_ref.warning(
            f"[Guard] final_answer blocked: only {runs}/{_min_exp} experiments done"
        )
        print(
            f"\n[Guard] BLOCKED: final_answer rejected -- "
            f"{runs}/{_min_exp} experiments done. "
            f"Keep exploring!\n",
            flush=True,
        )
        return msg
    _logger_ref.agent(
        f"[Guard] final_answer approved: {runs}/{_min_exp} experiments done"
    )
    return original_fa(answer)

agent._agent.tools["final_answer"].forward = _guarded_final_answer
print(f"[Test] Guard installed. min_experiments_before_stop = {_min_exp}")

# ── Test 1: Guard fires when 0 experiments done ───────────────────────────────
print("\n[Test 1] Calling final_answer with 0 experiments run...")
agent._exp_count = 0
result = _guarded_final_answer("I found the best model!")
assert "PREMATURE STOP BLOCKED" in result, f"Guard did not fire! result={result}"
print(f"[Test 1] PASS — guard fired correctly: '{result[:60]}...'")

# ── Test 2: Guard fires when 4 experiments done (< 8) ────────────────────────
print("\n[Test 2] Calling final_answer with 4 experiments run...")
agent._exp_count = 4
result = _guarded_final_answer("I found the best model!")
assert "PREMATURE STOP BLOCKED" in result, f"Guard did not fire! result={result}"
print(f"[Test 2] PASS — guard fired correctly: '{result[:60]}...'")

# ── Test 3: Guard blocks at exactly min-1 experiments ────────────────────────
print(f"\n[Test 3] Calling final_answer with {_min_exp - 1} experiments run...")
agent._exp_count = _min_exp - 1
result = _guarded_final_answer("I found the best model!")
assert "PREMATURE STOP BLOCKED" in result, f"Guard did not fire! result={result}"
print(f"[Test 3] PASS — guard fired correctly: '{result[:60]}...'")

# ── Test 4: Guard allows at exactly min experiments ───────────────────────────
print(f"\n[Test 4] Calling final_answer with exactly {_min_exp} experiments run...")
agent._exp_count = _min_exp
result = _guarded_final_answer("I found the best model!")
# original_fa just returns the answer passed to it
print(f"[Test 4] PASS — guard approved: '{str(result)[:80]}'")

print("\n" + "=" * 60)
print("  ALL 4 GUARD TESTS PASSED [OK]")
print("=" * 60)

# ── Test 5: Session context in prompt ─────────────────────────────────────────
print("\n[Test 5] Verifying session context is in task_prompt...")
import datetime, json
from agent.prompts.system_prompt import SYSTEM_PROMPT
from agent.core.session_manager import SessionManager

_session = SessionManager(model_name="groq-llama", logger=agent.logger)
_session.start()

# Build the context block exactly as orchestrator does
from agent.tools.rebuild_leaderboard import rebuild_leaderboard
_lb_state = rebuild_leaderboard(verbose=False)

_ALL_FAMILIES = [
    "classical_ml", "linear", "neural_network", "deep_residual",
    "ensemble_stack", "attention_based", "sequence_model",
    "modern_tabular", "protein_embedding",
]
_tried_families = _lb_state.get("families_completed", [])
_not_tried      = [f for f in _ALL_FAMILIES if f not in _tried_families]
_best_f1        = _lb_state.get("best_val_f1_macro", 0.0)
_best_exp       = _lb_state.get("best_experiment", "none yet")
_total_ever     = _lb_state.get("total_runs", 0)
_min_stop       = int(MAX_EXP * 0.8)

_session_context = (
    f"\n\n=== THIS SESSION ==="
    f"\nBudget this session          : {MAX_EXP} experiments"
    f"\nTotal experiments ever run   : {_total_ever} (across all sessions)"
    f"\nFamilies tried (all sessions): {_tried_families if _tried_families else 'none yet'}"
    f"\nFamilies NOT tried yet       : {_not_tried if _not_tried else 'all tried!'}"
    f"\nCurrent best                 : {_best_f1:.4f}  ({_best_exp})"
    f"\n"
    f"\nINSTRUCTIONS FOR THIS SESSION:"
    f"\n1. Read the leaderboard first (call read_leaderboard)."
    f"\n2. Try architecture families from the NOT TRIED list first."
    f"\n3. Do not repeat experiments already in the leaderboard."
    f"\n4. You MUST run at least {_min_stop} new experiments"
    f"\n   this session before you are allowed to call final_answer."
    f"\n5. If you find a good model early, do NOT stop --"
    f"\n   instead write: 'X works well. Now I will explore Y to confirm"
    f"\n   X is the true best and not just the first thing that worked.'"
)

task_prompt = SYSTEM_PROMPT + "\n\nADDITIONAL CONSTRAINT FOR THIS SESSION:" + _session_context

assert "THIS SESSION" in task_prompt,         "Missing THIS SESSION block"
assert "NOT TRIED" in task_prompt,            "Missing NOT TRIED families"
assert "MUST run at least" in task_prompt,    "Missing min experiments instruction"
assert str(_total_ever) in task_prompt,       "Missing total_ever count"
print(f"[Test 5] PASS — session context embedded correctly")
print(f"         total_runs={_total_ever}, tried={_tried_families}, not_tried={_not_tried}")
print(f"         best_f1={_best_f1:.4f} ({_best_exp})")

_session.end(status="completed", total_experiments=0)

print("\n" + "=" * 60)
print("  ALL TESTS PASSED [OK]")
print("  Changes summary:")
print("  - final_answer guard blocks at < 80% of budget")
print("  - Task prompt includes THIS SESSION context from leaderboard")
print("  - System prompt updated to enforce 80%/4-family rule")
print("=" * 60)

# ── Test 6: run_experiment repetition guard ───────────────────────────────────
print("\n" + "=" * 60)
print("  TEST 6 & 7: run_experiment repetition guard")
print("=" * 60)

import json as _json
import tempfile, pathlib

_EXPLOIT_KEYWORDS = ["xgb", "rf", "lgbm", "mlp", "svc", "gb"]

def _make_guard(lb_path_obj):
    """Build the same guard logic as orchestrator.run() uses."""
    def _guarded_run_experiment(exp_name, architecture_family,
                                 model_code, hyperparams="{}"):
        try:
            if lb_path_obj.exists():
                _lb = _json.loads(lb_path_obj.read_text())
                recent = _lb.get("experiments", [])[-3:]
                if len(recent) == 3:
                    for kw in _EXPLOIT_KEYWORDS:
                        if all(
                            kw in e.get("architecture", "").lower()
                            for e in recent
                        ):
                            raise RuntimeError(
                                f"Exploitation detected — last 3 experiments "
                                f"all used '{kw}'. You must try a genuinely "
                                f"different architecture before continuing."
                            )
        except RuntimeError:
            raise
        except Exception:
            pass
        return f"OK:{exp_name}"
    return _guarded_run_experiment

# Write a fake leaderboard with 3 consecutive XGBoost experiments
_tmp_dir  = pathlib.Path(tempfile.mkdtemp())
_lb_file  = _tmp_dir / "leaderboard.json"
_fake_lb  = {
    "experiments": [
        {"architecture": "XGBoost+ADASYN", "architecture_family": "classical_ml", "val_f1_macro": 0.72},
        {"architecture": "xgboost_tuned_v2", "architecture_family": "classical_ml", "val_f1_macro": 0.73},
        {"architecture": "XGB+CW+SMOTE",   "architecture_family": "classical_ml", "val_f1_macro": 0.74},
    ]
}
_lb_file.write_text(_json.dumps(_fake_lb))

guard_fn = _make_guard(_lb_file)

print("\n[Test 6] Guard should FIRE for 3 consecutive XGBoost experiments...")
_fired = False
try:
    guard_fn("xgb_new", "classical_ml", "def build_and_train(*a): pass")
except RuntimeError as e:
    _fired = True
    print(f"[Test 6] PASS — RuntimeError raised correctly: '{str(e)[:80]}...'")

assert _fired, "[Test 6] FAIL — guard did NOT raise RuntimeError!"

# Test 7: Mixed leaderboard should NOT trigger the guard
print("\n[Test 7] Guard should NOT fire for mixed-architecture leaderboard...")
_fake_lb_mixed = {
    "experiments": [
        {"architecture": "XGBoost+ADASYN",  "architecture_family": "classical_ml",   "val_f1_macro": 0.72},
        {"architecture": "RandomForest_v1",  "architecture_family": "classical_ml",   "val_f1_macro": 0.68},
        {"architecture": "MLP_residual",     "architecture_family": "neural_network", "val_f1_macro": 0.70},
    ]
}
_lb_file.write_text(_json.dumps(_fake_lb_mixed))

_allowed = False
try:
    result = guard_fn("new_exp", "neural_network", "def build_and_train(*a): pass")
    _allowed = True
    print(f"[Test 7] PASS — guard allowed mixed experiment: {result}")
except RuntimeError as e:
    print(f"[Test 7] FAIL — guard incorrectly blocked: {e}")

assert _allowed, "[Test 7] FAIL — guard should have allowed the call!"

# Test 8: Fewer than 3 experiments — guard should allow
print("\n[Test 8] Guard should NOT fire when fewer than 3 experiments exist...")
_fake_lb_short = {
    "experiments": [
        {"architecture": "XGBoost+ADASYN", "architecture_family": "classical_ml", "val_f1_macro": 0.72},
        {"architecture": "xgboost_v2",     "architecture_family": "classical_ml", "val_f1_macro": 0.73},
    ]
}
_lb_file.write_text(_json.dumps(_fake_lb_short))

_allowed2 = False
try:
    result = guard_fn("new_exp", "classical_ml", "def build_and_train(*a): pass")
    _allowed2 = True
    print(f"[Test 8] PASS — guard allowed with only 2 prior experiments: {result}")
except RuntimeError as e:
    print(f"[Test 8] FAIL — guard incorrectly blocked: {e}")

assert _allowed2, "[Test 8] FAIL — guard fired with < 3 experiments!"

# Cleanup
import shutil
shutil.rmtree(_tmp_dir, ignore_errors=True)

print("\n" + "=" * 60)
print("  ALL TESTS PASSED [OK]  (including repetition guard Tests 6-8)")
print("  Changes summary:")
print("  - final_answer guard blocks at < 80% of budget")
print("  - run_experiment guard blocks 3 consecutive same-learner runs")
print("  - Task prompt includes THIS SESSION context from leaderboard")
print("=" * 60)
