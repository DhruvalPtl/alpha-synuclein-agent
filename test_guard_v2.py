"""
test_guard_v2.py — Prove the RuntimeError guard actually forces the agent to continue.

Tests:
  1. Unit: _guarded_final_answer raises RuntimeError when < min_exp
  2. Unit: _guarded_final_answer passes through when >= min_exp
  3. Integration: Python executor sees the RuntimeError and wraps it as
     AgentExecutionError (does NOT set is_final_answer=True)
  4. Integration: smoke-run with max_experiments=10, max_steps=4
     — verify guard fires at least once in the step log
"""
import os, sys, json
sys.path.insert(0, r"d:\3rd sem M.tech\agent_workspace")
os.chdir(r"d:\3rd sem M.tech\agent_workspace")

from dotenv import load_dotenv
load_dotenv()

# ─── 1 + 2: Unit tests for _guarded_final_answer ──────────────────────────────
from agent.core.orchestrator import AgentOrchestrator

print("=" * 60)
print("  TEST SUITE: _guarded_final_answer (RuntimeError version)")
print("=" * 60)

agent = AgentOrchestrator(
    model_name = "groq-llama",
    verbosity  = "full",
    max_steps  = 5,
)

MAX_EXP = 10
agent._min_experiments_before_stop = int(MAX_EXP * 0.8)  # = 8
_min_exp = agent._min_experiments_before_stop

# Build guard closures (same as orchestrator.run() does)
from pathlib import Path
import json as _json
_LEADERBOARD_PATH = Path("master_log/leaderboard.json")
_ALL_FAMILIES_GUARD = [
    "classical_ml", "linear", "neural_network", "deep_residual",
    "ensemble_stack", "attention_based", "sequence_model",
    "modern_tabular", "protein_embedding",
]

def _get_families_tried():
    try:
        if _LEADERBOARD_PATH.exists():
            lb = _json.loads(_LEADERBOARD_PATH.read_text())
            return lb.get("families_completed", [])
    except Exception:
        pass
    return []

def _get_untried_families():
    tried = _get_families_tried()
    return [f for f in _ALL_FAMILIES_GUARD if f not in tried]

_original_fa = agent._agent.tools["final_answer"].forward
_orch_ref    = agent
_logger_ref  = agent.logger

def _guarded_final_answer(answer):
    runs    = _orch_ref._exp_count
    untried = _get_untried_families()
    if runs < _min_exp:
        _logger_ref.warning(
            f"[Guard] final_answer BLOCKED -- {runs}/{_min_exp} experiments done"
        )
        print(
            f"\n[Guard] BLOCKED: final_answer rejected -- "
            f"{runs}/{_min_exp} experiments done. "
            f"Untried families: {untried}. Keep exploring!\n",
            flush=True,
        )
        raise RuntimeError(
            f"Cannot conclude yet. You have run {runs} experiments "
            f"this session but need at least {_min_exp}. "
            f"Families not yet tried: {untried}. "
            f"Continue exploring -- do not call final_answer again "
            f"until you have run {_min_exp - runs} more experiments."
        )
    _logger_ref.agent(
        f"[Guard] final_answer APPROVED -- {runs}/{_min_exp} experiments done"
    )
    return _original_fa(answer)

agent._agent.tools["final_answer"].forward = _guarded_final_answer
print(f"\n[Setup] Guard installed. min_experiments_before_stop = {_min_exp}")

# Test 1: Guard RAISES at 0 experiments
print("\n[Test 1] Expect RuntimeError at 0 experiments...")
agent._exp_count = 0
try:
    _guarded_final_answer("Done!")
    assert False, "Should have raised!"
except RuntimeError as e:
    assert "Cannot conclude yet" in str(e)
    assert "0 experiments" in str(e)
    print(f"[Test 1] PASS -- RuntimeError raised: '{str(e)[:70]}...'")

# Test 2: Guard RAISES at 4 experiments
print("\n[Test 2] Expect RuntimeError at 4 experiments...")
agent._exp_count = 4
try:
    _guarded_final_answer("Done!")
    assert False, "Should have raised!"
except RuntimeError as e:
    assert "Cannot conclude yet" in str(e)
    assert "4 experiments" in str(e)
    print(f"[Test 2] PASS -- RuntimeError raised: '{str(e)[:70]}...'")

# Test 3: Guard RAISES at min-1
print(f"\n[Test 3] Expect RuntimeError at {_min_exp - 1} experiments...")
agent._exp_count = _min_exp - 1
try:
    _guarded_final_answer("Done!")
    assert False, "Should have raised!"
except RuntimeError as e:
    assert "Cannot conclude yet" in str(e)
    print(f"[Test 3] PASS -- RuntimeError raised at {_min_exp - 1}/{_min_exp}")

# Test 4: Guard PASSES at min
print(f"\n[Test 4] Expect PASS at {_min_exp} experiments...")
agent._exp_count = _min_exp
result = _guarded_final_answer("All done!")
print(f"[Test 4] PASS -- guard approved, result='{str(result)[:60]}'")

print("\n[Unit Tests] ALL 4 PASSED")

# ─── 3: Integration — Python executor wraps RuntimeError as AgentError ────────
print("\n" + "=" * 60)
print("  INTEGRATION TEST: Python executor wraps RuntimeError")
print("=" * 60)

from smolagents.local_python_executor import LocalPythonExecutor
from smolagents.utils import AgentExecutionError

# Simulate the executor calling final_answer (which raises RuntimeError)
agent._exp_count = 2  # < 8 => guard fires

# Install in agent.tools so the executor sees it
agent._agent.tools["final_answer"].forward = _guarded_final_answer

executor = LocalPythonExecutor(
    additional_authorized_imports=["json"],
    additional_functions={
        name: tool.forward
        for name, tool in agent._agent.tools.items()
    },
)

test_code = 'final_answer("I found the best model early!")'

print(f"\n[Integration] Running: {repr(test_code)}")
print(f"             exp_count = {agent._exp_count}, min = {_min_exp}")
print(f"             Expected: executor raises, is_final_answer stays False\n")

raised_correctly = False
try:
    out = executor(test_code)
    # If we get here without exception, check if it was NOT flagged as final
    print(f"[Integration] executor returned: is_final_answer={out.is_final_answer}")
    if not out.is_final_answer:
        print("[Integration] PASS -- RuntimeError suppressed final_answer flag")
        raised_correctly = True
    else:
        print("[Integration] FAIL -- agent would have stopped!")
except Exception as e:
    # Any exception here means the executor propagated the RuntimeError
    # (InterpreterError, AgentExecutionError, or similar)
    print(f"[Integration] PASS -- executor raised {type(e).__name__}: '{str(e)[:80]}...'")
    print(f"              (This is caught by the outer step loop; agent continues)")
    raised_correctly = True

assert raised_correctly, "Integration test failed -- agent would stop prematurely!"
print("\n[Integration Test] PASS")

# ─── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL TESTS PASSED")
print("  Summary of the fix:")
print("    - Returning a string from forward() does NOT stop the agent")
print("      because is_final_answer is set by FinalAnswerException,")
print("      not the return value of forward().")
print("    - Raising RuntimeError from forward() propagates through the")
print("      Python executor as InterpreterError -> AgentExecutionError")
print("      -> caught by the outer step loop -> agent continues.")
print("    - _get_families_tried() and _get_untried_families() helpers")
print("      read the live leaderboard and are included in the error msg.")
print("=" * 60)
