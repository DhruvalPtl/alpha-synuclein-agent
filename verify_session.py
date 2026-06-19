"""
Proof test for SessionManager:
  Test A — normal completion (session ends cleanly, summary says 'completed')
  Test B — simulated crash (KeyboardInterrupt mid-session, summary says 'interrupted')
Prints sessions/ folder tree after each test.
"""
import sys, os, json, time, shutil, pathlib
sys.path.insert(0, '.')

os.environ.setdefault('MACHINE_ID', 'laptop')

from agent.core.tee_logger import TeeLogger
from agent.core.session_manager import SessionManager

# Singleton TeeLogger — needs to be fresh for monkey-patching test
# (reset singleton to allow clean re-init in test)
TeeLogger._instance = None
logger = TeeLogger(master_log_dir='master_log')

def _tree(root: pathlib.Path, prefix='') -> None:
    for p in sorted(root.iterdir()):
        print(prefix + ('├── ' if p != sorted(root.iterdir())[-1] else '└── ') + p.name)
        if p.is_dir():
            _tree(p, prefix + ('│   ' if p != sorted(root.iterdir())[-1] else '    '))


# ═══════════════════════════════════════════════════════════════════════════════
# TEST A — Normal completion
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("TEST A — Normal completion")
print("=" * 65)

sm_a = SessionManager(model_name="groq-llama", logger=logger, heartbeat_interval=2)
sm_a.start()

print(f"Session ID : {sm_a.session_id}")
print(f"Session dir: {sm_a.session_dir}")

# Simulate 2 experiments
sm_a.tick(current_exp="linearsvc_baseline", step=1, status="running")
logger.info("Running experiment 1...")
time.sleep(3)   # let heartbeat fire once

sm_a.tick(current_exp="random_forest_baseline", step=2, status="running")
logger.info("Running experiment 2...")
time.sleep(1)

# Normal end
sm_a.end(status="completed", total_experiments=2)

# Verify files
print()
summary_a = json.loads(sm_a._summary_path.read_text())
print(f"session_summary.json:")
print(f"  session_id   : {summary_a['session_id']}")
print(f"  start_time   : {summary_a['start_time']}")
print(f"  end_time     : {summary_a['end_time']}")
print(f"  final_status : {summary_a['final_status']}")
print(f"  experiments  : {summary_a['total_experiments_this_session']}")
print(f"  model_used   : {summary_a['model_used']}")
print(f"  machine_id   : {summary_a['machine_id']}")

hb_a = json.loads(sm_a._heartbeat_path.read_text())
print(f"\nheartbeat.json:")
print(f"  status             : {hb_a['status']}")
print(f"  last_update        : {hb_a['last_update']}")
print(f"  current_experiment : {hb_a['current_experiment']}")

log_a_lines = sm_a._log_path.read_text(encoding='utf-8').splitlines()
print(f"\nsession_log.log: {len(log_a_lines)} lines")
print(f"  First non-empty: {next((l for l in log_a_lines if l.strip()), '(empty)')[:80]}")

assert summary_a['final_status'] == 'completed', "FAIL: status != completed"
assert summary_a['total_experiments_this_session'] == 2, "FAIL: exp count wrong"
assert sm_a._heartbeat_path.exists(), "FAIL: heartbeat.json missing"
assert sm_a._log_path.exists(), "FAIL: session_log.log missing"
print("\n>>> TEST A: PASS ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST B — Simulated crash (KeyboardInterrupt path)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("TEST B — Simulated crash (KeyboardInterrupt / interrupted)")
print("=" * 65)

# Reset singleton so we can create a fresh logger for the second session
TeeLogger._instance = None
logger_b = TeeLogger(master_log_dir='master_log')

sm_b = SessionManager(model_name="local-qwen", logger=logger_b, heartbeat_interval=2)
sm_b.start()
print(f"Session ID : {sm_b.session_id}")

sm_b.tick(current_exp="xgboost_baseline", step=1, status="running")
logger_b.info("Running xgboost experiment...")
time.sleep(3)

# Simulate the interrupted path (as orchestrator.run() does in its finally block)
fin_status = "interrupted"
fin_error  = "KeyboardInterrupt"
sm_b.end(status=fin_status, total_experiments=1, error_message=fin_error)

summary_b = json.loads(sm_b._summary_path.read_text())
print(f"\nsession_summary.json:")
print(f"  final_status : {summary_b['final_status']}")
print(f"  error_message: {summary_b['error_message']}")
print(f"  experiments  : {summary_b['total_experiments_this_session']}")

assert summary_b['final_status'] == 'interrupted', "FAIL: status != interrupted"
assert summary_b['error_message'] == 'KeyboardInterrupt', "FAIL: error_message wrong"
print("\n>>> TEST B: PASS ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# check_last_session output
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("check_last_session.py output (last 2 sessions)")
print("=" * 65)
from agent.tools.check_last_session import check_last_session
check_last_session(n=2)


# ═══════════════════════════════════════════════════════════════════════════════
# sessions/ folder tree
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("sessions/ folder structure")
print("=" * 65)
sessions_root = pathlib.Path('sessions')
print(f"sessions/")
_tree(sessions_root)

print()
print("=" * 65)
print("ALL TESTS PASSED")
print("=" * 65)
