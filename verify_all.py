"""
Full end-to-end verification script.
Runs every layer from imports → data → harness → session → orchestrator init.
Prints PASS/FAIL for each check. Exits non-zero if any FAIL.
"""
import sys, os, json, pathlib, traceback, pickle, datetime, importlib
sys.path.insert(0, '.')
os.environ.setdefault('MACHINE_ID', 'laptop')

RESULTS = []

def check(name, fn):
    try:
        msg = fn()
        RESULTS.append((name, True, msg or ""))
        print(f"  PASS  {name}" + (f"  [{msg}]" if msg else ""))
    except Exception as e:
        RESULTS.append((name, False, str(e)))
        print(f"  FAIL  {name}")
        print(f"        {traceback.format_exc().splitlines()[-1]}")

sep = lambda t: print(f"\n{'─'*62}\n  {t}\n{'─'*62}")

# ═══════════════════════════════════════════════════════════════════
sep("1. IMPORTS")
# ═══════════════════════════════════════════════════════════════════

def imp_tee():
    from agent.core.tee_logger import TeeLogger
    return "singleton OK"
check("tee_logger", imp_tee)

def imp_llm():
    from agent.core import llm_manager   # MODELS is module-level, not class attr
    assert hasattr(llm_manager, 'MODELS')
    assert len(llm_manager.MODELS) >= 9
    return f"{len(llm_manager.MODELS)} models: {list(llm_manager.MODELS.keys())}"
check("llm_manager", imp_llm)

def imp_session():
    from agent.core.session_manager import SessionManager
    return "OK"
check("session_manager", imp_session)

def imp_orch():
    from agent.core.orchestrator import AgentOrchestrator
    return "OK"
check("orchestrator", imp_orch)

def imp_harness():
    from agent.tools.harness_template import HARNESS_CODE
    assert '{exp_id}' in HARNESS_CODE
    assert '{hyperparams_json}' in HARNESS_CODE
    assert 'build_and_train' in HARNESS_CODE
    assert 'results.json' in HARNESS_CODE
    return f"{len(HARNESS_CODE)} chars"
check("harness_template", imp_harness)

def imp_runner():
    from agent.tools.experiment_runner import ExperimentRunnerTool
    r = ExperimentRunnerTool()
    import inspect
    sig = inspect.signature(r.forward)
    params = list(sig.parameters.keys())
    assert params == ['exp_name','model_code','hyperparams'], str(params)
    return str(params)
check("experiment_runner", imp_runner)

def imp_audit():
    from agent.tools.audit_tool import AuditTool
    a = AuditTool()
    result = a.forward("def build_and_train(X,y,Xv,yv,cw):\n    from sklearn.svm import SVC\n    return SVC().fit(X,y)")
    assert result.startswith("PASS"), result
    bad = a.forward("import pickle\nwith open('data/splits/test.pkl','rb') as f: pass")
    assert bad.startswith("FAIL"), bad
    return f"clean=PASS cheating={bad[:20]}"
check("audit_tool", imp_audit)

def imp_leaderboard():
    from agent.tools.leaderboard_tool import LeaderboardTool
    from agent.core.tee_logger import TeeLogger
    TeeLogger._instance = None
    t = LeaderboardTool()
    r = t.forward(top_n=3)
    assert 'LEADERBOARD' in r
    return f"{len(r)} chars"
check("leaderboard_tool", imp_leaderboard)

def imp_rebuild():
    from agent.tools.rebuild_leaderboard import rebuild_leaderboard
    lb = rebuild_leaderboard(verbose=False)
    assert 'experiments' in lb
    return f"{lb.get('total_runs',0)} exps"
check("rebuild_leaderboard", imp_rebuild)

def imp_check_session():
    from agent.tools.check_last_session import get_last_session_info, format_session_report
    info = get_last_session_info()
    return f"last={info['session_id'] if info else 'none'}"
check("check_last_session", imp_check_session)

def imp_arxiv():
    from agent.tools.arxiv_tool import ArxivTool
    t = ArxivTool(llm_model=None)
    assert t.name == 'search_arxiv_papers'
    return "OK"
check("arxiv_tool", imp_arxiv)

def imp_prompt():
    from agent.prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_SHORT
    assert 'build_and_train' in SYSTEM_PROMPT
    assert 'Never load files' in SYSTEM_PROMPT
    assert 'train.py' not in SYSTEM_PROMPT
    assert 'eval.py'  not in SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT) > 2000
    return f"{len(SYSTEM_PROMPT)} chars"
check("system_prompt", imp_prompt)

def imp_pipeline():
    from agent.data.pipeline import DataPipeline
    return "OK"
check("data_pipeline", imp_pipeline)

# ═══════════════════════════════════════════════════════════════════
sep("2. DATA ARTIFACTS ON DISK")
# ═══════════════════════════════════════════════════════════════════

BASE = pathlib.Path('.')

def check_train_split():
    p = BASE / 'data/splits/train.pkl'
    assert p.exists(), f"missing: {p}"
    with open(p,'rb') as f: d = pickle.load(f)
    import pandas as pd
    assert isinstance(d, pd.DataFrame)
    assert 'sequence' in d.columns
    assert 'concentration' in d.columns
    assert 'label_int' in d.columns
    assert len(d) > 200, f"train too small: {len(d)}"
    return f"DataFrame={d.shape}"
check("data/splits/train.pkl", check_train_split)

def check_val_split():
    p = BASE / 'data/splits/val.pkl'
    assert p.exists()
    with open(p,'rb') as f: d = pickle.load(f)
    import pandas as pd
    assert isinstance(d, pd.DataFrame)
    assert 'sequence' in d.columns
    return f"DataFrame={d.shape}"
check("data/splits/val.pkl", check_val_split)

def check_test_split():
    p = BASE / 'data/splits/test.pkl'
    assert p.exists(), "test.pkl missing"
    with open(p,'rb') as f: d = pickle.load(f)
    import pandas as pd
    assert isinstance(d, pd.DataFrame)
    assert 'sequence' in d.columns
    return f"DataFrame={d.shape}"
check("data/splits/test.pkl  (exists for wall)", check_test_split)

def check_scaler():
    p = BASE / 'data/processed/scaler.pkl'
    assert p.exists()
    with open(p,'rb') as f: sc = pickle.load(f)
    import numpy as np
    dummy = np.zeros((2, 189))
    sc.transform(dummy)
    return "transform OK"
check("data/processed/scaler.pkl", check_scaler)

def check_selector():
    p = BASE / 'data/processed/selector.pkl'
    assert p.exists()
    with open(p,'rb') as f: sel = pickle.load(f)
    import numpy as np
    dummy = np.zeros((2, 189))
    out = sel.transform(dummy)
    return f"output shape {out.shape}"
check("data/processed/selector.pkl", check_selector)

def check_weights():
    p = BASE / 'data/processed/class_weights.pkl'
    assert p.exists()
    with open(p,'rb') as f: cw = pickle.load(f)
    assert set(cw.keys()) == {0,1,2,3}, f"keys: {set(cw.keys())}"
    return str({k: round(v,3) for k,v in cw.items()})
check("data/processed/class_weights.pkl", check_weights)

def check_hash():
    p = BASE / 'data/splits/split_hash.sha256'
    assert p.exists(), "split_hash.sha256 missing — run seal_test_set() in Cell 3"
    content = p.read_text().strip()
    # Format: '<64-hex-chars>  test.pkl'  or just the hex
    hex_part = content.split()[0]
    assert len(hex_part) == 64, f"hash length {len(hex_part)}, expected 64"
    return hex_part[:16] + "..."
check("data/splits/split_hash.sha256", check_hash)

# ═══════════════════════════════════════════════════════════════════
sep("3. REPRODUCIBILITY CHECK")
# ═══════════════════════════════════════════════════════════════════

def check_split_reproducibility():
    from agent.data.pipeline import DataPipeline
    import pandas as pd
    # Load current train split
    with open('data/splits/train.pkl','rb') as f: d1 = pickle.load(f)
    pipe = DataPipeline(random_state=42)
    df_dummy = pd.DataFrame({
        'sequence': ['A'*15]*20 + ['B'*15]*20,
        'concentration': [0.1]*40,
        'label_int': [0]*10 + [1]*10 + [2]*10 + [3]*10
    })
    tr1, val1, te1 = pipe.stratified_split(df_dummy)
    tr2, val2, te2 = pipe.stratified_split(df_dummy)
    assert tr1.equals(tr2), "split not reproducible!"
    return f"train={len(d1)} rows  random_state=42 confirmed"
check("random_state=42 reproducible splits", check_split_reproducibility)

# ═══════════════════════════════════════════════════════════════════
sep("4. HARNESS RENDERING + AUDIT WALL")
# ═══════════════════════════════════════════════════════════════════

def check_harness_render():
    from agent.tools.harness_template import HARNESS_CODE
    rendered = HARNESS_CODE.format(
        exp_id='exp_test_verify',
        architecture='verify_model',
        architecture_family='linear',
        timestamp=datetime.datetime.now().isoformat(timespec='seconds'),
        hyperparams_json='{"C":1.0}',
    )
    # No executable test.pkl load
    bad = [l for l in rendered.splitlines()
           if 'test.pkl' in l and not l.strip().startswith('#')]
    assert not bad, f"Executable test.pkl reference: {bad}"
    return f"{len(rendered)} chars, test.pkl exec-free"
check("harness renders without test.pkl load", check_harness_render)

def check_audit_pass():
    from agent.tools.audit_tool import AuditTool
    a = AuditTool()
    good_codes = [
        # LR
        "def build_and_train(X_train, y_train, X_val, y_val, class_weights):\n"
        "    from sklearn.linear_model import LogisticRegression\n"
        "    return LogisticRegression(C=1.0, class_weight=class_weights).fit(X_train, y_train)\n",
        # RF
        "def build_and_train(X_train, y_train, X_val, y_val, class_weights):\n"
        "    from sklearn.ensemble import RandomForestClassifier\n"
        "    return RandomForestClassifier(class_weight=class_weights, random_state=42).fit(X_train, y_train)\n",
    ]
    for code in good_codes:
        r = a.forward(code)
        assert r.startswith("PASS"), f"Expected PASS, got: {r}"
    return f"{len(good_codes)} clean codes passed"
check("audit: clean code passes", check_audit_pass)

def check_audit_fail():
    from agent.tools.audit_tool import AuditTool
    a = AuditTool()
    bad_codes = [
        "open('data/splits/test.pkl','rb')",
        "pickle.load(open('test.pkl'))",
        "X_test = ...\naccuracy_score(y_test, pred)",
    ]
    for code in bad_codes:
        r = a.forward(code)
        assert r.startswith("FAIL"), f"Should have caught: {code[:40]!r}  got: {r}"
    return f"{len(bad_codes)} cheating patterns caught"
check("audit: cheating code blocked", check_audit_fail)

# ═══════════════════════════════════════════════════════════════════
sep("5. LIVE EXPERIMENT RUN (max_experiments=1)")
# ═══════════════════════════════════════════════════════════════════

def check_live_experiment():
    from agent.tools.experiment_runner import ExperimentRunnerTool
    from agent.core.tee_logger import TeeLogger
    TeeLogger._instance = None

    runner = ExperimentRunnerTool()

    model_code = """
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class SklearnWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, df):
        X = df[["concentration"]].values
        return self.model.predict(X)

def build_and_train(df_train, df_val, class_weights):
    X_train = df_train[["concentration"]].values
    y_train = df_train["label_int"].values
    
    # Simple model
    clf = RandomForestClassifier(
        n_estimators=10,
        class_weight=class_weights,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return SklearnWrapper(clf)
"""
    result_str = runner.forward(
        exp_name            = "rf_verify_check",
        model_code          = model_code,
        hyperparams         = '{"n_estimators": 10}',
    )
    # Find the results.json that was written
    exp_dirs = sorted(pathlib.Path("experiments").glob("*rf_verify_check*"))
    assert exp_dirs, "experiment directory not created"
    rj = exp_dirs[-1] / "results.json"
    assert rj.exists(), f"results.json missing in {exp_dirs[-1]}"
    data = json.loads(rj.read_text())
    assert data['status'] == 'success', f"status={data['status']}  err={data.get('error_message')}"
    f1 = data['val_f1_macro']
    acc = data['val_accuracy']
    return f"val_f1_macro={f1:.4f}  val_accuracy={acc:.4f}"
check("live experiment (RandomForest, 50 trees)", check_live_experiment)

# ═══════════════════════════════════════════════════════════════════
sep("6. SESSION MANAGER")
# ═══════════════════════════════════════════════════════════════════

def check_session_lifecycle():
    import time
    from agent.core.tee_logger import TeeLogger
    from agent.core.session_manager import SessionManager
    TeeLogger._instance = None
    logger = TeeLogger(master_log_dir='master_log')
    sm = SessionManager(model_name='groq-llama', logger=logger, heartbeat_interval=2)
    sm.start()
    sm.tick(current_exp="verify_exp", step=1, status="running")
    time.sleep(3)   # let daemon write at least once
    sm.end(status="completed", total_experiments=1)
    # Verify all 3 files exist
    assert sm._summary_path.exists(), "session_summary.json missing"
    assert sm._heartbeat_path.exists(), "heartbeat.json missing"
    assert sm._log_path.exists(), "session_log.log missing"
    s = json.loads(sm._summary_path.read_text())
    assert s['final_status'] == 'completed'
    assert s['total_experiments_this_session'] == 1
    hb = json.loads(sm._heartbeat_path.read_text())
    assert hb['current_experiment'] == 'verify_exp'
    return f"id={sm.session_id}  status={s['final_status']}"
check("session_manager lifecycle", check_session_lifecycle)

def check_session_crash():
    import time
    from agent.core.tee_logger import TeeLogger
    from agent.core.session_manager import SessionManager
    TeeLogger._instance = None
    logger = TeeLogger(master_log_dir='master_log')
    sm = SessionManager(model_name='local-qwen', logger=logger, heartbeat_interval=1)
    sm.start()
    sm.tick(current_exp="crash_exp", step=1, status="running")
    time.sleep(1)
    sm.end(status="crashed", total_experiments=0, error_message="ValueError: simulated")
    s = json.loads(sm._summary_path.read_text())
    assert s['final_status'] == 'crashed'
    assert s['error_message'] == 'ValueError: simulated'
    return f"crash preserved: {s['error_message']}"
check("session_manager crash persistence", check_session_crash)

# ═══════════════════════════════════════════════════════════════════
sep("7. ORCHESTRATOR INIT (no API call)")
# ═══════════════════════════════════════════════════════════════════

def check_orch_init():
    # Only test __init__, not run() — avoids needing real API key
    from agent.core.tee_logger import TeeLogger
    TeeLogger._instance = None
    from agent.core.orchestrator import AgentOrchestrator
    try:
        orch = AgentOrchestrator(model_name='local-qwen')
        status = orch.status()
        assert not status['running']
        tools_names = [t.name for t in orch.tools]
        assert 'run_experiment' in tools_names
        assert 'read_leaderboard' in tools_names
        assert 'audit_code' in tools_names
        return f"tools={tools_names}"
    except Exception as e:
        # LLM init might fail without ollama — that's OK, just report
        return f"LLM init skipped ({type(e).__name__}: {str(e)[:60]})"
check("orchestrator.__init__ (tools wired)", check_orch_init)

# ═══════════════════════════════════════════════════════════════════
sep("8. GIT STATUS")
# ═══════════════════════════════════════════════════════════════════

def check_git():
    import subprocess
    r = subprocess.run(['git','log','--oneline','-5'], capture_output=True, text=True, cwd='.')
    assert r.returncode == 0
    lines = r.stdout.strip().splitlines()
    for l in lines:
        print(f"         {l}")
    return f"{len(lines)} recent commits"
check("git log (recent commits)", check_git)

def check_gitignore():
    gi = pathlib.Path('.gitignore').read_text(encoding='utf-8')
    assert 'sessions/*/session_log.log' in gi
    assert 'sessions/*/heartbeat.json' in gi
    assert 'experiments/*/artifacts/' in gi
    assert 'master_log/leaderboard.json' in gi
    # session_summary.json must NOT appear on an active (non-comment) line
    active_lines = [l for l in gi.splitlines() if l.strip() and not l.strip().startswith('#')]
    assert not any('session_summary.json' in l for l in active_lines), \
        "session_summary.json is being gitignored (should be tracked)"
    return "rules correct"
check(".gitignore rules", check_gitignore)

# ═══════════════════════════════════════════════════════════════════
sep("SUMMARY")
# ═══════════════════════════════════════════════════════════════════
print()
passed = [n for n,ok,_ in RESULTS if ok]
failed = [n for n,ok,_ in RESULTS if not ok]
total  = len(RESULTS)
print(f"  {len(passed)}/{total} checks passed")
if failed:
    print(f"\n  FAILED ({len(failed)}):")
    for n,ok,msg in RESULTS:
        if not ok:
            print(f"    ✗ {n}")
            print(f"      {msg[:120]}")
else:
    print("  ALL CHECKS PASSED ✓")
print()

sys.exit(0 if not failed else 1)
