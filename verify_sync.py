import sys, os, platform, json
sys.path.insert(0, '.')

# ── 1. Print final .gitignore ──────────────────────────────────────────────────
print("=" * 65)
print("FINAL .gitignore")
print("=" * 65)
print(open('.gitignore').read())

# ── 2. Demo new ID format ──────────────────────────────────────────────────────
print("=" * 65)
print("NEW ID FORMAT — example")
print("=" * 65)
from agent.tools.experiment_runner import _get_machine_id, _sanitize

mid = _get_machine_id()
exp_name = "random_forest_baseline"
arch_slug = _sanitize(exp_name)[:30]
n = 1

example_id = f"exp_{n:03d}_{mid}_{arch_slug}"
print(f"  MACHINE_ID env var   : {os.environ.get('MACHINE_ID', '(not set)')}")
print(f"  platform.node()      : {platform.node()}")
print(f"  _get_machine_id()    : {mid}")
print(f"  exp_name             : {exp_name}")
print(f"  Example exp_id       : {example_id}")
print()
print("  If MACHINE_ID=gcloud in .env -> exp_001_gcloud_random_forest_baseline")
print("  If MACHINE_ID=laptop in .env -> exp_001_laptop_random_forest_baseline")
print("  Two DIFFERENT folders. No silent overwrite on git sync.")

# ── 3. Test rebuild_leaderboard.py against experiments/ (empty OK) ────────────
print()
print("=" * 65)
print("rebuild_leaderboard.py — run against experiments/ (empty now)")
print("=" * 65)
from agent.tools.rebuild_leaderboard import rebuild_leaderboard

# Create a fake experiment to verify parsing works
import tempfile, pathlib, datetime
fake_dir = pathlib.Path("experiments/exp_001_test_machine_smoke_test")
fake_dir.mkdir(parents=True, exist_ok=True)
fake_result = {
    "exp_id":              "exp_001_test_machine_smoke_test",
    "machine_id":          "test_machine",
    "architecture":        "SmokeTestClassifier",
    "architecture_family": "classical_ml",
    "val_f1_macro":        0.7234,
    "val_accuracy":        0.8500,
    "val_f1_per_class":    {"0": 0.95, "1": 0.45, "2": 0.70, "3": 0.68},
    "train_time_seconds":  3.2,
    "model_params_count":  100,
    "status":              "success",
    "error_message":       None,
    "timestamp":           datetime.datetime.now().isoformat(timespec="seconds"),
    "hyperparams":         {"n_estimators": 100}
}
(fake_dir / "results.json").write_text(json.dumps(fake_result, indent=2))

lb = rebuild_leaderboard(verbose=True)

print()
print("Leaderboard summary:")
print(f"  total_runs      : {lb['total_runs']}")
print(f"  best_f1_macro   : {lb['best_val_f1_macro']:.4f}")
print(f"  best_experiment : {lb['best_experiment']}")
print(f"  runs_per_machine: {lb['runs_per_machine']}")
print(f"  families        : {lb['families_completed']}")
print()

# Cleanup fake experiment
import shutil
shutil.rmtree(fake_dir)
print("(Fake experiment cleaned up.)")

print()
print("=" * 65)
print("ALL CHECKS COMPLETE")
print("=" * 65)
