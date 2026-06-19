"""
Simulate the two previously-failed experiments using the new harness.
This proves results.json always gets written.
"""
import sys, os, json, shutil
sys.path.insert(0, '.')

from agent.tools.experiment_runner import ExperimentRunnerTool

runner = ExperimentRunnerTool()

# ── Experiment 1: logistic_regression_c_0_1 ───────────────────────────────────
MODEL_LR = """
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(
        C=0.1,
        class_weight=class_weights,
        max_iter=2000,
        solver='lbfgs',
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model
"""

print("=" * 65)
print("EXP 1: logistic_regression_c_0_1")
print("=" * 65)
result_str = runner.forward(
    exp_name            = "logistic_regression_c_0_1",
    architecture_family = "linear",
    model_code          = MODEL_LR,
    hyperparams         = '{"C": 0.1, "solver": "lbfgs", "max_iter": 2000}',
)
print(result_str)

# Read results.json directly and confirm it exists
import pathlib
exp1_dirs = sorted(pathlib.Path("experiments").glob("*logistic_regression_c_0_1*"))
if exp1_dirs:
    rj = exp1_dirs[-1] / "results.json"
    data = json.loads(rj.read_text())
    print(f"\n>>> results.json EXISTS: {rj}")
    print(f">>> val_f1_macro  = {data['val_f1_macro']:.4f}")
    print(f">>> val_accuracy  = {data['val_accuracy']:.4f}")
    print(f">>> status        = {data['status']}")
    f1_lr = data['val_f1_macro']
else:
    print("ERROR: experiment dir not found!")
    f1_lr = None

print()

# ── Experiment 2: linearsvc_c_0_1 ─────────────────────────────────────────────
MODEL_SVC = """
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    model = LinearSVC(
        C=0.1,
        class_weight=class_weights,
        max_iter=5000,
        random_state=42,
    )
    # Wrap for .predict() consistency (LinearSVC always has .predict anyway)
    model.fit(X_train, y_train)
    return model
"""

print("=" * 65)
print("EXP 2: linearsvc_c_0_1")
print("=" * 65)
result_str = runner.forward(
    exp_name            = "linearsvc_c_0_1",
    architecture_family = "linear",
    model_code          = MODEL_SVC,
    hyperparams         = '{"C": 0.1, "class_weight": "balanced", "max_iter": 5000}',
)
print(result_str)

exp2_dirs = sorted(pathlib.Path("experiments").glob("*linearsvc_c_0_1*"))
if exp2_dirs:
    rj = exp2_dirs[-1] / "results.json"
    data = json.loads(rj.read_text())
    print(f"\n>>> results.json EXISTS: {rj}")
    print(f">>> val_f1_macro  = {data['val_f1_macro']:.4f}")
    print(f">>> val_accuracy  = {data['val_accuracy']:.4f}")
    print(f">>> status        = {data['status']}")
    f1_svc = data['val_f1_macro']
else:
    print("ERROR: experiment dir not found!")
    f1_svc = None

# ── Final confirmation table ───────────────────────────────────────────────────
print()
print("=" * 65)
print("FINAL CONFIRMATION")
print("=" * 65)
print(f"  logistic_regression_c_0_1  val_f1_macro = {f1_lr}")
print(f"  linearsvc_c_0_1            val_f1_macro = {f1_svc}")
print()
print("  results.json written by FIXED harness — not by LLM code.")
print("  The LLM defined only build_and_train(); harness did the rest.")
