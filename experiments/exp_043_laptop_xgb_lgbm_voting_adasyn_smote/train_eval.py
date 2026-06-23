#!/usr/bin/env python3
"""
train_eval.py  —  FIXED harness (not LLM-generated)
Experiment: exp_043_laptop_xgb_lgbm_voting_adasyn_smote
Architecture: xgb_lgbm_voting_adasyn_smote
Family: ensemble_stack
"""
import importlib.util
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)

# ── Resolve project root (works from any cwd) ─────────────────────────────────
_EXP_DIR      = Path(os.environ.get("EXP_DIR", Path(__file__).parent)).resolve()
_PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent.parent)).resolve()
sys.path.insert(0, str(_PROJECT_ROOT))

_SPLITS_DIR    = _PROJECT_ROOT / "data" / "splits"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

# ── Paths (hardcoded — LLM cannot change these) ───────────────────────────────
_TRAIN_PKL       = _SPLITS_DIR / "train.pkl"
_VAL_PKL         = _SPLITS_DIR / "val.pkl"
# test.pkl is intentionally NOT referenced anywhere in this file.
_SCALER_PKL      = _PROCESSED_DIR / "scaler.pkl"
_SELECTOR_PKL    = _PROCESSED_DIR / "selector.pkl"
_WEIGHTS_PKL     = _PROCESSED_DIR / "class_weights.pkl"
_MODEL_PY        = _EXP_DIR / "model.py"
_RESULTS_JSON    = _EXP_DIR / "results.json"

# ── Metadata (filled by ExperimentRunner template substitution) ───────────────
EXP_ID               = "exp_043_laptop_xgb_lgbm_voting_adasyn_smote"
ARCHITECTURE         = "xgb_lgbm_voting_adasyn_smote"
ARCHITECTURE_FAMILY  = "ensemble_stack"
TIMESTAMP            = "2026-06-23T12:02:25"
HYPERPARAMS: dict    = {"ensemble": "Voting", "models": ["XGBoost+ADASYN", "LGBM+SMOTE"]}

print(f"[harness] EXP_DIR      = {_EXP_DIR}")
print(f"[harness] PROJECT_ROOT = {_PROJECT_ROOT}")
print(f"[harness] Experiment   = {EXP_ID}")


def _write_results(result: dict) -> None:
    """Always write results.json, even on failure."""
    _RESULTS_JSON.write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(f"[harness] results.json written -> {_RESULTS_JSON}")


def _build_error_result(error_msg: str, train_time: float = 0.0) -> dict:
    return {
        "exp_id":               EXP_ID,
        "machine_id":           os.environ.get("MACHINE_ID", ""),
        "architecture":         ARCHITECTURE,
        "architecture_family":  ARCHITECTURE_FAMILY,
        "hyperparams":          HYPERPARAMS,
        "timestamp":            TIMESTAMP,
        "val_accuracy":         0.0,
        "val_f1_macro":         0.0,
        "val_f1_per_class":     {},
        "val_confusion_matrix": [],
        "train_time_seconds":   round(train_time, 3),
        "model_params_count":   0,
        "status":               "failed",
        "error_message":        error_msg,
    }


# ── Step 1: Load splits (ONLY train + val — test.pkl never touched) ───────────
print("[harness] Loading train split ...")
try:
    with open(_TRAIN_PKL, "rb") as f:
        d = pickle.load(f)
    X_train_raw = d.get("X", d.get("X_train"))
    y_train     = d.get("y", d.get("y_train"))
    if X_train_raw is None or y_train is None:
        raise KeyError(f"train.pkl keys: {list(d.keys())}")
except Exception as exc:
    _write_results(_build_error_result(f"Failed to load train.pkl: {exc}"))
    sys.exit(1)

print("[harness] Loading val split ...")
try:
    with open(_VAL_PKL, "rb") as f:
        d = pickle.load(f)
    X_val_raw = d.get("X", d.get("X_val"))
    y_val     = d.get("y", d.get("y_val"))
    if X_val_raw is None or y_val is None:
        raise KeyError(f"val.pkl keys: {list(d.keys())}")
except Exception as exc:
    _write_results(_build_error_result(f"Failed to load val.pkl: {exc}"))
    sys.exit(1)

print(f"[harness] Train: X={X_train_raw.shape}  y={y_train.shape}")
print(f"[harness] Val  : X={X_val_raw.shape}    y={y_val.shape}")

# ── Step 2: Apply scaler + selector (fit on train, transform both) ────────────
print("[harness] Applying scaler + selector ...")
try:
    with open(_SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)
    with open(_SELECTOR_PKL, "rb") as f:
        selector = pickle.load(f)
    X_train = selector.transform(scaler.transform(X_train_raw))
    X_val   = selector.transform(scaler.transform(X_val_raw))
    print(f"[harness] Reduced: train {X_train.shape}  val {X_val.shape}")
except Exception as exc:
    _write_results(_build_error_result(f"Scaler/selector failed: {exc}"))
    sys.exit(1)

# ── Step 3: Load class weights ────────────────────────────────────────────────
print("[harness] Loading class weights ...")
try:
    with open(_WEIGHTS_PKL, "rb") as f:
        class_weights = pickle.load(f)
    print(f"[harness] Class weights: {class_weights}")
except Exception as exc:
    print(f"[harness] WARNING: Could not load class weights: {exc}")
    # Fall back to balanced weights — still better than ignoring imbalance
    classes = np.unique(y_train)
    counts  = np.bincount(y_train.astype(int))
    total   = len(y_train)
    class_weights = {int(c): total / (len(classes) * counts[int(c)]) for c in classes}
    print(f"[harness] Computed fallback weights: {class_weights}")

# ── Step 4: Import LLM's build_and_train from model.py ───────────────────────
print(f"[harness] Importing build_and_train from {_MODEL_PY} ...")
try:
    spec   = importlib.util.spec_from_file_location("model", _MODEL_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_and_train = getattr(module, "build_and_train")
except Exception as exc:
    tb = traceback.format_exc()
    _write_results(_build_error_result(f"Import of model.py failed:\n{tb}"))
    sys.exit(1)

if not callable(build_and_train):
    _write_results(_build_error_result("model.py: build_and_train is not callable"))
    sys.exit(1)

# ── Step 5: Call build_and_train — wrapped in try/except ─────────────────────
print("[harness] Calling build_and_train() ...")
t0 = time.perf_counter()
try:
    model = build_and_train(X_train, y_train, X_val, y_val, class_weights)
    train_time = time.perf_counter() - t0
    print(f"[harness] build_and_train() returned in {train_time:.2f}s")
except Exception as exc:
    train_time = time.perf_counter() - t0
    tb = traceback.format_exc()
    print(f"[harness] build_and_train() RAISED:\n{tb}")
    _write_results(_build_error_result(
        f"build_and_train() raised {type(exc).__name__}: {exc}\n{tb}",
        train_time,
    ))
    sys.exit(1)

if model is None or not hasattr(model, "predict"):
    _write_results(_build_error_result(
        f"build_and_train() returned {type(model)} which has no .predict() method",
        train_time,
    ))
    sys.exit(1)

# ── Step 6: Evaluate on VAL set (ONLY) ───────────────────────────────────────
print("[harness] Evaluating on val set ...")
try:
    y_pred = model.predict(X_val)

    val_acc    = float(accuracy_score(y_val, y_pred))
    val_f1     = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
    val_f1_pc  = f1_score(y_val, y_pred, average=None, zero_division=0)
    val_cm     = confusion_matrix(y_val, y_pred).tolist()

    val_f1_per_class = {
        str(i): float(val_f1_pc[i]) if i < len(val_f1_pc) else 0.0
        for i in range(4)
    }

    print(f"[harness] val_accuracy  = {val_acc:.4f}")
    print(f"[harness] val_f1_macro  = {val_f1:.4f}")
    print(f"[harness] val_f1/class  = {val_f1_per_class}")

except Exception as exc:
    tb = traceback.format_exc()
    _write_results(_build_error_result(
        f"Evaluation failed: {exc}\n{tb}", train_time
    ))
    sys.exit(1)

# ── Step 7: Count model parameters (best-effort) ─────────────────────────────
params = 0
try:
    if hasattr(model, "coef_"):
        params = int(np.prod(model.coef_.shape))
    elif hasattr(model, "n_features_in_"):
        params = int(model.n_features_in_)
    elif hasattr(model, "parameters"):   # PyTorch
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
except Exception:
    params = 0

# ── Step 8: Write results.json — ALWAYS succeeds if we reach here ─────────────
result = {
    "exp_id":               EXP_ID,
    "machine_id":           os.environ.get("MACHINE_ID", ""),
    "architecture":         ARCHITECTURE,
    "architecture_family":  ARCHITECTURE_FAMILY,
    "hyperparams":          HYPERPARAMS,
    "timestamp":            TIMESTAMP,
    "val_accuracy":         round(val_acc,  6),
    "val_f1_macro":         round(val_f1,   6),
    "val_f1_per_class":     val_f1_per_class,
    "val_confusion_matrix": val_cm,
    "train_time_seconds":   round(train_time, 3),
    "model_params_count":   params,
    "status":               "success",
    "error_message":        None,
}

_write_results(result)
print(f"[harness] DONE — val_f1_macro = {val_f1:.4f}")
