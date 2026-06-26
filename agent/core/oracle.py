"""
agent/core/oracle.py
────────────────────────────────────────────────────────────────────────────
The Evaluation Oracle is a secure, tamper-proof evaluation module.
It isolates the evaluation logic from the Sandboxed Executer so the agent 
cannot write malicious code to alter `results.json` or overfit the test set.

Usage:
    oracle = EvaluationOracle()
    results = oracle.evaluate_model(model_pkl_path, exp_id)
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

class EvaluationOracle:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.splits_dir = self.project_root / "data" / "splits"
        self.processed_dir = self.project_root / "data" / "processed"
        
        # Paths
        self.val_pkl = self.splits_dir / "val.pkl"
        self.test_pkl = self.splits_dir / "test.pkl"
        self.scaler_pkl = self.processed_dir / "scaler.pkl"
        self.selector_pkl = self.processed_dir / "selector.pkl"
        
        # We load val data once upon initialization for fast repeated evaluations
        self.X_val, self.y_val = self._load_and_transform(self.val_pkl)
        
    def _load_and_transform(self, split_path: Path):
        """Load the validation DataFrame."""
        with open(split_path, "rb") as f:
            df_val = pickle.load(f)
        
        y = df_val["label_int"].values
        return df_val, y

    def evaluate_model(self, model_pkl_path: Path) -> dict:
        """
        Securely evaluate a saved model on the validation set.
        The model object MUST have a .predict() method.
        """
        model_pkl_path = Path(model_pkl_path)
        if not model_pkl_path.exists():
            return {"status": "failed", "error_message": f"Model file {model_pkl_path} not found."}
            
        try:
            exp_dir = model_pkl_path.parent.parent
            import sys
            import importlib.util
            spec = importlib.util.spec_from_file_location("model", exp_dir / "model.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules["model"] = module
            spec.loader.exec_module(module)
            
            with open(model_pkl_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            return {"status": "failed", "error_message": f"Failed to load model: {e}"}
            
        if not hasattr(model, "predict"):
            return {"status": "failed", "error_message": "Loaded model has no .predict() method"}
            
        # 1. Predict
        try:
            y_pred = model.predict(self.X_val)
        except Exception as e:
            return {"status": "failed", "error_message": f"Model predict() failed: {e}"}
            
        # 2. Compute metrics securely
        val_acc = float(accuracy_score(self.y_val, y_pred))
        val_f1_macro = float(f1_score(self.y_val, y_pred, average="macro", zero_division=0))
        val_mcc = float(matthews_corrcoef(self.y_val, y_pred))
        
        val_f1_pc = f1_score(self.y_val, y_pred, average=None, zero_division=0)
        val_f1_per_class = {
            str(i): float(val_f1_pc[i]) if i < len(val_f1_pc) else 0.0
            for i in range(4)
        }
        val_cm = confusion_matrix(self.y_val, y_pred).tolist()
        
        # 3. Model Params count (best effort)
        params = 0
        try:
            if hasattr(model, "coef_"):
                params = int(np.prod(model.coef_.shape))
            elif hasattr(model, "n_features_in_"):
                params = int(model.n_features_in_)
            elif hasattr(model, "parameters"):
                params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except:
            pass

        return {
            "status": "success",
            "val_accuracy": round(val_acc, 6),
            "val_f1_macro": round(val_f1_macro, 6),
            "val_mcc": round(val_mcc, 6),
            "val_f1_per_class": val_f1_per_class,
            "val_confusion_matrix": val_cm,
            "model_params_count": params,
            "error_message": None
        }
