import sys
import json
import pickle
import time
import traceback
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Extract paths from args
project_root = Path(sys.argv[1])
exp_dir = Path(sys.argv[2])

# Add project root to sys path
sys.path.insert(0, str(project_root))

# Load data
splits_dir = project_root / "data" / "splits"
processed_dir = project_root / "data" / "processed"

try:
    with open(splits_dir / "train.pkl", "rb") as f:
        df_train = pickle.load(f)
        
    with open(splits_dir / "val.pkl", "rb") as f:
        df_val = pickle.load(f)
        
    with open(processed_dir / "class_weights.pkl", "rb") as f:
        class_weights = pickle.load(f)
        
except Exception as e:
    print(json.dumps({"status": "failed", "error": f"Data load error: {e}"}))
    sys.exit(1)

# Import agent's model
model_py = exp_dir / "model.py"
import importlib.util
try:
    spec = importlib.util.spec_from_file_location("model", model_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_and_train = getattr(module, "build_and_train")
except Exception as e:
    tb = traceback.format_exc()
    print(json.dumps({"status": "failed", "error": f"Import error:\n{tb}"}))
    sys.exit(1)

# Train
try:
    t0 = time.perf_counter()
    model = build_and_train(df_train, df_val, class_weights)
    train_time = time.perf_counter() - t0
except Exception as e:
    tb = traceback.format_exc()
    print(json.dumps({"status": "failed", "error": f"Training failed:\n{tb}"}))
    sys.exit(1)
    
if model is None or not hasattr(model, "predict"):
    print(json.dumps({"status": "failed", "error": "Model has no .predict() method"}))
    sys.exit(1)

# Save
try:
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    with open(artifacts_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
except Exception as e:
    print(json.dumps({"status": "failed", "error": f"Failed to save model: {e}"}))
    sys.exit(1)

print(json.dumps({"status": "success", "train_time": train_time}))
