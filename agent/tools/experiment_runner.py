"""
agent/tools/experiment_runner.py
────────────────────────────────────────────────────────────────────────────
Strict JSON Tool Schemas for the LangGraph Architecture.
These tools use the secure Evaluation Oracle and Sandboxed Executor, 
ensuring the LLM cannot cheat, hang, or break the main loop.
"""

import os
import json
import uuid
import datetime
from pathlib import Path
from pydantic import BaseModel, Field

from langchain_core.tools import tool

# Internal modules
from agent.core.sandbox import SandboxedExecutor
from agent.core.oracle import EvaluationOracle

project_root = Path(os.environ.get("PROJECT_ROOT", os.getcwd()))
sandbox = SandboxedExecutor(project_root)
oracle = EvaluationOracle(project_root)
experiments_dir = project_root / "experiments"
experiments_dir.mkdir(exist_ok=True)

class RunExperimentInput(BaseModel):
    architecture: str = Field(..., description="Name of the model architecture (e.g., 'RandomForest', 'ESM_LSTM')")
    hypothesis_summary: str = Field(..., description="A 1-2 sentence summary of why you are running this specific config.")
    model_code: str = Field(..., description="The full Python code for the model. Must include a build_and_train function.")

@tool("run_experiment", args_schema=RunExperimentInput)
def run_experiment_tool(architecture: str, hypothesis_summary: str, model_code: str) -> str:
    """
    Executes a generated model in a secure sandbox, evaluates it via the Oracle, 
    and returns the metrics or error trace. Use this to test a new machine learning hypothesis.
    """
    # 1. Create a unique experiment ID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    exp_id = f"exp_{timestamp}_{architecture.lower().replace(' ', '_')}_{uid}"
    exp_dir = experiments_dir / exp_id
    
    # 2. Run the Sandboxed Executor (Training)
    train_result = sandbox.execute_training(exp_dir, model_code, timeout_seconds=300)
    
    if train_result.get("status") != "success":
        # If it crashed, return the trace immediately so the agent can fix it
        return f"EXPERIMENT FAILED DURING TRAINING\\nArchitecture: {architecture}\\nError:\\n{train_result.get('error')}"
        
    # 3. Securely Evaluate via Oracle
    model_pkl_path = exp_dir / "artifacts" / "model.pkl"
    eval_result = oracle.evaluate_model(model_pkl_path)
    
    if eval_result.get("status") != "success":
        return f"EXPERIMENT FAILED DURING EVALUATION\\nArchitecture: {architecture}\\nError:\\n{eval_result.get('error_message')}"
        
    # 4. Save results to disk for Leaderboard/Memory retrieval later
    full_result = {
        "exp_id": exp_id,
        "architecture": architecture,
        "hypothesis": hypothesis_summary,
        "timestamp": timestamp,
        "train_time_seconds": train_result.get("train_time", 0),
        "val_f1_macro": eval_result.get("val_f1_macro"),
        "val_mcc": eval_result.get("val_mcc"),
        "val_accuracy": eval_result.get("val_accuracy"),
        "model_params_count": eval_result.get("model_params_count"),
        "status": "success"
    }
    
    results_json = exp_dir / "results.json"
    with open(results_json, "w") as f:
        json.dump(full_result, f, indent=2)
        
    # 5. Return success summary to the agent
    return (
        f"EXPERIMENT SUCCESS\\n"
        f"Exp ID: {exp_id}\\n"
        f"Architecture: {architecture}\\n"
        f"Val F1 Macro: {full_result['val_f1_macro']:.4f}\\n"
        f"Val MCC: {full_result['val_mcc']:.4f}\\n"
        f"Params: {full_result['model_params_count']}\n"
        f"Train Time: {full_result['train_time_seconds']:.2f}s"
    )


class ExperimentRunnerTool:
    """Legacy compatibility class for ExperimentRunnerTool."""
    def __init__(self) -> None:
        pass

    def forward(self, exp_name: str, model_code: str, hyperparams: str) -> str:
        # Calls the new run_experiment_tool
        import json
        try:
            hp = json.loads(hyperparams)
        except Exception:
            hp = {}
        # Try to infer architecture and hypothesis
        architecture = exp_name
        hypothesis = f"Legacy experiment run via compatibility wrapper with hyperparams: {hyperparams}"
        return run_experiment_tool.invoke({
            "architecture": architecture,
            "hypothesis_summary": hypothesis,
            "model_code": model_code
        })

