"""
run_pipeline.py
────────────────────────────────────────────────────────────────────────────
Entry point for the Alpha-Synuclein Multi-Agent Platform.
Uses LangGraph to orchestrate the Lead Researcher, Coder, and Sandbox.
"""

import os
from pathlib import Path
from pprint import pprint

# Set up paths
os.environ["PROJECT_ROOT"] = os.getcwd()
project_root = Path(os.getcwd())

from agent.core.graph import create_workflow

def main():
    print("Initializing LangGraph Multi-Agent Workflow...")
    workflow = create_workflow(project_root)
    
    # Define initial state
    initial_state = {
        "hypothesis": "",
        "architecture": "",
        "model_code": "",
        "syntax_feedback": "",
        "sandbox_results": "",
        "error_count": 0
    }
    
    print("\\n=== STARTING RESEARCH LOOP ===")
    
    # We use a recursion limit to prevent true infinite loops in LangGraph
    # (though our logic handles this with error_count as well)
    try:
        final_state = workflow.invoke(initial_state, {"recursion_limit": 20})
        print("\\n=== LOOP FINISHED ===")
        print(f"Final Architecture: {final_state.get('architecture')}")
        print(f"Final Results:\\n{final_state.get('sandbox_results')}")
        
    except Exception as e:
        print(f"\\n[!] Workflow crashed: {e}")

if __name__ == "__main__":
    main()
