"""
agent/core/graph.py
────────────────────────────────────────────────────────────────────────────
The State-Based Execution Graph using LangGraph.
Orchestrates the Lead Researcher, Deduplicator, Coder, Reviewer, and Sandbox.
"""

try:
    import os
    import json
    from typing import TypedDict, Annotated
    from pathlib import Path
    from langgraph.graph import StateGraph, END
    from langchain_openai import ChatOpenAI
    from agent.core.memory import PersistentMemory
    from agent.tools.experiment_runner import ExperimentRunnerTool
    from agent.prompts.system_prompt import (
        LEAD_RESEARCHER_PROMPT,
        BIOINFORMATICS_CODER_PROMPT,
        SYNTAX_REVIEWER_PROMPT
    )
    _GRAPH_AVAILABLE = True
except ImportError:
    _GRAPH_AVAILABLE = False


class AgentState(TypedDict):
    hypothesis: str
    architecture: str
    model_code: str
    syntax_feedback: str
    sandbox_results: str
    error_count: int

def get_llm(model_name: str = None, temperature: float = 0.2):
    """
    Returns a LangChain LLM client pointing to the local OpenAI-compatible server or cloud provider.
    Restores support for multiple APIs (Groq, Gemini, OpenRouter, Mistral, Cerebras, Ollama).
    """
    from agent.core.llm_manager import MODELS
    
    if model_name is None:
        model_name = os.environ.get("ACTIVE_MODEL", "local-qwen")
        
    model_id = MODELS.get(model_name, model_name)
    provider = model_id.split("/")[0] if "/" in model_id else "ollama"
    actual_model = model_id.split("/", 1)[1] if "/" in model_id else model_id
    
    # Defaults for local ollama
    base_url = os.environ.get("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "ollama")
    
    if provider == "groq":
        base_url = "https://api.groq.com/openai/v1"
        api_key = os.environ.get("GROQ_API_KEY", "")
    elif provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
    elif provider == "mistral":
        base_url = "https://api.mistral.ai/v1"
        api_key = os.environ.get("MISTRAL_API_KEY", "")
    elif provider == "cerebras":
        base_url = "https://api.cerebras.ai/v1"
        api_key = os.environ.get("CEREBRAS_API_KEY", "")
    elif provider == "gemini":
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = os.environ.get("GEMINI_API_KEY", "")
    elif provider == "openai":
        if "lmstudio" in model_name or "llamafile" in model_name:
            base_url = "http://127.0.0.1:8080/v1"
            api_key = "local"
        else:
            base_url = "https://api.openai.com/v1"
            api_key = os.environ.get("OPENAI_API_KEY", "")
            
    return ChatOpenAI(
        model=actual_model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key
    )

def create_workflow(project_root: Path):
    memory = PersistentMemory(project_root)
    run_experiment_tool = ExperimentRunnerTool()
    
    def lead_researcher(state: AgentState):
        print("[Graph] -> Lead Researcher")
        
        llm = get_llm()
        if hasattr(llm, "with_structured_output"):
            try:
                llm = llm.with_structured_output(
                    schema={"title": "ResearcherOutput", "type": "object", "properties": {"architecture": {"type": "string"}, "hypothesis_summary": {"type": "string"}}}
                )
            except Exception:
                pass # fallback if provider lacks strict JSON structure
        
        past = memory.search_experiments("aggregation models", n_results=3)
        context = f"Past experiments:\n{past}"
        
        prompt = LEAD_RESEARCHER_PROMPT + f"\nContext:\n{context}\n\nPropose a novel hypothesis."
        if state.get("error_count", 0) > 0:
            prompt += f"\nPREVIOUS ATTEMPT FAILED. Feedback: {state.get('sandbox_results')}"
            
        res = llm.invoke(prompt)
        
        if isinstance(res, dict):
            arch = res.get("architecture", "Unknown")
            hypo = res.get("hypothesis_summary", "Unknown")
        else:
            # Fallback parsing
            content = getattr(res, "content", str(res))
            arch = "extracted_arch"
            hypo = content[:200]
            
        return {
            "architecture": arch,
            "hypothesis": hypo,
            "error_count": 0
        }
        
    def deduplicator(state: AgentState):
        print("[Graph] -> Deduplicator")
        is_dup = memory.check_is_duplicate(state["hypothesis"], state["architecture"])
        if is_dup:
            return {"sandbox_results": "DUPLICATE: This idea was already tried. Try something else.", "error_count": state.get("error_count", 0) + 1}
        return {}

    def bioinformatics_coder(state: AgentState):
        print("[Graph] -> Bioinformatics Coder")
        llm = get_llm()
        
        prompt = BIOINFORMATICS_CODER_PROMPT + f"\nArchitecture: {state['architecture']}\nHypothesis: {state['hypothesis']}"
        if state.get("syntax_feedback"):
            prompt += f"\n\nFIX THESE ERRORS:\n{state['syntax_feedback']}"
            
        res = llm.invoke(prompt)
        content = getattr(res, "content", str(res))
        
        # Extract code from Markdown blocks
        if "```python" in content:
            code = content.split("```python")[1].split("```")[0].strip()
        else:
            code = content.replace("```", "").strip()
            
        return {"model_code": code}

    def syntax_reviewer(state: AgentState):
        print("[Graph] -> Syntax Reviewer")
        llm = get_llm()
        schema = {"title": "ReviewerOutput", "type": "object", "properties": {"is_valid": {"type": "boolean"}, "feedback": {"type": "string"}}}
        
        if hasattr(llm, "with_structured_output"):
            try:
                llm = llm.with_structured_output(schema=schema)
            except Exception:
                pass

        prompt = SYNTAX_REVIEWER_PROMPT + f"\nCode to review:\n{state['model_code']}"
        res = llm.invoke(prompt)
        
        if isinstance(res, dict):
            is_valid = res.get("is_valid", False)
            feedback = res.get("feedback", "Looks good")
        else:
            content = getattr(res, "content", str(res)).lower()
            is_valid = "true" in content or "valid" in content
            feedback = getattr(res, "content", str(res))
        
        return {
            "syntax_feedback": feedback if not is_valid else "",
            "error_count": state.get("error_count", 0) + (0 if is_valid else 1)
        }

    def sandbox_runner(state: AgentState):
        print("[Graph] -> Sandbox Runner")
        exp_name = state["architecture"].replace(" ", "_").lower()[:20]
        
        # Invoke smolagents tool
        res_str = run_experiment_tool(
            exp_name=exp_name,
            model_code=state["model_code"],
            hyperparams="{}"
        )
        
        if "EXPERIMENT SUCCESS" in res_str:
            memory.add_experiment(
                exp_id=f"exp_{exp_name}", 
                hypothesis=state["hypothesis"],
                architecture=state["architecture"],
                f1_macro=0.0, 
                mcc=0.0
            )
            
        return {"sandbox_results": res_str}
        
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Researcher", lead_researcher)
    workflow.add_node("Deduplicator", deduplicator)
    workflow.add_node("Coder", bioinformatics_coder)
    workflow.add_node("Reviewer", syntax_reviewer)
    workflow.add_node("Sandbox", sandbox_runner)
    
    workflow.set_entry_point("Researcher")
    workflow.add_edge("Researcher", "Deduplicator")
    
    def route_dedup(state: AgentState):
        if "DUPLICATE" in state.get("sandbox_results", ""):
            return "Researcher"
        return "Coder"
    workflow.add_conditional_edges("Deduplicator", route_dedup, {"Researcher": "Researcher", "Coder": "Coder"})
    
    workflow.add_edge("Coder", "Reviewer")
    
    def route_reviewer(state: AgentState):
        if state.get("error_count", 0) > 3:
            return "Researcher"
        if state.get("syntax_feedback"):
            return "Coder"
        return "Sandbox"
    workflow.add_conditional_edges("Reviewer", route_reviewer, {"Researcher": "Researcher", "Coder": "Coder", "Sandbox": "Sandbox"})
    
    def route_sandbox(state: AgentState):
        if "EXPERIMENT SUCCESS" in state.get("sandbox_results", ""):
            return END
        return "Researcher" 
        
    workflow.add_conditional_edges("Sandbox", route_sandbox, {END: END, "Researcher": "Researcher"})
    
    return workflow.compile()
