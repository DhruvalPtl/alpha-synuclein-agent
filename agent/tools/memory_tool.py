"""
agent/tools/memory_tool.py
────────────────────────────────────────────────────────────────────────────
Two smolagents Tools backed by PersistentMemory (ChromaDB):

  SearchMemoryTool   — semantic search over past experiment summaries
  CheckDuplicateTool — novelty gate: blocks re-running too-similar experiments

Both degrade gracefully if ChromaDB is unavailable (returns a notice instead
of crashing so the agent can still run normally).
"""

import os
from pathlib import Path

try:
    from smolagents import Tool as _SmolTool
    _HAS_SMOLAGENTS = True
except ImportError:
    _HAS_SMOLAGENTS = False
    class _SmolTool:
        def __init__(self): pass

_PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent.parent)
).resolve()


def _get_memory():
    """Return a PersistentMemory instance, or raise if ChromaDB is unavailable."""
    from agent.core.memory import PersistentMemory
    return PersistentMemory(_PROJECT_ROOT)


# ── SearchMemoryTool ──────────────────────────────────────────────────────────

class SearchMemoryTool(_SmolTool):
    """
    Semantic search over past experiment summaries stored in ChromaDB.
    Use this to recall what happened with a technique without re-reading
    the entire leaderboard (e.g. "ESM-2 embeddings", "focal loss", "MCC > 0.5").
    """

    name = "search_memory"
    description = (
        "Semantically search the vector database of past experiment summaries.\n"
        "Examples:\n"
        "  search_memory('LSTM sequence model') → shows all LSTM experiments\n"
        "  search_memory('best F1 above 0.6')   → shows top-performing runs\n"
        "  search_memory('focal loss imbalance') → shows class-imbalance experiments\n"
        "Returns formatted text summaries of the most relevant past experiments."
    )
    inputs = {
        "query": {
            "type":        "string",
            "description": "Natural language description of what you want to find.",
        },
        "n_results": {
            "type":        "integer",
            "description": "Max number of past experiments to return (default: 5).",
            "nullable":    True,
        },
    }
    output_type = "string"

    def __init__(self):
        if _HAS_SMOLAGENTS:
            super().__init__()

    def forward(self, query: str, n_results: int = 5) -> str:
        try:
            mem     = _get_memory()
            results = mem.search_experiments(query=query, n_results=n_results)
        except Exception as exc:
            return f"[Memory] Search unavailable ({exc}). Use read_leaderboard instead."

        if not results or not results.get("documents"):
            return "[Memory] No matching experiments found yet. Run some experiments first."

        docs      = results.get("documents", [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]

        if not docs:
            return "[Memory] No matching experiments found."

        lines = [f"Memory search — query: '{query}'\n" + "─" * 60]
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            sim = round(1.0 - float(dist), 3)
            lines.append(
                f"\n[{i}] {meta.get('exp_id', '?')}  |  "
                f"F1={meta.get('f1_macro', 0):.4f}  "
                f"MCC={meta.get('mcc', 0):.4f}  "
                f"similarity={sim}\n"
                f"    {doc}"
            )
        return "\n".join(lines)


# ── CheckDuplicateTool ────────────────────────────────────────────────────────

class CheckDuplicateTool(_SmolTool):
    """
    Novelty gate: checks whether a proposed experiment is too similar to one
    already in the database.  Use this BEFORE run_experiment to avoid wasting
    your experiment budget on ideas already tried.
    """

    name = "check_duplicate"
    description = (
        "Check whether your proposed experiment is too similar to a past one.\n"
        "Returns NOVEL (safe to run) or DUPLICATE (try something different).\n"
        "Always call this before run_experiment when you are unsure if an idea "
        "has been tried before."
    )
    inputs = {
        "architecture": {
            "type":        "string",
            "description": "Short model name (e.g. 'XGBoost with SMOTE and focal loss')",
        },
        "hypothesis": {
            "type":        "string",
            "description": "1-2 sentence explanation of what makes this different from past runs",
        },
    }
    output_type = "string"

    def __init__(self):
        if _HAS_SMOLAGENTS:
            super().__init__()

    def forward(self, architecture: str, hypothesis: str) -> str:
        try:
            mem    = _get_memory()
            is_dup = mem.check_is_duplicate(
                new_hypothesis   = hypothesis,
                new_architecture = architecture,
                threshold        = 0.1,        # cosine distance < 0.1 → duplicate
            )
        except Exception as exc:
            # If memory is unavailable, don't block the agent
            return f"NOVEL (duplicate check unavailable: {exc}). Proceed."

        if is_dup:
            return (
                "DUPLICATE: This experiment is very similar to something already in memory. "
                "Use search_memory to find the closest past run and see its results. "
                "Then try a genuinely different architecture or approach."
            )
        return "NOVEL: This hypothesis looks sufficiently different from past experiments. You may proceed."
