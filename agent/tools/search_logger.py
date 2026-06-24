"""
agent/tools/search_logger.py
────────────────────────────────────────────────────────────────────────────
Logs every agent web/arxiv search and its findings to:
    master_log/agent_search_history.jsonl
One JSON line per search event.
"""
import json
from datetime import datetime
from pathlib import Path

_LOG_PATH = Path("master_log/agent_search_history.jsonl")


def log_search(
    query: str,
    source: str,
    results: list,
    exp_context: str = "",
) -> None:
    """
    Append one search event to agent_search_history.jsonl.

    Parameters
    ----------
    query       : search string sent to the provider
    source      : 'arxiv' or 'web'
    results     : list of dicts with title, summary/url
    exp_context : current experiment number or context label
    """
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp":   datetime.now().isoformat(),
        "source":      source,
        "query":       query,
        "exp_context": exp_context,
        "num_results": len(results),
        "results":     results[:5],   # store top 5 only
    }

    with open(_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def get_search_history(last_n: int = 20) -> list:
    """
    Return the last *last_n* search events from the log.

    Parameters
    ----------
    last_n : int   — number of most-recent entries to return

    Returns
    -------
    list[dict]
    """
    if not _LOG_PATH.exists():
        return []
    raw = _LOG_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    return [json.loads(ln) for ln in lines[-last_n:]]
