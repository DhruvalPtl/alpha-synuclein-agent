"""
agent/tools/arxiv_tool.py
────────────────────────────────────────────────────────────────────────────
Smolagents Tool: search arXiv for relevant ML papers and extract
actionable implementation ideas for peptide classification.

Workflow
--------
1.  Search arXiv via the `arxiv` Python library
2.  For each paper extract title, abstract, authors, year
3.  Optionally filter with an LLM: "Can this method work for
    4-class peptide classification with ~390 samples?"
4.  Return only actionable hits with implementation notes

Usage
-----
    from agent.tools.arxiv_tool import ArxivTool
    tool = ArxivTool()
    print(tool.forward(query="peptide classification small dataset", max_results=5))
"""

import time
from typing import List, Optional, Dict, Any

try:
    import arxiv as arxiv_lib
    _ARXIV_AVAILABLE = True
except ImportError:
    _ARXIV_AVAILABLE = False

try:
    from smolagents import Tool
    _SMOLAGENTS_AVAILABLE = True
except ImportError:
    _SMOLAGENTS_AVAILABLE = False
    class Tool:  # type: ignore[no-redef]
        pass

from agent.core.tee_logger import TeeLogger

# ── LLM relevance filter prompt ───────────────────────────────────────────────
_FILTER_PROMPT = """\
You are an ML researcher expert. Given this paper abstract, answer ONE question:

Can the method described in this abstract be usefully applied to a 4-class
classification problem on peptide sequences with ~390 samples and ~500 features?

Abstract:
{abstract}

Rules:
- If YES: reply with exactly "ACTIONABLE: " followed by a 1-2 sentence
  specific implementation suggestion (what architecture/trick to try, 
  concretely).
- If NO or UNCERTAIN: reply with exactly "SKIP"

Your answer:"""


class ArxivTool(Tool if _SMOLAGENTS_AVAILABLE else object):  # type: ignore[misc]
    """
    Search arXiv for relevant ML/bioinformatics papers and return
    actionable implementation ideas for peptide classification.
    """

    name        = "search_arxiv_papers"
    description = (
        "Search arXiv for ML papers relevant to protein/peptide classification. "
        "Returns titles, abstracts, and actionable implementation ideas. "
        "Use this to discover new methods — especially after hitting a plateau. "
        "Input: query string (e.g. 'peptide classification small dataset transformer'). "
        "Optional: max_results int (default 3, max 10). "
        "Optional: llm_filter bool (default True — filters for relevant papers; "
        "note: LLM filtering is slow on local models, ~1 min per paper). "
        "Output: formatted paper summaries with implementation suggestions."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "arXiv search query. Examples:\n"
                "  'peptide classification small dataset'\n"
                "  'protein sequence transformer few-shot 2024'\n"
                "  'tabular classification imbalanced deep learning'"
            ),
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of papers to retrieve (default 5, max 15).",
            "nullable": True,
        },
        "llm_filter": {
            "type": "boolean",
            "description": "Use LLM to filter only actionable papers (default True).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, llm_model=None) -> None:
        if _SMOLAGENTS_AVAILABLE:
            super().__init__()
        self.logger    = TeeLogger()
        self._llm      = llm_model   # optional LiteLLMModel for filtering

    def forward(
        self,
        query: str,
        max_results: int = 3,
        llm_filter: bool = True,
    ) -> str:
        """
        Search arXiv and return formatted, optionally LLM-filtered results.
        """
        if not _ARXIV_AVAILABLE:
            return (
                "[ArxivTool] 'arxiv' package not installed.\n"
                "Run: pip install arxiv"
            )

        max_results = min(int(max_results or 5), 15)
        self.logger.agent(
            f"[ArxivTool] Searching: '{query}'  (max={max_results})"
        )

        # ── 1. Fetch papers from arXiv ────────────────────────────────────────
        try:
            client = arxiv_lib.Client(
                page_size=max_results,
                delay_seconds=1.0,
                num_retries=3,
            )
            search = arxiv_lib.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv_lib.SortCriterion.Relevance,
            )
            papers: List[Dict[str, Any]] = []
            for result in client.results(search):
                papers.append({
                    "title":    result.title,
                    "abstract": result.summary[:1500],
                    "authors":  ", ".join(a.name for a in result.authors[:3]),
                    "year":     result.published.year if result.published else "?",
                    "url":      result.entry_id,
                    "pdf_url":  result.pdf_url or result.entry_id,
                })
                time.sleep(0.2)   # polite delay
        except Exception as exc:
            self.logger.error(f"[ArxivTool] Search failed: {exc}")
            return f"[ArxivTool] arXiv search failed: {exc}"

        if not papers:
            return f"[ArxivTool] No results found for query: '{query}'"

        self.logger.info(f"[ArxivTool] Retrieved {len(papers)} papers.")

        # ── 2. Optional LLM filtering ─────────────────────────────────────────
        results: List[Dict[str, Any]] = []
        for p in papers:
            suggestion = None

            if llm_filter and self._llm is not None:
                suggestion = self._llm_filter(p["abstract"])
                if suggestion is None:
                    self.logger.info(
                        f"[ArxivTool] SKIP: {p['title'][:60]}"
                    )
                    continue   # LLM said not applicable
                self.logger.agent(
                    f"[ArxivTool] ACTIONABLE: {p['title'][:60]}"
                )
            else:
                suggestion = "LLM filtering disabled — review abstract manually."

            results.append({**p, "suggestion": suggestion})

        # ── 3. Format output ──────────────────────────────────────────────────
        if not results:
            return (
                f"[ArxivTool] No actionable papers found for: '{query}'\n"
                f"(Searched {len(papers)} papers, all filtered out by LLM.)\n"
                "Try a different query or set llm_filter=False."
            )

        lines = [
            "",
            "=" * 68,
            f"  ARXIV SEARCH: '{query}'",
            f"  {len(results)} actionable paper(s) found",
            "=" * 68,
        ]
        for i, p in enumerate(results, start=1):
            lines += [
                "",
                f"  [{i}] {p['title']}",
                f"      Authors : {p['authors']} ({p['year']})",
                f"      URL     : {p['url']}",
                f"      Abstract: {p['abstract'][:400]}...",
                f"      --> IMPLEMENTATION IDEA:",
                f"          {p['suggestion']}",
            ]
        lines += ["", "=" * 68]

        report = "\n".join(lines)
        self.logger.agent(
            f"[ArxivTool] Report ready: {len(results)} actionable papers."
        )

        # ── Log search to master_log/agent_search_history.jsonl ───────────────
        try:
            from agent.tools.search_logger import log_search
            _log_results = [
                {"title": p.get("title", ""), "url": p.get("url", ""),
                 "suggestion": p.get("suggestion", "")}
                for p in results
            ]
            log_search(query=query, source="arxiv",
                       results=_log_results, exp_context="")
        except Exception as _log_exc:
            self.logger.warning(
                f"[ArxivTool] search_logger failed (non-fatal): {_log_exc}"
            )

        return report

    # ── Private helpers ────────────────────────────────────────────────────────

    def _llm_filter(self, abstract: str) -> Optional[str]:
        """
        Ask the LLM if this paper's method is applicable.
        Returns the implementation suggestion string, or None if SKIP.
        """
        prompt = _FILTER_PROMPT.format(abstract=abstract[:1200])
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self._llm(messages)
            if hasattr(response, "content"):
                text = str(response.content).strip()
            else:
                text = str(response).strip()

            if text.upper().startswith("ACTIONABLE:"):
                return text[len("ACTIONABLE:"):].strip()
            return None   # SKIP
        except Exception as exc:
            self.logger.warning(f"[ArxivTool] LLM filter error: {exc}")
            return "LLM filter failed — review abstract manually."
