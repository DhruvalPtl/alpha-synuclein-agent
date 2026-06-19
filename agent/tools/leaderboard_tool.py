"""
agent/tools/leaderboard_tool.py
────────────────────────────────────────────────────────────────────────────
Smolagents Tool: read leaderboard.json and return formatted analysis.

Provides:
  • Top-N experiments ranked by val_f1_macro
  • Which architecture families have NOT been tried yet
  • Performance gap analysis between families
  • Recommendations for next experiment

Usage (Smolagents agent)
------------------------
    from agent.tools.leaderboard_tool import LeaderboardTool
    tool = LeaderboardTool()
    print(tool.forward(top_n=5))

Usage (direct)
--------------
    tool = LeaderboardTool()
    print(tool.forward(top_n=10, show_gaps=True))
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from smolagents import Tool
    _SMOLAGENTS_AVAILABLE = True
except ImportError:
    _SMOLAGENTS_AVAILABLE = False
    class Tool:  # type: ignore[no-redef]
        pass

from agent.core.tee_logger import TeeLogger
from agent.tools.rebuild_leaderboard import rebuild_leaderboard

_LEADERBOARD_PATH = Path("master_log/leaderboard.json")

# All architecture families the agent should eventually explore
ALL_FAMILIES: List[str] = [
    "classical_ml",       # RandomForest, XGBoost, LightGBM, SVM, kNN
    "linear",             # LogisticRegression, ElasticNet, RidgeClassifier
    "neural_network",     # MLP with varying depth/width
    "deep_residual",      # Deep MLP with residual connections
    "ensemble_stack",     # Stacking / blending of diverse base models
    "attention_based",    # Self-attention / Transformer on feature tokens
    "graph_neural",       # GNN treating peptide as amino-acid graph
    "automl",             # AutoSklearn, FLAML, or Optuna sweep
]


class LeaderboardTool(Tool if _SMOLAGENTS_AVAILABLE else object):  # type: ignore[misc]
    """
    Read master_log/leaderboard.json and return a formatted analysis.
    """

    name        = "read_leaderboard"
    description = (
        "Read the experiment leaderboard. Returns a formatted table of the "
        "top-N experiments by val_f1_macro, lists untried architecture "
        "families, and highlights performance gaps. "
        "Use this before deciding what to try next. "
        "Input: top_n (int, default 10). "
        "Output: formatted text report."
    )
    inputs = {
        "top_n": {
            "type": "integer",
            "description": "Number of top experiments to show (default 10).",
            "nullable": True,
        }
    }
    output_type = "string"

    def __init__(self) -> None:
        if _SMOLAGENTS_AVAILABLE:
            super().__init__()
        self.logger = TeeLogger()

    def forward(self, top_n: int = 10) -> str:  # noqa: D102
        """
        Generate and return the leaderboard report.

        Calls rebuild_leaderboard() first so the report reflects ALL
        experiments from both machines (not just the local cache).

        Parameters
        ----------
        top_n : int   — how many top experiments to display

        Returns
        -------
        str — human-readable formatted report
        """
        # Rebuild from disk first — merges both machines' results.json files
        try:
            rebuild_leaderboard(verbose=False)
            self.logger.info("[LeaderboardTool] Leaderboard rebuilt from disk.")
        except Exception as exc:
            self.logger.warning(
                f"[LeaderboardTool] rebuild_leaderboard failed ({exc}); "
                "falling back to cached leaderboard.json."
            )

        if not _LEADERBOARD_PATH.exists():
            return "[LeaderboardTool] leaderboard.json not found."

        try:
            with open(_LEADERBOARD_PATH, "r", encoding="utf-8") as fh:
                lb: Dict[str, Any] = json.load(fh)
        except json.JSONDecodeError as exc:
            return f"[LeaderboardTool] JSON parse error: {exc}"

        experiments: List[Dict] = lb.get("experiments", [])
        total_runs:  int        = lb.get("total_runs", 0)
        best_f1:     float      = lb.get("best_val_f1_macro", 0.0)
        best_exp:    Optional[str] = lb.get("best_experiment")
        tried_fams:  List[str]  = lb.get("families_completed", [])
        agent_model: str        = lb.get("agent_model_used", "unknown")
        last_updated: str       = lb.get("last_updated", "never")

        lines: List[str] = []

        # ── Header ───────────────────────────────────────────────────────────
        lines.append("=" * 70)
        lines.append("  ALPHA-SYNUCLEIN AGENT — EXPERIMENT LEADERBOARD")
        lines.append("=" * 70)
        lines.append(f"  Total runs    : {total_runs}")
        lines.append(f"  Best F1-macro : {best_f1:.4f}  ({best_exp or 'none yet'})")
        lines.append(f"  Agent model   : {agent_model}")
        lines.append(f"  Last updated  : {last_updated}")
        lines.append("=" * 70)

        # ── Top-N experiments table ───────────────────────────────────────────
        if not experiments:
            lines.append("\n  No experiments recorded yet.")
        else:
            # Sort by val_f1_macro descending
            ranked = sorted(
                experiments,
                key=lambda e: e.get("val_f1_macro", 0.0),
                reverse=True,
            )[:top_n]

            lines.append(f"\n  TOP {min(top_n, len(ranked))} EXPERIMENTS")
            lines.append(
                f"  {'Rank':<5} {'Exp ID':<42} {'F1-macro':>9}"
                f" {'Accuracy':>9} {'Machine':<10} {'Status':<9}"
            )
            lines.append("  " + "-" * 88)
            for rank, exp in enumerate(ranked, start=1):
                lines.append(
                    f"  {rank:<5} "
                    f"{exp.get('exp_id','?'):<42} "
                    f"{exp.get('val_f1_macro', 0.0):>9.4f} "
                    f"{exp.get('val_accuracy', 0.0):>9.4f} "
                    f"{exp.get('machine_id','?'):<10} "
                    f"{exp.get('status','?'):<9}"
                )

        # ── Per-class F1 of best experiment ──────────────────────────────────
        if experiments:
            best_data = next(
                (e for e in experiments if e.get("exp_id") == best_exp), None
            )
            if best_data and "val_f1_per_class" in best_data:
                per_class = best_data["val_f1_per_class"]
                lines.append(f"\n  PER-CLASS F1 (best experiment: {best_exp})")
                label_names = {
                    "0": "No", "1": "Low", "2": "Medium", "3": "High",
                    0: "No",   1: "Low",   2: "Medium",   3: "High",
                }
                for cls, f1 in sorted(per_class.items(), key=lambda x: str(x[0])):
                    bar = "#" * int(f1 * 30)
                    lines.append(
                        f"    Class {cls} ({label_names.get(cls, '?'):<6}): "
                        f"{f1:.4f}  |{bar:<30}|"
                    )

        # ── Architecture family coverage ──────────────────────────────────────
        lines.append("\n  ARCHITECTURE FAMILY COVERAGE")
        lines.append(f"  {'Family':<22} {'Status':<12} {'Best F1':>9}")
        lines.append("  " + "-" * 45)

        family_scores: Dict[str, float] = {}
        for exp in experiments:
            fam = exp.get("architecture_family", "unknown")
            f1  = exp.get("val_f1_macro", 0.0)
            if fam not in family_scores or f1 > family_scores[fam]:
                family_scores[fam] = f1

        for fam in ALL_FAMILIES:
            if fam in family_scores:
                status = "[DONE]"
                score  = f"{family_scores[fam]:.4f}"
            elif fam in tried_fams:
                status = "[TRIED]"
                score  = "0.0000"
            else:
                status = "[ -- ]"
                score  = "  n/a "
            lines.append(f"  {fam:<22} {status:<12} {score:>9}")

        # Unknown families from actual experiments
        unknown_fams = [
            fam for fam in family_scores
            if fam not in ALL_FAMILIES
        ]
        for fam in unknown_fams:
            lines.append(
                f"  {fam:<22} {'[EXTRA]':<12}"
                f" {family_scores[fam]:>9.4f}"
            )

        # ── Performance gap analysis ──────────────────────────────────────────
        done_scores = {
            fam: sc for fam, sc in family_scores.items()
            if fam in ALL_FAMILIES
        }
        untried = [f for f in ALL_FAMILIES if f not in family_scores]

        lines.append("\n  UNTRIED FAMILIES (recommended next targets)")
        if untried:
            for fam in untried:
                lines.append(f"    >> {fam}")
        else:
            lines.append("    All families have been explored!")

        if done_scores:
            best_fam   = max(done_scores, key=lambda f: done_scores[f])
            worst_fam  = min(done_scores, key=lambda f: done_scores[f])
            gap        = done_scores[best_fam] - done_scores[worst_fam]
            lines.append(
                f"\n  PERFORMANCE GAP: {gap:.4f}"
                f"  (best={best_fam}: {done_scores[best_fam]:.4f}"
                f"  worst={worst_fam}: {done_scores[worst_fam]:.4f})"
            )

        # ── Class imbalance reminder ──────────────────────────────────────────
        lines.append("\n  [!] CLASS IMBALANCE: No=311, Med=37, High=32, Low=16")
        lines.append("      Use class_weight='balanced' or SMOTE for minority classes.")

        lines.append("\n" + "=" * 70)
        report = "\n".join(lines)

        self.logger.info(
            f"[LeaderboardTool] Report generated. "
            f"total_runs={total_runs}  best_f1={best_f1:.4f}  "
            f"untried_families={len(untried)}"
        )
        return report
