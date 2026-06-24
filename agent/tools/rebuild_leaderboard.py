"""
agent/tools/rebuild_leaderboard.py
────────────────────────────────────────────────────────────────────────────
Rebuild master_log/leaderboard.json from scratch by scanning every
experiments/*/results.json file on disk.

Why this exists
───────────────
leaderboard.json is gitignored (to prevent mid-run merge conflicts).
After `git pull`, each machine has only its own leaderboard state.
Running rebuild_leaderboard() merges results from ALL machines into
one combined leaderboard, giving the true global picture.

Run after every `git pull` that brings in cloud experiments, or call
rebuild_leaderboard() programmatically from the dashboard.

Usage
-----
    # From project root:
    python agent/tools/rebuild_leaderboard.py

    # Or programmatically:
    from agent.tools.rebuild_leaderboard import rebuild_leaderboard
    lb = rebuild_leaderboard()
    print(f"Total experiments: {lb['total_runs']}")
    print(f"Best F1-macro    : {lb['best_val_f1_macro']:.4f}")
    print(f"Best experiment  : {lb['best_experiment']}")
"""

import json
import datetime
from pathlib import Path
from typing import Dict, Any, List

_EXPERIMENTS_DIR  = Path("experiments")
_LEADERBOARD_PATH = Path("master_log/leaderboard.json")

# Required fields that every results.json must have
_REQUIRED_FIELDS = {
    "exp_id", "architecture",
    "val_f1_macro", "val_accuracy", "status",
}


def rebuild_leaderboard(
    experiments_dir: Path = _EXPERIMENTS_DIR,
    out_path:        Path = _LEADERBOARD_PATH,
    verbose:         bool = True,
) -> Dict[str, Any]:
    """
    Scan ALL experiments/*/results.json files on disk and build a fresh
    leaderboard.json that reflects every machine's work combined.

    Parameters
    ----------
    experiments_dir : Path  — root experiments directory
    out_path        : Path  — where to write leaderboard.json
    verbose         : bool  — print progress

    Returns
    -------
    dict  — the rebuilt leaderboard structure
    """
    experiments_dir = Path(experiments_dir)
    out_path        = Path(out_path)

    # ── 1. Collect all results.json files ─────────────────────────────────────
    results_files: List[Path] = sorted(
        experiments_dir.glob("*/results.json")
    )

    if verbose:
        print(f"[rebuild_leaderboard] Scanning: {experiments_dir.resolve()}")
        print(f"[rebuild_leaderboard] Found {len(results_files)} results.json files")

    # ── 2. Parse and validate each result ─────────────────────────────────────
    experiments: List[Dict[str, Any]] = []
    skipped = 0

    for rpath in results_files:
        try:
            with open(rpath, "r", encoding="utf-8") as fh:
                result = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            if verbose:
                print(f"  [SKIP] {rpath} — read error: {exc}")
            skipped += 1
            continue

        # Check required fields
        missing = _REQUIRED_FIELDS - set(result.keys())
        if missing:
            if verbose:
                print(f"  [SKIP] {rpath} — missing fields: {missing}")
            skipped += 1
            continue

        # Build a compact summary (same schema as leaderboard uses)
        summary = {
            "exp_id":               result.get("exp_id", "?"),
            "machine_id":           result.get("machine_id", "unknown"),
            "architecture":         result.get("architecture", "?"),
            "val_f1_macro":         float(result.get("val_f1_macro", 0.0)),
            "val_accuracy":         float(result.get("val_accuracy", 0.0)),
            "val_f1_per_class":     result.get("val_f1_per_class", {}),
            "val_confusion_matrix": result.get("val_confusion_matrix", []),
            "train_time_seconds":   float(result.get("train_time_seconds", 0.0)),
            "model_params_count":   int(result.get("model_params_count", 0)),
            "status":               result.get("status", "unknown"),
            "error_message":        result.get("error_message"),
            "timestamp":            result.get("timestamp", ""),
            "hyperparams":          result.get("hyperparams", {}),
            "results_path":         str(rpath),
        }
        experiments.append(summary)

        if verbose:
            f1  = summary["val_f1_macro"]
            mid = summary["machine_id"]
            print(f"  [OK]   {summary['exp_id']:<45} f1={f1:.4f}  machine={mid}")

    # ── 3. Sort by val_f1_macro descending ────────────────────────────────────
    experiments.sort(key=lambda e: e["val_f1_macro"], reverse=True)

    # ── 4. Compute summary statistics ─────────────────────────────────────────
    successful = [e for e in experiments if e["status"] == "success"]

    best_f1   = max((e["val_f1_macro"] for e in successful), default=0.0)
    best_exp  = next(
        (e["exp_id"] for e in successful if e["val_f1_macro"] == best_f1),
        None
    )

    # Per-machine stats
    machines: Dict[str, int] = {}
    for e in experiments:
        mid = e["machine_id"]
        machines[mid] = machines.get(mid, 0) + 1

    # Architectures tried
    architectures_tried = sorted(set(
        e["architecture"] for e in experiments
    ))

    # ── 5. Build leaderboard dict ─────────────────────────────────────────────
    leaderboard: Dict[str, Any] = {
        "total_runs":           len(experiments),
        "total_success":        len(successful),
        "total_failed":         len(experiments) - len(successful),
        "total_skipped":        skipped,
        "best_val_f1_macro":    best_f1,
        "best_experiment":      best_exp,
        "architectures_tried":  architectures_tried,
        "runs_per_machine":     machines,
        "last_updated":         datetime.datetime.now().isoformat(timespec="seconds"),
        "rebuilt_from_disk":    True,
        "experiments":          experiments,   # full list, sorted by f1
    }

    # ── 6. Write to disk ──────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(leaderboard, fh, indent=2)

    if verbose:
        print()
        print(f"[rebuild_leaderboard] Written to: {out_path.resolve()}")
        print(f"[rebuild_leaderboard] Total       : {len(experiments)} experiments")
        print(f"[rebuild_leaderboard] Successful  : {len(successful)}")
        print(f"[rebuild_leaderboard] Best F1     : {best_f1:.4f}  ({best_exp})")
        print(f"[rebuild_leaderboard] Machines    : {machines}")
        print(f"[rebuild_leaderboard] Families    : {families_done}")

    return leaderboard


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild leaderboard.json from experiments/*/results.json"
    )
    parser.add_argument("--experiments-dir", default="experiments",
                        help="Path to experiments directory (default: experiments/)")
    parser.add_argument("--out",             default="master_log/leaderboard.json",
                        help="Output path (default: master_log/leaderboard.json)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-experiment output")
    args = parser.parse_args()

    lb = rebuild_leaderboard(
        experiments_dir = Path(args.experiments_dir),
        out_path        = Path(args.out),
        verbose         = not args.quiet,
    )
    sys.exit(0)
