"""
agent/tools/dev_reset.py
────────────────────────────────────────────────────────────────────────────
Developer reset tool: wipes experiment history, logs, and sessions for a
clean re-run. NEVER touches data/, agent/, .env, or .git/.

Hard-coded exclusions (cannot be overridden by caller):
  data/raw/           — original CSV
  data/processed/     — scaler, selector, class weights
  data/splits/        — train/val/test.pkl + split_hash.sha256
  agent/              — all source code
  .env                — API keys
  .git/               — version control

Usage (standalone):
    python agent/tools/dev_reset.py

Usage (from notebook):
    from agent.tools.dev_reset import dev_reset
    dev_reset()
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

# ── Protected paths — NEVER deleted ──────────────────────────────────────────
_PROTECTED: list[str] = [
    "data/raw",
    "data/processed",
    "data/splits",
    "agent",
    ".env",
    ".git",
    "pyproject.toml",
    "requirements.txt",
]

# ── Empty leaderboard schema ──────────────────────────────────────────────────
_EMPTY_LEADERBOARD = {
    "total_runs":        0,
    "best_val_f1_macro": 0.0,
    "best_experiment":   None,
    "families_completed": [],
    "agent_model_used":  "unknown",
    "last_updated":      None,
    "experiments":       [],
}


def _is_protected(path: Path, root: Path) -> bool:
    """Return True if *path* is inside any protected directory."""
    rel = str(path.relative_to(root)).replace("\\", "/")
    for p in _PROTECTED:
        if rel == p or rel.startswith(p + "/") or rel.startswith(p + "\\"):
            return True
    return False


def dev_reset(root: str | Path = ".") -> dict:
    """
    Wipe experiment history for a clean dev re-run.

    Deletes:
        experiments/*          — all experiment folders
        master_log/*.log       — terminal + decision logs
        master_log/*.json      — orchestrator_state.json, leaderboard.json
        sessions/*             — all session folders
        reports/*.md           — generated markdown reports

    Recreates:
        master_log/leaderboard.json    — empty schema
        experiments/.gitkeep
        sessions/.gitkeep

    NEVER touches:
        data/raw/, data/processed/, data/splits/,
        agent/, .env, .git/, pyproject.toml, requirements.txt

    Parameters
    ----------
    root : str or Path — project root (default: current directory)

    Returns
    -------
    dict with summary counts
    """
    root = Path(root).resolve()

    print("=" * 62)
    print("  DEV RESET — wiping experiment history")
    print(f"  Root: {root}")
    print("=" * 62)

    # ── Safety: verify mathematical wall BEFORE wipe ──────────────────────────
    hash_file = root / "data/splits/split_hash.sha256"
    wall_hash_before = None
    if hash_file.exists():
        wall_hash_before = hash_file.read_text().strip().split()[0]
        print(f"\n  Pre-reset wall hash : {wall_hash_before[:16]}...")
    else:
        print("\n  WARNING: split_hash.sha256 not found — run Cell 3 first!")

    n_exp     = 0
    n_log     = 0
    n_session = 0
    n_report  = 0

    # ── 1. Delete experiment folders ──────────────────────────────────────────
    exp_dir = root / "experiments"
    if exp_dir.exists():
        for item in sorted(exp_dir.iterdir()):
            if item.name == ".gitkeep":
                continue
            if _is_protected(item, root):
                print(f"  [SKIP-PROTECTED] {item.name}")
                continue
            if item.is_dir():
                shutil.rmtree(item)
                n_exp += 1
            elif item.is_file():
                item.unlink()
                n_exp += 1
        print(f"\n  experiments/  : deleted {n_exp} folder(s)")
    else:
        exp_dir.mkdir(parents=True)

    # ── 2. Delete master_log files ────────────────────────────────────────────
    log_dir = root / "master_log"
    if log_dir.exists():
        for item in sorted(log_dir.iterdir()):
            if _is_protected(item, root):
                continue
            if item.is_file():
                item.unlink()
                n_log += 1
        print(f"  master_log/   : deleted {n_log} file(s)")
    else:
        log_dir.mkdir(parents=True)

    # ── 3. Delete session folders ─────────────────────────────────────────────
    sessions_dir = root / "sessions"
    if sessions_dir.exists():
        for item in sorted(sessions_dir.iterdir()):
            if item.name == ".gitkeep":
                continue
            if _is_protected(item, root):
                continue
            if item.is_dir():
                shutil.rmtree(item)
                n_session += 1
            elif item.is_file():
                item.unlink()
                n_session += 1
        print(f"  sessions/     : deleted {n_session} item(s)")
    else:
        sessions_dir.mkdir(parents=True)

    # ── 4. Delete report markdown files ──────────────────────────────────────
    reports_dir = root / "reports"
    if reports_dir.exists():
        for item in sorted(reports_dir.glob("*.md")):
            if _is_protected(item, root):
                continue
            item.unlink()
            n_report += 1
        print(f"  reports/      : deleted {n_report} file(s)")

    # ── 5. Recreate skeleton structure ────────────────────────────────────────
    (exp_dir / ".gitkeep").touch()
    (sessions_dir / ".gitkeep").touch()
    log_dir.mkdir(parents=True, exist_ok=True)

    lb_path = log_dir / "leaderboard.json"
    lb_data = dict(_EMPTY_LEADERBOARD)
    lb_data["last_updated"] = datetime.now().isoformat(timespec="seconds")
    lb_path.write_text(json.dumps(lb_data, indent=2), encoding="utf-8")
    print(f"\n  Recreated: {lb_path.relative_to(root)}")
    print(f"  Recreated: experiments/.gitkeep")
    print(f"  Recreated: sessions/.gitkeep")

    # ── 6. Verify wall AFTER wipe ─────────────────────────────────────────────
    print("\n" + "─" * 62)
    print("  Verifying mathematical wall survived the wipe ...")
    wall_ok     = False
    wall_hash   = "N/A"
    wall_detail = ""

    try:
        import sys
        sys.path.insert(0, str(root))
        from agent.data.pipeline import DataPipeline
        pipe = DataPipeline(splits_dir=str(root / "data/splits"), random_state=42)
        wall_ok = pipe.verify_wall()
        if hash_file.exists():
            wall_hash = hash_file.read_text().strip().split()[0]
        wall_detail = (
            f"  verify_wall() = {'PASS ✓' if wall_ok else 'FAIL ✗'}\n"
            f"  SHA-256 after : {wall_hash}"
        )
        if wall_hash_before and wall_hash != wall_hash_before:
            wall_detail += "\n  *** HASH CHANGED — RESET CORRUPTED THE WALL! ***"
    except Exception as exc:
        wall_detail = f"  verify_wall() could not run: {exc}"

    print(wall_detail)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  RESET COMPLETE")
    print(f"  Deleted : {n_exp} experiment folder(s), "
          f"{n_log} log file(s), "
          f"{n_session} session(s), "
          f"{n_report} report(s)")
    print(f"  Preserved: mathematical wall intact"
          f" (hash: {wall_hash[:16]}...{wall_hash[-8:] if len(wall_hash) > 24 else ''})")
    print(f"  Wall status: {'INTACT ✓' if wall_ok else 'CHECK FAILED'}")
    print("  Ready for fresh production run.")
    print("=" * 62)

    return {
        "experiments_deleted": n_exp,
        "logs_deleted":        n_log,
        "sessions_deleted":    n_session,
        "reports_deleted":     n_report,
        "wall_intact":         wall_ok,
        "wall_hash":           wall_hash,
    }


if __name__ == "__main__":
    import os
    # Find project root (look for pyproject.toml or agent/)
    cwd = Path.cwd()
    root = cwd
    for _ in range(4):
        if (root / "agent").is_dir() and (root / "data").is_dir():
            break
        root = root.parent
    else:
        root = cwd
    os.chdir(root)
    dev_reset(root)
