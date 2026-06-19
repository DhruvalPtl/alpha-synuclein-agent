"""
agent/tools/check_last_session.py
────────────────────────────────────────────────────────────────────────────
Standalone diagnostic script — run this after reconnecting to a cloud
instance to immediately know what happened while you were away.

Usage
-----
    python agent/tools/check_last_session.py          # most recent session
    python agent/tools/check_last_session.py --all    # all sessions (brief)
    python agent/tools/check_last_session.py --n 3    # last N sessions

Output example
--------------
    ══════════════════════════════════════════════════════════════
    Last session: 2026-06-19_14-00-00
      Machine    : gcloud
      Model      : groq-llama
      Started    : 2026-06-19T14:00:00
      Ended      : 2026-06-19T17:32:14  (3h 32m 14s)
      Status     : completed
      Experiments: 18 this session
      Last exp   : exp_018_gcloud_lightgbm_balanced
      Step #     : 18
      Heartbeat  : 4 minutes ago  ← if status is 'running'
    ══════════════════════════════════════════════════════════════

Importable as a function too:
    from agent.tools.check_last_session import get_last_session_info
    info = get_last_session_info()
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

_SESSIONS_DIR = Path("sessions")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ago(iso_str: Optional[str]) -> str:
    """Return '4 minutes ago', '1 hour 12 minutes ago', etc."""
    if not iso_str:
        return "never"
    try:
        then = datetime.datetime.fromisoformat(iso_str)
        delta = datetime.datetime.now() - then
        total_secs = int(delta.total_seconds())
        if total_secs < 60:
            return f"{total_secs} seconds ago"
        if total_secs < 3600:
            m = total_secs // 60
            return f"{m} minute{'s' if m != 1 else ''} ago"
        h = total_secs // 3600
        m = (total_secs % 3600) // 60
        return f"{h}h {m}m ago"
    except Exception:
        return iso_str or "?"


def _duration(start: Optional[str], end: Optional[str]) -> str:
    if not start:
        return "?"
    try:
        t0 = datetime.datetime.fromisoformat(start)
        t1 = datetime.datetime.fromisoformat(end) if end else datetime.datetime.now()
        secs = int((t1 - t0).total_seconds())
        h, rem = divmod(secs, 3600)
        m, s   = divmod(rem, 60)
        suffix = "" if end else " so far"
        return f"{h}h {m:02d}m {s:02d}s{suffix}"
    except Exception:
        return "?"


def _status_icon(status: Optional[str]) -> str:
    icons = {
        "completed":   "✓",
        "running":     "⟳",
        "interrupted": "⚠",
        "crashed":     "✗",
        "starting":    "…",
    }
    return icons.get(status or "", "?")


def get_session_info(session_dir: Path) -> Dict[str, Any]:
    """Load and merge summary + heartbeat for one session directory."""
    summary   = _load_json(session_dir / "session_summary.json") or {}
    heartbeat = _load_json(session_dir / "heartbeat.json") or {}

    sid    = summary.get("session_id") or session_dir.name
    start  = summary.get("start_time")
    end    = summary.get("end_time")
    status = summary.get("final_status") or heartbeat.get("status", "unknown")
    model  = summary.get("model_used", "?")
    machine= summary.get("machine_id", "?")
    exps   = summary.get("total_experiments_this_session", 0)
    err    = summary.get("error_message")

    hb_time = heartbeat.get("last_update")
    hb_exp  = heartbeat.get("current_experiment", "none")
    hb_step = heartbeat.get("step_number", 0)

    return {
        "session_id": sid,
        "start":      start,
        "end":        end,
        "status":     status,
        "model":      model,
        "machine":    machine,
        "experiments":exps,
        "error":      err,
        "hb_time":    hb_time,
        "hb_exp":     hb_exp,
        "hb_step":    hb_step,
        "duration":   _duration(start, end),
    }


def get_last_session_info(sessions_dir: Path = _SESSIONS_DIR) -> Optional[Dict[str, Any]]:
    """Return info dict for the most recent session, or None if none exist."""
    sessions_dir = Path(sessions_dir)
    if not sessions_dir.exists():
        return None
    dirs = sorted(
        [d for d in sessions_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not dirs:
        return None
    return get_session_info(dirs[0])


def format_session_report(info: Dict[str, Any], verbose: bool = True) -> str:
    """Format a session info dict into a human-readable string."""
    icon   = _status_icon(info["status"])
    lines  = [
        "═" * 62,
        f"  {icon}  Session : {info['session_id']}",
        f"     Machine   : {info['machine']}",
        f"     Model     : {info['model']}",
        f"     Started   : {info['start'] or '?'}",
    ]
    if info["end"]:
        lines.append(f"     Ended     : {info['end']}  ({info['duration']})")
    else:
        lines.append(f"     Running   : {info['duration']}")

    status_str = info["status"].upper() if info["status"] else "UNKNOWN"
    lines.append(f"     Status    : {status_str}")
    lines.append(f"     Experiments: {info['experiments']} this session")

    if verbose:
        lines.append(f"     Last exp  : {info['hb_exp']}")
        lines.append(f"     Step #    : {info['hb_step']}")

    if info["hb_time"]:
        ago = _ago(info["hb_time"])
        lines.append(f"     Heartbeat : {ago}")
        if info["status"] == "running" and "hour" in ago:
            lines.append(
                "     ⚠  Heartbeat > 1 hour old — session may have crashed silently"
            )

    if info["error"]:
        err = info["error"][:200]
        lines.append(f"     Error     : {err}")

    lines.append("═" * 62)
    return "\n".join(lines)


def check_last_session(
    sessions_dir: Path = _SESSIONS_DIR,
    n:            int  = 1,
    verbose:      bool = True,
) -> None:
    """Print session reports to stdout."""
    sessions_dir = Path(sessions_dir)
    if not sessions_dir.exists():
        print(f"No sessions directory found at: {sessions_dir.resolve()}")
        print("The agent has not been run yet, or sessions/ dir is missing.")
        return

    dirs = sorted(
        [d for d in sessions_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not dirs:
        print("sessions/ directory exists but contains no sessions yet.")
        return

    dirs_to_show = dirs[:n]

    print(f"\nFound {len(dirs)} session(s) total.  Showing last {len(dirs_to_show)}.\n")
    for d in dirs_to_show:
        info = get_session_info(d)
        print(format_session_report(info, verbose=verbose))
        print()


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check the status of the most recent agent session(s)."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show all sessions (brief summary each)"
    )
    parser.add_argument(
        "--n", type=int, default=1,
        help="Number of most recent sessions to show (default: 1)"
    )
    parser.add_argument(
        "--sessions-dir", default="sessions",
        help="Path to sessions directory (default: sessions/)"
    )
    args = parser.parse_args()

    n = 999 if args.all else args.n
    check_last_session(
        sessions_dir = Path(args.sessions_dir),
        n            = n,
        verbose      = True,
    )
