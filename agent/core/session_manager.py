"""
agent/core/session_manager.py
────────────────────────────────────────────────────────────────────────────
SessionManager — per-run session tracking with three-way logging,
heartbeat thread, and crash-proof session summaries.

Every AgentOrchestrator.run() creates one session:
  sessions/{session_id}/
    session_log.log         — all TeeLogger output for this session
    session_summary.json    — written at start + updated at end/crash
    heartbeat.json          — overwritten every 10 s by daemon thread

.gitignore keeps session_summary.json tracked but ignores
session_log.log and heartbeat.json (too noisy/large for git).
"""

import datetime
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
_SESSIONS_DIR = _PROJECT_ROOT / "sessions"


def _machine_id() -> str:
    raw = os.environ.get("MACHINE_ID", "").strip()
    if not raw:
        raw = platform.node()
    return raw or "unknown"


class SessionManager:
    """
    Lifecycle:
        sm = SessionManager(model_name="groq-llama", logger=tee_logger)
        sm.start()                           # writes summary, starts heartbeat
        sm.tick(current_exp="exp_003_...",
                step=12, status="running")   # call before/after each experiment
        sm.end(status="completed",
               total_experiments=12)         # call in finally block
    """

    def __init__(
        self,
        model_name: str,
        logger,                  # TeeLogger instance — already initialised
        heartbeat_interval: int = 10,
    ) -> None:
        self.model_name         = model_name
        self.logger             = logger
        self.heartbeat_interval = heartbeat_interval

        # Create a unique session ID from current timestamp
        self.session_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = _SESSIONS_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._log_path       = self.session_dir / "session_log.log"
        self._summary_path   = self.session_dir / "session_summary.json"
        self._heartbeat_path = self.session_dir / "heartbeat.json"

        # Heartbeat state (updated by caller via tick())
        self._hb_lock       = threading.Lock()
        self._hb_exp        = "none"
        self._hb_step       = 0
        self._hb_status     = "starting"
        self._hb_stop       = threading.Event()

        # Attach this session's log to TeeLogger as the third write target
        self._attach_session_log()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Write initial session_summary.json and launch heartbeat thread."""
        summary = {
            "session_id":   self.session_id,
            "start_time":   datetime.datetime.now().isoformat(timespec="seconds"),
            "model_used":   self.model_name,
            "machine_id":   _machine_id(),
            "end_time":     None,
            "total_experiments_this_session": 0,
            "final_status": "running",
            "error_message": None,
        }
        self._write_json(self._summary_path, summary)

        # Start daemon heartbeat thread
        t = threading.Thread(target=self._heartbeat_loop, daemon=True)
        t.start()

        self.logger.agent(
            f"[Session] Started  session_id={self.session_id}"
            f"  dir={self.session_dir}"
        )

    def tick(
        self,
        current_exp: str = "none",
        step:        int = 0,
        status:      str = "running",
    ) -> None:
        """Update the heartbeat payload. Thread-safe."""
        with self._hb_lock:
            self._hb_exp    = current_exp
            self._hb_step   = step
            self._hb_status = status
        # Force an immediate write so callers see fresh data without waiting
        self._write_heartbeat()

    def end(
        self,
        status:             str = "completed",
        total_experiments:  int = 0,
        error_message:      Optional[str] = None,
    ) -> None:
        """
        Finalise the session. Call this in a finally block so it runs even
        on crash / KeyboardInterrupt.
        """
        self._hb_stop.set()   # stop heartbeat daemon

        # Update the tick status so the final heartbeat reflects reality
        with self._hb_lock:
            self._hb_status = status
        self._write_heartbeat()

        # Update session_summary.json
        try:
            if self._summary_path.exists():
                summary = json.loads(self._summary_path.read_text(encoding="utf-8"))
            else:
                summary = {}
        except Exception:
            summary = {}

        summary["end_time"]   = datetime.datetime.now().isoformat(timespec="seconds")
        summary["total_experiments_this_session"] = total_experiments
        summary["final_status"]  = status
        summary["error_message"] = error_message
        self._write_json(self._summary_path, summary)

        self.logger.agent(
            f"[Session] Ended  status={status}"
            f"  experiments={total_experiments}"
            f"  session={self.session_id}"
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _attach_session_log(self) -> None:
        """
        Monkey-patch TeeLogger._emit so every log line also writes to
        this session's session_log.log.

        We do NOT replace the existing master_log or experiment log writes —
        this is purely additive (three-way becomes four-way if an experiment
        log is also active, but for the session log it becomes three-way).
        """
        session_log = self._log_path
        session_log.parent.mkdir(parents=True, exist_ok=True)

        # Write session header
        header = (
            f"\n{'=' * 80}\n"
            f"  SESSION LOG  {self.session_id}\n"
            f"{'=' * 80}\n"
        )
        try:
            with open(session_log, "a", encoding="utf-8") as fh:
                fh.write(header)
        except OSError:
            pass

        # Wrap _emit to also write to our session log
        original_emit = self.logger.__class__._emit

        def _patched_emit(logger_self, level: str, message: str) -> None:
            original_emit(logger_self, level, message)
            # Plain-text write to session log (same format as master log)
            plain = logger_self._format_plain(level, message)
            try:
                with open(session_log, "a", encoding="utf-8") as fh:
                    fh.write(plain + "\n")
            except OSError:
                pass

        # Bind to instance so only this process's logger gets patched
        import types
        self.logger._emit = types.MethodType(_patched_emit, self.logger)
        self._original_emit = original_emit   # keep reference for cleanup

    def _heartbeat_loop(self) -> None:
        """Daemon thread: overwrite heartbeat.json every heartbeat_interval s."""
        while not self._hb_stop.wait(timeout=self.heartbeat_interval):
            self._write_heartbeat()

    def _write_heartbeat(self) -> None:
        with self._hb_lock:
            payload = {
                "session_id":        self.session_id,
                "last_update":       datetime.datetime.now().isoformat(timespec="seconds"),
                "current_experiment": self._hb_exp,
                "step_number":       self._hb_step,
                "status":            self._hb_status,
            }
        self._write_json(self._heartbeat_path, payload)

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError:
            pass   # never crash the training loop for logging
