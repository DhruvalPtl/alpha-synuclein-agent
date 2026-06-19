"""
agent/core/watchdog.py
────────────────────────────────────────────────────────────────────────────
RunWatchdog — daemon thread that monitors agent liveness and triggers a
graceful stop when the agent is idle too long or exceeds wall-clock budget.

Two triggers (checked every 10 s):
  1. IDLE TIMEOUT   : heartbeat.json step_number unchanged for max_idle_seconds
  2. TOTAL TIMEOUT  : total elapsed exceeds max_total_seconds

Sets stop_event (threading.Event) — the orchestrator polls this between
steps and exits cleanly.  Never force-kills a running tool call.

Usage (wired automatically by AgentOrchestrator):
    watchdog = RunWatchdog(
        stop_event      = orchestrator._stop_event,
        heartbeat_path  = "sessions/<id>/heartbeat.json",
        max_idle_seconds  = 300,   # 5 min stuck = give up
        max_total_seconds = 3600,  # 1 h hard wall
        logger            = tee_logger,
    )
    watchdog.start()
    ...
    watchdog.stop()   # call in finally block
"""

import json
import time
import threading
from pathlib import Path
from typing import Optional


class RunWatchdog(threading.Thread):
    """
    Daemon thread: monitors heartbeat.json and wall-clock time.
    Sets stop_event when limits are hit.

    Parameters
    ----------
    stop_event        : threading.Event shared with the orchestrator
    heartbeat_path    : path to sessions/{id}/heartbeat.json
    max_idle_seconds  : stop if step_number doesn't change for this long
    max_total_seconds : stop if total elapsed exceeds this
    logger            : TeeLogger (optional — falls back to print)
    poll_interval     : how often to check (seconds), default 10
    """

    def __init__(
        self,
        stop_event:         threading.Event,
        heartbeat_path,
        max_idle_seconds:   int   = 300,
        max_total_seconds:  int   = 3600,
        logger              = None,
        poll_interval:      float = 10.0,
    ) -> None:
        super().__init__(daemon=True, name="RunWatchdog")
        self.stop_event        = stop_event
        self.heartbeat_path    = Path(heartbeat_path)
        self.max_idle          = max_idle_seconds
        self.max_total         = max_total_seconds
        self.logger            = logger
        self.poll_interval     = poll_interval

        self._start_time       = time.time()
        self._last_step_num:   Optional[int]   = None
        self._last_step_time:  float           = time.time()
        self._exit_event       = threading.Event()  # internal: tells run() to stop

        # Populated when a limit is hit
        self.stop_reason: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._start_time      = time.time()
        self._last_step_time  = time.time()
        super().start()
        self._log(
            f"[Watchdog] Started. "
            f"idle_limit={self.max_idle}s  "
            f"total_limit={self.max_total}s  "
            f"poll={self.poll_interval}s"
        )

    def stop(self) -> None:
        """Signal the watchdog thread to exit (call from finally block)."""
        self._exit_event.set()

    # ── Thread entry ─────────────────────────────────────────────────────────

    def run(self) -> None:
        while not self._exit_event.wait(self.poll_interval):
            if self.stop_event.is_set():
                break   # already stopped externally

            # ── 1. Total wall-clock check ─────────────────────────────────────
            total_elapsed = time.time() - self._start_time
            if total_elapsed > self.max_total:
                mins = int(total_elapsed // 60)
                self.stop_reason = "total timeout"
                self._log(
                    f"[Watchdog] ⏱  TOTAL TIME LIMIT reached "
                    f"({mins}m ≥ {self.max_total // 60}m). "
                    "Setting stop_event for graceful exit."
                )
                self.stop_event.set()
                break

            # ── 2. Idle (heartbeat) check ─────────────────────────────────────
            if self.heartbeat_path.exists():
                try:
                    hb = json.loads(self.heartbeat_path.read_text(encoding="utf-8"))
                    current_step = int(hb.get("step_number", hb.get("step", 0)) or 0)

                    if self._last_step_num is None:
                        # First read — initialise
                        self._last_step_num  = current_step
                        self._last_step_time = time.time()
                    elif current_step != self._last_step_num:
                        # Progress detected — reset idle clock
                        self._last_step_num  = current_step
                        self._last_step_time = time.time()
                    else:
                        # No progress
                        idle = time.time() - self._last_step_time
                        if idle > self.max_idle:
                            self.stop_reason = "idle timeout"
                            self._log(
                                f"[Watchdog] 💤  IDLE TIMEOUT: step_number={current_step} "
                                f"unchanged for {idle:.0f}s "
                                f"(limit={self.max_idle}s). "
                                "Setting stop_event for graceful exit."
                            )
                            self.stop_event.set()
                            break
                        elif idle > self.max_idle * 0.5:
                            # Early warning at 50 % of idle limit
                            self._log(
                                f"[Watchdog] ⚠  Idle warning: "
                                f"{idle:.0f}s / {self.max_idle}s "
                                f"(step_number={current_step})"
                            )
                except Exception:
                    pass  # heartbeat temporarily unreadable — skip

        self._log(
            f"[Watchdog] Exited. stop_reason={self.stop_reason or 'external'}"
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.agent(msg)
        else:
            print(msg, flush=True)
