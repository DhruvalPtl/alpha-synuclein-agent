"""
agent/core/tee_logger.py
────────────────────────────────────────────────────────────────────────────
Singleton TeeLogger.

Every call to logger.info() / .warning() / .error() / .agent() writes:
  (a) Jupyter cell output  — with ANSI colour by level
  (b) master_log/master_terminal.log  — timestamped, appended forever
  (c) current experiment log  — set via set_experiment_log(path)

Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] message

Usage
-----
    from agent.core.tee_logger import TeeLogger
    logger = TeeLogger()                        # always returns the same instance
    logger.set_experiment_log("experiments/001/run.log")
    logger.info("Pipeline started")
    logger.agent("Searching HuggingFace for pretrained encoder …")
    logger.warning("Class imbalance detected — consider oversampling")
    logger.error("CSV column 'sequence' not found")
"""

import sys
import os
import datetime
import threading
from pathlib import Path
from typing import Optional

# ── ANSI colour codes ──────────────────────────────────────────────────────────
_RESET   = "\033[0m"
_GREEN   = "\033[32m"   # INFO
_YELLOW  = "\033[33m"   # WARNING
_RED     = "\033[31m"   # ERROR
_CYAN    = "\033[36m"   # AGENT
_BOLD    = "\033[1m"

_LEVEL_COLOURS = {
    "INFO":    _GREEN,
    "WARNING": _YELLOW,
    "ERROR":   _RED,
    "AGENT":   _CYAN,
}

_MASTER_LOG_DIR  = "master_log"
_MASTER_LOG_FILE = "master_terminal.log"


class TeeLogger:
    """
    Thread-safe singleton logger that tees every message to:
      • sys.__stdout__ (Jupyter cell / terminal)
      • master_log/master_terminal.log
      • (optional) current experiment log
    """

    _instance: Optional["TeeLogger"] = None
    _lock = threading.Lock()

    # ── Singleton constructor ──────────────────────────────────────────────────
    def __new__(cls, master_log_dir: str = _MASTER_LOG_DIR):
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self, master_log_dir: str = _MASTER_LOG_DIR):
        # Guard: only run __init__ body once even if __new__ is called many times
        if self._initialized:
            return

        self._master_log_dir  = Path(master_log_dir)
        self._master_log_path = self._master_log_dir / _MASTER_LOG_FILE
        self._exp_log_path: Optional[Path] = None
        self._write_lock = threading.Lock()

        # Ensure master log directory exists
        self._master_log_dir.mkdir(parents=True, exist_ok=True)

        # Write a session-start separator to master log
        self._append_to_file(
            self._master_log_path,
            f"\n{'═' * 80}\n"
            f"  SESSION START  {self._now()}\n"
            f"{'═' * 80}\n",
        )

        self._initialized = True

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_experiment_log(self, path: str) -> None:
        """Point the per-experiment log file at *path*.  Created if absent."""
        self._exp_log_path = Path(path)
        self._exp_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._append_to_file(
            self._exp_log_path,
            f"\n{'─' * 60}\n"
            f"  EXPERIMENT LOG START  {self._now()}\n"
            f"{'─' * 60}\n",
        )

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warning(self, message: str) -> None:
        self._emit("WARNING", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)

    def agent(self, message: str) -> None:
        self._emit("AGENT", message)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _now(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _format_plain(self, level: str, message: str) -> str:
        return f"[{self._now()}] [{level:<7}] {message}"

    def _format_colour(self, level: str, message: str) -> str:
        colour = _LEVEL_COLOURS.get(level, "")
        tag    = f"{_BOLD}{colour}[{level:<7}]{_RESET}"
        ts     = f"\033[90m[{self._now()}]{_RESET}"   # dim grey timestamp
        body   = f"{colour}{message}{_RESET}"
        return f"{ts} {tag} {body}"

    def _emit(self, level: str, message: str) -> None:
        plain   = self._format_plain(level, message)
        coloured = self._format_colour(level, message)

        with self._write_lock:
            # (a) Jupyter / terminal output (colour)
            print(coloured, file=sys.__stdout__, flush=True)

            # (b) Master log (plain text, timestamped)
            self._append_to_file(self._master_log_path, plain + "\n")

            # (c) Experiment log (plain text, timestamped)
            if self._exp_log_path is not None:
                self._append_to_file(self._exp_log_path, plain + "\n")

    @staticmethod
    def _append_to_file(path: Path, text: str) -> None:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            # Never crash the training loop just because logging failed
            print(
                f"\033[31m[TeeLogger] WRITE ERROR: {exc}\033[0m",
                file=sys.__stderr__,
                flush=True,
            )

    # ── Convenience: intercept bare print() calls ──────────────────────────────
    def redirect_print(self) -> None:
        """
        After calling this, any plain print() call is captured and forwarded
        through logger.info().  Useful when third-party libraries print progress.
        """
        outer = self

        class _PrintCapture:
            def write(self, msg: str) -> None:
                msg = msg.rstrip("\n")
                if msg:
                    outer.info(msg)

            def flush(self) -> None:
                pass

        sys.stdout = _PrintCapture()  # type: ignore[assignment]

    def restore_print(self) -> None:
        """Undo redirect_print()."""
        sys.stdout = sys.__stdout__
