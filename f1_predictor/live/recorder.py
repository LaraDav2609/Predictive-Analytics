"""Optional FastF1 live timing recorder manager.

This starts FastF1's free live timing recorder as a local subprocess. It does
not use credentials or paid APIs. The recorder writes to a local file that
free_sources.py can ingest as `fastf1_recorded_file`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from time import monotonic
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class FastF1LiveRecorderManager:
    def __init__(self, directory: str | None = None) -> None:
        root = directory or os.getenv("F1_FASTF1_LIVE_DIR")
        if not root:
            root = str(Path(__file__).resolve().parents[2] / "data" / "live_timing")
        self.directory = Path(root)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._process: subprocess.Popen | None = None
        self._file: Path | None = None
        self._log_file: Path | None = None
        self._log_handle = None
        self._started_at: str | None = None
        self._last_error: str | None = None
        self._fastf1_available_cache: bool | None = None
        self._fastf1_available_checked_at: float = 0.0

    def status(self) -> dict[str, Any]:
        running = self._process is not None and self._process.poll() is None
        return {
            "ok": True,
            "running": running,
            "pid": self._process.pid if running and self._process else None,
            "file": str(self._file) if self._file else None,
            "file_size": self._file.stat().st_size if self._file and self._file.exists() else None,
            "log_file": str(self._log_file) if self._log_file else None,
            "log_tail": _tail(self._log_file) if self._log_file else "",
            "directory": str(self.directory),
            "started_at": self._started_at,
            "last_error": self._last_error,
            "fastf1_available": self._fastf1_available(),
        }

    def start(self, round_num: int, session: str = "race") -> dict[str, Any]:
        if self._process is not None and self._process.poll() is None:
            return {**self.status(), "reason": "already_running"}

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_session = _safe_token(session or "race")
        self._file = self.directory / f"round_{int(round_num):02d}_{safe_session}_{stamp}.txt"
        self._log_file = self.directory / f"round_{int(round_num):02d}_{safe_session}_{stamp}.log"
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._last_error = None
        self._close_log()

        executable = os.getenv("F1_FASTF1_PYTHON") or sys.executable
        args = [
            executable,
            "-m",
            "f1_predictor.live.fastf1_runner",
            "--append",
            "--timeout",
            "0",
            str(self._file),
        ]
        try:
            self._log_handle = self._log_file.open("ab") if self._log_file else subprocess.DEVNULL
            self._process = subprocess.Popen(
                args,
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=self._log_handle,
                stderr=self._log_handle,
                creationflags=_creation_flags(),
            )
        except Exception as exc:
            self._last_error = str(exc)
            self._process = None
            self._close_log()
            return {**self.status(), "ok": False, "reason": f"recorder_start_failed: {exc}"}
        return {**self.status(), "reason": "started"}

    def stop(self) -> dict[str, Any]:
        if self._process is None or self._process.poll() is not None:
            return {**self.status(), "reason": "not_running"}
        self._process.terminate()
        try:
            self._process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self._process.kill()
        status = {**self.status(), "reason": "stopped"}
        self._close_log()
        return status

    def _fastf1_available(self) -> bool:
        now = monotonic()
        if self._fastf1_available_cache is not None and now - self._fastf1_available_checked_at < 60.0:
            return self._fastf1_available_cache
        executable = os.getenv("F1_FASTF1_PYTHON") or sys.executable
        try:
            result = subprocess.run(
                [executable, "-c", "import fastf1"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
            )
            available = result.returncode == 0
        except Exception:
            available = False
        self._fastf1_available_cache = available
        self._fastf1_available_checked_at = now
        return available

    def _close_log(self) -> None:
        if self._log_handle and self._log_handle is not subprocess.DEVNULL:
            try:
                self._log_handle.close()
            except Exception:
                pass
        self._log_handle = None


def _safe_token(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "race"


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _tail(path: Path | None, limit: int = 1200) -> str:
    if not path or not path.exists():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-limit:].decode("utf-8", errors="ignore").strip()
