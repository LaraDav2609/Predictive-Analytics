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
        self._module_cache: dict[tuple[str, str], tuple[float, bool]] = {}
        self._growth_cache: dict[str, tuple[int, float]] = {}

    def status(self, round_num: int | None = None, session: str = "race") -> dict[str, Any]:
        running = self._process is not None and self._process.poll() is None
        parsed = self._recording_metrics(round_num=round_num, session=session)
        executable = self._python_executable()
        fastf1_available = self._module_available(executable, "fastf1")
        signalrcore_available = self._module_available(executable, "signalrcore")
        setup = self._setup_status(fastf1_available, signalrcore_available, parsed, running)
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
            "python_executable": executable,
            "fastf1_available": fastf1_available,
            "signalrcore_available": signalrcore_available,
            "setup_status": setup["status"],
            "next_setup_action": setup["next_action"],
            "recording_file_growth": parsed["file_growth_status"],
            "parsed_driver_count": parsed["parsed_driver_count"],
            "parsed_row_count": parsed["parsed_row_count"],
            "parsed_real_gap_count": parsed["parsed_real_gap_count"],
            "parsed_tyre_count": parsed["parsed_tyre_count"],
            "last_parsed_timestamp": parsed["last_parsed_timestamp"],
            "recording_source_mode": setup["recording_source_mode"],
            "install_command": f"\"{executable}\" -m pip install fastf1 signalrcore",
        }

    def start(self, round_num: int, session: str = "race") -> dict[str, Any]:
        if self._process is not None and self._process.poll() is None:
            return {**self.status(round_num, session), "reason": "already_running"}

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_session = _safe_token(session or "race")
        self._file = self.directory / f"round_{int(round_num):02d}_{safe_session}_{stamp}.txt"
        self._log_file = self.directory / f"round_{int(round_num):02d}_{safe_session}_{stamp}.log"
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._last_error = None
        self._close_log()

        executable = self._python_executable()
        if not self._module_available(executable, "fastf1"):
            self._last_error = "fastf1_unavailable"
            return {**self.status(round_num, session), "ok": False, "reason": "fastf1_unavailable"}
        args = [
            executable,
            "-m",
            "sports.f1.predictor.live.fastf1_runner",
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
            return {**self.status(round_num, session), "ok": False, "reason": f"recorder_start_failed: {exc}"}
        return {**self.status(round_num, session), "reason": "started"}

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

    def _python_executable(self) -> str:
        configured = os.getenv("F1_FASTF1_PYTHON")
        if configured and Path(configured).exists():
            return configured
        candidates = [
            Path(__file__).resolve().parents[3] / "Sentiment" / ".venv" / "Scripts" / "python.exe",
            Path(__file__).resolve().parents[2] / ".venv" / "Scripts" / "python.exe",
            Path(sys.executable),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return sys.executable

    def _fastf1_available(self) -> bool:
        return self._module_available(self._python_executable(), "fastf1")

    def _module_available(self, executable: str, module: str) -> bool:
        now = monotonic()
        key = (executable, module)
        cached = self._module_cache.get(key)
        if cached and now - cached[0] < 60.0:
            return cached[1]
        try:
            result = subprocess.run(
                [executable, "-c", f"import {module}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
            )
            available = result.returncode == 0
        except Exception:
            available = False
        if module == "fastf1":
            self._fastf1_available_cache = available
            self._fastf1_available_checked_at = now
        self._module_cache[key] = (now, available)
        return available

    def _recording_metrics(self, round_num: int | None = None, session: str = "race") -> dict[str, Any]:
        file = self._file if self._file and self._file.exists() else _find_latest_recording(self.directory, round_num, session)
        if not file:
            return _empty_recording_metrics("missing")
        try:
            stat = file.stat()
        except OSError:
            return _empty_recording_metrics("missing")

        previous = self._growth_cache.get(str(file))
        self._growth_cache[str(file)] = (stat.st_size, monotonic())
        if stat.st_size <= 0:
            growth = "empty"
        elif previous and stat.st_size > previous[0]:
            growth = "growing"
        else:
            growth = "stable"

        rows = _read_recording_rows_safe(file)
        numbers = {_driver_number(row) for row in rows}
        numbers.discard(None)
        timestamps = [_row_time(row) for row in rows if _row_time(row)]
        return {
            "file": str(file),
            "file_size": stat.st_size,
            "file_growth_status": growth,
            "parsed_row_count": len(rows),
            "parsed_driver_count": len(numbers),
            "parsed_real_gap_count": sum(1 for row in rows if _first(row, "gap_to_leader", "GapToLeader", "gap", "IntervalToPositionAhead", "interval")),
            "parsed_tyre_count": sum(1 for row in rows if _first(row, "compound", "Compound", "tyre_compound") or _first(row, "Stints", "stints")),
            "last_parsed_timestamp": timestamps[-1] if timestamps else None,
        }

    def _setup_status(self, fastf1_available: bool, signalrcore_available: bool, parsed: dict[str, Any], running: bool) -> dict[str, str]:
        parsed_drivers = int(parsed.get("parsed_driver_count") or 0)
        growth = parsed.get("file_growth_status")
        if not fastf1_available or not signalrcore_available:
            return {
                "status": "missing_dependencies",
                "recording_source_mode": "unavailable",
                "next_action": "Install FastF1 dependencies for the configured Python runtime.",
            }
        if running and parsed_drivers <= 0:
            return {
                "status": "recording_pending",
                "recording_source_mode": "recording_pending",
                "next_action": "Recorder is running; wait for live timing messages and file growth.",
            }
        if parsed_drivers >= 18:
            return {
                "status": "recorded_confident",
                "recording_source_mode": "recorded_confident",
                "next_action": "Keep recorder running and monitor parsed driver count.",
            }
        if parsed_drivers > 0:
            return {
                "status": "recorded_partial",
                "recording_source_mode": "recorded",
                "next_action": "Recording parsed partial timing; keep recorder running until more drivers appear.",
            }
        if growth == "empty":
            return {
                "status": "recording_empty",
                "recording_source_mode": "recording_pending",
                "next_action": "Recording file exists but has no usable timing rows yet.",
            }
        return {
            "status": "ready",
            "recording_source_mode": "estimated",
            "next_action": "Start FastF1 recorder before or during the session.",
        }

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


def _find_latest_recording(directory: Path, round_num: int | None, session: str = "race") -> Path | None:
    if not directory.exists():
        return None
    session_token = _safe_token(session or "race")
    round_tokens = []
    if round_num is not None:
        try:
            round_tokens = [f"round_{int(round_num):02d}", f"round_{int(round_num)}", f"r{int(round_num)}"]
        except (TypeError, ValueError):
            round_tokens = []
    candidates = [
        item for item in directory.glob("*")
        if item.is_file()
        and item.suffix.lower() in {".json", ".jsonl", ".txt"}
        and session_token in item.name.lower()
        and (not round_tokens or any(token in item.name.lower() for token in round_tokens))
    ]
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def _empty_recording_metrics(status: str) -> dict[str, Any]:
    return {
        "file": None,
        "file_size": None,
        "file_growth_status": status,
        "parsed_row_count": 0,
        "parsed_driver_count": 0,
        "parsed_real_gap_count": 0,
        "parsed_tyre_count": 0,
        "last_parsed_timestamp": None,
    }


def _read_recording_rows_safe(path: Path) -> list[dict[str, Any]]:
    try:
        from sports.f1.predictor.live.free_sources import _read_recording_rows

        return _read_recording_rows(path)
    except Exception:
        return []


def _driver_number(row: dict[str, Any]) -> int | None:
    return _safe_int(_first(row, "driver_number", "RacingNumber", "racing_number", "number", "Number"))


def _row_time(row: dict[str, Any]) -> str:
    return str(_first(row, "date", "timestamp", "Utc", "utc", "Time") or "")


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
