"""Disk cache for provider-normalized F1 telemetry snapshots."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "f1_telemetry_cache_v1"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[4] / ".cache" / "f1" / "telemetry"


class TelemetryCache:
    """Small JSON snapshot cache for telemetry provider payloads.

    The implementation intentionally keeps the file contract close to the
    planned Parquet layout: one session directory per source/season/round with
    row groups and a manifest. JSON keeps this dependency-free for the first
    implementation slice.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.getenv("F1_TELEMETRY_CACHE_DIR") or DEFAULT_CACHE_DIR)

    def session_dir(self, source: str, season: int, round_num: int, session: str) -> Path:
        return self.root / _safe_part(source) / str(int(season)) / f"{int(round_num):02d}" / _safe_part(session)

    def read_snapshot(
        self,
        *,
        source: str,
        season: int,
        round_num: int,
        session: str,
        live: bool = False,
    ) -> dict[str, Any] | None:
        directory = self.session_dir(source, season, round_num, session)
        manifest = _read_json(directory / "manifest.json")
        if not manifest or manifest.get("schema_version") != SCHEMA_VERSION:
            return None
        if live and _is_expired(manifest, default_ttl_seconds=8):
            return None
        payload = {
            "ok": True,
            "source": source,
            "session": session,
            "cache": {
                "hit": True,
                "schema_version": manifest.get("schema_version"),
                "fetched_at": manifest.get("fetched_at"),
                "path": str(directory),
            },
            "manifest": manifest,
        }
        for name in ("openf1_session", "car_data", "location", "trace_points", "laps", "stints", "intervals", "weather"):
            data = _read_json(directory / f"{name}.json")
            if data is not None:
                payload[name] = data
        payload["raw_counts"] = dict(manifest.get("raw_counts") or {})
        return payload

    def write_snapshot(
        self,
        *,
        source: str,
        season: int,
        round_num: int,
        session: str,
        payload: dict[str, Any],
        live: bool = False,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        directory = self.session_dir(source, season, round_num, session)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source": source,
            "season": int(season),
            "round": int(round_num),
            "session": session,
            "live": bool(live),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "ttl_seconds": ttl_seconds,
            "raw_counts": dict(payload.get("raw_counts") or {}),
            "rate_limit": payload.get("rate_limit") or payload.get("last_error"),
        }
        for name in ("openf1_session", "car_data", "location", "trace_points", "laps", "stints", "intervals", "weather"):
            if name in payload:
                _write_json(directory / f"{name}.json", payload.get(name))
        _write_json(directory / "manifest.json", manifest)
        cached = deepcopy(payload)
        cached["cache"] = {
            "hit": False,
            "schema_version": SCHEMA_VERSION,
            "fetched_at": manifest["fetched_at"],
            "path": str(directory),
        }
        cached["manifest"] = manifest
        return cached


def _read_json(path: Path) -> Any:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _is_expired(manifest: dict[str, Any], *, default_ttl_seconds: int) -> bool:
    ttl = manifest.get("ttl_seconds")
    ttl_seconds = int(ttl if ttl is not None else default_ttl_seconds)
    if ttl_seconds <= 0:
        return False
    fetched_at = manifest.get("fetched_at")
    if not fetched_at:
        return True
    try:
        fetched = datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
    except ValueError:
        return True
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - fetched).total_seconds() > ttl_seconds


def _safe_part(value: str | None) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value or "unknown").lower())
    return text.strip("-") or "unknown"
