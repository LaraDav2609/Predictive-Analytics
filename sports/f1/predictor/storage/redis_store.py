"""Redis hot-state store for F1 dashboard data."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from time import monotonic
from typing import Any

import redis


DEFAULT_REDIS_URL = "redis://localhost:6379/0"


@dataclass(frozen=True)
class F1RedisKeys:
    """Centralized F1 Redis key policy."""

    prefix: str = "f1"

    def live_state(self, season: int, round_num: int, session: str) -> str:
        return f"{self.prefix}:live:{season}:{round_num}:{_session(session)}:state"

    def truth_snapshot(self, season: int, round_num: int, session: str) -> str:
        return f"{self.prefix}:truth:{season}:{round_num}:{_session(session)}:latest"

    def probability_latest(self, season: int, round_num: int, session: str, model_id: str) -> str:
        return f"{self.prefix}:prob:{season}:{round_num}:{_session(session)}:{model_id or 'production_v1'}:latest"

    def probability_channel(self, season: int, round_num: int, session: str, model_id: str) -> str:
        return f"{self.prefix}:pub:prob:{season}:{round_num}:{_session(session)}:{model_id or 'production_v1'}"

    def racehub(self, round_num: int, tab: str, payload_hash: str) -> str:
        return f"{self.prefix}:racehub:{round_num}:{tab}:{payload_hash}"

    def track(self, track_key: str, geometry_hash: str) -> str:
        return f"{self.prefix}:track:{track_key}:{geometry_hash}"

    def sentiment_latest(self) -> str:
        return f"{self.prefix}:sentiment:latest"

    def race_sentiment(self, season: int, round_num: int, session: str) -> str:
        return f"{self.prefix}:sentiment:race:{season}:{round_num}:{_session(session)}"


class F1RedisStore:
    """Volatile Redis storage for hot F1 UI state.

    Every method is best-effort and returns a compact status dict instead of
    raising, so storage outages never break prediction routes.
    """

    def __init__(self, redis_url: str | None = None, prefix: str = "f1") -> None:
        self.redis_url = redis_url or os.getenv("F1_REDIS_URL") or os.getenv("REDIS_URL") or DEFAULT_REDIS_URL
        self.keys = F1RedisKeys(prefix=prefix)
        self._client: redis.Redis | None = None
        self._last_error: str | None = None
        self._disabled_until = 0.0

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True, socket_connect_timeout=1.0, socket_timeout=1.0)
        return self._client

    def health(self) -> dict[str, Any]:
        if monotonic() < self._disabled_until:
            return {
                "available": False,
                "url": _redact_url(self.redis_url),
                "used_memory_human": None,
                "last_error": self._last_error,
                "backoff_seconds": round(self._disabled_until - monotonic(), 2),
            }
        try:
            self.client.ping()
            info = self.client.info("memory")
            return {
                "available": True,
                "url": _redact_url(self.redis_url),
                "used_memory_human": info.get("used_memory_human"),
                "last_error": None,
            }
        except Exception as exc:  # pragma: no cover - depends on local service
            self._last_error = str(exc)
            self._disabled_until = monotonic() + 15.0
            return {
                "available": False,
                "url": _redact_url(self.redis_url),
                "used_memory_human": None,
                "last_error": self._last_error,
            }

    def set_live_state(self, season: int, round_num: int, session: str, payload: dict[str, Any], ttl_seconds: int = 60) -> dict[str, Any]:
        key = self.keys.live_state(season, round_num, session)
        return self._set_json(key, payload, ttl_seconds)

    def set_truth_snapshot(self, season: int, round_num: int, session: str, payload: dict[str, Any], ttl_seconds: int = 120) -> dict[str, Any]:
        key = self.keys.truth_snapshot(season, round_num, session)
        return self._set_json(key, payload, ttl_seconds)

    def set_probability_snapshot(
        self,
        season: int,
        round_num: int,
        session: str,
        model_id: str,
        payload: dict[str, Any],
        ttl_seconds: int = 900,
        publish: bool = True,
    ) -> dict[str, Any]:
        key = self.keys.probability_latest(season, round_num, session, model_id)
        status = self._set_json(key, payload, ttl_seconds)
        if publish and status.get("ok"):
            try:
                self.client.publish(self.keys.probability_channel(season, round_num, session, model_id), json.dumps(payload, default=_json_default))
            except Exception as exc:  # pragma: no cover - depends on local service
                self._last_error = str(exc)
                status["publish_ok"] = False
                status["publish_error"] = self._last_error
            else:
                status["publish_ok"] = True
        return status

    def set_racehub_cache(self, round_num: int, tab: str, payload: dict[str, Any], ttl_seconds: int = 180) -> dict[str, Any]:
        payload_hash = sha1(json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")).hexdigest()[:16]
        key = self.keys.racehub(round_num, tab, payload_hash)
        return self._set_json(key, payload, ttl_seconds)

    def set_track_geometry(self, track_key: str, geometry_hash: str, payload: dict[str, Any], ttl_seconds: int = 86400) -> dict[str, Any]:
        return self._set_json(self.keys.track(track_key, geometry_hash), payload, ttl_seconds)

    def set_sentiment_latest(self, payload: dict[str, Any], ttl_seconds: int = 3600) -> dict[str, Any]:
        return self._set_json(self.keys.sentiment_latest(), payload, ttl_seconds)

    def set_race_sentiment(self, season: int, round_num: int, session: str, payload: dict[str, Any], ttl_seconds: int = 3600) -> dict[str, Any]:
        return self._set_json(self.keys.race_sentiment(season, round_num, session), payload, ttl_seconds)

    def _set_json(self, key: str, payload: dict[str, Any], ttl_seconds: int) -> dict[str, Any]:
        if monotonic() < self._disabled_until:
            return {"ok": False, "key": key, "reason": self._last_error or "redis_backoff"}
        try:
            self.client.set(key, json.dumps(payload, ensure_ascii=False, default=_json_default), ex=max(1, int(ttl_seconds)))
            return {"ok": True, "key": key, "ttl_seconds": max(1, int(ttl_seconds))}
        except Exception as exc:  # pragma: no cover - depends on local service
            self._last_error = str(exc)
            self._disabled_until = monotonic() + 15.0
            return {"ok": False, "key": key, "reason": self._last_error}


def _session(session: str) -> str:
    value = (session or "race").lower()
    if value.startswith("qual"):
        return "qualifying"
    if value.startswith("sprint"):
        return "sprint"
    return "race"


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _redact_url(url: str) -> str:
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1) if "://" in url else ("", url)
    host = rest.split("@", 1)[1]
    return f"{scheme}://***@{host}" if scheme else f"***@{host}"
