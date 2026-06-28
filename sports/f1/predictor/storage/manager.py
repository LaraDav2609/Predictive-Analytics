"""Best-effort F1 storage facade."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from copy import deepcopy
from time import monotonic
from typing import Any

from .clickhouse_store import F1ClickHouseStore
from .redis_store import F1RedisStore


class F1Storage:
    """Coordinates Redis hot-state writes and ClickHouse analytics appends."""

    def __init__(self, redis_store: F1RedisStore | None = None, clickhouse_store: F1ClickHouseStore | None = None) -> None:
        self.redis = redis_store or F1RedisStore()
        self.clickhouse = clickhouse_store or F1ClickHouseStore()
        self._health_cache: tuple[float, dict[str, Any]] | None = None
        self._health_cache_ttl_seconds = 5.0
        self._clickhouse_health_timeout_seconds = 0.35
        self._clickhouse_write_timeout_seconds = 0.35

    async def close(self) -> None:
        await self.clickhouse.close()

    async def health(self) -> dict[str, Any]:
        now = monotonic()
        if self._health_cache and now - self._health_cache[0] < self._health_cache_ttl_seconds:
            payload = deepcopy(self._health_cache[1])
            payload["cached"] = True
            payload["cache_age_seconds"] = round(now - self._health_cache[0], 3)
            return payload
        payload = {
            "redis": self.redis.health(),
            "clickhouse": await self._clickhouse_health_best_effort(),
        }
        self._health_cache = (now, deepcopy(payload))
        payload["cached"] = False
        payload["cache_age_seconds"] = 0.0
        return payload

    async def ensure_schema(self) -> dict[str, Any]:
        return await self.clickhouse.ensure_schema()

    async def persist_live_state(self, season: int, race: Any, session: str, state: dict[str, Any]) -> dict[str, Any]:
        redis_status = self.redis.set_live_state(season, int(getattr(race, "round", state.get("round", 0)) or 0), session, state, ttl_seconds=60)
        clickhouse_status = await self._clickhouse_best_effort(
            lambda: self.clickhouse.append_live_state(season, race, session, state),
            "f1_live_snapshots",
        )
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_truth_snapshot(self, season: int, race: Any, session: str, truth: dict[str, Any]) -> dict[str, Any]:
        round_num = int(getattr(race, "round", truth.get("round", 0)) or 0)
        ttl = 60 if truth.get("source_mode") == "live" else 300
        redis_status = self.redis.set_truth_snapshot(season, round_num, session, truth, ttl_seconds=ttl)
        clickhouse_status = await self._clickhouse_best_effort(
            lambda: self.clickhouse.append_live_state(season, race, session, truth),
            "f1_live_snapshots",
        )
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_probability_snapshot(self, season: int, race: Any, session: str, payload: dict[str, Any], model_id: str = "production_v1") -> dict[str, Any]:
        payload = {
            **payload,
            "generated_at": payload.get("generated_at") or datetime.now(timezone.utc).isoformat(),
            "model_id": model_id or payload.get("model_id") or "production_v1",
        }
        round_num = int(getattr(race, "round", payload.get("round", 0)) or 0)
        redis_status = self.redis.set_probability_snapshot(season, round_num, session, model_id, payload, ttl_seconds=900)
        clickhouse_status = await self._clickhouse_best_effort(
            lambda: self.clickhouse.append_probability_snapshot(season, race, session, model_id, payload),
            "f1_probability_snapshots",
        )
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_telemetry_snapshot(self, season: int, race: Any, session: str, payload: dict[str, Any], model_id: str = "telemetry_simulator_v1") -> dict[str, Any]:
        model = payload.get("telemetry_model") or payload.get("model") or payload
        resolved_model_id = model_id or model.get("model_id") or "telemetry_simulator_v1"
        payload = {
            **payload,
            "generated_at": payload.get("generated_at") or model.get("generated_at") or datetime.now(timezone.utc).isoformat(),
            "model_id": resolved_model_id,
        }
        round_num = int(getattr(race, "round", payload.get("round", 0)) or 0)
        redis_status = self.redis.set_telemetry_snapshot(season, round_num, session, resolved_model_id, payload, ttl_seconds=900)
        clickhouse_status = await self._clickhouse_best_effort(
            lambda: self.clickhouse.append_telemetry_snapshot(season, race, session, payload),
            "f1_telemetry_model_outputs",
        )
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_simulation_run(self, season: int, race: Any, session: str, payload: dict[str, Any], live: bool = False) -> dict[str, Any]:
        clickhouse_status = await self._clickhouse_best_effort(
            lambda: self.clickhouse.append_simulation_run(season, race, session, payload, live=live),
            "f1_simulation_runs",
        )
        return {"clickhouse": clickhouse_status}

    async def persist_track_geometry(self, race: Any, track: dict[str, Any]) -> dict[str, Any]:
        if not track:
            return {"redis": {"ok": True, "rows": 0}, "clickhouse": {"ok": True, "rows": 0}}
        geometry_hash = str(track.get("geometry_hash") or track.get("hash") or "")
        if not geometry_hash:
            geometry_hash = str(abs(hash(str(track.get("path") or track.get("display_points") or track.get("points") or ""))))
        track_key = str(track.get("track_key") or getattr(race, "circuit", "") or "unknown")
        redis_status = self.redis.set_track_geometry(track_key, geometry_hash, track, ttl_seconds=86400)
        clickhouse_status = await self._clickhouse_best_effort(
            lambda: self.clickhouse.append_track_geometry(race, track),
            "f1_track_geometry_versions",
        )
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_sentiment(self, season: int, items: list[dict[str, Any]], aggregates: dict[str, Any]) -> dict[str, Any]:
        redis_status = self.redis.set_sentiment_latest(aggregates, ttl_seconds=3600)
        clickhouse_status = await self._clickhouse_best_effort(
            lambda: self.clickhouse.append_sentiment(season, items, aggregates),
            "f1_sentiment",
        )
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_race_sentiment_impact(self, season: int, race: Any, session: str, impact: dict[str, Any]) -> dict[str, Any]:
        round_num = int(getattr(race, "round", impact.get("race_round", 0)) or 0)
        redis_status = self.redis.set_race_sentiment(season, round_num, session, impact, ttl_seconds=3600)
        # Reuse existing entity sentiment archive shape for v1; full race-impact tables can be added later.
        clickhouse_status = await self._clickhouse_best_effort(
            lambda: self.clickhouse.append_sentiment(season, [], {
                "composite": {
                    "MarketCategory": "f1",
                    "Composite": impact.get("confidence") or 0.0,
                    "Label": "RaceImpact",
                    "TotalItems": impact.get("article_count") or 0,
                    "Sources": {},
                    "Topics": {},
                },
                "drivers": impact.get("drivers") or {},
                "teams": impact.get("constructors") or {},
            }),
            "f1_sentiment",
        )
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_backtest_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        clickhouse_status = await self._clickhouse_best_effort(
            lambda: self.clickhouse.append_backtest_result(payload),
            "f1_backtest_results",
        )
        return {"clickhouse": clickhouse_status}

    async def _clickhouse_best_effort(self, operation_factory, table: str) -> dict[str, Any]:
        skipped = self._clickhouse_write_preflight(table)
        if skipped is not None:
            return skipped

        try:
            operation = operation_factory() if callable(operation_factory) else operation_factory
            return await asyncio.wait_for(operation, timeout=self._clickhouse_write_timeout_seconds)
        except asyncio.TimeoutError:
            if hasattr(self.clickhouse, "_disabled_until"):
                self.clickhouse._disabled_until = monotonic() + 30.0
            if hasattr(self.clickhouse, "_last_error"):
                self.clickhouse._last_error = "clickhouse_write_timeout"
            return {
                "ok": False,
                "rows": 0,
                "table": table,
                "reason": "clickhouse_write_timeout",
                "best_effort": True,
                "timeout_seconds": self._clickhouse_write_timeout_seconds,
            }

    def _clickhouse_write_preflight(self, table: str) -> dict[str, Any] | None:
        if getattr(self.clickhouse, "enabled", True) is False:
            return {
                "ok": False,
                "rows": 0,
                "table": table,
                "reason": "clickhouse_disabled",
                "best_effort": True,
                "skipped": True,
            }
        disabled_until = float(getattr(self.clickhouse, "_disabled_until", 0.0) or 0.0)
        now = monotonic()
        if disabled_until > now:
            return {
                "ok": False,
                "rows": 0,
                "table": table,
                "reason": getattr(self.clickhouse, "_last_error", None) or "clickhouse_backoff",
                "best_effort": True,
                "skipped": True,
                "backoff_seconds": round(disabled_until - now, 2),
            }
        return None

    async def _clickhouse_health_best_effort(self) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(self.clickhouse.health(), timeout=self._clickhouse_health_timeout_seconds)
        except asyncio.TimeoutError:
            if hasattr(self.clickhouse, "_disabled_until"):
                self.clickhouse._disabled_until = monotonic() + 30.0
            if hasattr(self.clickhouse, "_last_error"):
                self.clickhouse._last_error = "clickhouse_health_timeout"
            return {
                "available": False,
                "enabled": True,
                "database": getattr(self.clickhouse, "database", None),
                "last_error": "clickhouse_health_timeout",
                "best_effort": True,
                "timeout_seconds": self._clickhouse_health_timeout_seconds,
                "backoff_seconds": 30.0,
            }
