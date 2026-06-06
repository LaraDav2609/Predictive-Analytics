"""Best-effort F1 storage facade."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .clickhouse_store import F1ClickHouseStore
from .redis_store import F1RedisStore


class F1Storage:
    """Coordinates Redis hot-state writes and ClickHouse analytics appends."""

    def __init__(self, redis_store: F1RedisStore | None = None, clickhouse_store: F1ClickHouseStore | None = None) -> None:
        self.redis = redis_store or F1RedisStore()
        self.clickhouse = clickhouse_store or F1ClickHouseStore()

    async def close(self) -> None:
        await self.clickhouse.close()

    async def health(self) -> dict[str, Any]:
        return {
            "redis": self.redis.health(),
            "clickhouse": await self.clickhouse.health(),
        }

    async def ensure_schema(self) -> dict[str, Any]:
        return await self.clickhouse.ensure_schema()

    async def persist_live_state(self, season: int, race: Any, session: str, state: dict[str, Any]) -> dict[str, Any]:
        redis_status = self.redis.set_live_state(season, int(getattr(race, "round", state.get("round", 0)) or 0), session, state, ttl_seconds=60)
        clickhouse_status = await self.clickhouse.append_live_state(season, race, session, state)
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_truth_snapshot(self, season: int, race: Any, session: str, truth: dict[str, Any]) -> dict[str, Any]:
        round_num = int(getattr(race, "round", truth.get("round", 0)) or 0)
        ttl = 60 if truth.get("source_mode") == "live" else 300
        redis_status = self.redis.set_truth_snapshot(season, round_num, session, truth, ttl_seconds=ttl)
        clickhouse_status = await self.clickhouse.append_live_state(season, race, session, truth)
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_probability_snapshot(self, season: int, race: Any, session: str, payload: dict[str, Any], model_id: str = "production_v1") -> dict[str, Any]:
        payload = {
            **payload,
            "generated_at": payload.get("generated_at") or datetime.now(timezone.utc).isoformat(),
            "model_id": model_id or payload.get("model_id") or "production_v1",
        }
        round_num = int(getattr(race, "round", payload.get("round", 0)) or 0)
        redis_status = self.redis.set_probability_snapshot(season, round_num, session, model_id, payload, ttl_seconds=900)
        clickhouse_status = await self.clickhouse.append_probability_snapshot(season, race, session, model_id, payload)
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_simulation_run(self, season: int, race: Any, session: str, payload: dict[str, Any], live: bool = False) -> dict[str, Any]:
        clickhouse_status = await self.clickhouse.append_simulation_run(season, race, session, payload, live=live)
        return {"clickhouse": clickhouse_status}

    async def persist_track_geometry(self, race: Any, track: dict[str, Any]) -> dict[str, Any]:
        if not track:
            return {"redis": {"ok": True, "rows": 0}, "clickhouse": {"ok": True, "rows": 0}}
        geometry_hash = str(track.get("geometry_hash") or track.get("hash") or "")
        if not geometry_hash:
            geometry_hash = str(abs(hash(str(track.get("path") or track.get("display_points") or track.get("points") or ""))))
        track_key = str(track.get("track_key") or getattr(race, "circuit", "") or "unknown")
        redis_status = self.redis.set_track_geometry(track_key, geometry_hash, track, ttl_seconds=86400)
        clickhouse_status = await self.clickhouse.append_track_geometry(race, track)
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_sentiment(self, season: int, items: list[dict[str, Any]], aggregates: dict[str, Any]) -> dict[str, Any]:
        redis_status = self.redis.set_sentiment_latest(aggregates, ttl_seconds=3600)
        clickhouse_status = await self.clickhouse.append_sentiment(season, items, aggregates)
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_race_sentiment_impact(self, season: int, race: Any, session: str, impact: dict[str, Any]) -> dict[str, Any]:
        round_num = int(getattr(race, "round", impact.get("race_round", 0)) or 0)
        redis_status = self.redis.set_race_sentiment(season, round_num, session, impact, ttl_seconds=3600)
        # Reuse existing entity sentiment archive shape for v1; full race-impact tables can be added later.
        clickhouse_status = await self.clickhouse.append_sentiment(season, [], {
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
        })
        return {"redis": redis_status, "clickhouse": clickhouse_status}

    async def persist_backtest_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        clickhouse_status = await self.clickhouse.append_backtest_result(payload)
        return {"clickhouse": clickhouse_status}
