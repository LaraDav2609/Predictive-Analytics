"""ClickHouse analytics store for durable F1 history."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from hashlib import sha1
from time import monotonic
from typing import Any

import httpx


DEFAULT_CLICKHOUSE_URL = "http://localhost:8123"
DEFAULT_DATABASE = "f1_analytics"
DEFAULT_TIMEOUT_SECONDS = 0.5


class F1ClickHouseStore:
    """Append-only ClickHouse storage using the HTTP interface."""

    def __init__(
        self,
        url: str | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.url = (url or os.getenv("F1_CLICKHOUSE_URL") or os.getenv("CLICKHOUSE_URL") or DEFAULT_CLICKHOUSE_URL).rstrip("/")
        self.database = database or os.getenv("F1_CLICKHOUSE_DATABASE") or DEFAULT_DATABASE
        self.user = user or os.getenv("F1_CLICKHOUSE_USER") or os.getenv("CLICKHOUSE_USER")
        self.password = password or os.getenv("F1_CLICKHOUSE_PASSWORD") or os.getenv("CLICKHOUSE_PASSWORD")
        raw_enabled = os.getenv("F1_CLICKHOUSE_ENABLED")
        self.enabled = bool(enabled) if enabled is not None else str(raw_enabled or "true").lower() not in {"0", "false", "no"}
        timeout = float(os.getenv("F1_CLICKHOUSE_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS)
        self._client = httpx.AsyncClient(timeout=timeout)
        self._initialized = False
        self._last_error: str | None = None
        self._disabled_until = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {"available": False, "enabled": False, "database": self.database, "last_error": None}
        if monotonic() < self._disabled_until:
            return {
                "available": False,
                "enabled": True,
                "url": self.url,
                "database": self.database,
                "last_error": self._last_error,
                "backoff_seconds": round(self._disabled_until - monotonic(), 2),
            }
        try:
            await self._query("SELECT 1")
            return {"available": True, "enabled": True, "url": self.url, "database": self.database, "last_error": None}
        except Exception as exc:  # pragma: no cover - depends on local service
            self._last_error = str(exc)
            self._disabled_until = monotonic() + 30.0
            return {
                "available": False,
                "enabled": True,
                "url": self.url,
                "database": self.database,
                "last_error": self._last_error,
                "backoff_seconds": 30.0,
            }

    async def ensure_schema(self) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "reason": "clickhouse_disabled"}
        if monotonic() < self._disabled_until:
            return {"ok": False, "reason": self._last_error or "clickhouse_backoff"}
        if self._initialized:
            return {"ok": True, "cached": True}
        try:
            await self._query(f"CREATE DATABASE IF NOT EXISTS {_ident(self.database)}")
            for ddl in _table_ddls(self.database):
                await self._query(ddl)
            self._initialized = True
            return {"ok": True, "cached": False}
        except Exception as exc:  # pragma: no cover - depends on local service
            self._last_error = str(exc)
            self._disabled_until = monotonic() + 30.0
            return {"ok": False, "reason": self._last_error}

    async def append_live_state(self, season: int, race: Any, session: str, state: dict[str, Any]) -> dict[str, Any]:
        rows = live_state_rows(season, race, session, state)
        return await self._insert_rows("f1_live_snapshots", rows)

    async def append_probability_snapshot(self, season: int, race: Any, session: str, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        rows = probability_rows(season, race, session, model_id, payload)
        return await self._insert_rows("f1_probability_snapshots", rows)

    async def append_telemetry_snapshot(self, season: int, race: Any, session: str, payload: dict[str, Any]) -> dict[str, Any]:
        rows = telemetry_model_rows(season, race, session, payload)
        return await self._insert_rows("f1_telemetry_model_outputs", rows)

    async def append_simulation_run(self, season: int, race: Any, session: str, payload: dict[str, Any], live: bool = False) -> dict[str, Any]:
        rows = simulation_run_rows(season, race, session, payload, live=live)
        return await self._insert_rows("f1_simulation_runs", rows)

    async def append_track_geometry(self, race: Any, track: dict[str, Any]) -> dict[str, Any]:
        row = track_geometry_row(race, track)
        return await self._insert_rows("f1_track_geometry_versions", [row] if row else [])

    async def append_sentiment(self, season: int, items: list[dict[str, Any]], aggregates: dict[str, Any]) -> dict[str, Any]:
        item_rows = sentiment_item_rows(season, items)
        score_rows = entity_sentiment_rows(season, aggregates)
        item_status = await self._insert_rows("f1_sentiment_items", item_rows)
        score_status = await self._insert_rows("f1_entity_sentiment_scores", score_rows)
        return {"ok": bool(item_status.get("ok") and score_status.get("ok")), "items": item_status, "entity_scores": score_status}

    async def append_backtest_result(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows = backtest_rows(payload)
        return await self._insert_rows("f1_backtest_results", rows)

    async def _insert_rows(self, table: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"ok": True, "rows": 0, "table": table}
        schema = await self.ensure_schema()
        if not schema.get("ok"):
            return {"ok": False, "rows": 0, "table": table, "reason": schema.get("reason")}
        try:
            content = "\n".join(json.dumps(row, ensure_ascii=False, default=_json_default) for row in rows)
            await self._query(f"INSERT INTO {_ident(self.database)}.{_ident(table)} FORMAT JSONEachRow", content=content)
            return {"ok": True, "rows": len(rows), "table": table}
        except Exception as exc:  # pragma: no cover - depends on local service
            self._last_error = str(exc)
            self._disabled_until = monotonic() + 30.0
            return {"ok": False, "rows": 0, "table": table, "reason": self._last_error}

    async def _query(self, query: str, content: str | None = None) -> str:
        params = {"query": query} if content is not None else {}
        auth = (self.user, self.password or "") if self.user else None
        response = await self._client.post(self.url, params=params, content=content or query, auth=auth)
        response.raise_for_status()
        return response.text


def live_state_rows(season: int, race: Any, session: str, state: dict[str, Any]) -> list[dict[str, Any]]:
    captured_at = _timestamp(state.get("refreshed_at"))
    rows = []
    for driver in state.get("drivers") or []:
        rows.append({
            "captured_at": captured_at,
            "season": int(season),
            "round": _round(race),
            "session": _session(session),
            "driver_id": str(driver.get("driver_id") or ""),
            "driver_code": str(driver.get("driver_code") or ""),
            "team": str(driver.get("team") or ""),
            "position": _int(driver.get("position")),
            "lap": _int(driver.get("lap") or driver.get("laps")),
            "progress": _float(driver.get("estimated_progress") or driver.get("progress")),
            "gap_to_leader": _str(driver.get("gap_to_leader")),
            "interval": _str(driver.get("interval")),
            "compound": _first_compound(driver.get("compound") or driver.get("compounds")),
            "tyre_age": _int(driver.get("tyre_age")),
            "pit_stops": _int(driver.get("pit_stops")),
            "source_mode": str(driver.get("source_mode") or state.get("source_mode") or state.get("mode") or ""),
            "confidence": _float(driver.get("confidence") if driver.get("confidence") is not None else state.get("confidence")),
            "payload_json": _json(driver),
        })
    return rows


def probability_rows(season: int, race: Any, session: str, model_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    generated_at = _timestamp(payload.get("generated_at"))
    source_rows = payload.get("probabilities") or payload.get("simulations") or []
    rows = []
    for item in source_rows:
        rows.append({
            "generated_at": generated_at,
            "season": int(season),
            "round": _round(race),
            "session": _session(session),
            "model_id": model_id or str(payload.get("model_id") or payload.get("model_version") or "production_v1"),
            "driver_id": str(item.get("driver_id") or ""),
            "driver_code": str(item.get("driver_code") or ""),
            "win_probability": _float(item.get("win_probability") if item.get("win_probability") is not None else item.get("win_prob")),
            "podium_probability": _float(item.get("podium_probability") if item.get("podium_probability") is not None else item.get("podium_prob")),
            "top5_probability": _float(item.get("top5_probability") if item.get("top5_probability") is not None else item.get("top5_prob")),
            "points_probability": _float(item.get("points_probability") if item.get("points_probability") is not None else item.get("points_prob")),
            "dnf_probability": _float(item.get("dnf_probability") if item.get("dnf_probability") is not None else item.get("dnf_prob")),
            "wdc_probability": _float(item.get("wdc_probability") if item.get("wdc_probability") is not None else item.get("wdc_prob")),
            "expected_finish": _float(item.get("expected_finish") if item.get("expected_finish") is not None else item.get("predicted_position")),
            "confidence": _float(item.get("confidence") if item.get("confidence") is not None else payload.get("confidence")),
            "component_scores_json": _json(item.get("component_scores") or item.get("components") or {}),
            "payload_json": _json(item),
        })
    return rows


def telemetry_model_rows(season: int, race: Any, session: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    model = payload.get("telemetry_model") or payload.get("model") or payload
    features = payload.get("telemetry_features") or {}
    generated_at = _timestamp(model.get("generated_at") or payload.get("generated_at"))
    explanations = model.get("probability_delta_explanations") or []
    factors_by_driver = {
        str(item.get("driver_code") or "").upper(): item.get("factors") or []
        for item in explanations
        if isinstance(item, dict)
    }
    rows = []
    for driver_code, adjustment in (model.get("driver_adjustments") or {}).items():
        code = str(driver_code or "").upper()
        rows.append({
            "generated_at": generated_at,
            "season": int(season),
            "round": _round(race),
            "session": _session(session),
            "model_id": str(model.get("model_id") or payload.get("model_id") or "telemetry_simulator_v1"),
            "model_version": str(model.get("model_version") or ""),
            "source_mode": str(model.get("source_mode") or features.get("source_mode") or ""),
            "confidence": _float(adjustment.get("confidence") if isinstance(adjustment, dict) else None),
            "driver_code": code,
            "clean_air_pace_shift_s": _float(adjustment.get("clean_air_pace_shift_s") if isinstance(adjustment, dict) else None),
            "pace_sigma_multiplier": _float(adjustment.get("pace_sigma_multiplier") if isinstance(adjustment, dict) else None),
            "tire_deg_slope_delta": _float(adjustment.get("tire_deg_slope_delta") if isinstance(adjustment, dict) else None),
            "dnf_hazard_multiplier": _float(adjustment.get("dnf_hazard_multiplier") if isinstance(adjustment, dict) else None),
            "overtake_score_delta": _float(adjustment.get("overtake_score_delta") if isinstance(adjustment, dict) else None),
            "pit_window_value_s": _float(adjustment.get("pit_window_value_s") if isinstance(adjustment, dict) else None),
            "feature_factors_json": _json(factors_by_driver.get(code) or []),
            "missing_groups_json": _json(model.get("missing_groups") or features.get("missing_groups") or []),
            "payload_json": _json({"adjustment": adjustment, "model": model, "features": features}),
        })
    return rows


def simulation_run_rows(season: int, race: Any, session: str, payload: dict[str, Any], live: bool = False) -> list[dict[str, Any]]:
    source = payload.get("track", {}) if isinstance(payload.get("track"), dict) else {}
    summary = {
        "winner": (payload.get("simulations") or [{}])[0] if payload.get("simulations") else {},
        "drivers": len(payload.get("simulations") or []),
        "status": payload.get("status"),
        "mode": payload.get("mode"),
    }
    input_hash = sha1(_json({"race": _round(race), "session": session, "live": live, "source": source.get("source")}).encode("utf-8")).hexdigest()
    run_id = sha1(_json({"input_hash": input_hash, "generated_at": payload.get("generated_at") or _now()}).encode("utf-8")).hexdigest()
    return [{
        "generated_at": _timestamp(payload.get("generated_at")),
        "run_id": run_id,
        "season": int(season),
        "round": _round(race),
        "session": _session(session),
        "model_id": str(payload.get("model_id") or payload.get("model_version") or "production_v1"),
        "seed": _int(payload.get("seed")),
        "source_mode": str(payload.get("source_mode") or payload.get("mode") or source.get("mode") or ""),
        "confidence": _float(payload.get("confidence") or source.get("confidence")),
        "inputs_hash": input_hash,
        "summary_json": _json(summary),
        "payload_json": _json(payload),
    }]


def track_geometry_row(race: Any, track: dict[str, Any]) -> dict[str, Any] | None:
    if not track:
        return None
    points_json = _json({
        "display_points": track.get("display_points") or track.get("points") or [],
        "racing_line": track.get("racing_line") or [],
        "markers": track.get("markers") or {},
    })
    geometry_hash = sha1(points_json.encode("utf-8")).hexdigest()
    return {
        "created_at": _now(),
        "track_key": str(track.get("track_key") or getattr(race, "circuit", "") or getattr(race, "name", "")),
        "season": _season(race),
        "round": _round(race),
        "source": str(track.get("source") or ""),
        "mapping_source": str(track.get("mapping_source") or ""),
        "confidence": _float(track.get("geometry_confidence") if track.get("geometry_confidence") is not None else track.get("confidence")),
        "geometry_hash": geometry_hash,
        "points_json": points_json,
        "payload_json": _json(track),
    }


def sentiment_item_rows(season: int, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        title = str(item.get("Title") or item.get("title") or "")
        url = str(item.get("Url") or item.get("url") or "")
        item_hash = sha1(f"{url}|{title}".encode("utf-8")).hexdigest()
        rows.append({
            "published_at": _timestamp(item.get("PublishedAt") or item.get("Published") or item.get("Timestamp")),
            "season": int(season),
            "item_hash": item_hash,
            "source": str(item.get("Source") or item.get("source") or ""),
            "url": url,
            "title": title,
            "score": _float(item.get("Score") if item.get("Score") is not None else item.get("score")),
            "label": str(item.get("Label") or item.get("label") or ""),
            "topics_json": _json(item.get("Topics") or item.get("topics") or item.get("topic_scores") or {}),
            "payload_json": _json(item),
        })
    return rows


def entity_sentiment_rows(season: int, aggregates: dict[str, Any]) -> list[dict[str, Any]]:
    captured_at = _timestamp((aggregates.get("composite") or {}).get("Timestamp"))
    rows = []
    for entity_type, collection in [("driver", aggregates.get("drivers") or {}), ("constructor", aggregates.get("teams") or {})]:
        for entity_id, item in collection.items():
            rows.append({
                "captured_at": captured_at,
                "season": int(season),
                "entity_type": entity_type,
                "entity_id": str(entity_id),
                "topic": "overall",
                "source": "f1_sentiment",
                "score": _float(item.get("overall_score") if item.get("overall_score") is not None else item.get("score")),
                "confidence": _float(item.get("confidence")),
                "mentions": _int(item.get("mentions")),
                "label": str(item.get("overall_label") or item.get("label") or ""),
                "payload_json": _json(item),
            })
            for topic, score in (item.get("topic_scores") or {}).items():
                rows.append({
                    "captured_at": captured_at,
                    "season": int(season),
                    "entity_type": entity_type,
                    "entity_id": str(entity_id),
                    "topic": str(topic),
                    "source": "f1_sentiment",
                    "score": _float(score),
                    "confidence": _float(item.get("confidence")),
                    "mentions": _int(item.get("mentions")),
                    "label": str(item.get("label") or ""),
                    "payload_json": _json(item),
                })
    return rows


def backtest_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    if payload.get("races"):
        source = payload.get("races") or []
    elif payload.get("season") and payload.get("round"):
        source = [payload]
    else:
        source = []
    for item in source:
        rows.append({
            "generated_at": _timestamp(payload.get("generated_at") or item.get("generated_at")),
            "season": _int(item.get("season")),
            "round": _int(item.get("round")),
            "model_id": str(item.get("model_id") or payload.get("model_id") or "production_v1"),
            "race_name": str(item.get("race_name") or ""),
            "actual_winner": str(item.get("actual_winner") or ""),
            "predicted_winner": str(item.get("predicted_winner") or ""),
            "metrics_json": _json(item.get("metrics") or {}),
            "probability_distribution_json": _json(item.get("probability_distribution") or []),
            "calibration_json": _json(payload.get("calibration_buckets") or item.get("calibration_buckets") or {}),
            "payload_json": _json(item),
        })
    return rows


def _table_ddls(database: str) -> list[str]:
    db = _ident(database)
    return [
        f"""
        CREATE TABLE IF NOT EXISTS {db}.f1_live_snapshots (
            captured_at DateTime64(3, 'UTC'),
            season UInt16,
            round UInt8,
            session LowCardinality(String),
            driver_id String,
            driver_code LowCardinality(String),
            team LowCardinality(String),
            position Nullable(UInt8),
            lap Nullable(UInt16),
            progress Nullable(Float64),
            gap_to_leader String,
            interval String,
            compound LowCardinality(String),
            tyre_age Nullable(UInt16),
            pit_stops Nullable(UInt8),
            source_mode LowCardinality(String),
            confidence Nullable(Float64),
            payload_json String CODEC(ZSTD(3))
        ) ENGINE = MergeTree
        ORDER BY (season, round, session, captured_at, driver_id)
        TTL captured_at + INTERVAL 730 DAY DELETE
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {db}.f1_probability_snapshots (
            generated_at DateTime64(3, 'UTC'),
            season UInt16,
            round UInt8,
            session LowCardinality(String),
            model_id LowCardinality(String),
            driver_id String,
            driver_code LowCardinality(String),
            win_probability Nullable(Float64),
            podium_probability Nullable(Float64),
            top5_probability Nullable(Float64),
            points_probability Nullable(Float64),
            dnf_probability Nullable(Float64),
            wdc_probability Nullable(Float64),
            expected_finish Nullable(Float64),
            confidence Nullable(Float64),
            component_scores_json String CODEC(ZSTD(3)),
            payload_json String CODEC(ZSTD(3))
        ) ENGINE = MergeTree
        ORDER BY (season, round, session, model_id, generated_at, driver_id)
        TTL generated_at + INTERVAL 1825 DAY DELETE
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {db}.f1_simulation_runs (
            generated_at DateTime64(3, 'UTC'),
            run_id String,
            season UInt16,
            round UInt8,
            session LowCardinality(String),
            model_id LowCardinality(String),
            seed Nullable(Int64),
            source_mode LowCardinality(String),
            confidence Nullable(Float64),
            inputs_hash String,
            summary_json String CODEC(ZSTD(3)),
            payload_json String CODEC(ZSTD(3))
        ) ENGINE = MergeTree
        ORDER BY (season, round, session, model_id, generated_at, run_id)
        TTL generated_at + INTERVAL 1825 DAY DELETE
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {db}.f1_telemetry_model_outputs (
            generated_at DateTime64(3, 'UTC'),
            season UInt16,
            round UInt8,
            session LowCardinality(String),
            model_id LowCardinality(String),
            model_version String,
            source_mode LowCardinality(String),
            confidence Nullable(Float64),
            driver_code LowCardinality(String),
            clean_air_pace_shift_s Nullable(Float64),
            pace_sigma_multiplier Nullable(Float64),
            tire_deg_slope_delta Nullable(Float64),
            dnf_hazard_multiplier Nullable(Float64),
            overtake_score_delta Nullable(Float64),
            pit_window_value_s Nullable(Float64),
            feature_factors_json String CODEC(ZSTD(3)),
            missing_groups_json String CODEC(ZSTD(3)),
            payload_json String CODEC(ZSTD(3))
        ) ENGINE = MergeTree
        ORDER BY (season, round, session, model_id, generated_at, driver_code)
        TTL generated_at + INTERVAL 1825 DAY DELETE
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {db}.f1_session_events (
            event_at DateTime64(3, 'UTC'),
            season UInt16,
            round UInt8,
            session LowCardinality(String),
            event_type LowCardinality(String),
            driver_id String,
            lap Nullable(UInt16),
            payload_json String CODEC(ZSTD(3))
        ) ENGINE = MergeTree
        ORDER BY (season, round, session, event_at, event_type)
        TTL event_at + INTERVAL 1825 DAY DELETE
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {db}.f1_track_geometry_versions (
            created_at DateTime64(3, 'UTC'),
            track_key String,
            season UInt16,
            round UInt8,
            source LowCardinality(String),
            mapping_source LowCardinality(String),
            confidence Nullable(Float64),
            geometry_hash String,
            points_json String CODEC(ZSTD(3)),
            payload_json String CODEC(ZSTD(3))
        ) ENGINE = ReplacingMergeTree(created_at)
        ORDER BY (track_key, geometry_hash)
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {db}.f1_sentiment_items (
            published_at DateTime64(3, 'UTC'),
            season UInt16,
            item_hash String,
            source LowCardinality(String),
            url String,
            title String,
            score Nullable(Float64),
            label LowCardinality(String),
            topics_json String CODEC(ZSTD(3)),
            payload_json String CODEC(ZSTD(3))
        ) ENGINE = ReplacingMergeTree(published_at)
        ORDER BY (season, item_hash)
        TTL published_at + INTERVAL 1095 DAY DELETE
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {db}.f1_entity_sentiment_scores (
            captured_at DateTime64(3, 'UTC'),
            season UInt16,
            entity_type LowCardinality(String),
            entity_id String,
            topic LowCardinality(String),
            source LowCardinality(String),
            score Nullable(Float64),
            confidence Nullable(Float64),
            mentions Nullable(UInt32),
            label LowCardinality(String),
            payload_json String CODEC(ZSTD(3))
        ) ENGINE = MergeTree
        ORDER BY (season, entity_type, entity_id, topic, captured_at)
        TTL captured_at + INTERVAL 1825 DAY DELETE
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {db}.f1_backtest_results (
            generated_at DateTime64(3, 'UTC'),
            season UInt16,
            round UInt8,
            model_id LowCardinality(String),
            race_name String,
            actual_winner String,
            predicted_winner String,
            metrics_json String CODEC(ZSTD(3)),
            probability_distribution_json String CODEC(ZSTD(3)),
            calibration_json String CODEC(ZSTD(3)),
            payload_json String CODEC(ZSTD(3))
        ) ENGINE = ReplacingMergeTree(generated_at)
        ORDER BY (season, round, model_id)
        """,
    ]


def _ident(value: str) -> str:
    cleaned = "".join(ch for ch in str(value) if ch.isalnum() or ch == "_")
    if not cleaned:
        raise ValueError("invalid_clickhouse_identifier")
    return cleaned


def _timestamp(value: Any) -> str:
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    elif value:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _now() -> str:
    return _timestamp(datetime.now(timezone.utc))


def _season(race: Any) -> int:
    return _int(getattr(race, "season", None) or datetime.now(timezone.utc).year) or datetime.now(timezone.utc).year


def _round(race: Any) -> int:
    return _int(getattr(race, "round", None) if race is not None else None) or 0


def _session(session: str) -> str:
    value = (session or "race").lower()
    if value.startswith("qual"):
        return "qualifying"
    if value.startswith("sprint"):
        return "sprint"
    return "race"


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _str(value: Any) -> str:
    return "" if value is None else str(value)


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _first_compound(value: Any) -> str:
    if isinstance(value, list):
        return str(value[-1] if value else "")
    return str(value or "")
