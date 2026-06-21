"""In-memory live F1 session state built from OpenF1 data."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sports.f1.models.f1 import Driver, Race
from sports.f1.predictor.live.free_sources import (
    enrich_free_live_state,
    load_fastf1_recorded_state,
    source_diagnostics,
)


class F1LiveSessionEngine:
    def __init__(self, openf1_client=None, ttl_seconds: int = 25, max_timeline: int = 160):
        self._openf1 = openf1_client
        self._ttl_seconds = max(5, int(ttl_seconds))
        self._max_timeline = max(20, int(max_timeline))
        self._cache: dict[tuple[int, str], dict[str, Any]] = {}
        self._timeline: dict[tuple[int, str], list[dict[str, Any]]] = {}
        self._last_refresh: dict[str, Any] = {}

    def set_openf1_client(self, openf1_client) -> None:
        self._openf1 = openf1_client

    async def get_state(
        self,
        race: Race,
        drivers: list[Driver],
        session: str = "race",
        force: bool = False,
    ) -> dict[str, Any]:
        session_key = _session_kind(session)
        key = (int(race.round), session_key)
        cached = self._cache.get(key)
        if cached and not force:
            state = self._with_age(cached)
            if not state.get("stale"):
                return state
        return await self.refresh_state(race, drivers, session_key)

    async def refresh_state(self, race: Race, drivers: list[Driver], session: str = "race") -> dict[str, Any]:
        session_key = _session_kind(session)
        key = (int(race.round), session_key)
        if not self._openf1:
            state = self._unavailable_state(race, session_key, "openf1_client_unavailable")
            recorded = load_fastf1_recorded_state(race, drivers, session_key)
            if recorded.get("ok"):
                state = enrich_free_live_state(state, race, drivers, {}, {}, recorded)
            self._store(key, state)
            return self._with_age(state)

        try:
            session_features = await self._openf1.get_session_features(race, session_key, drivers, live=True)
            track = await self._openf1.get_track_data(
                race,
                drivers,
                session=session_key,
                live=True,
                static_only=not _has_timing_rows(session_features),
            )
        except Exception as exc:
            state = self._unavailable_state(race, session_key, f"openf1_refresh_failed: {exc}")
            recorded = load_fastf1_recorded_state(race, drivers, session_key)
            state = enrich_free_live_state(state, race, drivers, {}, {}, recorded)
            self._store(key, state)
            return self._with_age(state)

        recorded = load_fastf1_recorded_state(race, drivers, session_key)
        state = build_live_state(
            race=race,
            drivers=drivers,
            session=session_key,
            openf1_session=session_features if isinstance(session_features, dict) else {},
            track=track if isinstance(track, dict) else {},
            ttl_seconds=self._ttl_seconds,
        )
        state = enrich_free_live_state(
            state=state,
            race=race,
            drivers=drivers,
            openf1_session=session_features if isinstance(session_features, dict) else {},
            track=track if isinstance(track, dict) else {},
            recorded=recorded,
        )
        self._store(key, state)
        return self._with_age(state)

    def get_sources(self, race: Race, session: str = "race") -> dict[str, Any]:
        session_key = _session_kind(session)
        key = (int(race.round), session_key)
        return source_diagnostics(self._with_age(self._cache.get(key)))

    def get_timeline(self, race: Race, session: str = "race") -> dict[str, Any]:
        session_key = _session_kind(session)
        key = (int(race.round), session_key)
        current = self._cache.get(key)
        return {
            "ok": True,
            "round": race.round,
            "session": session_key,
            "current": self._with_age(current) if current else None,
            "events": list(self._timeline.get(key, [])),
        }

    def record_probability_snapshot(self, race: Race, session: str, payload: dict[str, Any]) -> dict[str, Any]:
        session_key = _session_kind(session)
        key = (int(race.round), session_key)
        rows = payload.get("probabilities") or payload.get("simulations") or []
        top = []
        for row in rows[:5]:
            top.append({
                "driver_id": row.get("driver_id"),
                "driver_code": row.get("driver_code") or row.get("code"),
                "driver_name": row.get("driver_name") or row.get("name"),
                "team": row.get("team"),
                "win_probability": row.get("win_probability"),
                "raw_probability": row.get("raw_probability"),
                "live_adjusted_probability": row.get("live_adjusted_probability"),
                "live_probability_delta": row.get("live_probability_delta"),
                "expected_finish": row.get("expected_finish"),
            })
        movers = payload.get("top_probability_movers") or []
        dynamics = payload.get("live_dynamics") or {}
        event = {
            "at": payload.get("generated_at") or datetime.now(timezone.utc).isoformat(),
            "type": "probability_snapshot",
            "mode": payload.get("source_mode"),
            "source_mode": payload.get("source_mode"),
            "confidence": payload.get("confidence"),
            "stage": payload.get("stage"),
            "top": top,
            "top5": top,
            "movers": movers[:5] if isinstance(movers, list) else [],
            "live_dynamics_summary": dynamics.get("summary") or [],
            "chaos_score": ((payload.get("truth") or {}).get("signals") or {}).get("chaos_score") or (payload.get("weather") or {}).get("chaos_score"),
            "fallback_reason": payload.get("fallback_reason"),
        }
        events = self._timeline.setdefault(key, [])
        events.append(event)
        if len(events) > self._max_timeline:
            del events[: len(events) - self._max_timeline]
        return {"ok": True, "events": len(events), "latest_type": event["type"]}

    def health(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        latest = None
        for state in self._cache.values():
            if not latest or str(state.get("refreshed_at") or "") > str(latest.get("refreshed_at") or ""):
                latest = state
        age = _age_seconds(latest, now) if latest else None
        return {
            "cache_entries": len(self._cache),
            "timeline_entries": sum(len(items) for items in self._timeline.values()),
            "ttl_seconds": self._ttl_seconds,
            "latest_age_seconds": age,
            "latest_stale": bool(age is not None and age > self._ttl_seconds),
            "active_session_key": latest.get("session_key") if latest else None,
            "latest_mode": latest.get("mode") if latest else None,
            "latest_reason": latest.get("reason") if latest else None,
            "last_refresh": self._last_refresh,
        }

    def _store(self, key: tuple[int, str], state: dict[str, Any]) -> None:
        self._cache[key] = state
        self._last_refresh = {
            "round": state.get("round"),
            "session": state.get("session"),
            "mode": state.get("mode"),
            "source_mode": state.get("source_mode"),
            "confidence": state.get("confidence"),
            "ok": state.get("ok"),
            "reason": state.get("reason"),
            "fallback_reason": state.get("fallback_reason"),
            "refreshed_at": state.get("refreshed_at"),
        }
        event = {
            "at": state.get("refreshed_at"),
            "mode": state.get("mode"),
            "source_mode": state.get("source_mode"),
            "status": state.get("status"),
            "leader": state.get("leader"),
            "confidence": state.get("confidence"),
            "fallback_reason": state.get("fallback_reason"),
            "chaos_score": state.get("signals", {}).get("chaos_score"),
            "driver_count": len(state.get("drivers") or []),
        }
        events = self._timeline.setdefault(key, [])
        events.append(event)
        if len(events) > self._max_timeline:
            del events[: len(events) - self._max_timeline]

    def _with_age(self, state: dict[str, Any] | None) -> dict[str, Any]:
        if not state:
            return {}
        now = datetime.now(timezone.utc)
        age = _age_seconds(state, now)
        enriched = {**state}
        enriched["age_seconds"] = age
        enriched["data_age_seconds"] = age
        enriched["stale"] = bool(age is not None and age > self._ttl_seconds)
        if enriched["stale"]:
            enriched["status"] = "stale"
        return enriched

    def _unavailable_state(self, race: Race, session: str, reason: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "ok": False,
            "round": race.round,
            "race_name": race.name,
            "session": session,
            "mode": "unavailable",
            "status": "unavailable",
            "source_mode": "unavailable",
            "source_chain": [],
            "confidence": 0.0,
            "last_successful_source": None,
            "fallback_reason": reason,
            "is_estimated": False,
            "reason": reason,
            "session_key": None,
            "meeting_key": None,
            "refreshed_at": now,
            "expires_at": now,
            "drivers": [],
            "by_driver_id": {},
            "live_positions": {},
            "signals": {"chaos_score": 0.0},
            "sources": {"openf1": "unavailable"},
        }


def build_live_state(
    race: Race,
    drivers: list[Driver],
    session: str,
    openf1_session: dict[str, Any],
    track: dict[str, Any],
    ttl_seconds: int = 25,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    position_rows = ((openf1_session.get("positions") or {}).get("drivers") or {})
    interval_rows = ((openf1_session.get("intervals") or {}).get("drivers") or {})
    lap_rows = ((openf1_session.get("laps") or {}).get("drivers") or {})
    stint_rows = ((openf1_session.get("stints") or {}).get("drivers") or {})
    pit_rows = ((openf1_session.get("pits") or {}).get("drivers") or {})
    track_locations = track.get("live_positions") or []
    location_by_id = {
        item.get("driver_id"): item
        for item in track_locations
        if item.get("driver_id")
    }
    location_by_number = {
        int(item.get("driver_number")): item
        for item in track_locations
        if str(item.get("driver_number") or "").isdigit()
    }

    driver_states = []
    for driver in drivers:
        number = int(driver.number) if driver.number is not None else None
        number_key = str(number) if number is not None else ""
        position = position_rows.get(number_key) or {}
        interval = interval_rows.get(number_key) or {}
        lap = lap_rows.get(number_key) or {}
        stint = stint_rows.get(number_key) or {}
        pit = pit_rows.get(number_key) or {}
        location = location_by_id.get(driver.id) or (location_by_number.get(number) if number is not None else {}) or {}
        state = {
            "driver_id": driver.id,
            "driver_number": number,
            "driver_code": driver.code,
            "driver_name": f"{driver.first_name} {driver.last_name}".strip(),
            "team": driver.team,
            "position": _safe_int(position.get("position")),
            "gap_to_leader": interval.get("gap_to_leader"),
            "interval": interval.get("interval"),
            "best_lap": lap.get("best_lap"),
            "representative_lap": lap.get("representative_lap"),
            "median_lap": lap.get("median_lap"),
            "laps": lap.get("laps"),
            "stints": stint.get("stints"),
            "compounds": stint.get("compounds") or [],
            "avg_stint_laps": stint.get("avg_stint_laps"),
            "pit_stops": pit.get("pit_stops"),
            "avg_pit_duration": pit.get("avg_pit_duration"),
            "x": location.get("x"),
            "y": location.get("y"),
            "location_date": location.get("date"),
            "data_status": _driver_status(position, lap, location),
        }
        driver_states.append(state)

    driver_states.sort(key=lambda item: item.get("position") or 99)
    by_driver_id = {item["driver_id"]: item for item in driver_states}
    live_positions = {
        item["driver_id"]: {
            "position": item.get("position"),
            "x": item.get("x"),
            "y": item.get("y"),
            "gap_to_leader": item.get("gap_to_leader"),
            "interval": item.get("interval"),
            "representative_lap": item.get("representative_lap"),
        }
        for item in driver_states
        if item.get("position") is not None or item.get("x") is not None
    }
    weather = openf1_session.get("weather") or {}
    race_control = openf1_session.get("race_control") or {}
    chaos = max(float(weather.get("chaos_score") or 0.0), float(race_control.get("chaos_score") or 0.0))
    mode = _state_mode(openf1_session, track, live_positions)
    status = "unavailable" if mode == "unavailable" else mode
    return {
        "ok": mode != "unavailable",
        "round": race.round,
        "race_name": race.name,
        "session": session,
        "mode": mode,
        "status": status,
        "reason": openf1_session.get("reason") or track.get("reason"),
        "session_key": openf1_session.get("session_key") or track.get("session_key"),
        "meeting_key": openf1_session.get("meeting_key") or track.get("meeting_key"),
        "session_name": track.get("session_name"),
        "refreshed_at": now.isoformat(),
        "expires_at": (now.timestamp() + ttl_seconds),
        "drivers": driver_states,
        "by_driver_id": by_driver_id,
        "leader": driver_states[0]["driver_id"] if driver_states and driver_states[0].get("position") == 1 else None,
        "live_positions": live_positions,
        "signals": {
            "chaos_score": round(chaos, 4),
            "weather": weather,
            "race_control": race_control,
            "raw_counts": openf1_session.get("raw_counts") or {},
        },
        "track": {
            "source": track.get("source"),
            "mode": track.get("mode"),
            "path": track.get("path"),
            "points": track.get("points") or [],
            "live_positions": track_locations,
        },
        "sources": {
            "openf1_session": openf1_session.get("source"),
            "track": track.get("source"),
        },
    }


def _session_kind(session: str) -> str:
    value = (session or "race").lower()
    if value.startswith("qual"):
        return "qualifying"
    if value.startswith("sprint"):
        return "sprint"
    return "race"


def _state_mode(openf1_session: dict[str, Any], track: dict[str, Any], live_positions: dict[str, Any]) -> str:
    if live_positions and track.get("mode") == "live":
        return "live"
    if openf1_session.get("ok"):
        raw_counts = openf1_session.get("raw_counts") or {}
        if any(int(raw_counts.get(key) or 0) > 0 for key in ["positions", "laps", "intervals"]):
            return "historical"
    if track.get("source") == "estimated":
        return "estimated"
    return "unavailable"


def _has_timing_rows(openf1_session: dict[str, Any]) -> bool:
    raw_counts = (openf1_session or {}).get("raw_counts") or {}
    return any(int(raw_counts.get(key) or 0) > 0 for key in ["positions", "laps", "intervals"])


def _driver_status(position: dict[str, Any], lap: dict[str, Any], location: dict[str, Any]) -> str:
    if location.get("x") is not None and position.get("position") is not None:
        return "live"
    if position or lap:
        return "historical"
    return "unavailable"


def _safe_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _age_seconds(state: dict[str, Any] | None, now: datetime) -> float | None:
    if not state:
        return None
    try:
        refreshed = datetime.fromisoformat(str(state.get("refreshed_at")))
    except ValueError:
        return None
    if refreshed.tzinfo is None:
        refreshed = refreshed.replace(tzinfo=timezone.utc)
    return round(max(0.0, (now - refreshed).total_seconds()), 3)
