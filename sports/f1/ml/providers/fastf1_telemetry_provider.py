"""FastF1 telemetry bridge for telemetry simulator features."""

from __future__ import annotations

from typing import Any

from sports.f1.ml.common.types import Race, SessionType
from sports.f1.ml.providers.fastf1_provider import FastF1Provider
from sports.f1.ml.telemetry.cache import TelemetryCache
from sports.f1.ml.telemetry.features import build_telemetry_features
from sports.f1.ml.telemetry.types import TelemetryFeaturePayload, TelemetryTracePoint


class FastF1TelemetryProvider:
    """Fetch, cache, and feature-engineer FastF1 traces."""

    def __init__(self, fastf1_provider: FastF1Provider | None = None, cache: TelemetryCache | None = None) -> None:
        self.fastf1 = fastf1_provider or FastF1Provider()
        self.cache = cache or TelemetryCache()

    def session_snapshot(
        self,
        *,
        race: Race,
        season: int,
        round_num: int,
        session: str,
        live: bool = False,
    ) -> dict[str, Any]:
        session_key = (session or "race").lower()
        if not live:
            cached = self.cache.read_snapshot(
                source="fastf1",
                season=season,
                round_num=round_num,
                session=session_key,
                live=False,
            )
            if cached:
                return cached

        session_type = _session_type(session_key)
        points = self.fastf1.trace_points(race, session_type)
        laps = self.fastf1.laps(race, session_type)
        diagnostics = self.fastf1.session_diagnostics(race, session_type)
        raw_counts = {
            "trace_points": len(points),
            "laps": len(laps),
            **dict((diagnostics or {}).get("raw_counts") or {}),
        }
        snapshot = {
            "ok": bool(points),
            "source": "fastf1",
            "session": session_key,
            "race_id": _race_id(race),
            "trace_points": [point.model_dump(mode="json") for point in points],
            "laps": [lap.model_dump(mode="json") for lap in laps],
            "diagnostics": diagnostics,
            "raw_counts": raw_counts,
        }
        return self.cache.write_snapshot(
            source="fastf1",
            season=season,
            round_num=round_num,
            session=session_key,
            payload=snapshot,
            live=live,
            ttl_seconds=8 if live else None,
        )

    def telemetry_features(
        self,
        *,
        race: Race,
        season: int,
        round_num: int,
        session: str,
        live: bool = False,
    ) -> tuple[TelemetryFeaturePayload, dict[str, Any]]:
        snapshot = self.session_snapshot(
            race=race,
            season=season,
            round_num=round_num,
            session=session,
            live=live,
        )
        points = _trace_points_from_snapshot(snapshot.get("trace_points") or [])
        diagnostics = snapshot.get("diagnostics") or {}
        payload = build_telemetry_features(
            race_id=_race_id(race),
            session=(session or "race").lower(),
            trace_points=points,
            laps=snapshot.get("laps") or [],
            stints=[],
            intervals=[],
            source_mode="fastf1_live_raw" if live else "fastf1_historical_raw",
            live=live,
        )
        payload.raw_counts.update(snapshot.get("raw_counts") or {})
        payload.data_quality.update({
            "cache_hit": bool((snapshot.get("cache") or {}).get("hit")),
            "raw_trace": bool(points),
            "summary_only": False,
            "diagnostics": diagnostics.get("data_quality") or {},
            "track_status_count": int(((diagnostics.get("raw_counts") or {}).get("track_status")) or 0),
            "race_control_message_count": int(((diagnostics.get("raw_counts") or {}).get("race_control_messages")) or 0),
        })
        return payload, snapshot


def _session_type(session: str) -> SessionType:
    normalized = (session or "race").lower()
    mapping = {
        "fp1": SessionType.FP1,
        "practice1": SessionType.FP1,
        "practice_1": SessionType.FP1,
        "fp2": SessionType.FP2,
        "practice2": SessionType.FP2,
        "practice_2": SessionType.FP2,
        "fp3": SessionType.FP3,
        "practice3": SessionType.FP3,
        "practice_3": SessionType.FP3,
        "qualifying": SessionType.QUALIFYING,
        "q": SessionType.QUALIFYING,
        "sprint_qualifying": SessionType.SPRINT_QUALI,
        "sprint_shootout": SessionType.SPRINT_QUALI,
        "sq": SessionType.SPRINT_QUALI,
        "sprint": SessionType.SPRINT,
        "race": SessionType.RACE,
        "r": SessionType.RACE,
    }
    return mapping.get(normalized, SessionType.RACE)


def _race_id(race: Race) -> str:
    return f"{int(race.season)}-{int(race.round):02d}-{race.track_code}"


def _trace_points_from_snapshot(rows: list[dict[str, Any]]) -> list[TelemetryTracePoint]:
    points: list[TelemetryTracePoint] = []
    for row in rows or []:
        try:
            points.append(TelemetryTracePoint.model_validate(row))
        except Exception:
            continue
    return points
