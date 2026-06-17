"""Build local race-weekend evidence cache files for deeper F1 backtests."""

from __future__ import annotations

import asyncio
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sports.f1.models.f1 import Driver, Race


DEFAULT_EVIDENCE_CACHE_DIR = Path.cwd() / "artifacts" / "f1_weekend_evidence_cache"
DEFAULT_SESSIONS = ("fp1", "fp2", "fp3", "qualifying", "race")


class WeekendEvidenceCacheWriter:
    """Create local evidence artifacts that HistoricalRaceLoader can merge.

    The writer intentionally stores compact summaries, not raw telemetry. Raw
    OpenF1/FastF1 streams can be huge; backtesting needs stable practice/grid/
    race-input facts that are cheap to replay offline.
    """

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_EVIDENCE_CACHE_DIR

    async def build_round(
        self,
        *,
        season: int,
        race: Race,
        drivers: list[Driver],
        profile: dict[str, Any] | None = None,
        openf1=None,
        fastf1_provider=None,
        fastf1_mode: str = "off",
        sessions: list[str] | tuple[str, ...] | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        session_codes = _normalize_sessions(sessions)
        fastf1_mode = _fastf1_mode(fastf1_mode)
        path = self.path_for(int(season), int(race.round))
        old_payload = _read_existing_payload(path)
        payload = {
            "schema_version": "f1-weekend-evidence-cache-v1",
            "source": "openf1_profile_compact_summary",
            "season": int(season),
            "round": int(race.round),
            "race_name": race.name,
            "circuit": race.circuit,
            "country": race.country,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "sessions_requested": session_codes,
            "PracticeResults": [],
            "QualifyingResults": _qualifying_rows(profile or {}),
            "GridResults": _grid_rows(profile or {}),
            "RaceInputs": {},
            "raw_counts": {},
            "session_status": {},
            "coverage": {},
            "errors": [],
            "warnings": [],
        }

        openf1_payloads: dict[str, dict[str, Any]] = {}
        if openf1:
            for session in session_codes:
                try:
                    row = await _openf1_session_features(openf1, race, session, drivers)
                except Exception as exc:
                    row = {
                        "ok": False,
                        "source": "openf1",
                        "session": session,
                        "reason": f"openf1_error:{exc.__class__.__name__}",
                    }
                openf1_payloads[session] = row
                payload["raw_counts"][session] = row.get("raw_counts") or {}
                payload["session_status"][session] = {
                    "ok": bool(row.get("ok")),
                    "source": row.get("source") or "openf1",
                    "session_key": row.get("session_key"),
                    "reason": row.get("reason"),
                }
                if session in {"fp1", "fp2", "fp3"} and row.get("ok"):
                    raw_counts = row.get("raw_counts") or {}
                    if int(raw_counts.get("laps") or 0) <= 0:
                        payload["warnings"].append({
                            "session": session,
                            "code": "openf1_practice_lap_rows_empty",
                            "message": "OpenF1 returned a practice session but no lap-duration rows, so practice pace could not be built.",
                            "raw_counts": raw_counts,
                        })
                if not row.get("ok") and row.get("reason"):
                    payload["errors"].append({"session": session, "reason": row.get("reason")})

        number_lookup = {str(driver.number): driver for driver in drivers if driver.number is not None}
        practice_rows = []
        for session in ("fp1", "fp2", "fp3"):
            session_rows = _practice_rows_from_openf1(session, openf1_payloads.get(session) or {}, number_lookup)
            should_try_fastf1 = (
                session in session_codes
                and (
                    fastf1_mode == "force"
                    or (fastf1_mode == "fallback" and not session_rows)
                )
            )
            if should_try_fastf1:
                fastf1_rows, fastf1_status = await _practice_rows_from_fastf1(
                    season=int(season),
                    race=race,
                    session=session,
                    drivers=drivers,
                    provider=fastf1_provider,
                )
                payload["session_status"][f"{session}_fastf1"] = fastf1_status
                if fastf1_status.get("ok"):
                    payload["raw_counts"][f"{session}_fastf1"] = {"laps": fastf1_status.get("raw_laps") or 0}
                if fastf1_rows:
                    session_rows = fastf1_rows
                    if fastf1_mode == "force":
                        payload["session_status"].setdefault(session, {
                            "ok": False,
                            "source": "openf1_disabled",
                            "session_key": None,
                            "reason": None,
                        })["replaced_by_fastf1"] = True
                    payload["warnings"] = [
                        warning for warning in payload["warnings"]
                        if not (
                            warning.get("session") == session
                            and warning.get("code") == "openf1_practice_lap_rows_empty"
                        )
                    ]
                elif fastf1_status.get("reason"):
                    payload["warnings"].append({
                        "session": session,
                        "code": "fastf1_practice_fallback_unavailable",
                        "message": "FastF1 practice fallback did not return usable lap rows.",
                        "reason": fastf1_status.get("reason"),
                    })
            practice_rows.extend(session_rows)
        payload["PracticeResults"] = practice_rows

        race_payload = openf1_payloads.get("race") or {}
        payload["RaceInputs"] = _race_inputs_from_openf1(race_payload, number_lookup)

        if not payload["PracticeResults"] and old_payload:
            old_practice_rows = old_payload.get("PracticeResults") or old_payload.get("practice_results") or []
            if isinstance(old_practice_rows, list) and old_practice_rows:
                payload["PracticeResults"] = [row for row in old_practice_rows if isinstance(row, dict)]
                payload["warnings"].append({
                    "code": "preserved_existing_practice_rows",
                    "message": "New refresh returned no usable practice rows, so existing cached practice evidence was preserved.",
                    "rows": len(payload["PracticeResults"]),
                })

        if (
            any(session in session_codes for session in ("race", "sprint"))
            and not ((payload.get("RaceInputs") or {}).get("drivers") or {})
            and old_payload
        ):
            old_race_inputs = old_payload.get("RaceInputs") or old_payload.get("race_inputs") or {}
            old_drivers = old_race_inputs.get("drivers") if isinstance(old_race_inputs, dict) else {}
            if isinstance(old_drivers, dict) and old_drivers:
                payload["RaceInputs"] = old_race_inputs
                payload["warnings"].append({
                    "code": "preserved_existing_race_inputs",
                    "message": "New refresh returned no usable race input rows, so existing cached race evidence was preserved.",
                    "rows": len(old_drivers),
                })

        payload["coverage"] = _coverage(payload, openf1_payloads)
        payload["available"] = any(
            int(payload["coverage"].get(key) or 0) > 0
            for key in ("practice_rows", "qualifying_rows", "grid_rows", "race_input_drivers")
        )

        result = {
            "ok": True,
            "season": int(season),
            "round": int(race.round),
            "path": str(path),
            "write": bool(write),
            "payload": payload,
            "coverage": payload["coverage"],
            "available": payload["available"],
            "warnings": payload["warnings"],
        }
        if write:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                result["bytes"] = path.stat().st_size
            except Exception as exc:
                result.update({"ok": False, "reason": str(exc)})
        return result

    def path_for(self, season: int, round_num: int) -> Path:
        return self.cache_dir / f"season_{int(season)}_round_{int(round_num):02d}.json"


def _coverage(payload: dict[str, Any], openf1_payloads: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {
        "practice_rows": len(payload.get("PracticeResults") or []),
        "qualifying_rows": len(payload.get("QualifyingResults") or []),
        "grid_rows": len(payload.get("GridResults") or []),
        "race_input_drivers": len(((payload.get("RaceInputs") or {}).get("drivers") or {})),
        "openf1_sessions_ok": sum(1 for item in openf1_payloads.values() if item.get("ok")),
        "openf1_sessions_requested": len(openf1_payloads),
    }


def race_from_historical_raw(raw: dict[str, Any]) -> Race:
    circuit = raw.get("Circuit") or {}
    location = circuit.get("Location") or {}
    date_str = raw.get("date") or f"{raw.get('season', 2000)}-01-01"
    time_str = str(raw.get("time") or "14:00:00Z").rstrip("Z")
    try:
        race_date = datetime.fromisoformat(f"{date_str}T{time_str}").replace(tzinfo=timezone.utc)
    except ValueError:
        race_date = datetime.now(timezone.utc)
    return Race(
        round=int(raw.get("round") or 0),
        name=raw.get("raceName") or "",
        circuit=circuit.get("circuitName") or "",
        country=location.get("country") or "",
        date=race_date,
        circuit_id=circuit.get("circuitId"),
        locality=location.get("locality"),
        latitude=_float(location.get("lat")),
        longitude=_float(location.get("long")),
        has_sprint=bool(raw.get("SprintResults") or raw.get("Sprint") or raw.get("SprintQualifying")),
        status="COMPLETED" if raw.get("Results") else "SCHEDULED",
    )


def drivers_from_historical_raw(raw: dict[str, Any]) -> list[Driver]:
    rows = raw.get("Results") or raw.get("QualifyingResults") or []
    drivers = []
    seen = set()
    for item in rows:
        driver = item.get("Driver") or {}
        constructor = item.get("Constructor") or {}
        driver_id = driver.get("driverId")
        if not driver_id or driver_id in seen:
            continue
        seen.add(driver_id)
        drivers.append(Driver(
            id=driver_id,
            number=_int(driver.get("permanentNumber")),
            code=driver.get("code") or str(driver_id)[:3].upper(),
            first_name=driver.get("givenName") or "",
            last_name=driver.get("familyName") or "",
            nationality=driver.get("nationality") or "",
            team=constructor.get("name") or "",
        ))
    return drivers


def profile_from_historical_raw(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "race": race_from_historical_raw(raw).model_dump(mode="json"),
        "qualifying": _historical_qualifying_rows(raw),
        "results": _historical_result_rows(raw),
        "context": {
            "has_sprint": bool(raw.get("SprintResults") or raw.get("Sprint") or raw.get("SprintQualifying")),
            "has_results": bool(raw.get("Results")),
            "has_qualifying": bool(raw.get("QualifyingResults")),
        },
    }


def _normalize_sessions(sessions: list[str] | tuple[str, ...] | None) -> list[str]:
    if sessions is None:
        return list(DEFAULT_SESSIONS)
    output = []
    for session in sessions:
        code = _canonical_session(session)
        if code and code not in output:
            output.append(code)
    return output


def _read_existing_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _canonical_session(session: str | None) -> str:
    value = str(session or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "practice": "fp1",
        "practice1": "fp1",
        "practice_1": "fp1",
        "p1": "fp1",
        "practice2": "fp2",
        "practice_2": "fp2",
        "p2": "fp2",
        "practice3": "fp3",
        "practice_3": "fp3",
        "p3": "fp3",
        "quali": "qualifying",
        "qualification": "qualifying",
        "sq": "sprint_qualifying",
        "sprint_quali": "sprint_qualifying",
        "sprint_shootout": "sprint_qualifying",
        "grand_prix": "race",
    }
    value = aliases.get(value, value)
    return value if value in {"fp1", "fp2", "fp3", "qualifying", "sprint_qualifying", "sprint", "race"} else ""


def _openf1_session_arg(session_code: str) -> str:
    return {
        "fp1": "fp1",
        "fp2": "fp2",
        "fp3": "fp3",
        "qualifying": "qualifying",
        "sprint_qualifying": "sprint_qualifying",
        "sprint": "sprint",
        "race": "race",
    }.get(session_code, "race")


async def _openf1_session_features(openf1, race: Race, session_code: str, drivers: list[Driver]) -> dict[str, Any]:
    endpoints = _openf1_endpoint_profile(session_code)
    try:
        return await openf1.get_session_features(
            race=race,
            session=_openf1_session_arg(session_code),
            drivers=drivers,
            live=False,
            endpoints=endpoints,
        )
    except TypeError:
        return await openf1.get_session_features(
            race=race,
            session=_openf1_session_arg(session_code),
            drivers=drivers,
            live=False,
        )


def _openf1_endpoint_profile(session_code: str) -> tuple[str, ...]:
    if session_code in {"fp1", "fp2", "fp3", "qualifying", "sprint_qualifying"}:
        return ("laps", "stints")
    if session_code == "sprint":
        return ("laps", "positions", "intervals", "stints", "pits")
    return ("laps", "positions", "intervals", "stints", "pits", "weather", "race_control")


def _fastf1_mode(value: str | None) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in {"off", "fallback", "force"} else "off"


def _practice_rows_from_openf1(session: str, payload: dict[str, Any], number_lookup: dict[str, Driver]) -> list[dict[str, Any]]:
    if not payload.get("ok"):
        return []
    laps = ((payload.get("laps") or {}).get("drivers") or {})
    stints = ((payload.get("stints") or {}).get("drivers") or {})
    rows = []
    for number, lap in laps.items():
        driver = number_lookup.get(str(number))
        if not driver:
            continue
        stint = stints.get(str(number)) or {}
        compounds = sorted({
            str(item).upper()
            for item in [*(lap.get("compounds") or []), *(stint.get("compounds") or []), stint.get("compound")]
            if item
        })
        rows.append({
            "session": session,
            "driver_id": driver.id,
            "driver_code": driver.code,
            "driver_number": driver.number,
            "team": driver.team,
            "best_lap": lap.get("best_lap"),
            "representative_lap": lap.get("representative_lap") or lap.get("median_lap") or lap.get("best_lap"),
            "median_lap": lap.get("median_lap"),
            "long_run_lap": lap.get("long_run_lap") or lap.get("median_lap"),
            "best_sector_1": lap.get("best_sector_1"),
            "best_sector_2": lap.get("best_sector_2"),
            "best_sector_3": lap.get("best_sector_3"),
            "representative_sector_1": lap.get("representative_sector_1"),
            "representative_sector_2": lap.get("representative_sector_2"),
            "representative_sector_3": lap.get("representative_sector_3"),
            "sector_coverage": lap.get("sector_coverage"),
            "track_evolution_delta": lap.get("track_evolution_delta"),
            "lap_time_stddev": lap.get("lap_time_stddev"),
            "pace_stability": lap.get("pace_stability"),
            "lap_distribution": lap.get("lap_distribution") if isinstance(lap.get("lap_distribution"), dict) else {},
            "laps": lap.get("laps") or 0,
            "compounds": compounds,
            "source": "openf1_compact_laps",
        })
    return rows


async def _practice_rows_from_fastf1(
    *,
    season: int,
    race: Race,
    session: str,
    drivers: list[Driver],
    provider=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        if provider is None:
            from sports.f1.ml.providers.fastf1_provider import FastF1Provider
            provider = FastF1Provider(cache_dir=str(DEFAULT_EVIDENCE_CACHE_DIR.parent / "fastf1_cache"))
        from sports.f1.ml.common.types import Race as MlRace
        from sports.f1.ml.common.types import SessionType
    except Exception as exc:
        return [], {"ok": False, "source": "fastf1", "session": session, "reason": f"fastf1_import_error:{exc.__class__.__name__}"}

    session_type = {
        "fp1": SessionType.FP1,
        "fp2": SessionType.FP2,
        "fp3": SessionType.FP3,
    }.get(session)
    if session_type is None:
        return [], {"ok": False, "source": "fastf1", "session": session, "reason": "fastf1_session_unsupported"}

    ml_race = MlRace(
        season=int(season),
        round=int(race.round),
        track_code=_track_code(race),
        name=race.name,
        scheduled_start=race.date,
    )
    try:
        laps = await asyncio.to_thread(provider.laps, ml_race, session_type)
    except Exception as exc:
        return [], {"ok": False, "source": "fastf1", "session": session, "reason": f"fastf1_load_error:{exc.__class__.__name__}"}

    by_code: dict[str, list[Any]] = {}
    for lap in laps or []:
        code = str(getattr(lap, "driver_code", "") or "").upper()
        duration = _float(getattr(lap, "lap_time_s", None))
        if not code or duration is None or duration <= 0:
            continue
        by_code.setdefault(code, []).append(lap)

    driver_by_code = {str(driver.code or "").upper(): driver for driver in drivers if driver.code}
    rows = []
    for code, driver_laps in sorted(by_code.items()):
        driver = driver_by_code.get(code)
        if not driver:
            continue
        usable_laps = _clean_practice_laps(driver_laps)
        durations = [
            _float(getattr(lap, "lap_time_s", None))
            for lap in usable_laps
            if _float(getattr(lap, "lap_time_s", None)) is not None
        ]
        durations = [value for value in durations if value is not None and value > 0]
        if not durations:
            continue
        ordered = sorted(usable_laps, key=lambda item: int(getattr(item, "lap_number", 0) or 0))
        split_at = max(1, len(ordered) // 2)
        late_laps = [
            _float(getattr(lap, "lap_time_s", None))
            for lap in ordered[split_at:]
            if _float(getattr(lap, "lap_time_s", None)) is not None
        ] or durations
        early_laps = [
            _float(getattr(lap, "lap_time_s", None))
            for lap in ordered[:split_at]
            if _float(getattr(lap, "lap_time_s", None)) is not None
        ] or durations
        sector_summary = _fastf1_sector_summary(usable_laps)
        lap_stddev = statistics.pstdev(durations) if len(durations) >= 2 else 0.0
        pace_stability = _pace_stability(lap_stddev)
        lap_distribution = _lap_distribution(durations)
        compounds = sorted({
            str(getattr(getattr(lap, "compound", None), "value", getattr(lap, "compound", "")) or "").upper()
            for lap in usable_laps
            if getattr(lap, "compound", None)
        })
        telemetry_quality = _practice_telemetry_quality(
            usable_laps=len(durations),
            raw_laps=len(driver_laps),
            sector_coverage=_float(sector_summary.get("sector_coverage")) or 0.0,
            pace_stability=pace_stability,
        )
        rows.append({
            "session": session,
            "driver_id": driver.id,
            "driver_code": driver.code,
            "driver_number": driver.number,
            "team": driver.team,
            "best_lap": round(min(durations), 3),
            "representative_lap": round(statistics.median(durations), 3),
            "median_lap": round(statistics.median(durations), 3),
            "long_run_lap": round(statistics.median(late_laps), 3) if late_laps else round(statistics.median(durations), 3),
            "track_evolution_delta": (
                round(min(early_laps) - min(late_laps), 3)
                if early_laps and late_laps else None
            ),
            "lap_time_stddev": round(lap_stddev, 3),
            "pace_stability": round(pace_stability, 4),
            "lap_distribution": lap_distribution,
            "telemetry_quality": round(telemetry_quality, 4),
            **sector_summary,
            "laps": len(durations),
            "raw_laps": len(driver_laps),
            "usable_laps": len(durations),
            "compounds": compounds,
            "source": "fastf1_cached_practice_laps",
        })
    return rows, {
        "ok": bool(rows),
        "source": "fastf1",
        "session": session,
        "raw_laps": len(laps or []),
        "driver_rows": len(rows),
        "reason": None if rows else "fastf1_no_matching_practice_laps",
    }


def _clean_practice_laps(laps: list[Any]) -> list[Any]:
    timed = [
        lap for lap in laps
        if (_float(getattr(lap, "lap_time_s", None)) or 0.0) > 0
        and not bool(getattr(lap, "pit_in", False))
        and not bool(getattr(lap, "pit_out", False))
    ]
    if not timed:
        return []
    best = min(_float(getattr(lap, "lap_time_s", None)) or 99999.0 for lap in timed)
    # Practice feeds include cooldown, aborted and traffic laps. Keep the
    # performance-relevant window while leaving long-run medians possible.
    threshold = min(best * 1.08, best + 7.5)
    clean = [
        lap for lap in timed
        if (_float(getattr(lap, "lap_time_s", None)) or 99999.0) <= threshold
    ]
    return clean or sorted(timed, key=lambda lap: _float(getattr(lap, "lap_time_s", None)) or 99999.0)[:5]


def _fastf1_sector_summary(laps: list[Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for index, attr in ((1, "sector1_s"), (2, "sector2_s"), (3, "sector3_s")):
        values = [
            _float(getattr(lap, attr, None))
            for lap in laps
            if _float(getattr(lap, attr, None)) is not None and (_float(getattr(lap, attr, None)) or 0) > 0
        ]
        values = [value for value in values if value is not None and value > 0]
        if not values:
            continue
        clean = sorted(values)[: max(1, min(5, len(values)))]
        payload[f"best_sector_{index}"] = round(min(values), 3)
        payload[f"representative_sector_{index}"] = round(sum(clean) / len(clean), 3)
    payload["sector_coverage"] = round(
        sum(1 for key in payload if key.startswith("representative_sector_")) / 3.0,
        4,
    )
    return payload


def _pace_stability(lap_stddev: float | None) -> float:
    if lap_stddev is None:
        return 0.50
    return max(0.0, min(1.0, 1.0 - (float(lap_stddev) / 3.0)))


def _lap_distribution(values: list[float]) -> dict[str, Any]:
    clean = sorted(float(value) for value in values if value is not None and value > 0)
    if not clean:
        return {}
    return {
        "sample_size": len(clean),
        "p10": round(_percentile(clean, 0.10), 3),
        "p25": round(_percentile(clean, 0.25), 3),
        "median": round(statistics.median(clean), 3),
        "p75": round(_percentile(clean, 0.75), 3),
        "p90": round(_percentile(clean, 0.90), 3),
        "best": round(clean[0], 3),
        "spread_p90_p10": round(_percentile(clean, 0.90) - _percentile(clean, 0.10), 3),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = max(0.0, min(1.0, quantile)) * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _practice_telemetry_quality(
    *,
    usable_laps: int,
    raw_laps: int,
    sector_coverage: float,
    pace_stability: float,
) -> float:
    lap_depth = min(int(usable_laps or 0), 12) / 12.0
    usable_share = min(1.0, int(usable_laps or 0) / max(1, int(raw_laps or 0)))
    return max(0.0, min(
        1.0,
        0.38 * lap_depth
        + 0.28 * max(0.0, min(1.0, sector_coverage))
        + 0.22 * max(0.0, min(1.0, pace_stability))
        + 0.12 * usable_share,
    ))


def _track_code(race: Race) -> str:
    raw = race.circuit_id or race.circuit or race.country or race.name or "track"
    return str(raw).upper().replace(" ", "_").replace("-", "_")


def _race_inputs_from_openf1(payload: dict[str, Any], number_lookup: dict[str, Driver]) -> dict[str, Any]:
    if not payload.get("ok"):
        return {}
    drivers = {}
    laps = ((payload.get("laps") or {}).get("drivers") or {})
    positions = ((payload.get("positions") or {}).get("drivers") or {})
    intervals = ((payload.get("intervals") or {}).get("drivers") or {})
    stints = ((payload.get("stints") or {}).get("drivers") or {})
    pits = ((payload.get("pits") or {}).get("drivers") or {})
    for number in set(laps) | set(positions) | set(intervals) | set(stints) | set(pits):
        driver = number_lookup.get(str(number))
        if not driver:
            continue
        lap = laps.get(str(number)) or {}
        position = positions.get(str(number)) or {}
        interval = intervals.get(str(number)) or {}
        stint = stints.get(str(number)) or {}
        pit = pits.get(str(number)) or {}
        compounds = stint.get("compounds") or lap.get("compounds") or []
        drivers[driver.id] = {
            "driver_id": driver.id,
            "driver_code": driver.code,
            "driver_number": driver.number,
            "team": driver.team,
            "position": position.get("position"),
            "gap_to_leader": interval.get("gap_to_leader"),
            "interval": interval.get("interval"),
            "lap": lap.get("lap"),
            "laps": lap.get("laps"),
            "best_lap": lap.get("best_lap"),
            "representative_lap": lap.get("representative_lap") or lap.get("median_lap"),
            "compound": stint.get("compound") or (compounds[-1] if compounds else None),
            "tyre_age": stint.get("tyre_age") or stint.get("avg_stint_laps"),
            "stints": stint.get("stints"),
            "compound_sequence": stint.get("compound_sequence") or compounds,
            "avg_stint_laps": stint.get("avg_stint_laps"),
            "max_stint_laps": stint.get("max_stint_laps"),
            "final_stint_laps": stint.get("final_stint_laps"),
            "stint_lap_distribution": stint.get("stint_lap_distribution") if isinstance(stint.get("stint_lap_distribution"), dict) else {},
            "estimated_tyre_age": stint.get("estimated_tyre_age"),
            "pit_stops": pit.get("pit_stops"),
            "source": "openf1_compact_race_inputs",
        }
    return {
        "source": "openf1_compact_race_inputs",
        "drivers": drivers,
        "weather": payload.get("weather") or {},
        "race_control": payload.get("race_control") or {},
    }


def _qualifying_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in profile.get("qualifying") or []:
        if not row.get("driver_id"):
            continue
        rows.append({
            "position": row.get("position"),
            "driver_id": row.get("driver_id"),
            "driver_code": row.get("driver_code"),
            "driver_name": row.get("driver_name"),
            "team": row.get("team"),
            "grid": row.get("grid") or row.get("position"),
            "q1": row.get("q1"),
            "q2": row.get("q2"),
            "q3": row.get("q3"),
            "source": row.get("source") or "profile_qualifying",
        })
    return rows


def _grid_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    results = profile.get("results") or []
    rows = []
    for row in results:
        if not row.get("driver_id"):
            continue
        grid = row.get("grid")
        if grid in {None, ""}:
            continue
        rows.append({
            "driver_id": row.get("driver_id"),
            "driver_code": row.get("driver_code"),
            "driver_name": row.get("driver_name"),
            "team": row.get("team"),
            "grid": grid,
            "position": row.get("position"),
            "status": row.get("status"),
            "source": row.get("source") or "profile_grid",
        })
    return rows


def _historical_qualifying_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in raw.get("QualifyingResults") or []:
        driver = item.get("Driver") or {}
        constructor = item.get("Constructor") or {}
        driver_id = driver.get("driverId")
        if not driver_id:
            continue
        rows.append({
            "position": _int(item.get("position")),
            "driver_id": driver_id,
            "driver_code": driver.get("code") or str(driver_id)[:3].upper(),
            "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
            "team": constructor.get("name"),
            "grid": _int(item.get("position")),
            "q1": item.get("Q1") or item.get("q1"),
            "q2": item.get("Q2") or item.get("q2"),
            "q3": item.get("Q3") or item.get("q3"),
            "source": "historical_jolpica_qualifying",
        })
    return sorted(rows, key=lambda row: row.get("position") or 99)


def _historical_result_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in raw.get("Results") or []:
        driver = item.get("Driver") or {}
        constructor = item.get("Constructor") or {}
        driver_id = driver.get("driverId")
        if not driver_id:
            continue
        rows.append({
            "position": _int(item.get("position")),
            "driver_id": driver_id,
            "driver_code": driver.get("code") or str(driver_id)[:3].upper(),
            "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
            "team": constructor.get("name"),
            "grid": _int(item.get("grid")),
            "points": _float(item.get("points")) or 0.0,
            "status": item.get("status") or "Unknown",
            "source": "historical_jolpica_results",
        })
    return sorted(rows, key=lambda row: row.get("position") or 99)


def _float(value: Any) -> float | None:
    try:
        parsed = float(value) if value is not None and value != "" else None
        return parsed if parsed is not None and math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None
