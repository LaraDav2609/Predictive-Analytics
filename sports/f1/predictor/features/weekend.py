"""Race-weekend evidence normalization for F1 predictions.

This module turns official results, OpenF1 timing summaries, and live state into
one source-labeled evidence object. It stays intentionally conservative: missing
real timing never creates high-confidence pace evidence.
"""

from __future__ import annotations

from typing import Any

from sports.f1.models.f1 import Driver, Race
from sports.f1.predictor.scoring.normalization import clamp01


PRACTICE_SESSIONS = ("fp1", "fp2", "fp3")


def build_weekend_evidence(
    *,
    race: Race,
    drivers: list[Driver],
    profile: dict[str, Any] | None = None,
    openf1_sessions: dict[str, dict[str, Any]] | None = None,
    live_state: dict[str, Any] | None = None,
    session: str = "race",
    live: bool = False,
    weather: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build normalized race-weekend evidence for feature and probability code."""

    profile = profile or {}
    openf1_sessions = openf1_sessions or {}
    live_state = live_state or {}
    session_key = normalize_weekend_session(session)
    drivers_by_number = {str(driver.number): driver for driver in drivers if driver.number is not None}
    drivers_by_id = {driver.id: driver for driver in drivers}

    practice = _practice_evidence(openf1_sessions, drivers_by_number, len(drivers))
    grid = _grid_evidence(profile, drivers_by_id)
    race_inputs = _race_input_evidence(
        session_key=session_key,
        drivers=drivers,
        openf1_sessions=openf1_sessions,
        live_state=live_state,
        live=live,
        weather=weather,
    )

    combined_drivers: dict[str, dict[str, Any]] = {}
    for driver in drivers:
        combined_drivers[driver.id] = {
            "driver_id": driver.id,
            "driver_code": driver.code,
            "team": driver.team,
            "practice": (practice.get("drivers") or {}).get(driver.id),
            "grid": (grid.get("drivers") or {}).get(driver.id),
            "race_inputs": (race_inputs.get("drivers") or {}).get(driver.id),
        }

    coverage_counts = {
        "practice_drivers": len(practice.get("drivers") or {}),
        "grid_drivers": len(grid.get("drivers") or {}),
        "race_input_drivers": len(race_inputs.get("drivers") or {}),
        "weather_available": 1 if race_inputs.get("weather") and not (race_inputs.get("weather") or {}).get("missing_data") else 0,
        "race_control_available": 1 if race_inputs.get("race_control") and not (race_inputs.get("race_control") or {}).get("missing_data") else 0,
    }
    missing_groups = []
    if not practice.get("available"):
        missing_groups.append("practice_timing")
    if session_key in {"race", "sprint"} and not grid.get("available"):
        missing_groups.append("grid_truth")
    if session_key == "race" and not race_inputs.get("available"):
        missing_groups.append("race_timing_inputs")
    if coverage_counts["weather_available"] == 0:
        missing_groups.append("weather")
    if coverage_counts["race_control_available"] == 0:
        missing_groups.append("race_control")

    confidence = _overall_confidence(practice, grid, race_inputs, session_key, live)
    source_mode = _source_mode(practice, grid, race_inputs, live)
    return {
        "ok": True,
        "round": race.round,
        "race_name": race.name,
        "session": session_key,
        "live": live,
        "source_mode": source_mode,
        "confidence": confidence,
        "practice": practice,
        "grid": grid,
        "race_inputs": race_inputs,
        "drivers": combined_drivers,
        "coverage_counts": coverage_counts,
        "missing_groups": sorted(set(missing_groups)),
        "explanations": _explanations(practice, grid, race_inputs, session_key, live),
    }


def apply_weekend_evidence_adjustments(
    drivers: list[Driver],
    driver_features: dict[str, dict[str, Any]],
    weekend_evidence: dict[str, Any] | None,
    *,
    session_stage: str = "race",
) -> dict[str, dict[str, Any]]:
    """Blend normalized weekend evidence into existing per-driver features."""

    adjusted = {driver_id: dict(values or {}) for driver_id, values in (driver_features or {}).items()}
    evidence = weekend_evidence or {}
    if not evidence.get("ok"):
        return adjusted

    session = normalize_weekend_session(session_stage)
    for driver in drivers:
        feature = adjusted.setdefault(driver.id, {})
        driver_evidence = (evidence.get("drivers") or {}).get(driver.id) or {}
        practice = driver_evidence.get("practice") or {}
        grid = driver_evidence.get("grid") or {}
        race_inputs = driver_evidence.get("race_inputs") or {}

        if practice:
            feature.update(_practice_feature_fields(practice))
            old_quali = _float(feature.get("qualifying_pace_score")) or _float(feature.get("form_score")) or 0.50
            old_race = _float(feature.get("race_pace_score")) or _float(feature.get("form_score")) or 0.50
            confidence = _float(practice.get("confidence")) or 0.0
            fuel_uncertainty = _float(practice.get("fuel_uncertainty")) or 0.35
            if session == "qualifying":
                quali_weight, race_weight = 0.34, 0.10
            elif session in PRACTICE_SESSIONS:
                quali_weight, race_weight = 0.38, 0.32
            else:
                quali_weight, race_weight = 0.16, 0.28
            quali_weight *= 0.65 + confidence * 0.35
            race_weight *= (0.65 + confidence * 0.35) * (1.0 - min(0.40, fuel_uncertainty * 0.35))
            feature["qualifying_pace_score"] = round(_blend(old_quali, _float(practice.get("qualifying_evidence_score")) or old_quali, quali_weight), 4)
            feature["race_pace_score"] = round(_blend(old_race, _float(practice.get("race_evidence_score")) or old_race, race_weight), 4)

        if grid:
            feature["qualifying_position"] = grid.get("qualifying_position")
            feature["grid_position"] = grid.get("grid_position")
            feature["grid_penalty"] = grid.get("grid_penalty") or 0
            feature["pit_lane_start"] = bool(grid.get("pit_lane_start"))
            feature["grid_source"] = grid.get("source")
            feature["grid_confidence"] = grid.get("confidence")

        if race_inputs:
            feature["weekend_position"] = race_inputs.get("position")
            feature["weekend_gap_to_leader"] = race_inputs.get("gap_to_leader")
            feature["weekend_interval"] = race_inputs.get("interval")
            feature["weekend_lap"] = race_inputs.get("lap")
            feature["weekend_compound"] = race_inputs.get("compound")
            feature["weekend_tyre_age"] = race_inputs.get("tyre_age")
            feature["weekend_stints"] = race_inputs.get("stints")
            feature["weekend_compound_sequence"] = race_inputs.get("compound_sequence") or []
            feature["weekend_avg_stint_laps"] = race_inputs.get("avg_stint_laps")
            feature["weekend_max_stint_laps"] = race_inputs.get("max_stint_laps")
            feature["weekend_final_stint_laps"] = race_inputs.get("final_stint_laps")
            feature["weekend_stint_lap_distribution"] = race_inputs.get("stint_lap_distribution") or {}
            feature["weekend_tyre_phase"] = race_inputs.get("tyre_phase")
            feature["weekend_pit_stops"] = race_inputs.get("pit_stops")
            feature["weekend_input_source"] = race_inputs.get("source")
            feature["weekend_input_confidence"] = race_inputs.get("confidence")

    return adjusted


def normalize_weekend_session(session: str | None) -> str:
    value = (session or "race").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "practice": "fp1",
        "practice1": "fp1",
        "practice_1": "fp1",
        "free_practice_1": "fp1",
        "p1": "fp1",
        "practice2": "fp2",
        "practice_2": "fp2",
        "free_practice_2": "fp2",
        "p2": "fp2",
        "practice3": "fp3",
        "practice_3": "fp3",
        "free_practice_3": "fp3",
        "p3": "fp3",
        "quali": "qualifying",
        "qualification": "qualifying",
        "sprint_quali": "sprint_qualifying",
        "sprint_shootout": "sprint_qualifying",
    }
    return aliases.get(value, value if value in {*PRACTICE_SESSIONS, "sprint_qualifying", "qualifying", "sprint", "race"} else "race")


def has_real_practice_evidence(evidence: dict[str, Any] | None) -> bool:
    practice = (evidence or {}).get("practice") or {}
    return bool(practice.get("available") and (practice.get("coverage_count") or 0) > 0)


def has_real_grid_evidence(evidence: dict[str, Any] | None) -> bool:
    grid = (evidence or {}).get("grid") or {}
    return bool(grid.get("available") and (grid.get("coverage_count") or 0) > 0)


def has_real_live_evidence(evidence: dict[str, Any] | None) -> bool:
    race_inputs = (evidence or {}).get("race_inputs") or {}
    return bool(race_inputs.get("available") and (race_inputs.get("coverage_count") or 0) >= 10)


def _practice_evidence(openf1_sessions: dict[str, dict[str, Any]], drivers_by_number: dict[str, Driver], field_size: int) -> dict[str, Any]:
    rows_by_driver: dict[str, list[dict[str, Any]]] = {}
    session_summaries = {}
    for session in PRACTICE_SESSIONS:
        payload = openf1_sessions.get(session) or {}
        laps = ((payload.get("laps") or {}).get("drivers") or {}) if payload.get("ok") else {}
        stints = ((payload.get("stints") or {}).get("drivers") or {}) if payload.get("ok") else {}
        session_summaries[session] = {
            "ok": bool(payload.get("ok")),
            "source": payload.get("source") or "openf1",
            "raw_counts": payload.get("raw_counts") or {},
            "driver_count": len(laps),
            "reason": payload.get("reason"),
        }
        for number, lap in laps.items():
            driver = drivers_by_number.get(str(number))
            if not driver:
                continue
            representative = _float(lap.get("representative_lap") or lap.get("median_lap") or lap.get("best_lap"))
            best = _float(lap.get("best_lap"))
            long_run = _float(lap.get("long_run_lap") or lap.get("median_lap"))
            if not representative and not best:
                continue
            stint = stints.get(str(number)) or {}
            rows_by_driver.setdefault(driver.id, []).append({
                "session": session,
                "representative_lap": representative or best,
                "best_lap": best or representative,
                "long_run_lap": long_run,
                "lap_count": int(lap.get("laps") or 0),
                "compounds": sorted({
                    str(item).upper()
                    for item in [*(lap.get("compounds") or []), *(stint.get("compounds") or []), stint.get("compound")]
                    if item
                }),
                "track_evolution_delta": _float(lap.get("track_evolution_delta")),
                "lap_time_stddev": _float(lap.get("lap_time_stddev")),
                "pace_stability": _float(lap.get("pace_stability")),
                "sectors": {
                    "1": _float(lap.get("representative_sector_1") or lap.get("best_sector_1")),
                    "2": _float(lap.get("representative_sector_2") or lap.get("best_sector_2")),
                    "3": _float(lap.get("representative_sector_3") or lap.get("best_sector_3")),
                },
            })

    if not rows_by_driver:
        return {
            "available": False,
            "source": "unavailable",
            "confidence": 0.0,
            "coverage_count": 0,
            "sessions": session_summaries,
            "drivers": {},
            "reason": "no_openf1_practice_lap_rows",
        }

    aggregated = {driver_id: _aggregate_practice_rows(rows) for driver_id, rows in rows_by_driver.items()}
    _score_practice_rows(aggregated)
    coverage = len(aggregated) / max(field_size, 1)
    confidence = clamp01(0.25 + coverage * 0.22 + min(0.28, _avg([row.get("lap_count") for row in aggregated.values()]) / 20.0 * 0.28))
    top_short = _top_driver_rows(aggregated, "best_lap")
    top_long = _top_driver_rows(aggregated, "long_run_lap")
    return {
        "available": True,
        "source": "openf1_practice_laps",
        "confidence": round(confidence, 4),
        "coverage_count": len(aggregated),
        "sessions": session_summaries,
        "drivers": aggregated,
        "top_short_run": top_short,
        "top_long_run": top_long,
    }


def _aggregate_practice_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_row = min(rows, key=lambda row: _float(row.get("best_lap")) or 99999)
    representative_values = [_float(row.get("representative_lap")) for row in rows if _float(row.get("representative_lap"))]
    long_values = [_float(row.get("long_run_lap")) for row in rows if _float(row.get("long_run_lap"))]
    compounds = sorted({compound for row in rows for compound in (row.get("compounds") or [])})
    sectors = {}
    for index in ("1", "2", "3"):
        values = [_float((row.get("sectors") or {}).get(index)) for row in rows if _float((row.get("sectors") or {}).get(index))]
        sectors[index] = round(min(values), 3) if values else None
    stddev_values = [_float(row.get("lap_time_stddev")) for row in rows if _float(row.get("lap_time_stddev")) is not None]
    stability_values = [_float(row.get("pace_stability")) for row in rows if _float(row.get("pace_stability")) is not None]
    lap_count = sum(int(row.get("lap_count") or 0) for row in rows)
    best_lap = _float(best_row.get("best_lap"))
    representative = _avg(representative_values) or best_lap
    long_run = _avg(long_values)
    pace_stability = _avg(stability_values)
    if pace_stability is None and stddev_values:
        pace_stability = clamp01(1.0 - (_avg(stddev_values) or 0.0) / 3.0)
    fuel_uncertainty = clamp01(
        0.40
        - min(lap_count, 30) * 0.008
        - (0.05 if long_run else 0.0)
        - (0.04 if compounds else 0.0)
        - (0.04 if pace_stability is not None else 0.0)
    )
    return {
        "sessions": [row["session"] for row in rows],
        "best_lap": round(best_lap, 3) if best_lap else None,
        "representative_lap": round(representative, 3) if representative else None,
        "long_run_lap": round(long_run, 3) if long_run else None,
        "lap_time_stddev": round(_avg(stddev_values), 3) if stddev_values else None,
        "pace_stability": round(pace_stability, 4) if pace_stability is not None else None,
        "lap_count": lap_count,
        "compounds": compounds,
        "sector_times": sectors,
        "fuel_uncertainty": round(fuel_uncertainty, 4),
        "source": "openf1_practice_laps",
    }


def _score_practice_rows(rows: dict[str, dict[str, Any]]) -> None:
    best_values = [_float(row.get("best_lap")) for row in rows.values() if _float(row.get("best_lap"))]
    representative_values = [_float(row.get("representative_lap")) for row in rows.values() if _float(row.get("representative_lap"))]
    long_values = [_float(row.get("long_run_lap")) for row in rows.values() if _float(row.get("long_run_lap"))]
    best_min, best_max = min(best_values or [0.0]), max(best_values or [0.0])
    rep_min, rep_max = min(representative_values or [0.0]), max(representative_values or [0.0])
    long_min, long_max = min(long_values or representative_values or [0.0]), max(long_values or representative_values or [0.0])
    sector_bounds = {}
    for index in ("1", "2", "3"):
        values = [_float((row.get("sector_times") or {}).get(index)) for row in rows.values() if _float((row.get("sector_times") or {}).get(index))]
        if values:
            sector_bounds[index] = (min(values), max(values))

    for row in rows.values():
        best_score = _inverse_score(row.get("best_lap"), best_min, best_max, minimum_spread=1.6)
        pace_score = _inverse_score(row.get("representative_lap"), rep_min, rep_max, minimum_spread=2.2)
        long_score = _inverse_score(row.get("long_run_lap") or row.get("representative_lap"), long_min, long_max, minimum_spread=2.8)
        sector_scores = {}
        for index, bounds in sector_bounds.items():
            sector_scores[index] = _inverse_score((row.get("sector_times") or {}).get(index), bounds[0], bounds[1], minimum_spread=0.25)
        sector_score = _avg(sector_scores.values()) if sector_scores else None
        stability = _float(row.get("pace_stability"))
        confidence = clamp01(
            0.18
            + min(int(row.get("lap_count") or 0), 24) / 24 * 0.32
            + (0.10 if row.get("long_run_lap") else 0.0)
            + (0.08 if row.get("compounds") else 0.0)
            + (0.08 if sector_scores else 0.0)
            + (0.06 if stability is not None else 0.0)
            - (_float(row.get("fuel_uncertainty")) or 0.35) * 0.10
        )
        row["practice_pace_score"] = round(pace_score, 4)
        row["qualifying_evidence_score"] = round(clamp01(0.48 * best_score + 0.28 * pace_score + 0.24 * (sector_score if sector_score is not None else pace_score)), 4)
        row["race_evidence_score"] = round(clamp01(
            0.42 * pace_score
            + 0.34 * long_score
            + 0.10 * (stability if stability is not None else pace_score)
            + 0.14 * (1.0 - (_float(row.get("fuel_uncertainty")) or 0.35))
        ), 4)
        row["long_run_score"] = round(long_score, 4)
        row["sector_score"] = round(sector_score, 4) if sector_score is not None else None
        row["sector_scores"] = {key: round(value, 4) for key, value in sector_scores.items()}
        row["confidence"] = round(confidence, 4)


def _grid_evidence(profile: dict[str, Any], drivers_by_id: dict[str, Driver]) -> dict[str, Any]:
    rows = profile.get("qualifying") or []
    race_rows = {row.get("driver_id"): row for row in profile.get("results") or [] if row.get("driver_id")}
    drivers = {}
    penalties = []
    for index, row in enumerate(rows, start=1):
        driver_id = row.get("driver_id")
        if not driver_id:
            continue
        race_row = race_rows.get(driver_id) or {}
        qualifying_position = _int(row.get("position")) or index
        grid_position = _int(row.get("grid")) or _int(race_row.get("grid")) or qualifying_position
        pit_lane_start = grid_position in {0, -1} or "pit" in str(race_row.get("status") or row.get("status") or "").lower()
        penalty = max(0, (grid_position or qualifying_position) - qualifying_position) if grid_position else 0
        item = {
            "driver_id": driver_id,
            "driver_code": row.get("driver_code") or getattr(drivers_by_id.get(driver_id), "code", None),
            "qualifying_position": qualifying_position,
            "grid_position": grid_position,
            "grid_penalty": penalty,
            "pit_lane_start": pit_lane_start,
            "status": race_row.get("status") or row.get("status"),
            "source": "jolpica_qualifying_grid" if rows else "unavailable",
            "confidence": 0.86 if grid_position else 0.74,
        }
        if penalty or pit_lane_start:
            penalties.append(item)
        drivers[driver_id] = item

    return {
        "available": bool(drivers),
        "source": "jolpica_qualifying_grid" if drivers else "unavailable",
        "confidence": 0.86 if drivers else 0.0,
        "coverage_count": len(drivers),
        "drivers": drivers,
        "penalties": penalties,
        "reason": None if drivers else "no_official_qualifying_or_grid_rows",
    }


def _race_input_evidence(
    *,
    session_key: str,
    drivers: list[Driver],
    openf1_sessions: dict[str, dict[str, Any]],
    live_state: dict[str, Any],
    live: bool,
    weather: dict[str, Any] | None,
) -> dict[str, Any]:
    source_payload = live_state if live and live_state else openf1_sessions.get(session_key) or openf1_sessions.get("race") or {}
    rows = {}
    if live and live_state:
        rows = _live_driver_rows(live_state)
    if not rows and source_payload.get("ok"):
        rows = _openf1_driver_rows(source_payload, drivers)

    open_weather = source_payload.get("weather") or {}
    race_control = source_payload.get("race_control") or {}
    weather_payload = weather or (open_weather if not open_weather.get("missing_data") else {})
    confidence = 0.0
    if rows:
        confidence = 0.72 if live and str(live_state.get("source_mode") or "").startswith("live") else 0.62
        if len(rows) >= max(10, len(drivers) - 2):
            confidence += 0.08
        if any(row.get("gap_to_leader") is not None or row.get("interval") is not None for row in rows.values()):
            confidence += 0.06
        if any(row.get("compound") for row in rows.values()):
            confidence += 0.04
    return {
        "available": bool(rows),
        "source": live_state.get("source_mode") or source_payload.get("source") or ("openf1_session_facts" if rows else "unavailable"),
        "confidence": round(clamp01(confidence), 4),
        "coverage_count": len(rows),
        "drivers": rows,
        "weather": weather_payload,
        "race_control": race_control,
        "reason": None if rows else "no_race_or_live_timing_inputs",
    }


def _openf1_driver_rows(payload: dict[str, Any], drivers: list[Driver]) -> dict[str, dict[str, Any]]:
    by_number = {str(driver.number): driver for driver in drivers if driver.number is not None}
    laps = ((payload.get("laps") or {}).get("drivers") or {})
    positions = ((payload.get("positions") or {}).get("drivers") or {})
    intervals = ((payload.get("intervals") or {}).get("drivers") or {})
    stints = ((payload.get("stints") or {}).get("drivers") or {})
    pits = ((payload.get("pits") or {}).get("drivers") or {})
    rows = {}
    for number in set(laps) | set(positions) | set(intervals) | set(stints) | set(pits):
        driver = by_number.get(str(number))
        if not driver:
            continue
        lap = laps.get(str(number)) or {}
        position = positions.get(str(number)) or {}
        interval = intervals.get(str(number)) or {}
        stint = stints.get(str(number)) or {}
        pit = pits.get(str(number)) or {}
        compounds = stint.get("compounds") or lap.get("compounds") or []
        compound = stint.get("compound") or (compounds[-1] if compounds else None)
        tyre_age = _int(stint.get("tyre_age") or stint.get("estimated_tyre_age") or stint.get("avg_stint_laps"))
        rows[driver.id] = {
            "driver_id": driver.id,
            "driver_code": driver.code,
            "position": _int(position.get("position")),
            "gap_to_leader": interval.get("gap_to_leader"),
            "interval": interval.get("interval"),
            "lap": _int(lap.get("lap")),
            "laps": _int(lap.get("laps")),
            "best_lap": lap.get("best_lap"),
            "representative_lap": lap.get("representative_lap"),
            "compound": compound,
            "tyre_age": tyre_age,
            "stints": _int(stint.get("stints")),
            "compound_sequence": stint.get("compound_sequence") or compounds,
            "avg_stint_laps": stint.get("avg_stint_laps"),
            "max_stint_laps": stint.get("max_stint_laps"),
            "final_stint_laps": stint.get("final_stint_laps"),
            "stint_lap_distribution": stint.get("stint_lap_distribution") if isinstance(stint.get("stint_lap_distribution"), dict) else {},
            "tyre_phase": _tyre_phase(compound, tyre_age),
            "pit_stops": _int(pit.get("pit_stops")),
            "source": "openf1_session_facts",
            "confidence": 0.66,
        }
    return rows


def _live_driver_rows(live_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_rows = live_state.get("drivers")
    if isinstance(raw_rows, list):
        return {
            row["driver_id"]: {
                **row,
                "source": row.get("source_mode") or live_state.get("source_mode") or "live_state",
                "confidence": row.get("confidence") if row.get("confidence") is not None else live_state.get("confidence"),
            }
            for row in raw_rows
            if row.get("driver_id")
        }
    live_positions = live_state.get("live_positions") or {}
    if isinstance(live_positions, dict):
        return {
            driver_id: {
                "driver_id": driver_id,
                **row,
                "source": row.get("source_mode") or live_state.get("source_mode") or "live_state",
                "confidence": row.get("confidence") if row.get("confidence") is not None else live_state.get("confidence"),
            }
            for driver_id, row in live_positions.items()
            if isinstance(row, dict)
        }
    return {}


def _practice_feature_fields(practice: dict[str, Any]) -> dict[str, Any]:
    return {
        "practice_pace_score": practice.get("practice_pace_score"),
        "practice_qualifying_evidence_score": practice.get("qualifying_evidence_score"),
        "practice_race_evidence_score": practice.get("race_evidence_score"),
        "practice_long_run_score": practice.get("long_run_score"),
        "practice_sector_score": practice.get("sector_score"),
        "practice_sector_scores": practice.get("sector_scores") or {},
        "practice_pace_stability": practice.get("pace_stability"),
        "practice_lap_time_stddev": practice.get("lap_time_stddev"),
        "practice_fuel_uncertainty": practice.get("fuel_uncertainty"),
        "practice_compounds": practice.get("compounds") or [],
        "practice_best_lap": practice.get("best_lap"),
        "practice_long_run_lap": practice.get("long_run_lap"),
        "practice_representative_lap": practice.get("representative_lap"),
        "practice_lap_count": practice.get("lap_count"),
        "practice_source": practice.get("source"),
        "practice_confidence": practice.get("confidence"),
    }


def _top_driver_rows(rows: dict[str, dict[str, Any]], key: str) -> list[dict[str, Any]]:
    ranked = sorted(
        [
            {"driver_id": driver_id, "value": row.get(key), "sessions": row.get("sessions") or []}
            for driver_id, row in rows.items()
            if row.get(key) is not None
        ],
        key=lambda item: _float(item.get("value")) or 99999,
    )
    return ranked[:5]


def _overall_confidence(practice: dict[str, Any], grid: dict[str, Any], race_inputs: dict[str, Any], session: str, live: bool) -> float:
    if live:
        return round(max(_float(race_inputs.get("confidence")) or 0.0, 0.10), 4)
    if session == "qualifying":
        return round(_float(practice.get("confidence")) or 0.0, 4)
    if session in {"race", "sprint"}:
        return round(clamp01(0.42 * (_float(practice.get("confidence")) or 0.0) + 0.40 * (_float(grid.get("confidence")) or 0.0) + 0.18 * (_float(race_inputs.get("confidence")) or 0.0)), 4)
    return round(_float(practice.get("confidence")) or 0.0, 4)


def _source_mode(practice: dict[str, Any], grid: dict[str, Any], race_inputs: dict[str, Any], live: bool) -> str:
    if live and race_inputs.get("available"):
        return str(race_inputs.get("source") or "live")
    if race_inputs.get("available"):
        return "recent"
    if grid.get("available"):
        return "historical"
    if practice.get("available"):
        return "recent_practice"
    return "estimated"


def _explanations(practice: dict[str, Any], grid: dict[str, Any], race_inputs: dict[str, Any], session: str, live: bool) -> list[str]:
    messages = []
    if practice.get("available"):
        messages.append(f"practice timing available for {practice.get('coverage_count')} drivers")
    else:
        messages.append("practice timing unavailable; historical pace remains dominant")
    if grid.get("available"):
        messages.append("official qualifying/grid rows available")
    elif session in {"race", "sprint"}:
        messages.append("final grid unavailable; race projection uses qualifying/history fallback")
    if live:
        messages.append("live probability movement is capped by timing source confidence")
    if race_inputs.get("available"):
        messages.append(f"race/live timing inputs available for {race_inputs.get('coverage_count')} drivers")
    return messages


def _blend(base: float, evidence: float, weight: float) -> float:
    return clamp01(base * (1.0 - weight) + evidence * max(0.0, min(1.0, weight)))


def _inverse_score(value: Any, fastest: float, slowest: float, *, minimum_spread: float) -> float:
    numeric = _float(value)
    if numeric is None:
        return 0.50
    spread = max(minimum_spread, slowest - fastest)
    return clamp01(1.0 - ((numeric - fastest) / spread))


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _tyre_phase(compound: str | None, tyre_age: int | None) -> str | None:
    if tyre_age is None:
        return None
    compound_key = str(compound or "").upper()
    if compound_key == "SOFT":
        fresh, worn = 8, 20
    elif compound_key == "HARD":
        fresh, worn = 18, 36
    else:
        fresh, worn = 12, 28
    if tyre_age <= fresh:
        return "fresh"
    if tyre_age >= worn:
        return "worn"
    return "stable"


def _avg(values) -> float | None:
    clean = [_float(value) for value in values if _float(value) is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)
