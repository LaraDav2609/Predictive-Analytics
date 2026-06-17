"""Race replay builder that removes future information from backtests."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sports.f1.predictor.features.practice import apply_practice_pace_adjustments
from sports.f1.predictor.features.weekend import build_weekend_evidence
from sports.f1.models.f1 import Constructor, Driver, Race


@dataclass
class RaceReplay:
    season: int
    round: int
    stage: str
    race: Race
    drivers: list[Driver]
    constructors: list[Constructor]
    features: dict[str, Any]
    actual_results: list[dict[str, Any]]
    actual_winner: str | None
    actual_podium: list[str]
    actual_points: list[str]


class RaceReplayBuilder:
    def __init__(self, lookback_races: int = 8):
        self.lookback_races = lookback_races

    def build(self, season: int, races: list[dict[str, Any]], round_num: int, stage: str = "pre_weekend") -> RaceReplay:
        stage = _stage_key(stage)
        ordered = sorted(races, key=lambda race: int(race.get("round") or 0))
        target = next((race for race in ordered if int(race.get("round") or 0) == int(round_num)), None)
        if not target:
            raise ValueError(f"Race round {round_num} was not found in {season}")
        if not target.get("Results"):
            raise ValueError(f"Race round {round_num} has no results to backtest")

        previous = [race for race in ordered if int(race.get("round") or 0) < int(round_num) and race.get("Results")]
        race_model = _race_model(target)
        actual_results = _parse_results(target)
        drivers = _build_drivers(actual_results, previous)
        constructors = _build_constructors(drivers, previous)
        features = self._build_features(season, ordered, previous, drivers, constructors, target=target, stage=stage)

        return RaceReplay(
            season=season,
            round=round_num,
            stage=stage,
            race=race_model,
            drivers=drivers,
            constructors=constructors,
            features=features,
            actual_results=actual_results,
            actual_winner=next((item["driver_id"] for item in actual_results if item.get("position") == 1), None),
            actual_podium=[item["driver_id"] for item in actual_results if isinstance(item.get("position"), int) and item["position"] <= 3],
            actual_points=[item["driver_id"] for item in actual_results if float(item.get("points") or 0) > 0],
        )

    def _build_features(
        self,
        season: int,
        all_races: list[dict[str, Any]],
        previous_races: list[dict[str, Any]],
        drivers: list[Driver],
        constructors: list[Constructor],
        *,
        target: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        driver_results: dict[str, list[dict[str, Any]]] = {}
        constructor_results: dict[str, list[dict[str, Any]]] = {}
        track_results: dict[str, dict[str, list[dict[str, Any]]]] = {}
        teammate_results: dict[tuple[int, str], list[dict[str, Any]]] = {}

        for race in previous_races:
            round_num = int(race.get("round") or 0)
            track_key = _track_key_from_raw_race(race)
            for item in _parse_results(race):
                item = {**item, "round": round_num, "season": season, "track_key": track_key}
                driver_id = item.get("driver_id")
                constructor_name = str(item.get("team") or "").lower()
                if driver_id:
                    driver_results.setdefault(driver_id, []).append(item)
                    track_results.setdefault(track_key, {}).setdefault(driver_id, []).append(item)
                if constructor_name:
                    constructor_results.setdefault(constructor_name, []).append(item)
                    teammate_results.setdefault((round_num, constructor_name), []).append(item)

        teammate_scores = _teammate_scores(teammate_results)
        driver_features = {}
        for driver in drivers:
            results = sorted(driver_results.get(driver.id, []), key=lambda item: item["round"], reverse=True)
            recent = results[: self.lookback_races]
            driver_features[driver.id] = _driver_feature(driver, recent, results, teammate_scores.get(driver.id, 0.0), self.lookback_races)

        openf1_session = _practice_session_payload(target, drivers) if stage in {"practice_available", "post_qualifying", "live", "completed"} else {}
        if openf1_session:
            driver_features = apply_practice_pace_adjustments(
                drivers,
                driver_features,
                openf1_session,
                session_stage="practice",
            )

        qualifying_rows = []
        if stage in {"post_qualifying", "live", "completed"}:
            qualifying_rows = _qualifying_rows(target)
            _apply_qualifying_features(driver_features, qualifying_rows, drivers)

        race_input_session = _race_input_session_payload(target, drivers) if stage in {"live", "completed"} else {}
        weekend_evidence = (
            build_weekend_evidence(
                race=_race_model(target),
                drivers=drivers,
                profile={"ok": True, "qualifying": qualifying_rows, "results": []},
                openf1_sessions={
                    **({"fp1": openf1_session} if openf1_session else {}),
                    **({"race": race_input_session} if race_input_session else {}),
                },
                live_state={},
                session="race",
                live=stage == "live",
            )
            if race_input_session
            else {}
        )

        constructor_features = {}
        for constructor in constructors:
            key = constructor.name.lower()
            results = sorted(constructor_results.get(key, []), key=lambda item: item["round"], reverse=True)
            constructor_features[key] = _constructor_feature(constructor, results[: self.lookback_races * 2], self.lookback_races)

        return {
            "season": season,
            "lookback_races": self.lookback_races,
            "completed_races": len(previous_races),
            "total_races": len(all_races),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "drivers": driver_features,
            "constructors": constructor_features,
            "track_history": _track_history_features(track_results),
            "openf1_session": openf1_session,
            "weekend_evidence": weekend_evidence,
            "replay_profile": {
                "qualifying": qualifying_rows,
                "practice_available": bool(openf1_session),
                "race_inputs_available": bool(race_input_session),
            },
            "source_coverage": {
                "current_season_races": len(previous_races),
                "historical_races": 0,
                "weather_races": 0,
                "practice_sessions": 1 if openf1_session else 0,
                "qualifying_sessions": 1 if qualifying_rows else 0,
                "race_input_sessions": 1 if race_input_session else 0,
            },
            "backtest": {
                "replay_mode": True,
                "stage": stage,
                "future_rounds_removed": len([race for race in all_races if int(race.get("round") or 0) >= len(previous_races) + 1]),
                "target_race_results_removed": True,
                "target_qualifying_used": bool(qualifying_rows),
                "target_practice_used": bool(openf1_session),
                "target_race_inputs_used": bool(race_input_session),
            },
        }


def _race_model(raw: dict[str, Any]) -> Race:
    circuit = raw.get("Circuit") or {}
    location = circuit.get("Location") or {}
    date_str = raw.get("date") or f"{raw.get('season', 2000)}-01-01"
    time_str = str(raw.get("time") or "14:00:00Z").rstrip("Z")
    try:
        date = datetime.fromisoformat(f"{date_str}T{time_str}").replace(tzinfo=timezone.utc)
    except ValueError:
        date = datetime.now(timezone.utc)
    return Race(
        round=int(raw.get("round") or 0),
        name=raw.get("raceName") or "",
        circuit=circuit.get("circuitName") or "",
        country=location.get("country") or "",
        date=date,
        circuit_id=circuit.get("circuitId"),
        locality=location.get("locality"),
        latitude=_safe_float(location.get("lat")),
        longitude=_safe_float(location.get("long")),
        has_sprint=bool(raw.get("SprintResults") or raw.get("Sprint") or raw.get("SprintQualifying")),
        status="COMPLETED",
    )


def _parse_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in raw.get("Results") or []:
        driver = item.get("Driver") or {}
        constructor = item.get("Constructor") or {}
        position_raw = item.get("position")
        grid_raw = item.get("grid")
        rows.append({
            "position": int(position_raw) if str(position_raw).isdigit() else None,
            "driver_id": driver.get("driverId") or "",
            "driver_code": driver.get("code") or "",
            "driver_number": _safe_int(driver.get("permanentNumber")),
            "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
            "first_name": driver.get("givenName") or "",
            "last_name": driver.get("familyName") or "",
            "nationality": driver.get("nationality") or "",
            "team": constructor.get("name") or "",
            "constructor_id": constructor.get("constructorId") or _slug(constructor.get("name")),
            "constructor_nationality": constructor.get("nationality") or "",
            "grid": int(grid_raw) if str(grid_raw).lstrip("-").isdigit() else None,
            "points": float(item.get("points") or 0),
            "status": item.get("status") or "Unknown",
        })
    return rows


def _stage_key(stage: str | None) -> str:
    value = str(stage or "pre_weekend").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pre": "pre_weekend",
        "pre_race": "pre_weekend",
        "practice": "practice_available",
        "post_practice": "practice_available",
        "fp": "practice_available",
        "post_fp": "practice_available",
        "quali": "post_qualifying",
        "qualifying": "post_qualifying",
        "post_quali": "post_qualifying",
        "post_qualifying": "post_qualifying",
        "race_live": "live",
        "replay": "live",
        "complete": "completed",
    }
    value = aliases.get(value, value)
    if value not in {"pre_weekend", "practice_available", "post_qualifying", "live", "completed"}:
        return "pre_weekend"
    return value


def _qualifying_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in raw.get("QualifyingResults") or raw.get("qualifying") or []:
        driver = item.get("Driver") or {}
        constructor = item.get("Constructor") or {}
        driver_id = item.get("driver_id") or driver.get("driverId")
        position = _safe_int(item.get("position"))
        if not driver_id or not position:
            continue
        rows.append({
            "driver_id": driver_id,
            "driver_code": item.get("driver_code") or driver.get("code") or str(driver_id)[:3].upper(),
            "driver_name": item.get("driver_name") or f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
            "team": item.get("team") or constructor.get("name"),
            "position": position,
            "grid": position,
            "q1": item.get("q1") or item.get("Q1"),
            "q2": item.get("q2") or item.get("Q2"),
            "q3": item.get("q3") or item.get("Q3"),
            "source": "target_qualifying_results",
        })
    if rows:
        return sorted(rows, key=lambda row: row["position"])

    for item in _parse_results(raw):
        grid = item.get("grid")
        if isinstance(grid, int) and grid > 0:
            rows.append({
                "driver_id": item.get("driver_id"),
                "driver_code": item.get("driver_code"),
                "driver_name": item.get("driver_name"),
                "team": item.get("team"),
                "position": grid,
                "grid": grid,
                "source": "target_race_grid_only",
            })
    return sorted(rows, key=lambda row: row["position"])


def _apply_qualifying_features(
    driver_features: dict[str, dict[str, Any]],
    qualifying_rows: list[dict[str, Any]],
    drivers: list[Driver],
) -> None:
    if not qualifying_rows:
        return
    field_size = max(len(drivers), len(qualifying_rows), 1)
    for row in qualifying_rows:
        driver_id = row.get("driver_id")
        position = _safe_int(row.get("position"))
        if not driver_id or not position:
            continue
        feature = driver_features.setdefault(driver_id, {})
        grid_score = max(0.02, min(1.0, (field_size + 1 - position) / field_size))
        old_quali = _safe_float(feature.get("qualifying_pace_score")) or _safe_float(feature.get("form_score")) or 0.50
        old_race = _safe_float(feature.get("race_pace_score")) or _safe_float(feature.get("form_score")) or 0.50
        feature["qualifying_position"] = position
        feature["grid_position"] = row.get("grid") or position
        feature["grid_score"] = round(grid_score, 4)
        feature["grid_confidence"] = 0.86 if row.get("source") == "target_qualifying_results" else 0.72
        feature["qualifying_source"] = row.get("source") or "target_grid"
        feature["q1"] = row.get("q1")
        feature["q2"] = row.get("q2")
        feature["q3"] = row.get("q3")
        feature["qualifying_pace_score"] = round(old_quali * 0.40 + grid_score * 0.60, 4)
        feature["race_pace_score"] = round(old_race * 0.82 + grid_score * 0.18, 4)


def _practice_session_payload(raw: dict[str, Any], drivers: list[Driver]) -> dict[str, Any]:
    rows = _practice_rows(raw)
    if not rows:
        return {}
    driver_lookup = {driver.id: driver for driver in drivers}
    rows_by_driver: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        driver_id = row.get("driver_id")
        driver = driver_lookup.get(str(driver_id))
        if not driver or driver.number is None:
            continue
        representative = _lap_seconds(row.get("representative_lap") or row.get("median_lap") or row.get("best_lap"))
        best = _lap_seconds(row.get("best_lap") or row.get("representative_lap"))
        if not representative and not best:
            continue
        rows_by_driver.setdefault(str(driver.number), []).append({
            "driver_number": driver.number,
            "driver_code": driver.code,
            "laps": _safe_int(row.get("laps")) or _safe_int(row.get("lap_count")) or 0,
            "best_lap": best,
            "median_lap": representative,
            "representative_lap": representative or best,
            "long_run_lap": _lap_seconds(row.get("long_run_lap")),
            "best_sector_1": _lap_seconds(row.get("best_sector_1")),
            "best_sector_2": _lap_seconds(row.get("best_sector_2")),
            "best_sector_3": _lap_seconds(row.get("best_sector_3")),
            "representative_sector_1": _lap_seconds(row.get("representative_sector_1")),
            "representative_sector_2": _lap_seconds(row.get("representative_sector_2")),
            "representative_sector_3": _lap_seconds(row.get("representative_sector_3")),
            "sector_coverage": _safe_float(row.get("sector_coverage")),
            "track_evolution_delta": _safe_float(row.get("track_evolution_delta")),
            "lap_time_stddev": _safe_float(row.get("lap_time_stddev")),
            "pace_stability": _safe_float(row.get("pace_stability")),
            "telemetry_quality": _safe_float(row.get("telemetry_quality")),
            "lap_distribution": row.get("lap_distribution") if isinstance(row.get("lap_distribution"), dict) else {},
            "raw_laps": _safe_int(row.get("raw_laps")),
            "usable_laps": _safe_int(row.get("usable_laps")),
            "compounds": row.get("compounds") or [],
        })
    laps_by_number = {
        number: _aggregate_practice_driver_rows(number, driver_rows)
        for number, driver_rows in rows_by_driver.items()
    }
    if not laps_by_number:
        return {}
    return {
        "ok": True,
        "source": "backtest_target_practice_rows",
        "session": "practice",
        "laps": {"drivers": laps_by_number, "missing_data": False},
        "raw_counts": {"laps": sum(1 for _ in laps_by_number)},
    }


def _race_input_session_payload(raw: dict[str, Any], drivers: list[Driver]) -> dict[str, Any]:
    race_inputs = raw.get("RaceInputs") or raw.get("race_inputs") or {}
    driver_rows = race_inputs.get("drivers") if isinstance(race_inputs, dict) else {}
    if not isinstance(driver_rows, dict) or not driver_rows:
        return {}
    by_number = {
        str(driver.number): driver
        for driver in drivers
        if driver.number is not None
    }
    lap_rows = {}
    position_rows = {}
    interval_rows = {}
    stint_rows = {}
    pit_rows = {}
    for row in driver_rows.values():
        if not isinstance(row, dict):
            continue
        number = row.get("driver_number")
        if number is None:
            driver_id = str(row.get("driver_id") or "")
            driver = next((item for item in drivers if item.id == driver_id), None)
            number = driver.number if driver else None
        if number is None or str(number) not in by_number:
            continue
        number_key = str(number)
        lap_rows[number_key] = {
            "driver_number": number,
            "lap": _safe_int(row.get("lap")),
            "laps": _safe_int(row.get("laps")),
            "best_lap": _lap_seconds(row.get("best_lap")),
            "representative_lap": _lap_seconds(row.get("representative_lap")),
        }
        position_rows[number_key] = {
            "driver_number": number,
            "position": _safe_int(row.get("position")),
        }
        interval_rows[number_key] = {
            "driver_number": number,
            "gap_to_leader": row.get("gap_to_leader"),
            "interval": row.get("interval"),
        }
        sequence = row.get("compound_sequence") or ([row.get("compound")] if row.get("compound") else [])
        stint_rows[number_key] = {
            "driver_number": number,
            "compound": row.get("compound"),
            "compounds": sorted({str(item).upper() for item in sequence if item}),
            "compound_sequence": [str(item).upper() for item in sequence if item],
            "tyre_age": _safe_int(row.get("tyre_age") or row.get("estimated_tyre_age")),
            "estimated_tyre_age": _safe_int(row.get("estimated_tyre_age") or row.get("tyre_age")),
            "stints": _safe_int(row.get("stints")),
            "avg_stint_laps": _safe_float(row.get("avg_stint_laps")),
            "max_stint_laps": _safe_int(row.get("max_stint_laps")),
            "final_stint_laps": _safe_int(row.get("final_stint_laps")),
            "stint_lap_distribution": row.get("stint_lap_distribution") if isinstance(row.get("stint_lap_distribution"), dict) else {},
        }
        pit_rows[number_key] = {
            "driver_number": number,
            "pit_stops": _safe_int(row.get("pit_stops")),
        }
    if not any([lap_rows, position_rows, interval_rows, stint_rows, pit_rows]):
        return {}
    return {
        "ok": True,
        "source": race_inputs.get("source") or "backtest_target_race_inputs",
        "session": "race",
        "laps": {"drivers": lap_rows, "missing_data": not bool(lap_rows)},
        "positions": {"drivers": position_rows, "missing_data": not bool(position_rows)},
        "intervals": {"drivers": interval_rows, "missing_data": not bool(interval_rows)},
        "stints": {"drivers": stint_rows, "missing_data": not bool(stint_rows)},
        "pits": {"drivers": pit_rows, "missing_data": not bool(pit_rows)},
        "weather": race_inputs.get("weather") or {"missing_data": True},
        "race_control": race_inputs.get("race_control") or {"missing_data": True},
        "raw_counts": {
            "laps": len(lap_rows),
            "positions": len(position_rows),
            "intervals": len(interval_rows),
            "stints": len(stint_rows),
            "pits": len(pit_rows),
        },
    }


def _aggregate_practice_driver_rows(number: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    best_values = [_safe_float(row.get("best_lap")) for row in rows if _safe_float(row.get("best_lap")) is not None]
    representative_values = [
        _safe_float(row.get("representative_lap") or row.get("median_lap"))
        for row in rows
        if _safe_float(row.get("representative_lap") or row.get("median_lap")) is not None
    ]
    long_values = [_safe_float(row.get("long_run_lap")) for row in rows if _safe_float(row.get("long_run_lap")) is not None]
    stability_values = [_safe_float(row.get("pace_stability")) for row in rows if _safe_float(row.get("pace_stability")) is not None]
    stddev_values = [_safe_float(row.get("lap_time_stddev")) for row in rows if _safe_float(row.get("lap_time_stddev")) is not None]
    quality_values = [_safe_float(row.get("telemetry_quality")) for row in rows if _safe_float(row.get("telemetry_quality")) is not None]
    distribution_values = _merge_lap_distributions([row.get("lap_distribution") or {} for row in rows])
    compounds = sorted({compound for row in rows for compound in (row.get("compounds") or []) if compound})
    payload = {
        "driver_number": first.get("driver_number") or _safe_int(number),
        "driver_code": first.get("driver_code"),
        "laps": sum(_safe_int(row.get("laps")) or 0 for row in rows),
        "raw_laps": sum(_safe_int(row.get("raw_laps")) or _safe_int(row.get("laps")) or 0 for row in rows),
        "usable_laps": sum(_safe_int(row.get("usable_laps")) or _safe_int(row.get("laps")) or 0 for row in rows),
        "best_lap": min(best_values) if best_values else None,
        "median_lap": statistics.median(representative_values) if representative_values else (min(best_values) if best_values else None),
        "representative_lap": statistics.median(representative_values) if representative_values else (min(best_values) if best_values else None),
        "long_run_lap": statistics.median(long_values) if long_values else None,
        "lap_time_stddev": statistics.median(stddev_values) if stddev_values else None,
        "pace_stability": statistics.median(stability_values) if stability_values else None,
        "telemetry_quality": statistics.median(quality_values) if quality_values else None,
        "lap_distribution": distribution_values,
        "compounds": compounds,
    }
    for index in (1, 2, 3):
        best_sector_values = [
            _safe_float(row.get(f"best_sector_{index}") or row.get(f"representative_sector_{index}"))
            for row in rows
            if _safe_float(row.get(f"best_sector_{index}") or row.get(f"representative_sector_{index}")) is not None
        ]
        representative_sector_values = [
            _safe_float(row.get(f"representative_sector_{index}") or row.get(f"best_sector_{index}"))
            for row in rows
            if _safe_float(row.get(f"representative_sector_{index}") or row.get(f"best_sector_{index}")) is not None
        ]
        payload[f"best_sector_{index}"] = min(best_sector_values) if best_sector_values else None
        payload[f"representative_sector_{index}"] = (
            statistics.median(representative_sector_values)
            if representative_sector_values else payload[f"best_sector_{index}"]
        )
    payload["sector_coverage"] = round(
        sum(1 for index in (1, 2, 3) if payload.get(f"representative_sector_{index}") is not None) / 3.0,
        4,
    )
    return payload


def _merge_lap_distributions(distributions: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [item for item in distributions if isinstance(item, dict) and item.get("sample_size")]
    if not clean:
        return {}
    weighted_keys = ["p10", "p25", "median", "p75", "p90", "spread_p90_p10"]
    total = sum(_safe_int(item.get("sample_size")) or 0 for item in clean) or len(clean)
    merged = {"sample_size": total}
    for key in weighted_keys:
        pairs = [
            (float(item.get(key)), _safe_int(item.get("sample_size")) or 1)
            for item in clean
            if _safe_float(item.get(key)) is not None
        ]
        if pairs:
            weight_total = sum(weight for _, weight in pairs) or 1
            merged[key] = round(sum(value * weight for value, weight in pairs) / weight_total, 3)
    best_values = [_safe_float(item.get("best")) for item in clean if _safe_float(item.get("best")) is not None]
    if best_values:
        merged["best"] = round(min(best_values), 3)
    return merged


def _practice_rows(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(raw.get("PracticeResults"), list):
        return list(raw.get("PracticeResults") or [])
    rows = []
    for session in raw.get("PracticeSessions") or raw.get("Practice") or []:
        rows.extend(session.get("Results") or session.get("results") or [])
    return rows


def _lap_seconds(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        if ":" not in text:
            return float(text)
        minutes, seconds = text.split(":", 1)
        return int(minutes) * 60.0 + float(seconds)
    except (TypeError, ValueError):
        return None


def _build_drivers(actual_results: list[dict[str, Any]], previous_races: list[dict[str, Any]]) -> list[Driver]:
    prior_points: dict[str, float] = {}
    prior_wins: dict[str, int] = {}
    for race in previous_races:
        for item in _parse_results(race):
            driver_id = item["driver_id"]
            prior_points[driver_id] = prior_points.get(driver_id, 0.0) + float(item.get("points") or 0.0)
            if item.get("position") == 1:
                prior_wins[driver_id] = prior_wins.get(driver_id, 0) + 1

    ordered_ids = sorted({item["driver_id"] for item in actual_results}, key=lambda driver_id: prior_points.get(driver_id, 0.0), reverse=True)
    positions = {driver_id: index + 1 for index, driver_id in enumerate(ordered_ids)}
    drivers = []
    for item in actual_results:
        driver_id = item["driver_id"]
        drivers.append(Driver(
            id=driver_id,
            number=item.get("driver_number"),
            code=item.get("driver_code") or driver_id[:3].upper(),
            first_name=item.get("first_name") or item.get("driver_name", "").split(" ")[0],
            last_name=item.get("last_name") or item.get("driver_name", "").split(" ")[-1],
            nationality=item.get("nationality") or "",
            team=item.get("team") or "",
            points=round(prior_points.get(driver_id, 0.0), 2),
            wins=prior_wins.get(driver_id, 0),
            position=positions.get(driver_id),
        ))
    return sorted(drivers, key=lambda driver: driver.position or 99)


def _build_constructors(drivers: list[Driver], previous_races: list[dict[str, Any]]) -> list[Constructor]:
    points: dict[str, float] = {}
    wins: dict[str, int] = {}
    meta: dict[str, dict[str, str]] = {}
    for race in previous_races:
        for item in _parse_results(race):
            key = str(item.get("team") or "").lower()
            points[key] = points.get(key, 0.0) + float(item.get("points") or 0.0)
            if item.get("position") == 1:
                wins[key] = wins.get(key, 0) + 1
            meta[key] = {"id": item.get("constructor_id") or _slug(key), "name": item.get("team") or key, "nationality": item.get("constructor_nationality") or ""}
    for driver in drivers:
        key = driver.team.lower()
        meta.setdefault(key, {"id": _slug(driver.team), "name": driver.team, "nationality": ""})
        points.setdefault(key, 0.0)
        wins.setdefault(key, 0)
    ordered = sorted(meta, key=lambda key: points.get(key, 0.0), reverse=True)
    return [
        Constructor(
            id=meta[key]["id"],
            name=meta[key]["name"],
            nationality=meta[key]["nationality"],
            points=round(points.get(key, 0.0), 2),
            wins=wins.get(key, 0),
            position=index + 1,
        )
        for index, key in enumerate(ordered)
    ]


def _driver_feature(driver: Driver, recent: list[dict[str, Any]], all_results: list[dict[str, Any]], teammate_delta: float, lookback: int) -> dict[str, Any]:
    classified = [item for item in recent if isinstance(item.get("position"), int)]
    avg_finish = _avg([float(item["position"]) for item in classified]) if classified else None
    grids = [item["grid"] for item in recent if isinstance(item.get("grid"), int) and item.get("grid", 0) > 0]
    grid_deltas = [item["grid"] - item["position"] for item in recent if isinstance(item.get("grid"), int) and item.get("grid", 0) > 0 and isinstance(item.get("position"), int)]
    avg_grid = _avg([float(grid) for grid in grids]) if grids else None
    avg_grid_delta = _avg([float(delta) for delta in grid_deltas])
    recent_points = sum(float(item.get("points") or 0.0) for item in recent)
    podiums = sum(1 for item in recent if isinstance(item.get("position"), int) and item["position"] <= 3)
    wins = sum(1 for item in recent if item.get("position") == 1)
    dnfs = sum(1 for item in recent if not _is_finished_status(item.get("status")))
    mechanical_dnfs = sum(1 for item in recent if _is_mechanical_status(item.get("status")))
    incident_dnfs = sum(1 for item in recent if _is_incident_status(item.get("status")))
    finish_score = 0.45 if avg_finish is None else max(0.0, min(1.0, (21 - avg_finish) / 20))
    points_score = min(1.0, recent_points / max(1.0, lookback * 25.0))
    podium_score = min(1.0, podiums / max(1.0, min(lookback, 4)))
    reliability_score = 1.0 - min(0.7, dnfs / max(1, len(recent)) if recent else 0.15)
    qualifying_pace = 0.48 if avg_grid is None else max(0.02, min(1.0, (22 - avg_grid) / 21))
    race_pace = max(0.02, min(1.0, finish_score + max(-0.12, min(0.12, avg_grid_delta / 50.0))))
    teammate_score = max(0.02, min(1.0, 0.50 + teammate_delta / 12.0))
    trend_score = _trend_score(recent)
    standings_context = min(1.0, float(driver.points or 0.0) / max(1.0, lookback * 25.0))
    form_score = (
        0.34 * finish_score
        + 0.22 * points_score
        + 0.15 * podium_score
        + 0.04 * standings_context
        + 0.13 * qualifying_pace
        + 0.12 * trend_score
    )
    return {
        "starts": len(all_results),
        "current_season_starts": len(all_results),
        "recent_starts": len(recent),
        "recent_points": round(recent_points, 2),
        "recent_avg_finish": round(avg_finish, 2) if avg_finish is not None else None,
        "recent_avg_grid": round(avg_grid, 2) if avg_grid is not None else None,
        "recent_grid_delta": round(avg_grid_delta, 2),
        "recent_wins": wins,
        "recent_podiums": podiums,
        "recent_dnfs": dnfs,
        "mechanical_dnfs": mechanical_dnfs,
        "incident_dnfs": incident_dnfs,
        "form_score": round(form_score, 4),
        "reliability_score": round(reliability_score, 4),
        "qualifying_pace_score": round(qualifying_pace, 4),
        "race_pace_score": round(race_pace, 4),
        "teammate_score": round(teammate_score, 4),
        "trend_score": round(trend_score, 4),
        "recent_summary": f"{len(recent)} pre-race starts" if recent else "No pre-race starts available",
    }


def _constructor_feature(constructor: Constructor, recent: list[dict[str, Any]], lookback: int) -> dict[str, Any]:
    positions = [item["position"] for item in recent if isinstance(item.get("position"), int)]
    avg_finish = _avg([float(position) for position in positions]) if positions else None
    recent_points = sum(float(item.get("points") or 0.0) for item in recent)
    standings_score = min(1.0, float(constructor.points or 0.0) / max(1.0, lookback * 43.0))
    finish_score = 0.45 if avg_finish is None else max(0.0, min(1.0, (21 - avg_finish) / 20))
    points_score = min(1.0, recent_points / max(1.0, lookback * 43.0))
    dnfs = sum(1 for item in recent if not _is_finished_status(item.get("status")))
    return {
        "team_score": round(0.24 * standings_score + 0.34 * points_score + 0.42 * finish_score, 4),
        "recent_points": round(recent_points, 2),
        "recent_avg_finish": round(avg_finish, 2) if avg_finish is not None else None,
        "recent_starts": len(recent),
        "reliability_score": round(1.0 - min(0.55, dnfs / max(1, len(recent)) if recent else 0.10), 4),
    }


def _teammate_scores(teammate_results: dict[tuple[int, str], list[dict[str, Any]]]) -> dict[str, float]:
    deltas: dict[str, list[float]] = {}
    for entries in teammate_results.values():
        classified = [entry for entry in entries if isinstance(entry.get("position"), int)]
        if len(classified) < 2:
            continue
        for entry in classified:
            mates = [other for other in classified if other.get("driver_id") != entry.get("driver_id")]
            if mates:
                best_mate = min(mates, key=lambda item: item["position"])
                deltas.setdefault(entry["driver_id"], []).append(float(best_mate["position"] - entry["position"]))
    return {driver_id: _avg(values) for driver_id, values in deltas.items()}


def _track_history_features(track_results: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, Any]]:
    rows = {}
    for track_key, by_driver in track_results.items():
        driver_rows = {}
        for driver_id, results in by_driver.items():
            classified = [item for item in results if isinstance(item.get("position"), int)]
            if not classified:
                continue
            avg_finish = _avg([float(item["position"]) for item in classified])
            podiums = sum(1 for item in classified if item.get("position", 99) <= 3)
            driver_rows[driver_id] = {
                "starts": len(results),
                "avg_finish": round(avg_finish, 2),
                "points": round(sum(float(item.get("points") or 0.0) for item in results), 2),
                "wins": sum(1 for item in classified if item.get("position") == 1),
                "podiums": podiums,
                "track_score": round(max(0.02, min(1.0, (22 - avg_finish) / 21 + min(0.10, podiums * 0.025))), 4),
            }
        rows[track_key] = {"drivers": driver_rows, "source": "backtest_pre_race_history", "missing_data": not bool(driver_rows)}
    return rows


def _track_key_from_raw_race(race: dict[str, Any]) -> str:
    text = _slug(f"{race.get('raceName', '')} {((race.get('Circuit') or {}).get('circuitName') or '')} {(((race.get('Circuit') or {}).get('Location') or {}).get('country') or '')}")
    aliases = {
        "albertpark": ["albertpark", "australian"],
        "shanghai": ["shanghai", "chinese"],
        "suzuka": ["suzuka", "japanese"],
        "miami": ["miami"],
        "bahrain": ["bahrain", "sakhir"],
        "jeddah": ["jeddah", "saudi"],
        "monaco": ["monaco"],
        "monza": ["monza"],
        "spa": ["spa", "belgian"],
        "silverstone": ["silverstone", "british"],
        "baku": ["baku", "azerbaijan"],
        "marinabay": ["marinabay", "singapore"],
        "cota": ["circuitoftheamericas", "austin"],
        "yasmarina": ["yasmarina", "abudhabi"],
    }
    for key, values in aliases.items():
        if any(value in text for value in values):
            return key
    return text[:32] or "default"


def _is_finished_status(status: str | None) -> bool:
    normalized = str(status or "").lower()
    return normalized == "finished" or normalized.startswith("+") or "lap" in normalized


def _is_mechanical_status(status: str | None) -> bool:
    normalized = str(status or "").lower()
    return any(token in normalized for token in ["engine", "gearbox", "hydraulics", "power unit", "electrical", "brake", "suspension", "oil", "water", "fuel", "wheel", "puncture", "tyre", "tire"])


def _is_incident_status(status: str | None) -> bool:
    normalized = str(status or "").lower()
    return any(token in normalized for token in ["accident", "collision", "spun", "damage", "crash", "withdrew"])


def _trend_score(results: list[dict[str, Any]]) -> float:
    ordered = sorted(results, key=lambda item: item.get("round", 0))
    classified = [item for item in ordered if isinstance(item.get("position"), int)]
    if len(classified) < 2:
        return 0.50
    midpoint = max(1, len(classified) // 2)
    early = classified[:midpoint]
    late = classified[midpoint:]
    return max(0.02, min(1.0, 0.50 + (_avg([float(item["position"]) for item in early]) - _avg([float(item["position"]) for item in late])) / 18.0))


def _safe_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _slug(value: str | None) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())
