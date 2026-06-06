"""Constructor-level car model analysis.

The public predictor mostly works at driver level. This module keeps car
strength separate from driver skill by producing a source-labeled constructor
profile that can be used by race pages, constructor pages, and bounded
probability modifiers.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from models.f1 import Constructor, Driver, Race
from f1_predictor.scoring.normalization import clamp01, num


CAR_MODEL_VERSION = "f1-car-model-v1"


def build_car_model_analysis(
    *,
    race: Race | None,
    drivers: list[Driver],
    constructors: list[Constructor],
    features: dict | None = None,
    session: str = "race",
    track: dict | None = None,
    weather: dict | None = None,
    tires: dict | None = None,
    openf1_session: dict | None = None,
    sentiment_impact: dict | None = None,
    market_signals: dict | None = None,
) -> dict[str, Any]:
    features = features or {}
    session_key = _session_key(session)
    openf1_session = openf1_session or features.get("openf1_session") or {}
    driver_features = features.get("drivers") or {}
    constructor_features = features.get("constructors") or {}
    track = track or {}
    weather = weather or {}
    tires = tires or {}
    market_signals = market_signals or {}

    drivers_by_team: dict[str, list[Driver]] = {}
    for driver in drivers:
        drivers_by_team.setdefault(_team_key(driver.team), []).append(driver)

    lap_pace = _lap_pace_by_driver(openf1_session, drivers)
    car_data = _car_data_by_driver(openf1_session, drivers)
    pit_data = _pits_by_team(openf1_session, drivers)
    stint_data = _stints_by_team(openf1_session, drivers)
    position_data = _positions_by_team(openf1_session, drivers)
    interval_data = _intervals_by_team(openf1_session, drivers)
    location_trace = _location_trace_available(openf1_session)
    race_control_chaos = _race_control_chaos(openf1_session)
    field_pace = [value for value in lap_pace.values() if value > 0]
    fastest = min(field_pace) if field_pace else None
    slowest = max(field_pace) if len(field_pace) > 1 else None
    field_pit = [value.get("avg_pit_duration") for value in pit_data.values() if value.get("avg_pit_duration")]
    best_pit = min(field_pit) if field_pit else None
    slow_pit = max(field_pit) if len(field_pit) > 1 else None
    high_speed_need = 0.68 if track.get("high_speed") else 0.42
    quali_weight = float(track.get("qualifying_importance") or 0.58)
    tire_stress = float(track.get("tire_stress") or tires.get("degradation_rate") or 0.50)
    chaos = float(weather.get("chaos_score") or 0.0)

    constructor_rows: dict[str, dict[str, Any]] = {}
    driver_rows: dict[str, dict[str, Any]] = {}
    sources = set()

    for constructor in constructors:
        team_key = _team_key(constructor.name)
        team_drivers = drivers_by_team.get(team_key) or []
        c_feature = constructor_features.get(team_key) or {}
        standings_score = _constructor_standings_score(constructor, constructors)
        base_team_score = clamp01(num(c_feature.get("team_score"), standings_score))
        recent_points_score = clamp01(num(c_feature.get("recent_points"), 0.0) / 344.0)
        reliability = clamp01(num(c_feature.get("reliability_score"), _avg_driver_feature(team_drivers, driver_features, "reliability_score", 0.72)))

        team_laps = [lap_pace.get(driver.id) for driver in team_drivers if lap_pace.get(driver.id)]
        pace_score = _pace_score(team_laps, fastest, slowest, base_team_score)
        quali_driver_score = _avg_driver_feature(team_drivers, driver_features, "qualifying_pace_score", base_team_score)
        race_driver_score = _avg_driver_feature(team_drivers, driver_features, "race_pace_score", base_team_score)
        qualifying_pace = clamp01(0.58 * pace_score + 0.42 * quali_driver_score if session_key == "qualifying" else 0.42 * pace_score + 0.58 * quali_driver_score)
        race_pace = clamp01(0.58 * pace_score + 0.42 * race_driver_score if session_key == "race" else 0.36 * pace_score + 0.64 * race_driver_score)
        long_run = _long_run_score(team_drivers, openf1_session, pace_score, race_driver_score)
        teammate_norm = _teammate_normalized_score(team_drivers, lap_pace)
        stint = stint_data.get(team_key) or {}
        tire_behavior = _tire_behavior_score(stint, tire_stress, chaos)
        degradation_control = clamp01(0.68 * tire_behavior + 0.32 * reliability)
        pit = pit_data.get(team_key) or {}
        pit_performance = _pit_score(pit.get("avg_pit_duration"), best_pit, slow_pit)
        position = position_data.get(team_key) or {}
        interval = interval_data.get(team_key) or {}
        race_execution = _race_execution_score(position, interval, reliability)
        strategy_ops = clamp01(0.30 * pit_performance + 0.27 * tire_behavior + 0.18 * reliability + 0.15 * race_execution + 0.10 * base_team_score - race_control_chaos * 0.04)
        car_team_data = [car_data.get(driver.id) for driver in team_drivers if car_data.get(driver.id)]
        straight_line = _straight_line_score(car_team_data, high_speed_need, base_team_score)
        low_speed_fit = clamp01((1.0 - high_speed_need) * qualifying_pace + high_speed_need * 0.46 + 0.18 * reliability)
        high_speed_fit = clamp01(high_speed_need * straight_line + (1.0 - high_speed_need) * race_pace)
        braking_traction = clamp01(0.50 * low_speed_fit + 0.28 * qualifying_pace + 0.22 * reliability)
        track_fit = clamp01(
            quali_weight * low_speed_fit
            + (1.0 - quali_weight) * high_speed_fit
            + (0.08 if track.get("street_circuit") else 0.0) * braking_traction
        )
        upgrade_regulation = _upgrade_regulation_score(constructor, sentiment_impact)
        market_consensus = _market_team_score(constructor, market_signals)
        overall = clamp01(
            0.24 * race_pace
            + 0.18 * qualifying_pace
            + 0.13 * long_run
            + 0.12 * track_fit
            + 0.10 * tire_behavior
            + 0.09 * reliability
            + 0.07 * strategy_ops
            + 0.04 * straight_line
            + 0.03 * upgrade_regulation
        )
        confidence, coverage, missing = _confidence(
            has_laps=bool(team_laps),
            has_car_data=bool(car_team_data),
            has_positions=bool(position),
            has_intervals=bool(interval),
            has_stints=bool(stint),
            has_pits=bool(pit),
            has_race_control=bool(race_control_chaos),
            has_location_trace=location_trace,
            has_track=bool(track) and not track.get("missing_data"),
            has_weather=bool(weather) and not weather.get("missing_data"),
            has_sentiment=_has_constructor_sentiment(constructor, sentiment_impact),
            has_market=bool(market_consensus),
        )
        source_breakdown = {
            "historical_results": "available" if c_feature or team_drivers else "fallback",
            "openf1_laps": "available" if team_laps else "missing",
            "openf1_car_data": "available" if car_team_data else "missing",
            "openf1_positions": "available" if position else "missing",
            "openf1_intervals": "available" if interval else "missing",
            "openf1_stints": "available" if stint else "missing",
            "openf1_pits": "available" if pit else "missing",
            "openf1_race_control": "available" if race_control_chaos else "missing",
            "openf1_location_traces": "available" if location_trace else "missing",
            "track_traits": track.get("source") or "fallback",
            "weather": weather.get("source") or "fallback",
            "sentiment": "available" if _has_constructor_sentiment(constructor, sentiment_impact) else "neutral",
            "market": "available" if market_consensus else "neutral",
        }
        sources.update(value for value in source_breakdown.values() if value and value != "missing")

        row = {
            "constructor_id": constructor.id,
            "constructor_name": constructor.name,
            "drivers": [
                {
                    "driver_id": driver.id,
                    "driver_code": driver.code,
                    "driver_name": f"{driver.first_name} {driver.last_name}".strip(),
                    "lap_pace_seconds": lap_pace.get(driver.id),
                    "car_data": car_data.get(driver.id) or {},
                }
                for driver in team_drivers
            ],
            "session": session_key,
            "scores": {
                "overall_car_score": round(overall, 4),
                "car_pace": round(pace_score, 4),
                "qualifying_pace": round(qualifying_pace, 4),
                "race_pace": round(race_pace, 4),
                "long_run_pace": round(long_run, 4),
                "teammate_normalized_pace": round(teammate_norm, 4),
                "tire_behavior": round(tire_behavior, 4),
                "degradation_control": round(degradation_control, 4),
                "drs_straight_line_strength": round(straight_line, 4),
                "low_speed_track_fit": round(low_speed_fit, 4),
                "high_speed_track_fit": round(high_speed_fit, 4),
                "braking_traction_strength": round(braking_traction, 4),
                "strategy_operations": round(strategy_ops, 4),
                "race_execution": round(race_execution, 4),
                "pit_performance": round(pit_performance, 4),
                "reliability": round(reliability, 4),
                "upgrade_regulation_impact": round(upgrade_regulation, 4),
                "track_fit": round(track_fit, 4),
                "market_consensus": round(market_consensus, 4) if market_consensus is not None else None,
            },
            "bounded_modifier": round(_bounded_modifier(overall, confidence), 4),
            "confidence": confidence,
            "source_coverage": coverage,
            "source_breakdown": source_breakdown,
            "missing_data": missing,
            "telemetry": {
                "avg_lap_pace_seconds": round(mean(team_laps), 3) if team_laps else None,
                "avg_pit_duration": pit.get("avg_pit_duration"),
                "avg_stint_laps": stint.get("avg_stint_laps"),
                "avg_position": position.get("avg_position"),
                "avg_gap_to_leader": interval.get("avg_gap_to_leader"),
                "avg_interval": interval.get("avg_interval"),
                "race_control_chaos": round(race_control_chaos, 4),
                "location_trace_available": location_trace,
                "top_speed": max((item.get("max_speed") or 0 for item in car_team_data), default=None),
                "drs_usage": round(mean([item.get("drs_usage") for item in car_team_data if item.get("drs_usage") is not None]), 4) if car_team_data else None,
            },
            "explanations": _explanations(constructor.name, session_key, track, row_scores={
                "pace": pace_score,
                "track_fit": track_fit,
                "tire": tire_behavior,
                "strategy": strategy_ops,
                "reliability": reliability,
                "confidence": confidence,
            }, missing=missing),
        }
        constructor_rows[constructor.id] = row
        constructor_rows[team_key] = row
        for driver in team_drivers:
            driver_rows[driver.id] = {
                "driver_id": driver.id,
                "constructor_id": constructor.id,
                "constructor_name": constructor.name,
                "overall_car_score": row["scores"]["overall_car_score"],
                "car_modifier": row["bounded_modifier"],
                "confidence": confidence,
                "source_coverage": coverage,
                "scores": row["scores"],
                "missing_data": missing,
            }

    rankings = sorted(
        [row for key, row in constructor_rows.items() if key == row["constructor_id"]],
        key=lambda item: item["scores"]["overall_car_score"],
        reverse=True,
    )
    return {
        "ok": True,
        "model_version": CAR_MODEL_VERSION,
        "round": getattr(race, "round", None),
        "race_name": getattr(race, "name", None),
        "session": session_key,
        "source": "f1_car_model_feature_provider",
        "confidence": round(mean([row["confidence"] for row in rankings]), 4) if rankings else 0.0,
        "source_count": len(sources),
        "constructors": constructor_rows,
        "drivers": driver_rows,
        "rankings": rankings,
        "top_edges": rankings[:5],
        "missing_data": sorted({item for row in rankings for item in row.get("missing_data") or []}),
        "bias_controls": {
            "sentiment_cap": 0.03,
            "market_cap": 0.02,
            "car_modifier_range": "0.94-1.06",
            "driver_skill_separated": True,
        },
    }


def summarize_car_data(rows: list[dict[str, Any]], drivers: list[Driver]) -> dict[str, Any]:
    by_number: dict[int, list[dict[str, Any]]] = {}
    for row in rows or []:
        try:
            number = int(row.get("driver_number"))
        except (TypeError, ValueError):
            continue
        by_number.setdefault(number, []).append(row)
    code_by_number = {int(driver.number): driver.code for driver in drivers if driver.number is not None}
    summarized = {}
    for number, items in by_number.items():
        speeds = [_float(item.get("speed")) for item in items]
        speeds = [value for value in speeds if value is not None and value > 0]
        throttles = [_float(item.get("throttle")) for item in items]
        throttles = [value for value in throttles if value is not None]
        brakes = [_float(item.get("brake")) for item in items]
        brakes = [value for value in brakes if value is not None]
        drs = [_float(item.get("drs")) for item in items]
        drs = [value for value in drs if value is not None]
        summarized[str(number)] = {
            "driver_number": number,
            "driver_code": code_by_number.get(number),
            "samples": len(items),
            "max_speed": round(max(speeds), 2) if speeds else None,
            "avg_speed": round(mean(speeds), 2) if speeds else None,
            "avg_throttle": round(mean(throttles) / 100.0, 4) if throttles else None,
            "brake_usage": round(sum(1 for value in brakes if value > 0) / len(brakes), 4) if brakes else None,
            "drs_usage": round(sum(1 for value in drs if value and value >= 8) / len(drs), 4) if drs else None,
        }
    return {"drivers": summarized, "source": "openf1_car_data", "missing_data": not bool(summarized)}


def _session_key(session: str) -> str:
    value = (session or "race").lower()
    if value.startswith("qual"):
        return "qualifying"
    if value.startswith("sprint"):
        return "sprint"
    return "race"


def _team_key(team: str | None) -> str:
    return str(team or "").strip().lower()


def _constructor_standings_score(constructor: Constructor, constructors: list[Constructor]) -> float:
    max_points = max((item.points for item in constructors), default=1.0) or 1.0
    points = float(constructor.points or 0.0) / max_points
    position = max(0.0, 1.0 - ((constructor.position or 10) - 1) * 0.08)
    return clamp01(0.62 * points + 0.38 * position)


def _avg_driver_feature(drivers: list[Driver], features: dict[str, dict], key: str, default: float) -> float:
    values = [_float((features.get(driver.id) or {}).get(key)) for driver in drivers]
    values = [value for value in values if value is not None]
    return clamp01(mean(values) if values else default)


def _lap_pace_by_driver(openf1_session: dict, drivers: list[Driver]) -> dict[str, float]:
    by_number = {str(driver.number): driver.id for driver in drivers if driver.number is not None}
    rows = ((openf1_session.get("laps") or {}).get("drivers") or {})
    result = {}
    for number, driver_id in by_number.items():
        row = rows.get(number) or {}
        pace = _float(row.get("representative_lap") or row.get("median_lap") or row.get("best_lap"))
        if pace and pace > 0:
            result[driver_id] = pace
    return result


def _car_data_by_driver(openf1_session: dict, drivers: list[Driver]) -> dict[str, dict]:
    by_number = {str(driver.number): driver.id for driver in drivers if driver.number is not None}
    rows = ((openf1_session.get("car_data") or {}).get("drivers") or {})
    return {driver_id: rows.get(number) for number, driver_id in by_number.items() if rows.get(number)}


def _pits_by_team(openf1_session: dict, drivers: list[Driver]) -> dict[str, dict]:
    by_number = {str(driver.number): driver for driver in drivers if driver.number is not None}
    rows = ((openf1_session.get("pits") or {}).get("drivers") or {})
    by_team: dict[str, list[float]] = {}
    for number, row in rows.items():
        driver = by_number.get(str(number))
        duration = _float(row.get("avg_pit_duration"))
        if driver and duration:
            by_team.setdefault(_team_key(driver.team), []).append(duration)
    return {
        team: {"avg_pit_duration": round(mean(values), 3), "samples": len(values)}
        for team, values in by_team.items()
    }


def _stints_by_team(openf1_session: dict, drivers: list[Driver]) -> dict[str, dict]:
    by_number = {str(driver.number): driver for driver in drivers if driver.number is not None}
    rows = ((openf1_session.get("stints") or {}).get("drivers") or {})
    by_team: dict[str, list[float]] = {}
    compounds: dict[str, set[str]] = {}
    for number, row in rows.items():
        driver = by_number.get(str(number))
        laps = _float(row.get("avg_stint_laps"))
        if driver and laps:
            key = _team_key(driver.team)
            by_team.setdefault(key, []).append(laps)
            compounds.setdefault(key, set()).update(str(item) for item in (row.get("compounds") or []) if item)
    return {
        team: {"avg_stint_laps": round(mean(values), 2), "samples": len(values), "compounds": sorted(compounds.get(team) or [])}
        for team, values in by_team.items()
    }


def _positions_by_team(openf1_session: dict, drivers: list[Driver]) -> dict[str, dict]:
    by_number = {str(driver.number): driver for driver in drivers if driver.number is not None}
    rows = ((openf1_session.get("positions") or {}).get("drivers") or {})
    by_team: dict[str, list[float]] = {}
    for number, row in rows.items():
        driver = by_number.get(str(number))
        position = _float(row.get("position"))
        if driver and position:
            by_team.setdefault(_team_key(driver.team), []).append(position)
    return {
        team: {"avg_position": round(mean(values), 2), "samples": len(values)}
        for team, values in by_team.items()
    }


def _intervals_by_team(openf1_session: dict, drivers: list[Driver]) -> dict[str, dict]:
    by_number = {str(driver.number): driver for driver in drivers if driver.number is not None}
    rows = ((openf1_session.get("intervals") or {}).get("drivers") or {})
    gaps: dict[str, list[float]] = {}
    intervals: dict[str, list[float]] = {}
    for number, row in rows.items():
        driver = by_number.get(str(number))
        if not driver:
            continue
        team = _team_key(driver.team)
        gap = _gap_seconds(row.get("gap_to_leader"))
        interval = _gap_seconds(row.get("interval"))
        if gap is not None:
            gaps.setdefault(team, []).append(gap)
        if interval is not None:
            intervals.setdefault(team, []).append(interval)
    teams = sorted(set(gaps).union(intervals))
    return {
        team: {
            "avg_gap_to_leader": round(mean(gaps.get(team) or [0.0]), 3) if gaps.get(team) else None,
            "avg_interval": round(mean(intervals.get(team) or [0.0]), 3) if intervals.get(team) else None,
            "samples": len(gaps.get(team) or []) + len(intervals.get(team) or []),
        }
        for team in teams
    }


def _location_trace_available(openf1_session: dict) -> bool:
    location = openf1_session.get("location") or openf1_session.get("locations") or openf1_session.get("track")
    if isinstance(location, dict):
        return bool(location.get("drivers") or location.get("live_positions") or location.get("racing_line") or location.get("points"))
    if isinstance(location, list):
        return bool(location)
    return False


def _race_control_chaos(openf1_session: dict) -> float:
    race_control = openf1_session.get("race_control") or {}
    return clamp01(_float(race_control.get("chaos_score")) or 0.0)


def _race_execution_score(position: dict, interval: dict, reliability: float) -> float:
    avg_position = _float(position.get("avg_position"))
    avg_gap = _float(interval.get("avg_gap_to_leader"))
    avg_interval = _float(interval.get("avg_interval"))
    position_score = clamp01(1.0 - ((avg_position or 12.0) - 1.0) / 21.0)
    gap_score = 0.50 if avg_gap is None else clamp01(1.0 - min(1.0, avg_gap / 45.0) * 0.72)
    interval_score = 0.50 if avg_interval is None else clamp01(0.45 + min(1.0, 1.5 / max(avg_interval, 0.05)) * 0.30)
    return clamp01(0.42 * position_score + 0.28 * gap_score + 0.18 * interval_score + 0.12 * reliability)


def _pace_score(values: list[float], fastest: float | None, slowest: float | None, fallback: float) -> float:
    if not values or fastest is None or slowest is None or slowest <= fastest:
        return clamp01(fallback)
    avg = mean(values)
    return clamp01(1.0 - ((avg - fastest) / max(0.001, slowest - fastest)) * 0.72)


def _long_run_score(drivers: list[Driver], openf1_session: dict, pace_score: float, fallback: float) -> float:
    rows = ((openf1_session.get("laps") or {}).get("drivers") or {})
    by_number = {str(driver.number): driver for driver in drivers if driver.number is not None}
    lap_counts = []
    for number in by_number:
        row = rows.get(number) or {}
        laps = _float(row.get("laps"))
        if laps:
            lap_counts.append(laps)
    if not lap_counts:
        return clamp01(0.55 * fallback + 0.45 * pace_score)
    coverage = clamp01(mean(lap_counts) / 18.0)
    return clamp01(0.62 * pace_score + 0.38 * coverage)


def _teammate_normalized_score(drivers: list[Driver], lap_pace: dict[str, float]) -> float:
    values = [lap_pace.get(driver.id) for driver in drivers if lap_pace.get(driver.id)]
    if len(values) < 2:
        return 0.50
    gap = max(values) - min(values)
    return clamp01(1.0 - min(1.0, gap / 1.25) * 0.65)


def _tire_behavior_score(stint: dict, tire_stress: float, chaos: float) -> float:
    avg_laps = _float(stint.get("avg_stint_laps"))
    if avg_laps:
        endurance = clamp01(avg_laps / max(12.0, 30.0 - tire_stress * 8.0))
        return clamp01(endurance - chaos * 0.08)
    return clamp01(0.64 - tire_stress * 0.18 - chaos * 0.06)


def _pit_score(avg_pit: float | None, best: float | None, slow: float | None) -> float:
    if not avg_pit or best is None or slow is None or slow <= best:
        return 0.55
    return clamp01(1.0 - ((avg_pit - best) / max(0.001, slow - best)) * 0.62)


def _straight_line_score(car_data: list[dict], high_speed_need: float, fallback: float) -> float:
    speeds = [_float(item.get("max_speed")) for item in car_data if item]
    speeds = [value for value in speeds if value and value > 0]
    drs = [_float(item.get("drs_usage")) for item in car_data if item and item.get("drs_usage") is not None]
    if speeds:
        speed_score = clamp01((mean(speeds) - 260.0) / 75.0)
        drs_score = clamp01(mean(drs)) if drs else 0.55
        return clamp01(0.72 * speed_score + 0.18 * drs_score + 0.10 * fallback)
    return clamp01(fallback * (0.82 + high_speed_need * 0.18))


def _upgrade_regulation_score(constructor: Constructor, sentiment_impact: dict | None) -> float:
    impacts = ((sentiment_impact or {}).get("constructors") or {})
    item = impacts.get(constructor.id) or impacts.get(_team_key(constructor.name)) or {}
    topic_scores = item.get("topic_scores") or {}
    if not item:
        return 0.50
    performance = _float(topic_scores.get("performance")) or 0.0
    operations = _float(topic_scores.get("team_operations")) or 0.0
    regulation = _float(topic_scores.get("regulation")) or 0.0
    raw = performance + operations - abs(regulation) * 0.25
    impact = _float(item.get("driver_impact_score")) or 0.0
    return clamp01(0.50 + max(-0.12, min(0.12, raw * 0.06 + impact * 0.08)))


def _market_team_score(constructor: Constructor, market_signals: dict | None) -> float | None:
    if not market_signals:
        return None
    teams = market_signals.get("constructors") or market_signals.get("teams") or {}
    value = teams.get(constructor.id) or teams.get(_team_key(constructor.name))
    if value is None:
        return None
    return clamp01(_float(value) or 0.0)


def _has_constructor_sentiment(constructor: Constructor, sentiment_impact: dict | None) -> bool:
    impacts = ((sentiment_impact or {}).get("constructors") or {})
    item = impacts.get(constructor.id) or impacts.get(_team_key(constructor.name)) or {}
    return bool(item and int(item.get("article_count") or 0) > 0)


def _confidence(**coverage: bool) -> tuple[float, dict[str, bool], list[str]]:
    weights = {
        "has_laps": 0.18,
        "has_car_data": 0.12,
        "has_positions": 0.08,
        "has_intervals": 0.08,
        "has_stints": 0.10,
        "has_pits": 0.08,
        "has_race_control": 0.04,
        "has_location_trace": 0.04,
        "has_track": 0.12,
        "has_weather": 0.08,
        "has_sentiment": 0.04,
        "has_market": 0.02,
    }
    base = 0.18
    score = base + sum(weight for key, weight in weights.items() if coverage.get(key))
    missing = [key.replace("has_", "") for key in weights if not coverage.get(key)]
    return round(clamp01(score), 4), coverage, missing


def _bounded_modifier(overall: float, confidence: float) -> float:
    return max(0.94, min(1.06, 1.0 + (overall - 0.50) * 0.12 * confidence))


def _explanations(constructor_name: str, session: str, track: dict, row_scores: dict, missing: list[str]) -> list[str]:
    messages = []
    if row_scores["pace"] >= 0.68:
        messages.append(f"{constructor_name} has a strong measured or historical pace profile.")
    if row_scores["track_fit"] >= 0.66:
        messages.append("Circuit traits align with this car profile.")
    if row_scores["tire"] <= 0.45:
        messages.append("Tire behavior is a limiting factor for this session.")
    if track.get("street_circuit") and session == "qualifying":
        messages.append("Street-circuit qualifying importance increases track-fit relevance.")
    if missing:
        messages.append("Missing groups are discounted: " + ", ".join(missing[:5]) + ".")
    if not messages:
        messages.append("Car model is neutral until stronger session evidence arrives.")
    return messages[:4]


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _gap_seconds(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in {"-", "LEADER"}:
        return 0.0
    if "LAP" in text:
        return 75.0
    text = text.replace("+", "").replace("S", "")
    return _float(text)
