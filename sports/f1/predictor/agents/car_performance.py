"""Car Performance Analyst Agent for Formula 1.

The agent is intentionally read-only. It turns the existing car-model feature
provider into a compact analysis of car traits: low-speed strength, tyre
degradation, straight-line speed, sector strengths, reliability, and teammate
deltas. Missing OpenF1 telemetry is labeled as fallback instead of treated as
hard evidence.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from sports.f1.models.f1 import Constructor, Driver, Race
from sports.f1.predictor.features.car_model import build_car_model_analysis
from sports.f1.predictor.scoring.normalization import clamp01, num


AGENT_ID = "car_performance_analyst_v1"


def build_car_performance_agent(
    *,
    race: Race | None,
    drivers: list[Driver],
    constructors: list[Constructor],
    features: dict | None = None,
    session: str = "race",
    constructor_id: str | None = None,
    track: dict | None = None,
    weather: dict | None = None,
    tires: dict | None = None,
    weekend_evidence: dict | None = None,
    openf1_session: dict | None = None,
    car_model: dict | None = None,
    sentiment_impact: dict | None = None,
    market_signals: dict | None = None,
) -> dict[str, Any]:
    """Build source-labeled car performance analysis.

    The payload is designed for API, assistant, and dashboard consumption. It is
    additive and deterministic; invalid or missing inputs produce a low
    confidence fallback payload.
    """

    features = features or {}
    session_key = _session_key(session)
    openf1_session = openf1_session or features.get("openf1_session") or {}
    track = track or {}
    weather = weather or {}
    tires = tires or {}
    weekend_evidence = weekend_evidence or features.get("weekend_evidence") or {}
    if car_model is None:
        car_model = build_car_model_analysis(
            race=race,
            drivers=drivers,
            constructors=constructors,
            features=features,
            session=session_key,
            track=track,
            weather=weather,
            tires=tires,
            openf1_session=openf1_session,
            sentiment_impact=sentiment_impact,
            market_signals=market_signals,
        )

    constructors_map = car_model.get("constructors") or {}
    selected_ids = _selected_constructor_ids(constructor_id, constructors, constructors_map)
    selected_rows = [
        row
        for cid in selected_ids
        for row in [constructors_map.get(cid) or constructors_map.get(_team_key(cid))]
        if row
    ]
    if constructor_id and not selected_rows:
        return {
            "ok": False,
            "agent_id": AGENT_ID,
            "reason": "Constructor car performance analysis not found",
            "code": "constructor_car_performance_agent_not_found",
            "constructor_id": constructor_id,
            "round": getattr(race, "round", None),
            "session": session_key,
        }

    unique_rows = _unique_constructor_rows(selected_rows or car_model.get("rankings") or [])
    analyses = [_constructor_analysis(row, track, tires, weather, openf1_session, weekend_evidence) for row in unique_rows]
    rankings = sorted(
        [
            {
                "constructor_id": item["constructor_id"],
                "constructor_name": item["constructor_name"],
                "overall_car_score": item["overall_car_score"],
                "confidence": item["confidence"],
                "best_edges": item["best_edges"],
                "main_risks": item["main_risks"],
            }
            for item in analyses
        ],
        key=lambda item: item["overall_car_score"],
        reverse=True,
    )
    confidence_values = [item["confidence"] for item in analyses if item.get("confidence") is not None]
    missing_groups = sorted({group for item in analyses for group in item.get("missing_groups") or []})
    source_mode = _source_mode(car_model, openf1_session, weekend_evidence)
    return {
        "ok": True,
        "agent_id": AGENT_ID,
        "round": getattr(race, "round", None),
        "race_name": getattr(race, "name", None),
        "session": session_key,
        "constructor_id": constructor_id,
        "source_mode": source_mode,
        "confidence": round(mean(confidence_values), 4) if confidence_values else 0.0,
        "confidence_reason": _confidence_reason(source_mode, missing_groups),
        "missing_groups": missing_groups,
        "fallback_reason": _fallback_reason(source_mode, missing_groups),
        "source_breakdown": _aggregate_source_breakdown(analyses, car_model, openf1_session, weekend_evidence),
        "constructors": analyses,
        "rankings": rankings,
        "summary": _summary(analyses, source_mode, missing_groups),
        "constraints": {
            "read_only": True,
            "prediction_weights_changed": False,
            "bounded_to_existing_car_model": True,
            "include_car_data_required_for_top_speed_telemetry": True,
        },
    }


def _constructor_analysis(
    row: dict[str, Any],
    track: dict[str, Any],
    tires: dict[str, Any],
    weather: dict[str, Any],
    openf1_session: dict[str, Any],
    weekend_evidence: dict[str, Any],
) -> dict[str, Any]:
    scores = row.get("scores") or {}
    telemetry = row.get("telemetry") or {}
    source_breakdown = row.get("source_breakdown") or {}
    missing = list(row.get("missing_data") or [])
    low_speed_score = _avg(
        scores.get("low_speed_track_fit"),
        scores.get("braking_traction_strength"),
        scores.get("qualifying_pace"),
    )
    tyre_score = _avg(scores.get("tire_behavior"), scores.get("degradation_control"), scores.get("long_run_pace"))
    top_speed_score = _avg(scores.get("drs_straight_line_strength"), scores.get("high_speed_track_fit"))
    reliability_score = num(scores.get("reliability"), 0.5)
    teammate = _teammate_delta(row)
    sector_strengths = _sector_strengths(row, openf1_session, track)
    reliability = {
        "score": round(reliability_score, 4),
        "mechanical_risk": round(clamp01(1.0 - reliability_score), 4),
        "incident_risk": round(clamp01(num(track.get("safety_car_probability"), 0.35) * 0.55 + num(weather.get("chaos_score"), 0.0) * 0.45), 4),
        "source": "historical_and_race_control" if source_breakdown.get("openf1_race_control") == "available" else "historical_fallback",
        "explanation": _band_text(reliability_score, "reliability"),
    }
    low_speed = {
        "score": round(low_speed_score, 4),
        "source": "telemetry_and_track_traits" if source_breakdown.get("openf1_laps") == "available" else "track_traits_fallback",
        "track_dependency": "high" if track.get("street_circuit") or num(track.get("qualifying_importance"), 0.0) > 0.75 else "medium",
        "explanation": _band_text(low_speed_score, "low-speed performance"),
    }
    tyre_degradation = {
        "score": round(tyre_score, 4),
        "risk": round(clamp01(1.0 - tyre_score), 4),
        "avg_stint_laps": telemetry.get("avg_stint_laps"),
        "track_tire_stress": track.get("tire_stress") or tires.get("degradation_rate"),
        "source": "openf1_stints" if source_breakdown.get("openf1_stints") == "available" else "tire_model_fallback",
        "explanation": _band_text(tyre_score, "tyre degradation control"),
    }
    top_speed = {
        "score": round(top_speed_score, 4),
        "top_speed_kph": telemetry.get("top_speed"),
        "drs_usage": telemetry.get("drs_usage"),
        "source": "openf1_car_data" if source_breakdown.get("openf1_car_data") == "available" else "car_model_fallback",
        "explanation": _band_text(top_speed_score, "straight-line speed"),
    }
    best_edges = _best_edges({
        "low_speed": low_speed_score,
        "tyre_degradation": tyre_score,
        "top_speed": top_speed_score,
        "reliability": reliability_score,
        "sector_fit": sector_strengths.get("overall_sector_fit") or 0.5,
    })
    main_risks = _main_risks({
        "low_speed": low_speed_score,
        "tyre_degradation": tyre_score,
        "top_speed": top_speed_score,
        "reliability": reliability_score,
    }, missing)
    return {
        "constructor_id": row.get("constructor_id"),
        "constructor_name": row.get("constructor_name"),
        "session": row.get("session"),
        "overall_car_score": row.get("scores", {}).get("overall_car_score"),
        "confidence": row.get("confidence"),
        "source_coverage": row.get("source_coverage") or {},
        "source_breakdown": source_breakdown,
        "missing_groups": missing,
        "low_speed": low_speed,
        "tyre_degradation": tyre_degradation,
        "top_speed": top_speed,
        "sector_strengths": sector_strengths,
        "reliability": reliability,
        "teammate_deltas": teammate,
        "telemetry": telemetry,
        "best_edges": best_edges,
        "main_risks": main_risks,
        "explanations": _agent_explanations(row, low_speed, tyre_degradation, top_speed, sector_strengths, reliability, teammate),
    }


def _selected_constructor_ids(constructor_id: str | None, constructors: list[Constructor], constructors_map: dict[str, Any]) -> list[str]:
    if not constructor_id:
        return [constructor.id for constructor in constructors]
    needle = _team_key(constructor_id)
    matched = [
        constructor.id
        for constructor in constructors
        if _team_key(constructor.id) == needle or _team_key(constructor.name) == needle
    ]
    return matched or [constructor_id]


def _unique_constructor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        cid = str(row.get("constructor_id") or row.get("constructor_name") or "").lower()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        unique.append(row)
    return unique


def _sector_strengths(row: dict[str, Any], openf1_session: dict[str, Any], track: dict[str, Any]) -> dict[str, Any]:
    driver_numbers = {
        str(driver.get("driver_id") or driver.get("driver_code") or ""): driver
        for driver in row.get("drivers") or []
    }
    sectors = _extract_sector_values(openf1_session, set(driver_numbers))
    if sectors:
        sector_scores = {
            "sector_1": _pace_to_score([item.get("sector_1") for item in sectors]),
            "sector_2": _pace_to_score([item.get("sector_2") for item in sectors]),
            "sector_3": _pace_to_score([item.get("sector_3") for item in sectors]),
        }
        source = "openf1_lap_sectors"
    else:
        low = num((row.get("scores") or {}).get("low_speed_track_fit"), 0.5)
        high = num((row.get("scores") or {}).get("high_speed_track_fit"), 0.5)
        quali = num((row.get("scores") or {}).get("qualifying_pace"), 0.5)
        if track.get("street_circuit"):
            sector_scores = {"sector_1": round(low, 4), "sector_2": round(quali, 4), "sector_3": round(low * 0.65 + high * 0.35, 4)}
        elif track.get("high_speed"):
            sector_scores = {"sector_1": round(high, 4), "sector_2": round(high * 0.7 + quali * 0.3, 4), "sector_3": round(low * 0.35 + high * 0.65, 4)}
        else:
            sector_scores = {"sector_1": round(quali, 4), "sector_2": round(low, 4), "sector_3": round(high, 4)}
        source = "track_traits_fallback"
    return {
        **sector_scores,
        "overall_sector_fit": round(_avg(*sector_scores.values()), 4),
        "source": source,
        "strongest_sector": max(sector_scores, key=lambda key: sector_scores[key]),
        "weakest_sector": min(sector_scores, key=lambda key: sector_scores[key]),
    }


def _extract_sector_values(openf1_session: dict[str, Any], driver_ids: set[str]) -> list[dict[str, float]]:
    laps = ((openf1_session or {}).get("laps") or {}).get("drivers") or {}
    values: list[dict[str, float]] = []
    for payload in laps.values():
        if not isinstance(payload, dict):
            continue
        code = str(payload.get("driver_id") or payload.get("driver_code") or "").lower()
        if driver_ids and code and code not in driver_ids:
            continue
        sector_row = {
            "sector_1": _first_float(payload, ["sector_1", "sector1", "sector_1_time", "s1"]),
            "sector_2": _first_float(payload, ["sector_2", "sector2", "sector_2_time", "s2"]),
            "sector_3": _first_float(payload, ["sector_3", "sector3", "sector_3_time", "s3"]),
        }
        if any(value for value in sector_row.values()):
            values.append(sector_row)
    return values


def _teammate_delta(row: dict[str, Any]) -> dict[str, Any]:
    drivers = row.get("drivers") or []
    pace_rows = [
        {
            "driver_id": driver.get("driver_id"),
            "driver_code": driver.get("driver_code"),
            "lap_pace_seconds": _to_float(driver.get("lap_pace_seconds")),
        }
        for driver in drivers
        if _to_float(driver.get("lap_pace_seconds")) is not None
    ]
    if len(pace_rows) < 2:
        return {
            "available": False,
            "source": "missing_teammate_lap_pace",
            "delta_seconds": None,
            "leader": None,
            "explanation": "Teammate delta needs usable lap pace for both cars.",
        }
    ordered = sorted(pace_rows, key=lambda item: item["lap_pace_seconds"] or 9999)
    delta = (ordered[-1]["lap_pace_seconds"] or 0) - (ordered[0]["lap_pace_seconds"] or 0)
    return {
        "available": True,
        "source": "openf1_lap_pace",
        "delta_seconds": round(delta, 3),
        "leader": ordered[0],
        "trailer": ordered[-1],
        "explanation": f"{ordered[0].get('driver_code') or ordered[0].get('driver_id')} is ahead by {delta:.3f}s on representative lap pace.",
    }


def _agent_explanations(row: dict[str, Any], low_speed: dict, tyre: dict, top_speed: dict, sector: dict, reliability: dict, teammate: dict) -> list[str]:
    name = row.get("constructor_name") or "This team"
    explanations = [
        f"{name} low-speed read is {low_speed['score']:.0%} from {low_speed['source']}.",
        f"Tyre degradation control is {tyre['score']:.0%}; source {tyre['source']}.",
        f"Straight-line strength is {top_speed['score']:.0%}; source {top_speed['source']}.",
        f"Strongest sector signal is {sector['strongest_sector'].replace('_', ' ')}.",
        f"Reliability score is {reliability['score']:.0%}; {reliability['source']}.",
    ]
    if teammate.get("available"):
        explanations.append(teammate.get("explanation") or "")
    else:
        explanations.append("Teammate delta is unavailable until both cars have usable timing rows.")
    return [item for item in explanations if item]


def _aggregate_source_breakdown(
    analyses: list[dict[str, Any]],
    car_model: dict[str, Any],
    openf1_session: dict[str, Any],
    weekend_evidence: dict[str, Any],
) -> dict[str, Any]:
    raw_counts = openf1_session.get("raw_counts") or {}
    return {
        "car_model_version": car_model.get("model_version"),
        "openf1_raw_counts": raw_counts,
        "weekend_evidence_source_mode": weekend_evidence.get("source_mode"),
        "weekend_evidence_confidence": weekend_evidence.get("confidence"),
        "constructors_analyzed": len(analyses),
        "sources": sorted({value for item in analyses for value in (item.get("source_breakdown") or {}).values() if value}),
    }


def _source_mode(car_model: dict[str, Any], openf1_session: dict[str, Any], weekend_evidence: dict[str, Any]) -> str:
    if weekend_evidence.get("source_mode") in {"live", "recorded", "recent", "historical"}:
        return str(weekend_evidence["source_mode"])
    raw_counts = openf1_session.get("raw_counts") or {}
    if any(num(raw_counts.get(key), 0) > 0 for key in ["laps", "car_data", "stints", "pits", "positions", "intervals"]):
        return "recent"
    if not car_model.get("missing_data"):
        return "historical"
    return "estimated"


def _confidence_reason(source_mode: str, missing_groups: list[str]) -> str:
    if source_mode in {"live", "recorded"}:
        return "Real timing or recording evidence is contributing to the car analysis."
    if source_mode == "recent":
        return "Recent OpenF1/session rows are contributing, but not full live telemetry."
    if missing_groups:
        return "Analysis is capped because telemetry groups are missing: " + ", ".join(missing_groups[:4])
    return "Analysis is based on historical and track-trait fallback evidence."


def _fallback_reason(source_mode: str, missing_groups: list[str]) -> str | None:
    if source_mode in {"live", "recorded", "recent"}:
        return None
    if missing_groups:
        return "Missing car performance evidence: " + ", ".join(missing_groups[:5])
    return "Using bounded historical car model fallback."


def _summary(analyses: list[dict[str, Any]], source_mode: str, missing_groups: list[str]) -> str:
    if not analyses:
        return "No constructor car performance analysis is available."
    leader = max(analyses, key=lambda item: item.get("overall_car_score") or 0)
    edges = ", ".join(leader.get("best_edges") or []) or "balanced profile"
    suffix = ""
    if missing_groups:
        suffix = f" Missing evidence: {', '.join(missing_groups[:4])}."
    return f"{leader.get('constructor_name')} has the strongest car model read for this scope ({edges}). Source mode is {source_mode}.{suffix}"


def _best_edges(scores: dict[str, float]) -> list[str]:
    labels = {
        "low_speed": "low-speed",
        "tyre_degradation": "tyre deg",
        "top_speed": "top speed",
        "reliability": "reliability",
        "sector_fit": "sector fit",
    }
    return [labels[key] for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3] if value >= 0.56]


def _main_risks(scores: dict[str, float], missing: list[str]) -> list[str]:
    labels = {
        "low_speed": "low-speed weakness",
        "tyre_degradation": "tyre degradation risk",
        "top_speed": "straight-line speed risk",
        "reliability": "reliability risk",
    }
    risks = [labels[key] for key, value in sorted(scores.items(), key=lambda item: item[1]) if value < 0.48]
    if missing:
        risks.append("missing " + missing[0])
    return risks[:3]


def _pace_to_score(values: list[float | None]) -> float:
    usable = [value for value in values if value is not None and value > 0]
    if not usable:
        return 0.5
    avg_value = mean(usable)
    return round(clamp01(0.76 - (avg_value - min(usable)) / 4.0), 4)


def _avg(*values: Any) -> float:
    usable = [_to_float(value) for value in values if _to_float(value) is not None]
    return clamp01(mean(usable)) if usable else 0.5


def _first_float(payload: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _to_float(payload.get(key))
        if value is not None:
            return value
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace("+", "").replace("s", "").strip())
    except (TypeError, ValueError):
        return None


def _session_key(session: str | None) -> str:
    key = (session or "race").strip().lower().replace("-", "_")
    if key in {"q", "qualy", "quali"}:
        return "qualifying"
    return key or "race"


def _team_key(team: str | None) -> str:
    return (team or "").strip().lower().replace(" ", "_").replace("-", "_")


def _band_text(score: float, label: str) -> str:
    if score >= 0.68:
        return f"Strong {label} signal."
    if score >= 0.52:
        return f"Competitive {label} signal."
    if score >= 0.42:
        return f"Neutral-to-weak {label} signal."
    return f"Weak {label} signal."
