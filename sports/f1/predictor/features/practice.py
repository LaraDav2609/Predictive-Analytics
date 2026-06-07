"""Practice-session pace adapters for F1 feature snapshots."""

from __future__ import annotations

from typing import Any

from sports.f1.models.f1 import Driver
from sports.f1.predictor.scoring.normalization import clamp01


def apply_practice_pace_adjustments(
    drivers: list[Driver],
    driver_features: dict[str, dict[str, Any]],
    openf1_session: dict[str, Any] | None,
    *,
    session_stage: str = "race",
) -> dict[str, dict[str, Any]]:
    """Blend real practice lap evidence into driver pace features when present.

    The function is intentionally conservative: if OpenF1 has no usable lap rows,
    existing historical pace remains untouched.
    """

    adjusted = {driver_id: dict(values or {}) for driver_id, values in (driver_features or {}).items()}
    openf1_session = openf1_session or {}
    if not openf1_session.get("ok"):
        return adjusted

    laps_by_number = ((openf1_session.get("laps") or {}).get("drivers") or {})
    if not laps_by_number:
        return adjusted

    number_to_driver = {str(driver.number): driver for driver in drivers if driver.number is not None}
    stints_by_number = ((openf1_session.get("stints") or {}).get("drivers") or {})
    practice_rows: list[dict[str, Any]] = []
    for number, lap_summary in laps_by_number.items():
        driver = number_to_driver.get(str(number))
        if not driver:
            continue
        representative = _float(lap_summary.get("representative_lap") or lap_summary.get("median_lap"))
        best = _float(lap_summary.get("best_lap"))
        long_run = _float(lap_summary.get("long_run_lap") or lap_summary.get("median_lap"))
        lap_count = int(lap_summary.get("laps") or 0)
        pace_lap = representative or best
        if not pace_lap or pace_lap <= 0:
            continue
        stint = stints_by_number.get(str(number)) or {}
        compounds = sorted({
            str(item).upper()
            for item in [
                *(lap_summary.get("compounds") or []),
                *(stint.get("compounds") or []),
                stint.get("compound"),
            ]
            if item
        })
        practice_rows.append({
            "driver_id": driver.id,
            "team": driver.team,
            "representative_lap": pace_lap,
            "best_lap": best,
            "long_run_lap": long_run,
            "lap_count": lap_count,
            "compounds": compounds,
            "avg_stint_laps": _float(stint.get("avg_stint_laps")),
            "track_evolution_delta": _float(lap_summary.get("track_evolution_delta")),
            "sectors": {
                index: _float(lap_summary.get(f"representative_sector_{index}") or lap_summary.get(f"best_sector_{index}"))
                for index in (1, 2, 3)
            },
        })

    if len(practice_rows) < 2:
        return adjusted

    practice_rows.sort(key=lambda row: row["representative_lap"])
    fastest = practice_rows[0]["representative_lap"]
    slowest = practice_rows[-1]["representative_lap"]
    spread = max(2.5, slowest - fastest)
    best_laps = [row["best_lap"] for row in practice_rows if row.get("best_lap")]
    fastest_best = min(best_laps) if best_laps else fastest
    slowest_best = max(best_laps) if best_laps else slowest
    best_spread = max(2.0, slowest_best - fastest_best)
    long_runs = [row["long_run_lap"] for row in practice_rows if row.get("long_run_lap")]
    fastest_long = min(long_runs) if long_runs else fastest
    slowest_long = max(long_runs) if long_runs else slowest
    long_spread = max(3.0, slowest_long - fastest_long)
    sector_bounds = _sector_bounds(practice_rows)
    teammate_reference = _team_reference(practice_rows)
    session_evolution = _average([row.get("track_evolution_delta") for row in practice_rows])
    coverage = min(1.0, len(practice_rows) / max(len(drivers), 1))
    source = openf1_session.get("source") or "openf1_practice_laps"
    stage = (session_stage or "race").lower()
    if stage.startswith("qual"):
        quali_weight, race_weight = 0.30, 0.12
    elif stage.startswith("practice") or stage.startswith("fp"):
        quali_weight, race_weight = 0.45, 0.45
    else:
        quali_weight, race_weight = 0.12, 0.24

    for rank, row in enumerate(practice_rows, start=1):
        driver_id = row["driver_id"]
        feature = adjusted.setdefault(driver_id, {})
        gap = row["representative_lap"] - fastest
        pace_score = clamp01(1.0 - (gap / spread))
        best_score = clamp01(1.0 - (((row.get("best_lap") or row["representative_lap"]) - fastest_best) / best_spread))
        long_run_score = clamp01(1.0 - (((row.get("long_run_lap") or row["representative_lap"]) - fastest_long) / long_spread))
        sector_scores = _sector_scores(row.get("sectors") or {}, sector_bounds)
        sector_score = _average(sector_scores.values()) if sector_scores else None
        fuel_uncertainty = _fuel_uncertainty(row, session_evolution)
        teammate_delta = row["representative_lap"] - teammate_reference.get(str(row.get("team", "")).lower(), row["representative_lap"])
        teammate_adjustment = clamp01(0.50 - teammate_delta / 3.0)
        quali_evidence = clamp01(
            0.46 * best_score
            + 0.28 * pace_score
            + 0.18 * (sector_score if sector_score is not None else pace_score)
            + 0.08 * teammate_adjustment
        )
        race_evidence = clamp01(
            0.44 * pace_score
            + 0.34 * long_run_score
            + 0.12 * teammate_adjustment
            + 0.10 * (1.0 - fuel_uncertainty)
        )
        confidence = clamp01(
            0.20
            + coverage * 0.18
            + min(row["lap_count"], 20) / 20 * 0.24
            + (0.10 if sector_score is not None else 0.0)
            + (0.08 if row.get("long_run_lap") else 0.0)
            + (0.05 if row.get("compounds") else 0.0)
            - fuel_uncertainty * 0.12
        )
        old_quali = _float(feature.get("qualifying_pace_score")) or _float(feature.get("form_score")) or 0.50
        old_race = _float(feature.get("race_pace_score")) or _float(feature.get("form_score")) or 0.50
        effective_quali_weight = quali_weight * (0.65 + confidence * 0.35) * (1.0 - fuel_uncertainty * 0.35)
        effective_race_weight = race_weight * (0.65 + confidence * 0.35) * (1.0 - fuel_uncertainty * 0.30)

        feature["practice_pace_score"] = round(pace_score, 4)
        feature["practice_qualifying_evidence_score"] = round(quali_evidence, 4)
        feature["practice_race_evidence_score"] = round(race_evidence, 4)
        feature["practice_long_run_score"] = round(long_run_score, 4)
        feature["practice_sector_score"] = round(sector_score, 4) if sector_score is not None else None
        feature["practice_sector_scores"] = {str(key): round(value, 4) for key, value in sector_scores.items()}
        feature["practice_teammate_delta"] = round(teammate_delta, 3)
        feature["practice_teammate_score"] = round(teammate_adjustment, 4)
        feature["practice_fuel_uncertainty"] = round(fuel_uncertainty, 4)
        feature["practice_track_evolution_delta"] = round(session_evolution, 3) if session_evolution is not None else None
        feature["practice_compounds"] = row.get("compounds") or []
        feature["practice_best_lap_rank"] = rank
        feature["practice_best_lap"] = row.get("best_lap")
        feature["practice_long_run_lap"] = row.get("long_run_lap")
        feature["practice_representative_lap"] = round(row["representative_lap"], 3)
        feature["practice_lap_count"] = row["lap_count"]
        feature["practice_source"] = source
        feature["practice_confidence"] = round(confidence, 4)
        feature["practice_effective_quali_weight"] = round(effective_quali_weight, 4)
        feature["practice_effective_race_weight"] = round(effective_race_weight, 4)
        feature["qualifying_pace_score"] = round(_blend(old_quali, quali_evidence, effective_quali_weight), 4)
        feature["race_pace_score"] = round(_blend(old_race, race_evidence, effective_race_weight), 4)

    return adjusted


def _blend(base: float, evidence: float, weight: float) -> float:
    return clamp01(base * (1.0 - weight) + evidence * weight)


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _sector_bounds(rows: list[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    bounds = {}
    for index in (1, 2, 3):
        values = [row["sectors"].get(index) for row in rows if row.get("sectors", {}).get(index)]
        if len(values) >= 2:
            bounds[index] = (min(values), max(values))
    return bounds


def _sector_scores(sectors: dict[int, float | None], bounds: dict[int, tuple[float, float]]) -> dict[int, float]:
    scores = {}
    for index, value in sectors.items():
        if value is None or index not in bounds:
            continue
        fastest, slowest = bounds[index]
        spread = max(0.30, slowest - fastest)
        scores[index] = clamp01(1.0 - ((value - fastest) / spread))
    return scores


def _team_reference(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        team = str(row.get("team") or "").lower()
        if team:
            grouped.setdefault(team, []).append(float(row["representative_lap"]))
    return {team: sum(values) / len(values) for team, values in grouped.items() if values}


def _fuel_uncertainty(row: dict[str, Any], session_evolution: float | None) -> float:
    lap_count = int(row.get("lap_count") or 0)
    compounds = row.get("compounds") or []
    best = _float(row.get("best_lap"))
    representative = _float(row.get("representative_lap"))
    best_gap = max(0.0, (representative or 0.0) - (best or representative or 0.0))
    uncertainty = 0.38
    uncertainty -= min(lap_count, 20) * 0.010
    if compounds:
        uncertainty -= 0.05
    if row.get("long_run_lap"):
        uncertainty -= 0.05
    uncertainty += min(0.12, best_gap / 12.0)
    if session_evolution is not None and abs(session_evolution) >= 0.75:
        uncertainty += min(0.10, abs(session_evolution) / 12.0)
    return clamp01(uncertainty)


def _average(values) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)
