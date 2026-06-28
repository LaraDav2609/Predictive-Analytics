"""Build deterministic telemetry features from normalized trace points."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Any

from sports.f1.ml.telemetry.types import (
    TelemetryFeaturePayload,
    TelemetryFeatureVector,
    TelemetryTracePoint,
)


def build_telemetry_features(
    *,
    race_id: str,
    session: str,
    trace_points: list[TelemetryTracePoint],
    laps: list[dict[str, Any]] | None = None,
    stints: list[dict[str, Any]] | None = None,
    intervals: list[dict[str, Any]] | None = None,
    source_mode: str = "mixed",
    live: bool = False,
) -> TelemetryFeaturePayload:
    """Summarize normalized telemetry into per-driver simulator features."""

    laps = laps or []
    stints = stints or []
    intervals = intervals or []
    by_driver: dict[str, list[TelemetryTracePoint]] = defaultdict(list)
    for point in trace_points or []:
        by_driver[point.driver_code].append(point)

    lap_times = _lap_times_by_driver(laps)
    all_lap_times = [value for values in lap_times.values() for value in values]
    median_lap = median(all_lap_times) if all_lap_times else None

    top_speeds = {
        code: max((p.speed_kph for p in points if p.speed_kph is not None), default=None)
        for code, points in by_driver.items()
    }
    median_top_speed = median([v for v in top_speeds.values() if v is not None]) if any(v is not None for v in top_speeds.values()) else None
    min_corner_speeds = {
        code: _low_speed_proxy(points)
        for code, points in by_driver.items()
    }
    median_corner_speed = median([v for v in min_corner_speeds.values() if v is not None]) if any(v is not None for v in min_corner_speeds.values()) else None
    traffic = _traffic_penalty_by_driver(intervals)
    tire_deg = _stint_deg_by_driver(stints, lap_times)

    driver_features: dict[str, TelemetryFeatureVector] = {}
    for code, points in sorted(by_driver.items()):
        speeds = [p.speed_kph for p in points if p.speed_kph is not None]
        throttles = [p.throttle_pct for p in points if p.throttle_pct is not None]
        brakes = [p.brake_pct for p in points if p.brake_pct is not None]
        driver_laps = lap_times.get(code) or []
        pace_delta = _pace_delta(driver_laps, median_lap)
        top_speed_delta = _delta(top_speeds.get(code), median_top_speed)
        corner_delta = _delta(min_corner_speeds.get(code), median_corner_speed)
        full_throttle_pct = _ratio(throttles, lambda value: value >= 98.0)
        heavy_brake_pct = _ratio(brakes, lambda value: value >= 50.0)
        traction = _traction_score(points)
        stability = _stability_score(speeds)
        if pace_delta == 0.0 and top_speed_delta is not None:
            pace_delta = _proxy_pace_delta(top_speed_delta, corner_delta, full_throttle_pct, traction)
        confidence = _driver_confidence(points, driver_laps, stints, intervals, live=live)
        driver_features[code] = TelemetryFeatureVector(
            race_id=race_id,
            session=session,
            driver_code=code,
            source=_dominant_source(points),
            lap=max((p.lap or 0 for p in points), default=0) or None,
            clean_air_pace_delta_s=round(pace_delta, 4),
            pace_sigma_delta=round(_pace_sigma_delta(driver_laps, speeds), 4),
            full_throttle_pct=full_throttle_pct,
            heavy_brake_pct=heavy_brake_pct,
            corner_min_speed_delta_kph=corner_delta,
            top_speed_delta_kph=top_speed_delta,
            traction_score=traction,
            stability_score=stability,
            tire_deg_slope_delta=tire_deg.get(code),
            traffic_penalty_s=traffic.get(code),
            overtake_pressure=_overtake_pressure(top_speed_delta, traffic.get(code)),
            dnf_hazard_multiplier=_dnf_multiplier(stability, brakes),
            confidence=confidence,
            samples=len(points),
            raw={
                "lap_times": driver_laps,
                "speed_samples": len(speeds),
                "throttle_samples": len(throttles),
                "brake_samples": len(brakes),
            },
        )

    missing_groups = _missing_groups(trace_points, laps, stints, intervals)
    confidence = _payload_confidence(driver_features, missing_groups)
    return TelemetryFeaturePayload(
        ok=bool(driver_features),
        race_id=race_id,
        session=session,
        source_mode=source_mode,
        confidence=confidence,
        driver_features=driver_features,
        lap_features=[],
        segment_features=_top_segment_summaries(driver_features),
        stint_features=[{"driver_code": code, "tire_deg_slope_delta": value} for code, value in tire_deg.items()],
        raw_counts={
            "trace_points": len(trace_points or []),
            "laps": len(laps),
            "stints": len(stints),
            "intervals": len(intervals),
            "drivers": len(driver_features),
        },
        missing_groups=missing_groups,
        data_quality={
            "live": bool(live),
            "min_driver_samples": min((len(points) for points in by_driver.values()), default=0),
            "drivers_with_lap_times": sum(1 for values in lap_times.values() if values),
            "estimated": not bool(trace_points),
        },
    )


def build_telemetry_features_from_openf1_session(
    *,
    race_id: str,
    session: str,
    openf1_session: dict[str, Any] | None,
    live: bool = False,
) -> TelemetryFeaturePayload:
    """Build telemetry features from the existing OpenF1 summarized payload.

    The dashboard API already fetches summarized OpenF1 evidence for laps,
    intervals, stints, pits, weather, and race control. This bridge lets the
    telemetry model use that evidence before raw high-frequency `car_data`
    ingestion is fully wired.
    """

    payload = openf1_session or {}
    if not payload.get("ok"):
        return TelemetryFeaturePayload(
            ok=False,
            race_id=race_id,
            session=session,
            source_mode=str(payload.get("source_mode") or payload.get("source") or "estimated"),
            confidence=0.0,
            raw_counts=dict(payload.get("raw_counts") or {}),
            missing_groups=["openf1_session"],
            data_quality={
                "reason": payload.get("reason") or "openf1_session_unavailable",
                "live": bool(live),
                "estimated": True,
            },
        )

    lap_drivers = ((payload.get("laps") or {}).get("drivers") or {})
    interval_drivers = ((payload.get("intervals") or {}).get("drivers") or {})
    stint_drivers = ((payload.get("stints") or {}).get("drivers") or {})
    pit_drivers = ((payload.get("pits") or {}).get("drivers") or {})
    raw_counts = dict(payload.get("raw_counts") or {})
    representative_laps = [
        _safe_float(row.get("representative_lap") or row.get("median_lap") or row.get("best_lap"))
        for row in lap_drivers.values()
        if isinstance(row, dict)
    ]
    representative_laps = [value for value in representative_laps if value is not None]
    median_lap = median(representative_laps) if representative_laps else None
    driver_features: dict[str, TelemetryFeatureVector] = {}
    for key, row in sorted(lap_drivers.items()):
        if not isinstance(row, dict):
            continue
        code = str(row.get("driver_code") or key).upper()
        rep_lap = _safe_float(row.get("representative_lap") or row.get("median_lap") or row.get("best_lap"))
        lap_count = int(row.get("laps") or 0)
        interval = _find_driver_row(interval_drivers, row)
        stint = _find_driver_row(stint_drivers, row)
        pit = _find_driver_row(pit_drivers, row)
        pace_delta = (rep_lap - median_lap) if rep_lap is not None and median_lap is not None else 0.0
        lap_stddev = _safe_float(row.get("lap_time_stddev")) or 0.0
        tire_delta = _summary_tire_delta(stint, row)
        confidence = _summary_confidence(row, interval, stint, pit, raw_counts, live=live)
        driver_features[code] = TelemetryFeatureVector(
            race_id=race_id,
            session=session,
            driver_code=code,
            source=str((payload.get("laps") or {}).get("source") or payload.get("source") or "openf1_summary"),
            lap=lap_count or None,
            clean_air_pace_delta_s=round(pace_delta, 4),
            pace_sigma_delta=round(min(0.5, lap_stddev / 2.0), 4),
            tire_deg_slope_delta=tire_delta,
            traffic_penalty_s=_summary_traffic_penalty(interval),
            overtake_pressure=_summary_overtake_pressure(interval, row),
            dnf_hazard_multiplier=_summary_dnf_multiplier(row, pit),
            confidence=confidence,
            samples=lap_count,
            raw={
                "openf1_lap_summary": row,
                "openf1_interval_summary": interval,
                "openf1_stint_summary": stint,
                "openf1_pit_summary": pit,
            },
        )

    missing = []
    for group in ("laps", "intervals", "stints", "pits", "weather", "race_control"):
        summary = payload.get(group) or {}
        count = int(raw_counts.get(group) or 0)
        if summary.get("missing_data", count == 0):
            missing.append(group)
    confidence = _payload_confidence(driver_features, missing)
    return TelemetryFeaturePayload(
        ok=bool(driver_features),
        race_id=race_id,
        session=session,
        source_mode="openf1_live" if live else "openf1_historical",
        confidence=confidence,
        driver_features=driver_features,
        segment_features=_top_segment_summaries(driver_features),
        stint_features=[
            {
                "driver_code": code,
                "tire_deg_slope_delta": feature.tire_deg_slope_delta,
                "source": "openf1_stints",
            }
            for code, feature in driver_features.items()
            if feature.tire_deg_slope_delta is not None
        ],
        raw_counts=raw_counts,
        missing_groups=missing,
        data_quality={
            "live": bool(live),
            "estimated": False,
            "summary_only": True,
            "session_key": payload.get("session_key"),
            "meeting_key": payload.get("meeting_key"),
            "driver_count": len(driver_features),
        },
    )


def _lap_times_by_driver(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        code = str(_first(row, "driver_code", "Driver", "driver") or "").upper()
        value = _seconds(_first(row, "lap_time_s", "lap_duration", "LapTime", "lap_time"))
        if code and value is not None and 40.0 <= value <= 180.0:
            result[code].append(value)
    return result


def _pace_delta(values: list[float], median_lap: float | None) -> float:
    if not values or median_lap is None:
        return 0.0
    return mean(values[-3:]) - median_lap


def _pace_sigma_delta(laps: list[float], speeds: list[float]) -> float:
    if len(laps) >= 2:
        return min(0.4, pstdev(laps) / 2.0)
    if len(speeds) >= 3:
        return min(0.4, pstdev(speeds) / 60.0)
    return 0.12


def _proxy_pace_delta(
    top_speed_delta: float | None,
    corner_delta: float | None,
    full_throttle_pct: float | None,
    traction: float | None,
) -> float:
    delta = 0.0
    if top_speed_delta is not None:
        delta -= max(-0.25, min(0.25, top_speed_delta * 0.012))
    if corner_delta is not None:
        delta -= max(-0.35, min(0.35, corner_delta * 0.018))
    if full_throttle_pct is not None:
        delta -= (full_throttle_pct - 0.55) * 0.22
    if traction is not None:
        delta -= (traction - 0.5) * 0.18
    return max(-0.65, min(0.65, delta))


def _low_speed_proxy(points: list[TelemetryTracePoint]) -> float | None:
    speeds = sorted(p.speed_kph for p in points if p.speed_kph is not None and p.speed_kph > 0)
    if not speeds:
        return None
    cutoff = max(1, int(len(speeds) * 0.15))
    return mean(speeds[:cutoff])


def _traffic_penalty_by_driver(rows: list[dict[str, Any]]) -> dict[str, float]:
    penalties: dict[str, float] = {}
    for row in rows:
        code = str(_first(row, "driver_code", "Driver", "driver") or "").upper()
        interval = _seconds(_first(row, "interval", "gap_ahead_s", "IntervalToPositionAhead"))
        if not code or interval is None:
            continue
        if 0.0 <= interval <= 1.5:
            penalties[code] = max(penalties.get(code, 0.0), round((1.5 - interval) * 0.24, 4))
    return penalties


def _stint_deg_by_driver(stints: list[dict[str, Any]], lap_times: dict[str, list[float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in stints:
        code = str(_first(row, "driver_code", "Driver", "driver") or "").upper()
        if not code:
            continue
        explicit = _safe_float(_first(row, "deg_slope", "tire_deg_slope_delta"))
        if explicit is not None:
            result[code] = explicit
            continue
        values = lap_times.get(code) or []
        if len(values) >= 4:
            result[code] = round((mean(values[-2:]) - mean(values[:2])) / max(1, len(values) - 1), 4)
    return result


def _traction_score(points: list[TelemetryTracePoint]) -> float | None:
    usable = [p for p in points if p.speed_kph is not None and p.throttle_pct is not None]
    if not usable:
        return None
    low_speed = [p.throttle_pct for p in usable if p.speed_kph is not None and p.speed_kph <= 150.0]
    if not low_speed:
        return None
    return round(max(0.0, min(1.0, mean(low_speed) / 100.0)), 4)


def _stability_score(speeds: list[float]) -> float | None:
    if len(speeds) < 4:
        return None
    volatility = pstdev(speeds) / max(mean(speeds), 1.0)
    return round(max(0.0, min(1.0, 1.0 - volatility)), 4)


def _overtake_pressure(top_speed_delta: float | None, traffic_penalty: float | None) -> float | None:
    if top_speed_delta is None and traffic_penalty is None:
        return None
    score = 0.0
    if top_speed_delta is not None:
        score += max(0.0, min(0.7, top_speed_delta / 22.0))
    if traffic_penalty is not None:
        score += max(0.0, min(0.5, traffic_penalty / 0.36))
    return round(max(0.0, min(1.0, score)), 4)


def _dnf_multiplier(stability: float | None, brakes: list[float]) -> float:
    multiplier = 1.0
    if stability is not None and stability < 0.55:
        multiplier += (0.55 - stability) * 0.50
    if brakes and _ratio(brakes, lambda value: value >= 95.0) > 0.28:
        multiplier += 0.08
    return round(max(0.75, min(1.5, multiplier)), 4)


def _driver_confidence(
    points: list[TelemetryTracePoint],
    lap_times: list[float],
    stints: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
    *,
    live: bool,
) -> float:
    score = 0.18
    if len(points) >= 8:
        score += 0.24
    if any(p.speed_kph is not None for p in points):
        score += 0.16
    if any(p.throttle_pct is not None for p in points):
        score += 0.08
    if any(p.x is not None and p.y is not None for p in points):
        score += 0.12
    if lap_times:
        score += 0.16
    if stints:
        score += 0.05
    if intervals:
        score += 0.05
    if live:
        score = min(score, 0.90)
    return round(max(0.0, min(1.0, score)), 4)


def _payload_confidence(features: dict[str, TelemetryFeatureVector], missing: list[str]) -> float:
    if not features:
        return 0.0
    base = mean([item.confidence for item in features.values()])
    penalty = min(0.25, len(missing) * 0.04)
    return round(max(0.0, min(1.0, base - penalty)), 4)


def _missing_groups(
    points: list[TelemetryTracePoint],
    laps: list[dict[str, Any]],
    stints: list[dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> list[str]:
    missing: list[str] = []
    if not points:
        missing.append("car_data")
    if points and not any(p.x is not None and p.y is not None for p in points):
        missing.append("location")
    if not laps:
        missing.append("laps")
    if not stints:
        missing.append("stints")
    if not intervals:
        missing.append("intervals")
    return missing


def _top_segment_summaries(features: dict[str, TelemetryFeatureVector]) -> list[dict[str, Any]]:
    rows = []
    for code, feature in features.items():
        rows.append(
            {
                "driver_code": code,
                "kind": "pace_delta",
                "magnitude": abs(feature.clean_air_pace_delta_s),
                "label": "faster" if feature.clean_air_pace_delta_s < 0 else "slower",
            }
        )
    return sorted(rows, key=lambda row: row["magnitude"], reverse=True)[:6]


def _dominant_source(points: list[TelemetryTracePoint]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for point in points:
        counts[point.source] += 1
    return max(counts, key=counts.get) if counts else "unknown"


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return round(value - baseline, 4)


def _ratio(values: list[float], predicate) -> float:
    if not values:
        return 0.0
    return round(sum(1 for value in values if predicate(value)) / len(values), 4)


def _seconds(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    if isinstance(value, str) and ":" in value:
        try:
            minutes, seconds = value.split(":", 1)
            return int(minutes) * 60.0 + float(seconds)
        except ValueError:
            return None
    return _safe_float(value)


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _find_driver_row(rows: dict[str, Any], lap_row: dict[str, Any]) -> dict[str, Any]:
    number = str(lap_row.get("driver_number") or "")
    code = str(lap_row.get("driver_code") or "").upper()
    if number and isinstance(rows.get(number), dict):
        return rows.get(number) or {}
    for row in rows.values():
        if isinstance(row, dict) and str(row.get("driver_code") or "").upper() == code:
            return row
    return {}


def _summary_tire_delta(stint: dict[str, Any], lap_row: dict[str, Any]) -> float | None:
    if not stint:
        return None
    final_stint = _safe_float(stint.get("final_stint_laps") or stint.get("estimated_tyre_age"))
    stability = _safe_float(lap_row.get("pace_stability"))
    if final_stint is None:
        return None
    age_component = max(0.0, final_stint - 14.0) * 0.003
    stability_component = max(0.0, 0.65 - (stability if stability is not None else 0.65)) * 0.035
    return round(min(0.09, age_component + stability_component), 4)


def _summary_traffic_penalty(interval: dict[str, Any]) -> float | None:
    gap = _seconds(interval.get("interval")) if interval else None
    if gap is None:
        return None
    if 0.0 <= gap <= 1.5:
        return round((1.5 - gap) * 0.24, 4)
    return 0.0


def _summary_overtake_pressure(interval: dict[str, Any], lap_row: dict[str, Any]) -> float | None:
    traffic = _summary_traffic_penalty(interval)
    stability = _safe_float(lap_row.get("pace_stability"))
    if traffic is None and stability is None:
        return None
    score = 0.0
    if traffic is not None:
        score += min(0.65, traffic / 0.36)
    if stability is not None:
        score += max(0.0, min(0.35, (stability - 0.5) * 0.7))
    return round(max(0.0, min(1.0, score)), 4)


def _summary_dnf_multiplier(lap_row: dict[str, Any], pit_row: dict[str, Any]) -> float:
    stability = _safe_float(lap_row.get("pace_stability"))
    multiplier = 1.0
    if stability is not None and stability < 0.50:
        multiplier += (0.50 - stability) * 0.35
    pit_duration = _safe_float(pit_row.get("avg_pit_duration")) if pit_row else None
    if pit_duration is not None and pit_duration > 5.0:
        multiplier += min(0.12, (pit_duration - 5.0) * 0.01)
    return round(max(0.80, min(1.35, multiplier)), 4)


def _summary_confidence(
    lap_row: dict[str, Any],
    interval: dict[str, Any],
    stint: dict[str, Any],
    pit: dict[str, Any],
    raw_counts: dict[str, int],
    *,
    live: bool,
) -> float:
    laps = int(lap_row.get("laps") or 0)
    score = 0.18 + min(0.28, laps / 20.0 * 0.28)
    if lap_row.get("representative_lap") is not None:
        score += 0.15
    if lap_row.get("sector_coverage"):
        score += 0.08 * float(lap_row.get("sector_coverage") or 0.0)
    if interval:
        score += 0.08
    if stint:
        score += 0.08
    if pit:
        score += 0.03
    if int(raw_counts.get("weather") or 0) > 0:
        score += 0.04
    if int(raw_counts.get("race_control") or 0) > 0:
        score += 0.04
    return round(max(0.0, min(0.90 if live else 0.82, score)), 4)
