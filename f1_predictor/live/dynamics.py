"""Live race dynamics extracted from truth/live state.

The prediction model already has strong pre-race inputs. This module adds a
bounded live-race layer so timing facts can move probabilities without letting
noisy or estimated data dominate the model.
"""

from __future__ import annotations

from statistics import median
from typing import Any


def build_live_dynamics(
    truth: dict[str, Any] | None,
    *,
    session: str = "race",
    track: dict[str, Any] | None = None,
    tires: dict[str, Any] | None = None,
) -> dict[str, Any]:
    truth = truth or {}
    session = (session or "race").lower()
    if session != "race":
        return _empty("live_dynamics_race_only")

    rows = truth.get("drivers") or list((truth.get("by_driver_id") or {}).values())
    rows = [row for row in rows if row.get("driver_id")]
    if not rows:
        return _empty("no_live_truth_rows")

    source_mode = truth.get("source_mode") or "estimated"
    confidence = _source_confidence(source_mode, truth.get("confidence"))
    if confidence <= 0.02:
        return _empty("live_source_unavailable", source_mode=source_mode)

    field_size = max(1, len(rows))
    best_pace_values = [_seconds(row.get("representative_lap") or row.get("median_lap") or row.get("best_lap")) for row in rows]
    best_pace_values = [value for value in best_pace_values if value is not None and value > 0]
    field_pace = median(best_pace_values) if best_pace_values else None
    expected_pit_stops = _expected_pit_stops(track, tires)

    by_driver: dict[str, dict[str, Any]] = {}
    for row in rows:
        driver_id = str(row.get("driver_id"))
        position = _int(row.get("position"))
        position_score = _position_score(position, field_size)
        gap_seconds = _gap_seconds(row.get("gap_to_leader"))
        interval_seconds = _gap_seconds(row.get("interval"))
        pace_seconds = _seconds(row.get("representative_lap") or row.get("median_lap") or row.get("best_lap"))
        pace_delta = _pace_delta(pace_seconds, field_pace)
        compound = str(row.get("compound") or "").upper()
        tyre_age = _int(row.get("tyre_age"))
        pit_stops = _int(row.get("pit_stops"))
        evidence_score = _evidence_score(row, source_mode, confidence)
        tyre_delta = _tyre_delta(compound, tyre_age, track, tires)
        pit_delta = _pit_delta(pit_stops, expected_pit_stops)
        gap_delta = _gap_delta(position, gap_seconds, interval_seconds, field_size)
        live_delta = position_score + gap_delta + pace_delta + tyre_delta + pit_delta
        live_delta = _clamp(live_delta * confidence, -0.18, 0.18)
        strength_multiplier = round(_clamp(1.0 + live_delta, 0.78, 1.22), 4)
        by_driver[driver_id] = {
            "driver_id": driver_id,
            "position": position,
            "gap_seconds": gap_seconds,
            "interval_seconds": interval_seconds,
            "pace_seconds": pace_seconds,
            "pace_delta": round(pace_delta * confidence, 4),
            "tyre_delta": round(tyre_delta * confidence, 4),
            "pit_delta": round(pit_delta * confidence, 4),
            "gap_delta": round(gap_delta * confidence, 4),
            "position_delta": round(position_score * confidence, 4),
            "live_delta": round(live_delta, 4),
            "strength_multiplier": strength_multiplier,
            "time_anchor_seconds": round(_time_anchor_seconds(position, gap_seconds, interval_seconds, confidence), 4),
            "compound": compound or None,
            "tyre_age": tyre_age,
            "pit_stops": pit_stops,
            "evidence_score": evidence_score,
            "confidence": round(confidence, 4),
            "source_mode": source_mode,
            "explanations": _explanations(position, gap_seconds, pace_delta, tyre_delta, pit_delta, source_mode),
        }

    return {
        "ok": True,
        "source_mode": source_mode,
        "confidence": round(confidence, 4),
        "field_pace_seconds": round(field_pace, 4) if field_pace is not None else None,
        "expected_pit_stops": expected_pit_stops,
        "driver_count": len(by_driver),
        "drivers": by_driver,
        "summary": _summary(by_driver),
    }


def _empty(reason: str, source_mode: str = "unavailable") -> dict[str, Any]:
    return {
        "ok": False,
        "reason": reason,
        "source_mode": source_mode,
        "confidence": 0.0,
        "driver_count": 0,
        "drivers": {},
        "summary": [],
    }


def _source_confidence(source_mode: str, value: Any) -> float:
    try:
        base = float(value)
    except (TypeError, ValueError):
        base = {
            "live": 0.88,
            "recorded_confident": 0.80,
            "recorded": 0.76,
            "recording_pending": 0.18,
            "recent": 0.62,
            "historical": 0.58,
            "estimated": 0.24,
        }.get(source_mode, 0.0)
    source_cap = {
        "live": 0.95,
        "recorded_confident": 0.85,
        "recorded": 0.82,
        "recording_pending": 0.25,
        "recent": 0.68,
        "historical": 0.64,
        "estimated": 0.30,
        "unavailable": 0.0,
    }.get(source_mode, 0.30)
    return _clamp(base, 0.0, source_cap)


def _position_score(position: int | None, field_size: int) -> float:
    if position is None:
        return 0.0
    normalized = 1.0 - ((position - 1) / max(1, field_size - 1))
    return (normalized - 0.50) * 0.18


def _gap_delta(position: int | None, gap: float | None, interval: float | None, field_size: int) -> float:
    if position is None:
        return 0.0
    if position == 1:
        return 0.035
    if gap is None:
        return 0.0
    if gap <= 1.5:
        return 0.030
    if gap <= 5.0:
        return 0.015
    if gap >= 30.0:
        return -0.040
    penalty = -min(0.035, gap / max(30.0, field_size * 1.8) * 0.04)
    if interval is not None and interval <= 1.0:
        penalty += 0.012
    return penalty


def _pace_delta(pace: float | None, field_pace: float | None) -> float:
    if pace is None or field_pace is None or pace <= 0:
        return 0.0
    delta = field_pace - pace
    return _clamp(delta * 0.025, -0.035, 0.035)


def _tyre_delta(compound: str, tyre_age: int | None, track: dict[str, Any] | None, tires: dict[str, Any] | None) -> float:
    if not compound or tyre_age is None:
        return 0.0
    degradation = float((tires or {}).get("degradation_rate") or (track or {}).get("tire_stress") or 0.50)
    age_pressure = max(0.0, tyre_age - 12) / 28.0
    compound_factor = {
        "SOFT": 1.25,
        "S": 1.25,
        "MEDIUM": 0.90,
        "M": 0.90,
        "HARD": 0.58,
        "H": 0.58,
        "INTERMEDIATE": 1.05,
        "WET": 1.15,
    }.get(compound, 0.90)
    return -_clamp(age_pressure * degradation * compound_factor * 0.055, 0.0, 0.050)


def _pit_delta(pit_stops: int | None, expected: int) -> float:
    if pit_stops is None:
        return 0.0
    if pit_stops > expected:
        return -min(0.045, (pit_stops - expected) * 0.030)
    if pit_stops == expected and expected > 0:
        return 0.012
    return 0.0


def _expected_pit_stops(track: dict[str, Any] | None, tires: dict[str, Any] | None) -> int:
    degradation = float((tires or {}).get("degradation_rate") or (track or {}).get("tire_stress") or 0.50)
    if degradation >= 0.70:
        return 2
    return 1


def _evidence_score(row: dict[str, Any], source_mode: str, confidence: float) -> dict[str, Any]:
    position = 1.0 if row.get("position") is not None and source_mode not in {"estimated", "unavailable"} else 0.0
    gap = 1.0 if (row.get("gap_to_leader") is not None or row.get("interval") is not None) and source_mode not in {"estimated", "unavailable"} else 0.0
    pace = 1.0 if (row.get("representative_lap") is not None or row.get("median_lap") is not None or row.get("best_lap") is not None) and source_mode not in {"estimated", "unavailable"} else 0.0
    tyre = 1.0 if (row.get("compound") is not None or row.get("tyre_age") is not None) and source_mode not in {"estimated", "unavailable"} else 0.0
    pit = 1.0 if row.get("pit_stops") is not None and source_mode not in {"estimated", "unavailable"} else 0.0
    score = (position * 0.26 + gap * 0.22 + pace * 0.18 + tyre * 0.16 + pit * 0.08 + confidence * 0.10)
    return {
        "overall": round(_clamp(score, 0.0, 1.0), 4),
        "position": position,
        "gap_interval": gap,
        "pace": pace,
        "tyre_stint": tyre,
        "pit_strategy": pit,
        "source_confidence": round(confidence, 4),
    }


def _time_anchor_seconds(position: int | None, gap: float | None, interval: float | None, confidence: float) -> float:
    if position is None:
        return 0.0
    order_penalty = max(0, position - 1) * 1.10
    gap_penalty = min(35.0, max(0.0, gap or 0.0)) * 0.22
    close_bonus = -0.65 if interval is not None and 0.0 < interval <= 1.0 else 0.0
    return (order_penalty + gap_penalty + close_bonus) * confidence


def _summary(by_driver: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "driver_id": row["driver_id"],
                "position": row.get("position"),
                "live_delta": row.get("live_delta"),
                "strength_multiplier": row.get("strength_multiplier"),
                "confidence": row.get("confidence"),
                "top_reason": (row.get("explanations") or ["live state"])[0],
            }
            for row in by_driver.values()
        ],
        key=lambda row: abs(float(row.get("live_delta") or 0.0)),
        reverse=True,
    )[:6]


def _explanations(
    position: int | None,
    gap: float | None,
    pace_delta: float,
    tyre_delta: float,
    pit_delta: float,
    source_mode: str,
) -> list[str]:
    reasons = []
    if position is not None:
        reasons.append(f"live running order P{position}")
    if gap is not None:
        reasons.append(f"leader gap {gap:.1f}s")
    if abs(pace_delta) >= 0.012:
        reasons.append("live pace trend gain" if pace_delta > 0 else "live pace trend loss")
    if abs(tyre_delta) >= 0.010:
        reasons.append("tyre age risk")
    if abs(pit_delta) >= 0.010:
        reasons.append("pit strategy offset")
    if source_mode in {"estimated", "unavailable"}:
        reasons.append("low-confidence live source discounted")
    return reasons[:5]


def _gap_seconds(value: Any) -> float | None:
    if value in {None, "", "null"}:
        return None
    text = str(value).strip().upper().replace("+", "")
    if not text:
        return None
    if "LAP" in text:
        return 90.0
    try:
        return float(text)
    except ValueError:
        return None


def _seconds(value: Any) -> float | None:
    if value in {None, "", "null"}:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        try:
            minutes, seconds = text.split(":", 1)
            return float(minutes) * 60.0 + float(seconds)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
