"""Circuit archetypes and per-driver archetype-fit ratings.

Classifies each circuit into archetype tags from the real per-circuit ``TRACK_TRAITS``
registry (street / high-speed / low-speed / tire-limited / low-overtake /
high-safety-car / qualifying-critical, plus optional sprint / wet), then aggregates
a driver's existing per-track history (the ``track_score`` already computed by the
backtest replay layer) into per-archetype ratings. A driver's ``archetype_fit_score``
for an upcoming race is the start-weighted blend of their ratings for that race's
archetypes — "how well does this driver historically go at circuits like this one".

Pure + dependency-light (only the trait registry), so it is reusable and testable.
"""
from __future__ import annotations

from typing import Any

from sports.f1.predictor.features.track import ALIASES, TRACK_TRAITS

ARCHETYPES = (
    "street", "high_speed", "low_speed", "tire_limited",
    "low_overtake", "high_safety_car", "qualifying_critical", "sprint", "wet",
)


def classify_archetypes(traits: dict[str, Any], *, is_sprint: bool = False, is_wet: bool = False) -> list[str]:
    """Archetype tags for a circuit, from its trait scalars."""
    traits = traits or {}
    tags: list[str] = []
    if traits.get("street_circuit"):
        tags.append("street")
    if traits.get("high_speed"):
        tags.append("high_speed")
    else:
        tags.append("low_speed")
    if float(traits.get("tire_stress") or 0.0) >= 0.65:
        tags.append("tire_limited")
    if float(traits.get("overtaking_difficulty") or 0.0) >= 0.70:
        tags.append("low_overtake")
    if float(traits.get("safety_car_probability") or 0.0) >= 0.50:
        tags.append("high_safety_car")
    if float(traits.get("qualifying_importance") or 0.0) >= 0.78:
        tags.append("qualifying_critical")
    if is_sprint:
        tags.append("sprint")
    if is_wet:
        tags.append("wet")
    return tags


def resolve_track_key(text: str | None) -> str:
    """Resolve a free-text race/circuit/country name to a TRACK_TRAITS key."""
    low = (text or "").lower()
    for key, aliases in ALIASES.items():
        if key in low or any(alias in low for alias in aliases):
            return key
    return "default"


def archetypes_for_track(text: str | None, *, is_sprint: bool = False, is_wet: bool = False) -> list[str]:
    key = resolve_track_key(text)
    return classify_archetypes(TRACK_TRAITS.get(key) or {}, is_sprint=is_sprint, is_wet=is_wet)


def archetype_ratings(track_history: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate per-track driver scores into per-(driver, archetype) ratings.

    ``track_history`` is the ``_track_history_features`` shape:
    ``{track_key: {"drivers": {driver_id: {"track_score": float, "starts": int, ...}}}}``.
    Returns ``{driver_id: {archetype: {"rating": float, "tracks": int, "starts": int}}}``
    where rating is the start-weighted mean track_score across that archetype's circuits.
    """
    acc: dict[str, dict[str, dict[str, float]]] = {}
    for track_key, block in (track_history or {}).items():
        tags = classify_archetypes(TRACK_TRAITS.get(track_key) or {})
        if not tags:
            continue
        drivers = (block or {}).get("drivers") or {}
        for driver_id, stats in drivers.items():
            score = float((stats or {}).get("track_score") or 0.0)
            starts = int((stats or {}).get("starts") or 0)
            weight = max(1, starts)
            for tag in tags:
                d = acc.setdefault(str(driver_id), {}).setdefault(
                    tag, {"score_sum": 0.0, "weight": 0.0, "tracks": 0.0, "starts": 0.0})
                d["score_sum"] += score * weight
                d["weight"] += weight
                d["tracks"] += 1
                d["starts"] += starts

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for driver_id, archs in acc.items():
        out[driver_id] = {}
        for tag, d in archs.items():
            rating = d["score_sum"] / d["weight"] if d["weight"] else 0.0
            out[driver_id][tag] = {
                "rating": round(rating, 4),
                "tracks": int(d["tracks"]),
                "starts": int(d["starts"]),
            }
    return out


def archetype_fit_score(driver_ratings: dict[str, dict[str, Any]] | None,
                        race_archetypes: list[str]) -> float | None:
    """Start-weighted blend of a driver's ratings for the race's archetypes.
    Returns None when the driver has no history in any of the race's archetypes."""
    driver_ratings = driver_ratings or {}
    relevant = [driver_ratings[a] for a in race_archetypes if a in driver_ratings and driver_ratings[a]]
    total_w = sum(max(1, int(r.get("starts", 0))) for r in relevant)
    if not relevant or not total_w:
        return None
    score = sum(float(r["rating"]) * max(1, int(r.get("starts", 0))) for r in relevant) / total_w
    return round(score, 4)
