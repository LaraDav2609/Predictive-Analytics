"""Stage-aware feature leakage guard for F1 predictions and backtests."""

from __future__ import annotations

from typing import Any


ALLOWED_BY_STAGE: dict[str, set[str]] = {
    "pre_weekend": {"metadata", "historical", "performance", "track", "weather_forecast", "reliability", "tires", "sentiment_pre_race", "market_pre_race"},
    "practice_available": {"metadata", "historical", "performance", "track", "weather_forecast", "reliability", "tires", "sentiment_pre_race", "market_pre_race", "practice_pace"},
    "post_qualifying": {"metadata", "historical", "performance", "track", "weather_forecast", "reliability", "tires", "sentiment_pre_race", "market_pre_race", "practice_pace", "qualifying", "grid", "penalties"},
    "live": {"metadata", "historical", "performance", "track", "weather_live", "reliability", "tires_live", "sentiment_session", "market_live", "practice_pace", "qualifying", "grid", "penalties", "live_timing", "race_control"},
    "completed": {"official_results", "classification", "historical"},
}

FUTURE_GROUPS_BY_STAGE: dict[str, set[str]] = {
    "pre_weekend": {"practice_pace", "qualifying", "grid", "live_timing", "race_control", "official_results", "classification", "post_race_sentiment"},
    "practice_available": {"qualifying", "grid", "live_timing", "race_control", "official_results", "classification", "post_race_sentiment"},
    "post_qualifying": {"live_timing", "race_control", "official_results", "classification", "post_race_sentiment"},
    "live": {"official_results", "classification", "post_race_sentiment"},
    "completed": set(),
}


def leakage_guard_status(
    *,
    stage: str,
    truth: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    feature_groups: list[str] | None = None,
) -> dict[str, Any]:
    truth = truth or {}
    profile = profile or {}
    stage_key = stage if stage in ALLOWED_BY_STAGE else "pre_weekend"
    observed = set(feature_groups or _observed_groups(truth, profile))
    blocked = sorted(observed & FUTURE_GROUPS_BY_STAGE[stage_key])
    warnings = []
    if blocked:
        warnings.append({
            "type": "future_feature_blocked",
            "message": "Potential future-session features detected for this stage and should be excluded from training/backtests.",
            "groups": blocked,
        })
    if stage_key != "completed" and truth.get("status") in {"race_classification_available", "completed_missing_classification"}:
        warnings.append({
            "type": "completed_status_before_completed_stage",
            "message": "Completed-race status appeared before completed stage; replay should verify cutoff time.",
        })
    return {
        "status": "blocked" if blocked else "ok",
        "stage": stage_key,
        "allowed_feature_groups": sorted(ALLOWED_BY_STAGE[stage_key]),
        "blocked_feature_groups": blocked,
        "warnings": warnings,
    }


def _observed_groups(truth: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    groups = ["metadata", "historical", "performance", "track", "reliability", "tires"]
    status = truth.get("status") or ""
    if (truth.get("signals") or {}).get("weather"):
        groups.append("weather_live" if truth.get("source_mode") == "live" else "weather_forecast")
    if truth.get("sentiment_article_count") or truth.get("sentiment_impact"):
        groups.append("sentiment_session" if truth.get("source_mode") == "live" else "sentiment_pre_race")
    if profile.get("qualifying"):
        groups.extend(["qualifying", "grid"])
    if profile.get("sessions"):
        for session in profile.get("sessions") or []:
            name = str(session.get("name") or "").lower()
            if "practice" in name and session.get("status") == "completed":
                groups.append("practice_pace")
    if truth.get("source_mode") in {"live", "recent", "recorded"}:
        groups.append("live_timing")
    if (truth.get("signals") or {}).get("race_control"):
        groups.append("race_control")
    if "classification_available" in status or profile.get("results"):
        groups.extend(["official_results", "classification"])
    return groups
