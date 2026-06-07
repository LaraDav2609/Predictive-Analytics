"""Data quality scoring for F1 race truth and probability payloads."""

from __future__ import annotations

from typing import Any

from .source_registry import get_source_policy, policy_payload


def build_source_coverage(truth: dict[str, Any]) -> dict[str, Any]:
    sources = truth.get("sources") or {}
    chain = truth.get("source_chain") or []
    coverage: dict[str, Any] = {}
    for name in chain:
        policy = policy_payload(name)
        status = "available" if _source_available(name, sources, truth) else "unavailable"
        coverage[name] = {
            **policy,
            "status": status,
            "active": name == truth.get("last_successful_source"),
        }
    for name, status in sources.items():
        key = _coverage_key(name, status)
        if key not in coverage:
            coverage[key] = {
                **policy_payload(key),
                "status": "available" if status and str(status).lower() not in {"unavailable", "false", "none"} else "unavailable",
                "active": key == truth.get("last_successful_source"),
            }
    return coverage


def build_quality_score(
    truth: dict[str, Any] | None,
    *,
    probabilities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    truth = truth or {}
    probabilities = probabilities or []
    source_mode = truth.get("source_mode") or "unavailable"
    policy = get_source_policy(source_mode)
    source_coverage = build_source_coverage(truth)
    available = sum(1 for item in source_coverage.values() if item.get("status") == "available")
    total = max(1, len(source_coverage))
    coverage_score = available / total
    truth_confidence = _clamp(float(truth.get("confidence") or 0.0))
    freshness_score = _freshness_score(truth, policy.freshness_seconds)
    driver_confidence = _driver_confidence(truth)
    missing_penalty = min(0.35, len(truth.get("missing_groups") or []) * 0.035)
    fallback_penalty = 0.10 if truth.get("fallback_reason") else 0.0
    estimate_penalty = 0.18 if truth.get("is_estimated") or source_mode in {"estimated", "unavailable"} else 0.0
    overconfidence_penalty = _overconfidence_penalty(probabilities, truth_confidence, coverage_score)
    score = (
        truth_confidence * 0.34
        + coverage_score * 0.22
        + freshness_score * 0.18
        + policy.reliability * 0.16
        + driver_confidence * 0.10
        - missing_penalty
        - fallback_penalty
        - estimate_penalty
        - overconfidence_penalty
    )
    score = _clamp(score)
    return {
        "score": round(score, 4),
        "label": _quality_label(score),
        "source_mode": source_mode,
        "active_source": policy_payload(source_mode),
        "source_coverage": source_coverage,
        "coverage_score": round(coverage_score, 4),
        "freshness_score": round(freshness_score, 4),
        "driver_confidence": round(driver_confidence, 4),
        "truth_confidence": round(truth_confidence, 4),
        "missing_penalty": round(missing_penalty, 4),
        "fallback_penalty": round(fallback_penalty, 4),
        "estimated_penalty": round(estimate_penalty, 4),
        "overconfidence_penalty": round(overconfidence_penalty, 4),
    }


def build_influence_caps(truth: dict[str, Any] | None, quality: dict[str, Any]) -> dict[str, Any]:
    truth = truth or {}
    source_mode = truth.get("source_mode") or "unavailable"
    source_policy = get_source_policy(source_mode)
    quality_score = float(quality.get("score") or 0.0)
    estimated_discount = 0.35 if source_mode == "estimated" else 0.0
    unavailable_discount = 0.65 if source_mode == "unavailable" else 0.0
    disagreement_discount = 0.15 if quality_score < 0.45 else 0.0
    return {
        "live_position": round(source_policy.max_influence * max(0.2, quality_score), 4),
        "sentiment": 0.035,
        "market_consensus": 0.045,
        "weather": 0.08 if _weather_confirmed(truth) else 0.035,
        "estimated_source_discount": round(min(0.85, estimated_discount + unavailable_discount + disagreement_discount), 4),
        "quality_confidence_multiplier": round(max(0.35, min(1.0, 0.55 + quality_score * 0.55)), 4),
    }


def _source_available(name: str, sources: dict[str, Any], truth: dict[str, Any]) -> bool:
    if name == truth.get("last_successful_source"):
        return True
    aliases = {
        "f1_client_official_results": "official_results",
        "openf1_session_facts": "openf1_session",
        "live_session_engine": "live_state",
        "estimated_standings_order": "source_mode",
    }
    key = aliases.get(name, name)
    value = sources.get(key) if key != "source_mode" else truth.get("source_mode")
    return bool(value) and str(value).lower() not in {"unavailable", "none", "false"}


def _coverage_key(name: str, status: Any) -> str:
    if name == "weather" and status:
        return str(status) if "fallback" not in str(status).lower() else "weather_fallback"
    if name == "live_state":
        return "live_session_engine"
    if name == "openf1_session":
        return "openf1_session_facts"
    if name == "official_results":
        return "f1_client_official_results"
    return str(name)


def _freshness_score(truth: dict[str, Any], limit: int | None) -> float:
    age = truth.get("data_age_seconds")
    if age is None:
        return 0.92 if truth.get("source_mode") == "historical" else 0.62
    try:
        value = max(0.0, float(age))
    except (TypeError, ValueError):
        return 0.5
    if not limit:
        return 0.75
    return _clamp(1.0 - (value / max(1.0, float(limit))))


def _driver_confidence(truth: dict[str, Any]) -> float:
    rows = truth.get("drivers") or []
    if not rows:
        return 0.0
    values = []
    for row in rows:
        try:
            values.append(float(row.get("confidence") or 0.0))
        except (TypeError, ValueError):
            values.append(0.0)
    return _clamp(sum(values) / max(1, len(values)))


def _overconfidence_penalty(probabilities: list[dict[str, Any]], confidence: float, coverage: float) -> float:
    if not probabilities:
        return 0.0
    top = max(float(row.get("win_probability") or row.get("calibrated_probability") or row.get("win_prob") or 0.0) for row in probabilities)
    if top < 0.45:
        return 0.0
    weakness = max(0.0, 0.65 - ((confidence + coverage) / 2.0))
    return min(0.12, weakness * (top - 0.40))


def _weather_confirmed(truth: dict[str, Any]) -> bool:
    weather = (truth.get("signals") or {}).get("weather") or {}
    source = str(weather.get("source") or "").lower()
    return bool(weather) and "fallback" not in source and not weather.get("missing_data")


def _quality_label(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.55:
        return "usable"
    if score >= 0.35:
        return "thin"
    return "weak"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
