"""Source disagreement detection for F1 probability governance."""

from __future__ import annotations

from typing import Any


def detect_source_disagreement(
    truth: dict[str, Any] | None,
    *,
    probabilities: list[dict[str, Any]] | None = None,
    market: dict[str, Any] | None = None,
    sentiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    truth = truth or {}
    probabilities = probabilities or []
    items: list[dict[str, Any]] = []

    _prediction_vs_truth(items, truth, probabilities)
    _market_disagreement(items, probabilities, market or {})
    _sentiment_disagreement(items, probabilities, sentiment or truth.get("sentiment_impact") or {})
    _weather_disagreement(items, truth)

    score = min(1.0, sum(float(item.get("severity") or 0.0) for item in items))
    return {
        "score": round(score, 4),
        "label": _label(score),
        "items": items[:8],
    }


def _prediction_vs_truth(items: list[dict[str, Any]], truth: dict[str, Any], probabilities: list[dict[str, Any]]) -> None:
    rows = truth.get("drivers") or []
    if not rows or not probabilities:
        return
    truth_order = sorted(rows, key=lambda item: int(item.get("position") or 99))
    probability_order = sorted(probabilities, key=lambda item: float(item.get("win_probability") or item.get("calibrated_probability") or item.get("win_prob") or 0.0), reverse=True)
    truth_leader = str((truth_order[0] or {}).get("driver_id") or "")
    model_leader = str((probability_order[0] or {}).get("driver_id") or "")
    source_mode = truth.get("source_mode")
    if truth_leader and model_leader and truth_leader != model_leader and source_mode in {"live", "recent", "historical", "recorded"}:
        truth_pos = _position_of(probability_order, truth_leader)
        severity = 0.08 if truth_pos <= 3 else 0.16
        items.append({
            "type": "model_vs_truth_order",
            "severity": severity,
            "message": "Model leader differs from the latest timing/result leader.",
            "truth_driver_id": truth_leader,
            "model_driver_id": model_leader,
        })


def _market_disagreement(items: list[dict[str, Any]], probabilities: list[dict[str, Any]], market: dict[str, Any]) -> None:
    if not market or not probabilities:
        return
    market_leader = str(market.get("top_driver_id") or market.get("leader_driver_id") or "")
    model_leader = str(probabilities[0].get("driver_id") or "") if probabilities else ""
    market_confidence = float(market.get("confidence") or 0.0)
    if market_leader and model_leader and market_leader != model_leader and market_confidence > 0.45:
        items.append({
            "type": "market_vs_model",
            "severity": 0.06,
            "message": "Market consensus and model leader disagree; market signal remains capped.",
            "market_driver_id": market_leader,
            "model_driver_id": model_leader,
        })


def _sentiment_disagreement(items: list[dict[str, Any]], probabilities: list[dict[str, Any]], sentiment: dict[str, Any]) -> None:
    if not sentiment:
        return
    confidence = float(sentiment.get("confidence") or sentiment.get("sentiment_confidence") or 0.0)
    articles = int(sentiment.get("article_count") or sentiment.get("sentiment_article_count") or 0)
    sources = int(sentiment.get("source_count") or sentiment.get("sentiment_source_count") or 0)
    if articles >= 8 and sources <= 2:
        items.append({
            "type": "sentiment_source_concentration",
            "severity": 0.07,
            "message": "Many news items come from few sources; sentiment impact is treated as noisy.",
            "articles": articles,
            "sources": sources,
        })
    if articles and confidence < 0.35:
        items.append({
            "type": "sentiment_low_confidence",
            "severity": 0.05,
            "message": "Race sentiment coverage is low confidence and remains bounded.",
            "confidence": round(confidence, 4),
        })


def _weather_disagreement(items: list[dict[str, Any]], truth: dict[str, Any]) -> None:
    weather = (truth.get("signals") or {}).get("weather") or {}
    if not weather:
        return
    source = str(weather.get("source") or "").lower()
    confidence = float(weather.get("confidence") or 0.0)
    chaos = float(weather.get("chaos_score") or 0.0)
    if "fallback" in source and chaos > 0.05:
        items.append({
            "type": "weather_fallback_active",
            "severity": 0.05,
            "message": "Weather fallback is active, so weather influence is capped.",
        })
    if chaos > 0.25 and confidence < 0.55:
        items.append({
            "type": "weather_chaos_low_confidence",
            "severity": 0.08,
            "message": "Weather volatility exists but source confidence is limited.",
            "chaos_score": round(chaos, 4),
            "confidence": round(confidence, 4),
        })


def _position_of(rows: list[dict[str, Any]], driver_id: str) -> int:
    for index, row in enumerate(rows, start=1):
        if str(row.get("driver_id") or "") == driver_id:
            return index
    return 99


def _label(score: float) -> str:
    if score >= 0.25:
        return "high"
    if score >= 0.12:
        return "moderate"
    if score > 0:
        return "low"
    return "aligned"
