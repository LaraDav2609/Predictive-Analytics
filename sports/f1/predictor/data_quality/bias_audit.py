"""Bias warnings for source-heavy F1 prediction inputs."""

from __future__ import annotations

from typing import Any


def audit_biases(
    *,
    truth: dict[str, Any] | None = None,
    quality: dict[str, Any] | None = None,
    disagreement: dict[str, Any] | None = None,
    probabilities: list[dict[str, Any]] | None = None,
    stage: str = "pre_weekend",
) -> list[dict[str, Any]]:
    truth = truth or {}
    quality = quality or {}
    disagreement = disagreement or {}
    probabilities = probabilities or []
    warnings: list[dict[str, Any]] = []

    _source_bias(warnings, truth, quality)
    _overconfidence_bias(warnings, truth, quality, probabilities)
    _sentiment_bias(warnings, truth)
    _early_stage_bias(warnings, truth, stage)
    _disagreement_bias(warnings, disagreement)
    return warnings


def _source_bias(warnings: list[dict[str, Any]], truth: dict[str, Any], quality: dict[str, Any]) -> None:
    mode = truth.get("source_mode") or "unavailable"
    if mode in {"estimated", "unavailable"}:
        warnings.append({
            "type": "estimated_source_bias",
            "severity": "high" if mode == "unavailable" else "medium",
            "message": "Live/session facts are missing, so estimated standings/model signals are discounted.",
        })
    if len(truth.get("missing_groups") or []) >= 5:
        warnings.append({
            "type": "missing_source_coverage",
            "severity": "medium",
            "message": "Several source groups are missing; probabilities should stay flatter.",
            "missing_groups": (truth.get("missing_groups") or [])[:8],
        })
    if float(quality.get("score") or 0.0) < 0.35:
        warnings.append({
            "type": "weak_data_quality",
            "severity": "high",
            "message": "Overall data quality is weak; prediction confidence is capped.",
        })


def _overconfidence_bias(
    warnings: list[dict[str, Any]],
    truth: dict[str, Any],
    quality: dict[str, Any],
    probabilities: list[dict[str, Any]],
) -> None:
    if not probabilities:
        return
    top = max(float(row.get("win_probability") or row.get("calibrated_probability") or row.get("win_prob") or 0.0) for row in probabilities)
    if top >= 0.55 and float(quality.get("score") or 0.0) < 0.65:
        warnings.append({
            "type": "overconfidence_bias",
            "severity": "medium",
            "message": "Top win probability is high relative to source quality; calibration should flatten the board.",
            "top_probability": round(top, 4),
        })


def _sentiment_bias(warnings: list[dict[str, Any]], truth: dict[str, Any]) -> None:
    articles = int(truth.get("sentiment_article_count") or 0)
    sources = int(truth.get("sentiment_source_count") or 0)
    confidence = float(truth.get("sentiment_confidence") or 0.0)
    if articles >= 8 and sources <= 2:
        warnings.append({
            "type": "popularity_or_repost_bias",
            "severity": "medium",
            "message": "News volume is concentrated in few sources; article count cannot increase influence by itself.",
            "articles": articles,
            "sources": sources,
        })
    if articles and confidence < 0.35:
        warnings.append({
            "type": "sentiment_noise_bias",
            "severity": "low",
            "message": "Low-confidence sentiment is present but bounded below telemetry and results.",
            "confidence": round(confidence, 4),
        })


def _early_stage_bias(warnings: list[dict[str, Any]], truth: dict[str, Any], stage: str) -> None:
    round_num = int(truth.get("round") or 0)
    if round_num and round_num <= 4 and stage in {"pre_weekend", "practice_available"}:
        warnings.append({
            "type": "early_season_standings_bias",
            "severity": "low",
            "message": "Early-season standings can overstate title/race strength and remain downweighted.",
        })


def _disagreement_bias(warnings: list[dict[str, Any]], disagreement: dict[str, Any]) -> None:
    score = float(disagreement.get("score") or 0.0)
    if score > 0:
        warnings.append({
            "type": "source_disagreement",
            "severity": "low" if score < 0.12 else ("medium" if score < 0.25 else "high"),
            "message": "Important sources disagree; confidence is reduced rather than forcing one source.",
            "score": round(score, 4),
        })
