"""Data quality and bias governance for F1 predictions."""

from __future__ import annotations

from typing import Any

from sports.f1.predictor.probability.stage import detect_stage

from .bias_audit import audit_biases
from .disagreement import detect_source_disagreement
from .leakage_guard import leakage_guard_status
from .quality_score import build_influence_caps, build_quality_score
from .source_registry import get_source_policy, policy_payload


def build_data_quality_report(
    *,
    truth: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    probabilities: list[dict[str, Any]] | None = None,
    stage: str | None = "auto",
    live: bool = False,
    market: dict[str, Any] | None = None,
    sentiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reusable source-quality, disagreement, and bias report."""

    truth = truth or {}
    profile = profile or {}
    probabilities = probabilities or []
    detected_stage = detect_stage(profile, truth, live=live, requested_stage=stage)
    quality = build_quality_score(truth, probabilities=probabilities)
    disagreement = detect_source_disagreement(
        truth,
        probabilities=probabilities,
        market=market,
        sentiment=sentiment,
    )
    leakage = leakage_guard_status(stage=detected_stage, truth=truth, profile=profile)
    caps = build_influence_caps(truth, quality)
    warnings = audit_biases(
        truth=truth,
        quality=quality,
        disagreement=disagreement,
        probabilities=probabilities,
        stage=detected_stage,
    )
    report = {
        "ok": True,
        "stage": detected_stage,
        "data_quality_score": quality["score"],
        "data_quality_label": quality["label"],
        "confidence_multiplier": caps["quality_confidence_multiplier"],
        "source_mode": truth.get("source_mode") or "unavailable",
        "active_source": quality["active_source"],
        "source_coverage": quality["source_coverage"],
        "source_disagreement": disagreement,
        "bias_warnings": warnings,
        "influence_caps_applied": caps,
        "leakage_guard_status": leakage,
        "quality_components": quality,
        "estimated_data_discounted": bool(caps.get("estimated_source_discount")),
    }
    return report


def attach_data_quality(
    payload: dict[str, Any],
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach data-quality fields in a backwards-compatible shape."""

    report = report or {}
    if not report:
        return payload
    payload["data_quality"] = report
    payload["data_quality_score"] = report.get("data_quality_score")
    payload["source_coverage"] = report.get("source_coverage") or {}
    payload["source_disagreement"] = report.get("source_disagreement") or {}
    payload["bias_warnings"] = report.get("bias_warnings") or []
    payload["influence_caps_applied"] = report.get("influence_caps_applied") or {}
    payload["leakage_guard_status"] = report.get("leakage_guard_status") or {}
    return payload


__all__ = [
    "attach_data_quality",
    "build_data_quality_report",
    "get_source_policy",
    "policy_payload",
]
