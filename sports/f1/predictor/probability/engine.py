"""Stage-aware probability audit and row enrichment."""

from __future__ import annotations

import os
from typing import Any

from sports.f1.predictor.models.configs import get_model_config
from sports.f1.predictor.probability.calibration import apply_temperature, build_calibration_profile
from sports.f1.predictor.probability.empirical_calibration import (
    CALIBRATION_ARTIFACT_ENV,
    EmpiricalCalibrator,
    load_default as _load_empirical_calibrator,
)
from sports.f1.predictor.probability.stage import detect_stage


_EMPIRICAL_STATE = None


def _get_empirical_calibrator() -> EmpiricalCalibrator:
    """Lazily load the serve-time empirical calibrator, reloading if the
    configured artifact path changes. Identity (no-op) when unconfigured."""
    global _EMPIRICAL_STATE
    path = os.environ.get(CALIBRATION_ARTIFACT_ENV)
    if _EMPIRICAL_STATE is None or _EMPIRICAL_STATE[0] != path:
        _EMPIRICAL_STATE = (path, _load_empirical_calibrator())
    return _EMPIRICAL_STATE[1]


def build_probability_audit(
    simulation: dict[str, Any],
    *,
    profile: dict[str, Any] | None = None,
    truth: dict[str, Any] | None = None,
    stage: str | None = "auto",
    live: bool = False,
    model_id: str | None = None,
) -> dict[str, Any]:
    rows = list(simulation.get("simulations") or simulation.get("probabilities") or [])
    profile = profile or {}
    truth = truth or simulation.get("truth") or {}
    detected_stage = detect_stage(profile, truth, live=live, requested_stage=stage)
    model = get_model_config(model_id or simulation.get("model_id") or "production_v1")
    calibration = build_calibration_profile(
        detected_stage,
        track=simulation.get("track_features") or simulation.get("track") or {},
        truth=truth,
        model_temperature=getattr(model, "probability_temperature", 1.0),
        stage_profiles=getattr(model, "stage_probability_profiles", {}) or {},
    )

    raw = {
        str(row.get("driver_id")): float(row.get("raw_probability") or row.get("win_probability") or row.get("win_prob") or 0.0)
        for row in rows
        if row.get("driver_id")
    }
    raw_total = sum(raw.values()) or 1.0
    normalized_raw = {driver_id: round(value / raw_total, 4) for driver_id, value in raw.items()}
    governed_raw, governance = _apply_evidence_governance(normalized_raw, detected_stage, truth)
    calibrated = apply_temperature(governed_raw, calibration.temperature)
    # Real (data-fit) calibration on top of temperature scaling. No-op unless a
    # fitted artifact is configured (F1_CALIBRATION_ARTIFACT), so serving is
    # unchanged until a calibrator is explicitly trained and provided.
    empirical = _get_empirical_calibrator()
    empirical_meta: dict[str, Any] = {"applied": False}
    if not empirical.is_identity():
        calibrated = empirical.calibrate_field(calibrated, normalize_to=1.0)
        empirical_meta = {
            "applied": True,
            "method": empirical.method,
            "source": empirical.source,
            "sample_count": empirical.sample_count,
        }

    enriched = []
    finish_distribution = {}
    for row in rows:
        driver_id = row.get("driver_id")
        if not driver_id:
            continue
        driver_id = str(driver_id)
        raw_probability = normalized_raw.get(driver_id, 0.0)
        governed_probability = governed_raw.get(driver_id, raw_probability)
        calibrated_probability = calibrated.get(driver_id, raw_probability)
        distribution = _finish_distribution(row, calibrated_probability)
        finish_distribution[driver_id] = distribution
        enriched.append({
            "driver_id": driver_id,
            "driver_code": row.get("driver_code") or row.get("code"),
            "driver_name": row.get("driver_name") or row.get("name"),
            "team": row.get("team"),
            "raw_probability": raw_probability,
            "governed_probability": governed_probability,
            "governance_delta": round(governed_probability - raw_probability, 4),
            "calibrated_probability": calibrated_probability,
            "calibration_delta": round(calibrated_probability - raw_probability, 4),
            "finish_distribution": distribution,
            "probability_source_breakdown": _source_breakdown(row, detected_stage, calibration, truth, governance),
        })

    enriched.sort(key=lambda item: item["calibrated_probability"], reverse=True)
    confidence = _confidence(truth, calibration.confidence_scale)
    return {
        "ok": True,
        "stage": detected_stage,
        "model_id": model.model_id,
        "calibration_profile": calibration.as_dict(),
        "probability_governance": governance,
        "raw_probabilities": normalized_raw,
        "governed_probabilities": governed_raw,
        "calibrated_probabilities": calibrated,
        "empirical_calibration": empirical_meta,
        "finish_distribution": finish_distribution,
        "probabilities": enriched,
        "confidence": confidence,
        "explanations": _explanations(detected_stage, calibration, truth, governance),
        "top_movers": _top_movers(enriched),
    }


def enrich_probability_payload(
    payload: dict[str, Any],
    *,
    profile: dict[str, Any] | None = None,
    truth: dict[str, Any] | None = None,
    stage: str | None = "auto",
    live: bool = False,
    model_id: str | None = None,
) -> dict[str, Any]:
    audit = build_probability_audit(
        payload,
        profile=profile,
        truth=truth,
        stage=stage,
        live=live,
        model_id=model_id,
    )
    audit_by_driver = {row["driver_id"]: row for row in audit.get("probabilities") or []}
    rows = payload.get("simulations") or payload.get("probabilities") or []
    for row in rows:
        driver_id = str(row.get("driver_id") or "")
        audited = audit_by_driver.get(driver_id)
        if not audited:
            continue
        row["raw_probability"] = audited["raw_probability"]
        row["governed_probability"] = audited["governed_probability"]
        row["governance_delta"] = audited["governance_delta"]
        row["calibrated_probability"] = audited["calibrated_probability"]
        row["calibration_delta"] = audited["calibration_delta"]
        row["finish_distribution"] = audited["finish_distribution"]
        row["probability_source_breakdown"] = audited["probability_source_breakdown"]
        row["win_probability"] = audited["calibrated_probability"]
        row["podium_probability"] = audited["finish_distribution"].get("podium", 0.0)
        row["top5_probability"] = audited["finish_distribution"].get("top5", 0.0)
        row["points_probability"] = audited["finish_distribution"].get("points", 0.0)
        row["expected_finish"] = audited["finish_distribution"].get("expected_finish")

    rows.sort(key=lambda item: (float(item.get("win_probability") or 0.0), -float(item.get("expected_finish") or 99.0)), reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    payload["stage"] = audit["stage"]
    payload["calibration_profile"] = audit["calibration_profile"]
    payload["probability_governance"] = audit["probability_governance"]
    payload["raw_probabilities"] = audit["raw_probabilities"]
    payload["governed_probabilities"] = audit["governed_probabilities"]
    payload["calibrated_probabilities"] = audit["calibrated_probabilities"]
    payload["finish_distribution"] = audit["finish_distribution"]
    payload["probability_audit"] = audit
    payload["top_probability_movers"] = audit["top_movers"]
    payload["confidence"] = max(float(payload.get("confidence") or 0.0), audit["confidence"])
    return payload


def _finish_distribution(row: dict[str, Any], calibrated_win: float) -> dict[str, Any]:
    existing_distribution = row.get("finish_distribution") or row.get("position_distribution") or {}
    existing_summary = existing_distribution if isinstance(existing_distribution, dict) and "positions" in existing_distribution else {}
    positions = (existing_summary.get("positions") if existing_summary else existing_distribution) or {}
    podium = _position_probability(positions, 3)
    top5 = _position_probability(positions, 5)
    points = _position_probability(positions, 10)
    expected_finish = row.get("expected_finish")
    if expected_finish is None:
        expected_finish = existing_summary.get("expected_finish")
    if expected_finish is None and positions:
        expected_finish = sum(_position_key(position) * float(probability or 0.0) for position, probability in positions.items())
    return {
        "win": round(calibrated_win, 4),
        "podium": round(podium if podium is not None else float(existing_summary.get("podium") or row.get("podium_probability") or 0.0), 4),
        "top5": round(top5 if top5 is not None else float(existing_summary.get("top5") or row.get("top5_probability") or 0.0), 4),
        "points": round(points if points is not None else float(existing_summary.get("points") or row.get("points_probability") or 0.0), 4),
        "dnf": round(float(existing_summary.get("dnf") or row.get("dnf_probability") or 0.0), 4),
        "expected_finish": round(float(expected_finish or 0.0), 2),
        "positions": positions,
    }


def _position_probability(positions: dict[str, Any], top_n: int) -> float | None:
    if not positions:
        return None
    return sum(float(probability or 0.0) for position, probability in positions.items() if _position_key(position) <= top_n)


def _position_key(position: Any) -> int:
    try:
        return int(position)
    except (TypeError, ValueError):
        return 99


def _source_breakdown(
    row: dict[str, Any],
    stage: str,
    calibration: Any,
    truth: dict[str, Any],
    governance: dict[str, Any],
) -> dict[str, Any]:
    components = row.get("components") or {}
    source_mode = row.get("source_mode") or truth.get("source_mode") or "model"
    return {
        "stage": {"value": stage, "source": "race_profile"},
        "source_mode": {"value": source_mode, "confidence": row.get("source_confidence") or truth.get("confidence") or 0.0},
        "grid_quali": {"value": components.get("qualifying"), "source": "qualifying_or_projection"},
        "car_model": {
            "value": components.get("car_model"),
            "modifier": components.get("car_model_modifier"),
            "confidence": components.get("car_model_confidence"),
            "source": "constructor_car_model",
        },
        "championship_context": {
            "position": components.get("championship_position"),
            "points": components.get("championship_points"),
            "used_as": "bounded_context_not_primary_race_truth" if components.get("championship_context_only") else "model_context",
            "raw_model_win_prior": components.get("raw_model_win_prior"),
            "compressed_prior": components.get("model_prior"),
        },
        "live_position": {"value": components.get("live_position"), "source": source_mode},
        "tyres": {
            "compound": components.get("compound"),
            "tyre_age": components.get("tyre_age"),
            "pit_stops": components.get("pit_stops"),
            "source": source_mode,
        },
        "reliability": {"value": components.get("reliability"), "dnf_probability": components.get("dnf_probability")},
        "sentiment": {
            "value": components.get("race_sentiment_impact") or components.get("sentiment"),
            "delta": components.get("race_sentiment_delta"),
            "confidence": components.get("race_sentiment_confidence"),
        },
        "weather": {"chaos_score": (truth.get("signals") or {}).get("chaos_score") or truth.get("chaos_score")},
        "calibration": calibration.as_dict(),
        "probability_governance": governance,
    }


def _confidence(truth: dict[str, Any], scale: float) -> float:
    base = float(truth.get("confidence") or 0.35)
    missing_penalty = min(0.35, len(truth.get("missing_groups") or []) * 0.035)
    return round(max(0.05, min(0.98, base * scale - missing_penalty)), 4)


def _explanations(stage: str, calibration: Any, truth: dict[str, Any], governance: dict[str, Any]) -> list[dict[str, Any]]:
    explanations = [
        {
            "type": "stage",
            "message": f"{stage.replace('_', ' ').title()} stage uses temperature {calibration.temperature:.2f}.",
        }
    ]
    source_mode = truth.get("source_mode") or "model"
    if source_mode in {"estimated", "unavailable"}:
        explanations.append({
            "type": "source_quality",
            "message": "Estimated or unavailable timing keeps the model deliberately conservative.",
        })
    if governance.get("applied"):
        explanations.append({
            "type": "probability_governance",
            "message": governance.get("reason") or "Low-evidence probability cap applied.",
            "magnitude": governance.get("top_cap"),
        })
    for item in calibration.adjustments:
        explanations.append({"type": item["target"], "message": item["reason"], "magnitude": item["magnitude"]})
    return explanations


def _apply_evidence_governance(probabilities: dict[str, float], stage: str, truth: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    if len(probabilities) <= 1 or stage == "completed":
        return probabilities, {"applied": False, "reason": "not_required"}

    source_mode = str(truth.get("source_mode") or "model").lower()
    confidence = max(0.0, min(1.0, float(truth.get("confidence") or 0.0)))
    missing_groups = truth.get("missing_groups") or []
    cap = _top_probability_cap(stage, source_mode, confidence, len(missing_groups))
    cap = max(cap, round((1.0 / max(len(probabilities), 1)) + 0.0001, 4))
    leader = max(probabilities, key=probabilities.get)
    leader_probability = probabilities[leader]
    if leader_probability <= cap:
        return probabilities, {
            "applied": False,
            "source_mode": source_mode,
            "confidence": round(confidence, 4),
            "top_cap": cap,
            "leader_probability": round(leader_probability, 4),
            "reason": "within_evidence_cap",
        }

    governed, capped_driver_ids = _cap_distribution(probabilities, cap)
    return governed, {
        "applied": True,
        "source_mode": source_mode,
        "confidence": round(confidence, 4),
        "top_cap": cap,
        "capped_driver_id": leader,
        "capped_driver_ids": capped_driver_ids,
        "leader_probability_before": round(leader_probability, 4),
        "leader_probability_after": governed.get(leader),
        "reason": f"{stage.replace('_', ' ')} with {source_mode} evidence capped the top pick so standings/model priors cannot dominate.",
    }


def _top_probability_cap(stage: str, source_mode: str, confidence: float, missing_count: int) -> float:
    if source_mode in {"estimated", "unavailable"}:
        base = 0.30 if stage in {"pre_weekend", "practice_available"} else 0.36
    elif source_mode in {"model", "historical"}:
        base = 0.34 if stage in {"pre_weekend", "practice_available"} else 0.42
    elif source_mode == "recent":
        base = 0.48
    elif source_mode == "recording_pending":
        base = 0.34 if stage in {"pre_weekend", "practice_available"} else 0.38
    elif source_mode == "recorded":
        base = 0.58
    elif source_mode == "recorded_confident":
        base = 0.64
    elif source_mode == "live":
        base = 0.70
    else:
        base = 0.38

    confidence_bonus = max(0.0, confidence - 0.35) * 0.22
    missing_penalty = min(0.08, max(0, missing_count) * 0.012)
    return round(max(0.24, min(0.82, base + confidence_bonus - missing_penalty)), 4)


def _cap_distribution(probabilities: dict[str, float], cap: float) -> tuple[dict[str, float], list[str]]:
    governed = {driver_id: max(0.0, float(value or 0.0)) for driver_id, value in probabilities.items()}
    frozen: set[str] = set()
    capped_driver_ids: list[str] = []

    for _ in range(len(governed)):
        over_limit = [driver_id for driver_id, value in governed.items() if driver_id not in frozen and value > cap]
        if not over_limit:
            break
        excess = 0.0
        for driver_id in over_limit:
            excess += governed[driver_id] - cap
            governed[driver_id] = cap
            frozen.add(driver_id)
            capped_driver_ids.append(driver_id)

        recipients = [driver_id for driver_id in governed if driver_id not in frozen]
        if not recipients:
            break
        recipient_total = sum(governed[driver_id] for driver_id in recipients)
        if recipient_total <= 0:
            share = excess / len(recipients)
            for driver_id in recipients:
                governed[driver_id] += share
        else:
            for driver_id in recipients:
                governed[driver_id] += excess * (governed[driver_id] / recipient_total)

    total = sum(governed.values()) or 1.0
    return {driver_id: round(value / total, 4) for driver_id, value in governed.items()}, capped_driver_ids


def _top_movers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    movers = sorted(rows, key=lambda row: abs(row["calibration_delta"]), reverse=True)
    return [
        {
            "driver_id": row["driver_id"],
            "driver_code": row.get("driver_code"),
            "driver_name": row.get("driver_name"),
            "team": row.get("team"),
            "delta": row["calibration_delta"],
            "raw_probability": row["raw_probability"],
            "calibrated_probability": row["calibrated_probability"],
        }
        for row in movers[:5]
    ]
