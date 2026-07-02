"""Save, load, and validate F1 ML artifact bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sports.f1.ml.artifacts.schema import ARTIFACT_SCHEMA_VERSION, F1MLArtifactBundle


@dataclass
class ArtifactValidation:
    ok: bool
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)


def save_artifact_bundle(path: str | Path, bundle: F1MLArtifactBundle | dict[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    parsed = bundle if isinstance(bundle, F1MLArtifactBundle) else F1MLArtifactBundle.model_validate(bundle)
    target.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
    return {"ok": True, "path": str(target), "artifact_id": parsed.artifact_id, "schema_version": parsed.schema_version}


def load_artifact_bundle(path: str | Path) -> F1MLArtifactBundle:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    return F1MLArtifactBundle.model_validate(data)


def load_artifact_readiness(path: str | Path, *, target_race: str | None = None) -> dict[str, Any]:
    source = Path(path)
    try:
        bundle = load_artifact_bundle(source)
    except Exception as exc:
        return {
            "ok": False,
            "serving_ready": False,
            "live_trading_ready": False,
            "status": "blocked",
            "reason": "artifact_load_failed",
            "detail": exc.__class__.__name__,
            "path": str(source),
            "promotion_blockers": ["artifact_load_failed"],
        }
    return artifact_readiness(bundle, target_race=target_race, path=str(source))


def artifact_readiness(
    bundle: F1MLArtifactBundle | dict[str, Any],
    *,
    target_race: str | None = None,
    path: str | None = None,
    validation_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe serving/live-trading readiness summary for A5/A7."""

    try:
        parsed = bundle if isinstance(bundle, F1MLArtifactBundle) else F1MLArtifactBundle.model_validate(bundle)
    except (ValidationError, TypeError, ValueError) as exc:
        return {
            "ok": False,
            "serving_ready": False,
            "live_trading_ready": False,
            "status": "blocked",
            "reason": f"artifact_schema_invalid:{exc.__class__.__name__}",
            "path": path,
            "promotion_blockers": ["artifact_schema_invalid"],
        }

    validation = validate_artifact_bundle(parsed, target_race=target_race)
    coverage = _artifact_coverage(parsed)
    model_quality = (parsed.validation_metrics or {}).get("model_quality") or {}
    calibration_status = _calibration_status(parsed)
    gate = validation_gate or _embedded_validation_gate(parsed)
    gate_passed = bool(gate.get("passed") is True or str(gate.get("status") or "").lower() == "passed")
    blockers: list[str] = []
    if not validation.ok:
        blockers.append(validation.reason or "artifact_invalid")
    if not gate:
        blockers.append("out_of_sample_gate_missing")
    elif not gate_passed:
        blockers.append("out_of_sample_gate_blocked")
    if model_quality.get("pace_beats_baseline") is False:
        blockers.append("pace_model_underperformed_baseline")
    if model_quality.get("dnf_beats_baseline") is False:
        blockers.append("dnf_model_underperformed_baseline")
    if model_quality.get("overtake_beats_baseline") is False:
        blockers.append("overtake_model_underperformed_baseline")
    if calibration_status.get("status") == "blocked":
        blockers.append("calibration_underperformed_baseline")

    serving_ready = validation.ok
    live_ready = serving_ready and gate_passed and not blockers
    return {
        "ok": serving_ready,
        "serving_ready": serving_ready,
        "live_trading_ready": live_ready,
        "status": "ready" if serving_ready else "blocked",
        "live_status": "ready" if live_ready else "blocked",
        "reason": validation.reason,
        "warnings": validation.warnings,
        "promotion_blockers": blockers,
        "artifact_id": parsed.artifact_id,
        "model_id": parsed.model_id,
        "model_version": parsed.model_version,
        "schema_version": parsed.schema_version,
        "compatibility_version": parsed.compatibility_version,
        "created_at": parsed.created_at.isoformat() if parsed.created_at else None,
        "training_seasons": parsed.training_seasons,
        "training_races": parsed.training_races,
        "training_race_count": len(parsed.training_races or []),
        "training_cutoff": parsed.training_cutoff.isoformat() if parsed.training_cutoff else None,
        "knowable_as_of": parsed.knowable_as_of.isoformat() if parsed.knowable_as_of else None,
        "target_race": target_race,
        "path": path,
        "coverage": coverage,
        "model_quality": model_quality,
        "calibration_status": calibration_status,
        "validation_gate": gate,
        "validation_metrics": parsed.validation_metrics,
        "leakage_status": parsed.leakage_status,
        "source_provider": parsed.source_metadata.get("provider"),
        "fallback_groups": parsed.source_metadata.get("fallback_groups") or [],
    }


def validate_artifact_bundle(bundle: F1MLArtifactBundle | dict[str, Any], *, target_race: str | None = None) -> ArtifactValidation:
    try:
        parsed = bundle if isinstance(bundle, F1MLArtifactBundle) else F1MLArtifactBundle.model_validate(bundle)
    except (ValidationError, TypeError, ValueError) as exc:
        return ArtifactValidation(False, f"artifact_schema_invalid:{exc.__class__.__name__}")

    warnings: list[str] = []
    if parsed.schema_version != ARTIFACT_SCHEMA_VERSION:
        return ArtifactValidation(False, "unsupported_schema_version")
    if parsed.model_id != "ml_simulator_v1":
        return ArtifactValidation(False, "incompatible_model_id")
    if not parsed.feature_columns:
        return ArtifactValidation(False, "missing_feature_columns")
    if parsed.source_metadata.get("stale") is True:
        return ArtifactValidation(False, "stale_artifact")
    if not parsed.drivers:
        return ArtifactValidation(False, "missing_driver_payloads")

    has_pace = any(driver.pace_mean_seconds is not None for driver in parsed.drivers.values())
    has_dnf = any(driver.dnf_hazard_per_lap is not None for driver in parsed.drivers.values())
    if not has_pace and parsed.pace_model is None:
        return ArtifactValidation(False, "missing_pace_content")
    if not has_dnf and parsed.dnf_model is None and parsed.survival_model is None:
        return ArtifactValidation(False, "missing_dnf_content")

    leakage = parsed.leakage_status or {}
    if leakage.get("status") in {"failed", "leakage_detected"}:
        return ArtifactValidation(False, "leakage_guard_failed")
    if target_race and target_race in set(parsed.training_races or []):
        return ArtifactValidation(False, "target_race_in_training")
    if parsed.compatibility_version != "ml_simulator_v1":
        warnings.append("compatibility_version_not_ml_simulator_v1")

    return ArtifactValidation(True, warnings=warnings)


def _artifact_coverage(parsed: F1MLArtifactBundle) -> dict[str, Any]:
    drivers = parsed.drivers or {}
    pace_count = sum(1 for driver in drivers.values() if driver.pace_mean_seconds is not None)
    dnf_count = sum(1 for driver in drivers.values() if driver.dnf_hazard_per_lap is not None)
    rating_count = sum(1 for driver in drivers.values() if driver.rating_prior is not None)
    driver_count = len(drivers)
    return {
        "driver_count": driver_count,
        "pace_driver_count": pace_count,
        "dnf_driver_count": dnf_count,
        "rating_driver_count": rating_count,
        "pace_coverage": round(pace_count / driver_count, 4) if driver_count else 0.0,
        "dnf_coverage": round(dnf_count / driver_count, 4) if driver_count else 0.0,
        "rating_coverage": round(rating_count / driver_count, 4) if driver_count else 0.0,
        "feature_column_count": len(parsed.feature_columns or []),
        "model_adapters_present": [
            name for name, model in (
                ("pace", parsed.pace_model),
                ("dnf", parsed.dnf_model),
                ("survival", parsed.survival_model),
                ("overtake", parsed.overtake_model),
            )
            if model is not None
        ],
        "source_counts": parsed.source_metadata.get("coverage_counts") or {},
        "model_coverage_counts": parsed.source_metadata.get("model_coverage_counts") or {},
    }


def _embedded_validation_gate(parsed: F1MLArtifactBundle) -> dict[str, Any]:
    for container in (parsed.validation_metrics, parsed.source_metadata):
        gate = (container or {}).get("out_of_sample_gate") or (container or {}).get("validation_gate")
        if isinstance(gate, dict):
            return gate
    return {}


def _calibration_status(parsed: F1MLArtifactBundle) -> dict[str, Any]:
    """Summarize whether held-out calibration is proven useful for live promotion."""

    metrics = _calibration_metrics(parsed)
    if not metrics:
        return {
            "available": False,
            "status": "missing",
            "passed": None,
            "reason": "calibration_metrics_missing",
        }

    explicit_pass = _optional_bool(
        metrics,
        "passed",
        "improved",
        "beats_baseline",
        "brier_improved",
        "heldout_brier_improved",
    )
    improvement = _first_number(
        metrics,
        "heldout_brier_improvement",
        "brier_improvement",
        "brier_improvement_vs_baseline",
        "winner_brier_improvement",
        "delta_brier_improvement",
    )
    if improvement is None:
        baseline = _first_number(metrics, "baseline_brier", "uncalibrated_brier", "raw_brier")
        calibrated = _first_number(metrics, "calibrated_brier", "candidate_brier", "post_calibration_brier")
        if baseline is not None and calibrated is not None:
            improvement = baseline - calibrated

    raw_status = str(metrics.get("status") or "").strip().lower()
    explicit_block = raw_status in {"blocked", "failed", "rejected", "underperformed"} or explicit_pass is False
    passed = bool(explicit_pass is True or (improvement is not None and improvement > 0))
    if explicit_block or (improvement is not None and improvement <= 0):
        status = "blocked"
        reason = "heldout_brier_not_improved"
    elif passed:
        status = "passed"
        reason = "heldout_brier_improved"
    else:
        status = "unproven"
        reason = "heldout_brier_improvement_missing"

    return {
        "available": True,
        "status": status,
        "passed": passed if status != "unproven" else None,
        "reason": reason,
        "method": metrics.get("method") or metrics.get("kind"),
        "baseline_brier": _first_number(metrics, "baseline_brier", "uncalibrated_brier", "raw_brier"),
        "calibrated_brier": _first_number(metrics, "calibrated_brier", "candidate_brier", "post_calibration_brier"),
        "brier_improvement": improvement,
        "sample_count": _first_number(metrics, "sample_count", "n", "count", as_int=True),
    }


def _calibration_metrics(parsed: F1MLArtifactBundle) -> dict[str, Any]:
    for container in (parsed.validation_metrics, parsed.source_metadata):
        for key in ("calibration", "calibration_metrics", "empirical_calibration"):
            value = (container or {}).get(key)
            if isinstance(value, dict):
                return value
    return {}


def _first_number(metrics: dict[str, Any], *keys: str, as_int: bool = False) -> float | int | None:
    for key in keys:
        value = metrics.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        return int(number) if as_int else number
    return None


def _optional_bool(metrics: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, bool):
            return value
    return None
