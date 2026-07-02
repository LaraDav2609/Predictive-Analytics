"""Convert artifact bundles into production `ml_trained_inputs`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sports.f1.ml.artifacts.schema import F1MLArtifactBundle
from sports.f1.ml.artifacts.store import artifact_readiness, load_artifact_bundle, validate_artifact_bundle
from sports.f1.ml.simulator.model_contract import (
    SimulatorModelBundle,
    StaticDNFAdapter,
    StaticDriverPaceAdapter,
    StaticRatingPriorAdapter,
)


def load_ml_trained_inputs(path: str | Path, *, target_race: str | None = None) -> dict[str, Any]:
    try:
        bundle = load_artifact_bundle(path)
    except Exception as exc:
        return _fallback("artifact_load_failed", detail=exc.__class__.__name__, path=str(path))
    return artifact_to_ml_trained_inputs(bundle, target_race=target_race, path=str(path))


def load_simulator_model_bundle(path: str | Path, *, target_race: str | None = None) -> SimulatorModelBundle:
    try:
        bundle = load_artifact_bundle(path)
    except Exception as exc:
        return _fallback_model_bundle("artifact_load_failed", detail=exc.__class__.__name__, path=str(path))
    return artifact_to_simulator_model_bundle(bundle, target_race=target_race, path=str(path))


def artifact_to_simulator_model_bundle(
    bundle: F1MLArtifactBundle | dict[str, Any],
    *,
    target_race: str | None = None,
    path: str | None = None,
) -> SimulatorModelBundle:
    try:
        parsed = bundle if isinstance(bundle, F1MLArtifactBundle) else F1MLArtifactBundle.model_validate(bundle)
    except Exception as exc:
        return _fallback_model_bundle("artifact_schema_invalid", detail=exc.__class__.__name__, path=path)

    validation = validate_artifact_bundle(parsed, target_race=target_race)
    if not validation.ok:
        return _fallback_model_bundle(validation.reason or "artifact_invalid", path=path, artifact_id=parsed.artifact_id)
    readiness = artifact_readiness(parsed, target_race=target_race, path=path)

    rows = _artifact_driver_rows(parsed)
    pace_rows = {code: row for code, row in rows.items() if row.get("pace_mean_seconds") is not None or row.get("pace_sigma_seconds") is not None}
    dnf_rows = {code: row for code, row in rows.items() if row.get("dnf_hazard_per_lap") is not None}
    rating_priors = {
        code: float(row["rating_prior"])
        for code, row in rows.items()
        if row.get("rating_prior") is not None
    }
    rating_priors.update({str(code): float(value) for code, value in (parsed.rating_priors or {}).items()})

    pace_adapter = StaticDriverPaceAdapter(
        pace_rows,
        source=_artifact_model_source(parsed.pace_model, "artifact_static_pace"),
        confidence=_model_confidence(parsed.pace_model, rows, "pace_mean_seconds"),
    ) if pace_rows else None
    dnf_adapter = StaticDNFAdapter(
        dnf_rows,
        source=_artifact_model_source(parsed.dnf_model or parsed.survival_model, "artifact_static_dnf"),
        confidence=_model_confidence(parsed.dnf_model or parsed.survival_model, rows, "dnf_hazard_per_lap"),
    ) if dnf_rows else None
    rating_adapter = StaticRatingPriorAdapter(rating_priors) if rating_priors else None
    adapters = [adapter for adapter in (pace_adapter, dnf_adapter, rating_adapter) if adapter is not None]
    if not adapters:
        return _fallback_model_bundle("artifact_has_no_usable_model_adapters", path=path, artifact_id=parsed.artifact_id)

    confidence_values = [
        float(getattr(adapter, "confidence", 0.0) or 0.0)
        for adapter in adapters
    ]
    return SimulatorModelBundle(
        pace_adapter=pace_adapter,
        dnf_adapter=dnf_adapter,
        rating_adapter=rating_adapter,
        artifact_id=parsed.artifact_id,
        artifact_version=parsed.model_version,
        source="artifact_bundle",
        confidence=round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else 0.0,
        metadata={
            "artifact_path": path,
            "artifact_readiness": readiness,
            "schema_version": parsed.schema_version,
            "training_seasons": parsed.training_seasons,
            "training_races": parsed.training_races,
            "validation_warnings": validation.warnings,
            "leakage_status": parsed.leakage_status,
        },
    )


def artifact_to_ml_trained_inputs(
    bundle: F1MLArtifactBundle | dict[str, Any],
    *,
    target_race: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    try:
        parsed = bundle if isinstance(bundle, F1MLArtifactBundle) else F1MLArtifactBundle.model_validate(bundle)
    except Exception as exc:
        return _fallback("artifact_schema_invalid", detail=exc.__class__.__name__, path=path)

    validation = validate_artifact_bundle(parsed, target_race=target_race)
    if not validation.ok:
        return _fallback(validation.reason or "artifact_invalid", path=path, artifact_id=parsed.artifact_id)
    readiness = artifact_readiness(parsed, target_race=target_race, path=path)

    drivers = _artifact_driver_rows(parsed)

    return {
        "drivers": drivers,
        "feature_columns": parsed.feature_columns,
        "pace_model_confidence": _model_confidence(parsed.pace_model, drivers, "pace_mean_seconds"),
        "dnf_model_confidence": _model_confidence(parsed.dnf_model or parsed.survival_model, drivers, "dnf_hazard_per_lap"),
        "artifact_id": parsed.artifact_id,
        "artifact_version": parsed.model_version,
        "artifact_schema_version": parsed.schema_version,
        "artifact_path": path,
        "artifact_readiness": readiness,
        "artifact_metadata": {
            "training_seasons": parsed.training_seasons,
            "training_races": parsed.training_races,
            "training_cutoff": parsed.training_cutoff.isoformat() if parsed.training_cutoff else None,
            "knowable_as_of": parsed.knowable_as_of.isoformat() if parsed.knowable_as_of else None,
            "validation_metrics": parsed.validation_metrics,
            "source_metadata": parsed.source_metadata,
            "leakage_status": parsed.leakage_status,
            "warnings": validation.warnings,
        },
        "sources": ["artifact_bundle", "artifact_driver_rows"],
    }


def apply_ml_trained_inputs_to_initial_state(
    initial_state: dict[str, Any],
    trained_inputs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Overlay direct artifact rows onto a simulator initial state.

    This is used by the research CLI, while the production predictor uses the
    same converted artifact through `ml_simulator_v1`.
    """
    state = {**initial_state}
    codes = list(state.get("driver_codes") or [])
    means = list(state.get("driver_mean_pace_s") or [])
    sigmas = list(state.get("driver_pace_sigma_s") or [])
    dnfs = list(state.get("driver_dnf_rate_per_lap") or [])
    rows = trained_inputs.get("drivers") or {}
    applied = 0
    for index, code in enumerate(codes):
        row = rows.get(code)
        if not row:
            continue
        if row.get("pace_mean_seconds") is not None and index < len(means):
            means[index] = float(row["pace_mean_seconds"])
            applied += 1
        if row.get("pace_sigma_seconds") is not None and index < len(sigmas):
            sigmas[index] = max(0.08, min(2.25, float(row["pace_sigma_seconds"])))
        if row.get("dnf_hazard_per_lap") is not None and index < len(dnfs):
            dnfs[index] = max(0.00005, min(0.05, float(row["dnf_hazard_per_lap"])))
            applied += 1
    state["driver_mean_pace_s"] = means
    state["driver_pace_sigma_s"] = sigmas
    state["driver_dnf_rate_per_lap"] = dnfs
    return state, {
        "applied_rows": applied,
        "artifact_id": trained_inputs.get("artifact_id"),
        "artifact_version": trained_inputs.get("artifact_version"),
        "fallback_reason": trained_inputs.get("artifact_load_error") if applied == 0 else None,
    }


def _model_confidence(model: Any, drivers: dict[str, dict[str, Any]], key: str) -> float:
    if model is not None:
        return float(getattr(model, "confidence", 0.74) or 0.74)
    values = [float(row.get("confidence") or 0.0) for row in drivers.values() if row.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else 0.70


def _artifact_driver_rows(parsed: F1MLArtifactBundle) -> dict[str, dict[str, Any]]:
    drivers: dict[str, dict[str, Any]] = {}
    for key, row in parsed.drivers.items():
        driver_key = row.driver_id or row.driver_code or str(row.driver_number or key)
        sources = set(row.sources or [])
        if row.pace_mean_seconds is not None:
            sources.add("artifact_pace")
        if row.dnf_hazard_per_lap is not None:
            sources.add("artifact_dnf")
        if row.rating_prior is not None:
            sources.add("artifact_rating_prior")
        drivers[driver_key] = {
            "pace_mean_seconds": row.pace_mean_seconds,
            "pace_sigma_seconds": row.pace_sigma_seconds,
            "dnf_hazard_per_lap": row.dnf_hazard_per_lap,
            "rating_prior": row.rating_prior,
            "sources": sorted(sources),
            "confidence": row.confidence,
        }
    return drivers


def _artifact_model_source(model: Any, fallback: str) -> str:
    if model is None:
        return fallback
    return str(model.source or model.kind or fallback)


def _fallback_model_bundle(
    reason: str,
    *,
    detail: str | None = None,
    path: str | None = None,
    artifact_id: str | None = None,
) -> SimulatorModelBundle:
    return SimulatorModelBundle(
        artifact_id=artifact_id,
        source="artifact_invalid",
        confidence=0.0,
        fallback_reason=reason,
        metadata={"artifact_path": path, "detail": detail},
    )


def _fallback(reason: str, *, detail: str | None = None, path: str | None = None, artifact_id: str | None = None) -> dict[str, Any]:
    return {
        "drivers": {},
        "artifact_id": artifact_id,
        "artifact_path": path,
        "artifact_load_error": reason,
        "artifact_load_detail": detail,
        "sources": ["artifact_invalid"],
    }
