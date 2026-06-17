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
