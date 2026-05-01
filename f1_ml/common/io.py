"""Artifact IO — load/save trained models with version metadata.

Every artifact carries a manifest (model name, training-cutoff date, feature schema hash)
so the backtest harness can reject look-ahead violations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ArtifactManifest:
    model_name: str
    model_version: str
    trained_at: datetime
    training_cutoff: datetime  # critical for backtest look-ahead audit
    feature_schema_hash: str
    extra: dict[str, Any]


def save(artifact: Any, manifest: ArtifactManifest, path: Path) -> None:
    """Persist a model artifact + sidecar manifest. Format-specific (pickle/torch/joblib)
    is delegated to each model module — this is the manifest contract only."""
    raise NotImplementedError("delegate to per-model serializer; write manifest as path.with_suffix('.json')")


def load(path: Path) -> tuple[Any, ArtifactManifest]:
    """Load artifact + manifest. Caller validates training_cutoff against decision time."""
    raise NotImplementedError


def manifest_path(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(".json")


def write_manifest(manifest: ArtifactManifest, path: Path) -> None:
    payload = {
        "model_name": manifest.model_name,
        "model_version": manifest.model_version,
        "trained_at": manifest.trained_at.isoformat(),
        "training_cutoff": manifest.training_cutoff.isoformat(),
        "feature_schema_hash": manifest.feature_schema_hash,
        "extra": manifest.extra,
    }
    path.write_text(json.dumps(payload, indent=2))
