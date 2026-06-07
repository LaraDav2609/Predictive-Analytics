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
    """Persist a model artifact + sidecar JSON manifest.

    Default serialization is pickle, suitable for sklearn / lightgbm / xgboost
    / pymc trace objects. Torch and ONNX modules should serialize natively
    and call write_manifest separately.
    """
    import pickle
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(artifact, fh)
    write_manifest(manifest, manifest_path(path))


def load(path: Path) -> tuple[Any, ArtifactManifest]:
    """Load artifact + manifest. Caller validates training_cutoff against decision time."""
    import pickle
    path = Path(path)
    with path.open("rb") as fh:
        artifact = pickle.load(fh)
    manifest = read_manifest(manifest_path(path))
    return artifact, manifest


def manifest_path(artifact_path: Path) -> Path:
    return Path(artifact_path).with_suffix(".json")


def write_manifest(manifest: ArtifactManifest, path: Path) -> None:
    payload = {
        "model_name": manifest.model_name,
        "model_version": manifest.model_version,
        "trained_at": manifest.trained_at.isoformat(),
        "training_cutoff": manifest.training_cutoff.isoformat(),
        "feature_schema_hash": manifest.feature_schema_hash,
        "extra": manifest.extra,
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def read_manifest(path: Path) -> ArtifactManifest:
    payload = json.loads(Path(path).read_text())
    return ArtifactManifest(
        model_name=payload["model_name"],
        model_version=payload["model_version"],
        trained_at=datetime.fromisoformat(payload["trained_at"]),
        training_cutoff=datetime.fromisoformat(payload["training_cutoff"]),
        feature_schema_hash=payload["feature_schema_hash"],
        extra=payload.get("extra", {}),
    )
