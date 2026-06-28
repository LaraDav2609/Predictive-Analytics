"""Build portable telemetry learned-head artifacts.

This module exports the first trainable artifact shape for telemetry analysis.
The heads are JSON-linear so they can run inside the API without pulling in a
heavy model runtime. Real learned weights can replace these bootstrap
coefficients while keeping the same production contract.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sports.f1.ml.telemetry.artifacts import EXPECTED_HEADS, load_telemetry_artifact_manifest
from sports.f1.ml.telemetry.types import TelemetryFeaturePayload


TELEMETRY_LINEAR_FEATURES = (
    "clean_air_pace_delta_s",
    "pace_sigma_delta",
    "top_speed_delta_kph",
    "corner_min_speed_delta_kph",
    "traction_score",
    "stability_score",
    "tire_deg_slope_delta",
    "traffic_penalty_s",
    "overtake_pressure",
    "dnf_hazard_multiplier",
)


def build_telemetry_artifact_manifest(
    *,
    artifact_id: str = "telemetry-bootstrap-linear-v1",
    artifact_version: str = "telemetry_simulator_v1.telemetry_heads.v1",
    payloads: list[TelemetryFeaturePayload] | None = None,
    training_rows: list[dict[str, Any]] | None = None,
    season_start: int | None = None,
    season_end: int | None = None,
) -> dict[str, Any]:
    """Create a complete JSON-linear telemetry head manifest."""

    payloads = payloads or []
    training_rows = training_rows or []
    created_at = datetime.now(timezone.utc).isoformat()
    trained_heads, training_metrics = _fit_supervised_heads(training_rows)
    heads = {**_bootstrap_heads(), **trained_heads}
    return {
        "schema_version": "f1_telemetry_artifact_v1",
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "model_id": "telemetry_simulator_v1",
        "created_at": created_at,
        "training_window": {
            "season_start": season_start,
            "season_end": season_end,
            "payload_count": len(payloads),
        },
        "feature_schema": {
            "version": "telemetry_features_v0",
            "columns": list(TELEMETRY_LINEAR_FEATURES),
        },
        "heads": heads,
        "calibration": {
            "kind": "supervised_linear" if trained_heads else "deterministic_bootstrap_linear",
            "bounded_outputs": True,
            "canary_ready": True,
        },
        "metrics": {**_coverage_metrics(payloads), **training_metrics},
    }


def load_telemetry_training_payloads(path: str | Path | None) -> list[TelemetryFeaturePayload]:
    """Load saved telemetry feature payloads from a JSON file or directory."""

    if not path:
        return []
    root = Path(path)
    files = sorted(root.glob("*.json")) if root.is_dir() else [root]
    payloads: list[TelemetryFeaturePayload] = []
    for file in files:
        try:
            raw = _read_json_file(file)
        except (OSError, json.JSONDecodeError):
            continue
        payloads.extend(_payloads_from_json(raw))
    return payloads


def load_telemetry_training_rows(path: str | Path | None) -> list[dict[str, Any]]:
    """Load supervised telemetry rows with feature and target dictionaries."""

    if not path:
        return []
    root = Path(path)
    files = sorted(root.glob("*.json")) if root.is_dir() else [root]
    rows: list[dict[str, Any]] = []
    for file in files:
        try:
            raw = _read_json_file(file)
        except (OSError, json.JSONDecodeError):
            continue
        rows.extend(_training_rows_from_json(raw))
    return rows


def load_telemetry_label_rows(path: str | Path | None) -> list[dict[str, Any]]:
    """Load realized telemetry labels for joining with feature payloads."""

    if not path:
        return []
    root = Path(path)
    files = sorted(root.glob("*.json")) if root.is_dir() else [root]
    rows: list[dict[str, Any]] = []
    for file in files:
        try:
            raw = _read_json_file(file)
        except (OSError, json.JSONDecodeError):
            continue
        rows.extend(_label_rows_from_json(raw))
    return rows


def export_telemetry_training_rows(
    payloads: list[TelemetryFeaturePayload],
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join telemetry feature payloads with realized labels into training rows."""

    label_index = {_label_key(row): _target_values(row) for row in labels if _label_key(row)}
    rows: list[dict[str, Any]] = []
    for payload in payloads or []:
        for code, feature in (payload.driver_features or {}).items():
            key = (payload.race_id, payload.session, str(code).upper())
            targets = label_index.get(key)
            if not targets:
                continue
            feature_values = feature.model_dump(mode="python")
            rows.append({
                "race_id": payload.race_id,
                "session": payload.session,
                "driver_code": str(code).upper(),
                "source_mode": payload.source_mode,
                "confidence": feature.confidence,
                "samples": feature.samples,
                "features": {
                    name: _safe_float(feature_values.get(name)) or 0.0
                    for name in TELEMETRY_LINEAR_FEATURES
                },
                "targets": targets,
            })
    return rows


def save_telemetry_training_rows(path: str | Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Persist exported supervised telemetry rows."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "f1_telemetry_training_rows_v1",
        "row_count": len(rows),
        "target_counts": {
            head: sum(1 for row in rows if (row.get("targets") or {}).get(head) is not None)
            for head in EXPECTED_HEADS
        },
        "training_rows": rows,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "ok": True,
        "path": str(target),
        "row_count": len(rows),
        "target_counts": payload["target_counts"],
    }


def save_telemetry_artifact_manifest(path: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Persist a telemetry artifact manifest and validate it through the loader."""

    target = Path(path)
    manifest_path = target / "metadata.json" if target.suffix.lower() != ".json" else target
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    status = load_telemetry_artifact_manifest(manifest_path)
    return {
        "ok": bool(status.get("ok")),
        "path": str(manifest_path),
        "artifact_id": status.get("artifact_id"),
        "artifact_version": status.get("artifact_version"),
        "trained_heads": status.get("trained_heads") or [],
        "missing_heads": status.get("missing_heads") or [],
        "fallback_reason": status.get("fallback_reason"),
    }


def _read_json_file(path: Path) -> Any:
    """Read JSON fixtures written by Python or Windows PowerShell."""

    return json.loads(path.read_text(encoding="utf-8-sig"))


def _bootstrap_heads() -> dict[str, dict[str, Any]]:
    return {
        "pace_delta": {
            "intercept": 0.09,
            "coefficients": {
                "clean_air_pace_delta_s": 0.60,
                "top_speed_delta_kph": -0.010,
                "corner_min_speed_delta_kph": -0.015,
                "traction_score": -0.18,
                "traffic_penalty_s": 1.0,
                "tire_deg_slope_delta": 1.5,
            },
            "clamp": [-1.20, 1.20],
        },
        "pace_quantile": {
            "intercept": 1.0,
            "coefficients": {
                "pace_sigma_delta": 1.0,
                "stability_score": -0.08,
            },
            "clamp": [0.78, 1.25],
        },
        "dnf_hazard": {
            "intercept": 0.0,
            "coefficients": {
                "dnf_hazard_multiplier": 1.0,
            },
            "clamp": [0.75, 1.50],
        },
        "overtake": {
            "intercept": -0.07,
            "coefficients": {
                "overtake_pressure": 0.20,
            },
            "clamp": [-0.07, 0.10],
        },
        "pit_value": {
            "intercept": 0.0,
            "coefficients": {
                "overtake_pressure": 0.35,
                "tire_deg_slope_delta": 1.8,
                "traffic_penalty_s": 1.0,
            },
            "clamp": [-0.25, 0.55],
        },
    }


def _coverage_metrics(payloads: list[TelemetryFeaturePayload]) -> dict[str, Any]:
    driver_count = sum(len(payload.driver_features or {}) for payload in payloads)
    sample_count = sum(
        int(feature.samples or 0)
        for payload in payloads
        for feature in (payload.driver_features or {}).values()
    )
    return {
        "payload_count": len(payloads),
        "driver_feature_count": driver_count,
        "trace_sample_count": sample_count,
        "race_ids": sorted({payload.race_id for payload in payloads if payload.race_id}),
        "sessions": sorted({payload.session for payload in payloads if payload.session}),
        "source_modes": sorted({payload.source_mode for payload in payloads if payload.source_mode}),
        "complete_heads": list(EXPECTED_HEADS),
    }


def _fit_supervised_heads(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    heads: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {
        "supervised_row_count": len(rows),
        "supervised_head_counts": {},
        "supervised_head_mae": {},
    }
    for head in EXPECTED_HEADS:
        x_rows: list[list[float]] = []
        y_values: list[float] = []
        for row in rows:
            features = row.get("features") or row.get("telemetry_features") or {}
            targets = row.get("targets") or row.get("labels") or {}
            if not isinstance(features, dict) or not isinstance(targets, dict):
                continue
            target = _safe_float(targets.get(head))
            if target is None:
                continue
            x_rows.append([_safe_float(features.get(name)) or 0.0 for name in TELEMETRY_LINEAR_FEATURES])
            y_values.append(target)
        metrics["supervised_head_counts"][head] = len(y_values)
        if len(y_values) < 2:
            continue
        intercept, coefficients = _fit_linear(x_rows, y_values, ridge=0.05)
        predictions = [
            intercept + sum(coefficients[index] * row[index] for index in range(len(coefficients)))
            for row in x_rows
        ]
        mae = sum(abs(predictions[index] - y_values[index]) for index in range(len(y_values))) / len(y_values)
        metrics["supervised_head_mae"][head] = round(mae, 5)
        heads[head] = {
            "intercept": round(intercept, 6),
            "coefficients": {
                name: round(value, 6)
                for name, value in zip(TELEMETRY_LINEAR_FEATURES, coefficients)
                if abs(value) >= 0.000001
            },
            "clamp": _head_clamp(head),
            "calibration": {
                "kind": "supervised_ridge_linear",
                "row_count": len(y_values),
                "train_mae": round(mae, 5),
            },
        }
    metrics["supervised_heads_trained"] = sorted(heads.keys())
    return heads, metrics


def _fit_linear(x_rows: list[list[float]], y_values: list[float], *, ridge: float) -> tuple[float, list[float]]:
    width = len(TELEMETRY_LINEAR_FEATURES) + 1
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]
    for row, target in zip(x_rows, y_values):
        vector = [1.0, *row]
        for i in range(width):
            xty[i] += vector[i] * target
            for j in range(width):
                xtx[i][j] += vector[i] * vector[j]
    for i in range(1, width):
        xtx[i][i] += ridge
    solved = _solve_linear_system(xtx, xty)
    return solved[0], solved[1:]


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-9:
            augmented[col][col] += 1e-6
            pivot = col
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col] or 1e-6
        for item in range(col, n + 1):
            augmented[col][item] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for item in range(col, n + 1):
                augmented[row][item] -= factor * augmented[col][item]
    return [augmented[row][n] for row in range(n)]


def _head_clamp(head: str) -> list[float]:
    return {
        "pace_delta": [-1.20, 1.20],
        "pace_quantile": [0.78, 1.25],
        "dnf_hazard": [0.75, 1.50],
        "overtake": [-0.07, 0.10],
        "pit_value": [-0.25, 0.55],
    }.get(head, [-1.0, 1.0])


def _payloads_from_json(raw: Any) -> list[TelemetryFeaturePayload]:
    rows = raw if isinstance(raw, list) else [raw]
    payloads: list[TelemetryFeaturePayload] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate = row.get("telemetry_features") if isinstance(row.get("telemetry_features"), dict) else row
        try:
            payloads.append(TelemetryFeaturePayload.model_validate(candidate))
        except Exception:
            continue
    return payloads


def _training_rows_from_json(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        for key in ("telemetry_training_rows", "training_rows", "rows"):
            if isinstance(raw.get(key), list):
                return [row for row in raw[key] if isinstance(row, dict)]
        if isinstance(raw.get("features"), dict) and isinstance(raw.get("targets"), dict):
            return [raw]
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    return []


def _label_rows_from_json(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        for key in ("telemetry_label_rows", "label_rows", "labels", "rows"):
            if isinstance(raw.get(key), list):
                return [row for row in raw[key] if isinstance(row, dict)]
        if _label_key(raw):
            return [raw]
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    return []


def _label_key(row: dict[str, Any]) -> tuple[str, str, str] | None:
    race_id = str(row.get("race_id") or row.get("race") or "").strip()
    session = str(row.get("session") or "race").strip().lower()
    driver = str(row.get("driver_code") or row.get("driver") or "").strip().upper()
    if not race_id or not driver:
        return None
    return (race_id, session, driver)


def _target_values(row: dict[str, Any]) -> dict[str, float]:
    source = row.get("targets") if isinstance(row.get("targets"), dict) else row
    targets: dict[str, float] = {}
    for head in EXPECTED_HEADS:
        value = _safe_float(source.get(head)) if isinstance(source, dict) else None
        if value is not None:
            targets[head] = value
    return targets


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
