"""Input providers for the production ML simulator adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Protocol

from sports.f1.ml.artifacts.loader import (
    artifact_to_ml_trained_inputs,
    artifact_to_simulator_model_bundle,
    load_ml_trained_inputs,
    load_simulator_model_bundle,
)
from sports.f1.ml.simulator.model_contract import SimulatorModelBundle
from sports.f1.models.f1 import Constructor, Driver, Race
from sports.f1.predictor.config import DEFAULT_TOTAL_RACES
from sports.f1.predictor.features.sentiment import driver_sentiment_component
from sports.f1.predictor.scoring.normalization import clamp01, num


@dataclass
class MLRaceInputContext:
    race: Race
    drivers: list[Driver]
    constructors: list[Constructor]
    performance: dict[str, dict]
    driver_features: dict[str, dict]
    constructor_features: dict[str, dict]
    evidence: dict[str, Any]
    race_truth: dict[str, Any]
    track: dict[str, Any]
    weather: dict[str, Any]
    tires: dict[str, Any]
    reliability: dict[str, dict]
    sentiment: dict[str, Any]


@dataclass
class MLRaceInputPayload:
    initial_state: dict[str, Any]
    driver_inputs: dict[str, dict[str, Any]]
    input_source: str
    provider_sources: list[str]
    confidence: float
    fallback_reason: str | None = None
    trained_artifacts_used: bool = False
    evidence_groups_used: list[str] = field(default_factory=list)
    artifact_id: str | None = None
    artifact_version: str | None = None


class MLRaceInputProvider(Protocol):
    def build(self, context: MLRaceInputContext) -> MLRaceInputPayload:
        ...


class EvidenceFallbackRaceInputProvider:
    """Build simulator inputs from already-normalized production evidence."""

    def build(self, context: MLRaceInputContext) -> MLRaceInputPayload:
        initial_state, driver_inputs = _initial_state_from_evidence(context)
        return MLRaceInputPayload(
            initial_state=initial_state,
            driver_inputs=driver_inputs,
            input_source="evidence_fallback",
            provider_sources=["weekend_evidence", "race_truth", "production_features"],
            confidence=_input_confidence(context, trained=False),
            fallback_reason="trained_ml_artifacts_unavailable",
            trained_artifacts_used=False,
            evidence_groups_used=_evidence_groups(context),
        )


class TrainedArtifactRaceInputProvider:
    """Overlay optional trained artifact outputs on top of evidence fallback.

    v1 intentionally accepts plain dict overlays and optional fitted model
    objects so tests and future artifact loaders can supply the same shape
    without a new runtime dependency:

    {
      "pace_model": fitted object with predict/predict_with_quantiles,
      "dnf_model": fitted object with hazard_per_lap/predict_proba,
      "drivers": {
        "antonelli": {
          "pace_mean_seconds": 72.3,
          "pace_sigma_seconds": 0.35,
          "dnf_hazard_per_lap": 0.0012,
          "sources": ["trained_gbm_pace", "trained_gbm_dnf"],
          "confidence": 0.81
        }
      }
    }
    """

    def __init__(self, artifacts: dict[str, Any] | None = None, fallback: MLRaceInputProvider | None = None):
        self._artifacts = artifacts or {}
        self._fallback = fallback or EvidenceFallbackRaceInputProvider()

    def build(self, context: MLRaceInputContext) -> MLRaceInputPayload:
        payload = self._fallback.build(context)
        drivers = dict(self._artifacts.get("drivers") or {})
        model_rows, model_sources, model_confidence = _trained_model_driver_rows(self._artifacts, context)
        for driver_id, row in model_rows.items():
            merged = {**(drivers.get(driver_id) or {}), **row}
            drivers[driver_id] = merged
        if not drivers:
            if self._artifacts:
                payload.provider_sources = sorted(set(payload.provider_sources) | set(self._artifacts.get("sources") or []) | {"trained_artifact_unusable"})
                payload.fallback_reason = self._artifacts.get("artifact_load_error") or "trained_ml_artifacts_unusable"
                payload.artifact_id = self._artifacts.get("artifact_id")
                payload.artifact_version = self._artifacts.get("artifact_version")
            return payload

        codes = list(payload.initial_state.get("driver_codes") or [])
        means = list(payload.initial_state.get("driver_mean_pace_s") or [])
        sigmas = list(payload.initial_state.get("driver_pace_sigma_s") or [])
        dnfs = list(payload.initial_state.get("driver_dnf_rate_per_lap") or [])
        source_set = set(payload.provider_sources)
        source_set.update(model_sources)
        source_set.update(str(source) for source in (self._artifacts.get("sources") or []))
        confidence_values: list[float] = [payload.confidence]
        if model_confidence is not None:
            confidence_values.append(model_confidence)
        used = 0

        for index, driver in enumerate(context.drivers):
            if index >= len(codes):
                continue
            row = drivers.get(driver.id) or drivers.get(driver.code) or drivers.get(str(driver.number))
            if not row:
                continue
            pace = _maybe_num(row.get("pace_mean_seconds"))
            if pace is not None:
                means[index] = round(pace, 4)
                used += 1
            sigma = _maybe_num(row.get("pace_sigma_seconds") or row.get("pace_sigma"))
            if sigma is not None:
                sigmas[index] = round(max(0.08, min(2.25, sigma)), 4)
                used += 1
            dnf = _maybe_num(row.get("dnf_hazard_per_lap") or row.get("dnf_rate_per_lap"))
            if dnf is not None:
                dnfs[index] = round(max(0.00005, min(0.05, dnf)), 6)
                used += 1
            for source in row.get("sources") or []:
                source_set.add(str(source))
            if row.get("confidence") is not None:
                confidence_values.append(float(row.get("confidence") or 0.0))
            payload.driver_inputs.setdefault(driver.id, {}).update({
                "trained_pace_mean_seconds": pace,
                "trained_pace_sigma_seconds": sigma,
                "trained_dnf_hazard_per_lap": dnf,
                "trained_source": row.get("sources") or [],
            })

        if used == 0:
            if self._artifacts:
                payload.provider_sources = sorted(source_set | {"trained_artifact_unusable"})
                payload.fallback_reason = "trained_ml_artifacts_unusable"
            return payload

        payload.initial_state["driver_mean_pace_s"] = means
        payload.initial_state["driver_pace_sigma_s"] = sigmas
        payload.initial_state["driver_dnf_rate_per_lap"] = dnfs
        return MLRaceInputPayload(
            initial_state=payload.initial_state,
            driver_inputs=payload.driver_inputs,
            input_source="trained_artifacts",
            provider_sources=sorted(source_set | {"trained_artifact_overlay"}),
            confidence=round(max(payload.confidence, min(0.94, mean(confidence_values))), 4),
            fallback_reason=None,
            trained_artifacts_used=True,
            evidence_groups_used=payload.evidence_groups_used,
            artifact_id=self._artifacts.get("artifact_id"),
            artifact_version=self._artifacts.get("artifact_version"),
        )


def select_ml_input_provider(features: dict[str, Any] | None) -> MLRaceInputProvider:
    features = features or {}
    provider = features.get("ml_input_provider")
    if provider and hasattr(provider, "build"):
        return provider
    return TrainedArtifactRaceInputProvider(
        artifacts=_artifact_inputs_from_features(features),
        fallback=EvidenceFallbackRaceInputProvider(),
    )


def select_simulator_model_bundle(features: dict[str, Any] | None) -> SimulatorModelBundle | None:
    features = features or {}
    bundle = features.get("ml_model_bundle") or features.get("simulator_model_bundle")
    if isinstance(bundle, SimulatorModelBundle):
        return bundle
    if features.get("ml_artifact_bundle"):
        return artifact_to_simulator_model_bundle(features.get("ml_artifact_bundle"))
    path = features.get("ml_artifact_path") or os.environ.get("F1_ML_ARTIFACT_PATH")
    if path:
        return load_simulator_model_bundle(str(path))
    return None


def _artifact_inputs_from_features(features: dict[str, Any]) -> dict[str, Any]:
    if features.get("ml_trained_inputs"):
        return dict(features.get("ml_trained_inputs") or {})
    if features.get("ml_artifact_bundle"):
        return artifact_to_ml_trained_inputs(features.get("ml_artifact_bundle"))
    if features.get("ml_artifacts"):
        return dict(features.get("ml_artifacts") or {})
    path = features.get("ml_artifact_path") or os.environ.get("F1_ML_ARTIFACT_PATH")
    if path:
        return load_ml_trained_inputs(str(path))
    return {}


def _trained_model_driver_rows(artifacts: dict[str, Any], context: MLRaceInputContext) -> tuple[dict[str, dict[str, Any]], set[str], float | None]:
    rows: dict[str, dict[str, Any]] = {}
    sources: set[str] = set()
    confidence_values: list[float] = []
    feature_rows = _trained_feature_rows(context)
    if not feature_rows:
        return rows, sources, None

    pace_model = artifacts.get("pace_model") or artifacts.get("gbm_pace_model")
    if pace_model is not None:
        pace_rows, pace_sources, pace_conf = _predict_pace_artifact(pace_model, feature_rows, artifacts)
        sources.update(pace_sources)
        if pace_conf is not None:
            confidence_values.append(pace_conf)
        for driver_id, row in pace_rows.items():
            rows.setdefault(driver_id, {}).update(row)

    dnf_model = artifacts.get("dnf_model") or artifacts.get("gbm_dnf_model") or artifacts.get("survival_dnf_model")
    if dnf_model is not None:
        dnf_rows, dnf_sources, dnf_conf = _predict_dnf_artifact(dnf_model, feature_rows, artifacts)
        sources.update(dnf_sources)
        if dnf_conf is not None:
            confidence_values.append(dnf_conf)
        for driver_id, row in dnf_rows.items():
            rows.setdefault(driver_id, {}).update(row)

    return rows, sources, round(mean(confidence_values), 4) if confidence_values else None


def _trained_feature_rows(context: MLRaceInputContext) -> list[dict[str, Any]]:
    rows = []
    for driver in context.drivers:
        feature = context.driver_features.get(driver.id) or {}
        perf = context.performance.get(driver.id) or {}
        reliability = context.reliability.get(driver.id) or {}
        evidence_row = (context.evidence.get("drivers") or {}).get(driver.id) or {}
        practice = evidence_row.get("practice") or {}
        grid = evidence_row.get("grid") or {}
        race_inputs = evidence_row.get("race_inputs") or {}
        truth = (context.race_truth.get("by_driver_id") or {}).get(driver.id) or {}
        rows.append({
            "driver_id": driver.id,
            "driver_code": driver.code,
            "driver_number": driver.number or 0,
            "championship_points": float(driver.points or 0.0),
            "championship_position": float(driver.position or 20),
            "form_score": num(feature.get("form_score"), perf.get("form_score") or 0.50),
            "driver_skill_score": num(perf.get("driver_skill_score"), feature.get("form_score") or 0.50),
            "car_performance_score": num(perf.get("car_performance_score"), 0.50),
            "performance_score": num(perf.get("performance_score"), 0.50),
            "qualifying_pace_score": num(feature.get("qualifying_pace_score"), perf.get("qualifying_pace_score") or 0.50),
            "race_pace_score": num(feature.get("race_pace_score"), perf.get("race_pace_score") or 0.50),
            "reliability_score": num(reliability.get("reliability_score"), feature.get("reliability_score") or 0.75),
            "practice_best_lap": _maybe_num(practice.get("best_lap")) or 0.0,
            "practice_representative_lap": _maybe_num(practice.get("representative_lap")) or 0.0,
            "practice_long_run_lap": _maybe_num(practice.get("long_run_lap")) or 0.0,
            "practice_lap_count": float(practice.get("lap_count") or 0.0),
            "practice_fuel_uncertainty": num(practice.get("fuel_uncertainty"), 0.35),
            "practice_pace_stability": num(practice.get("pace_stability"), 0.50),
            "grid_position": float(grid.get("grid_position") or feature.get("grid_position") or 20),
            "grid_penalty": float(grid.get("grid_penalty") or feature.get("grid_penalty") or 0),
            "pit_lane_start": 1.0 if grid.get("pit_lane_start") or feature.get("pit_lane_start") else 0.0,
            "live_position": float(truth.get("position") or race_inputs.get("position") or 20),
            "live_gap_to_leader": _gap_seconds(truth.get("gap_to_leader") or race_inputs.get("gap_to_leader")) or 0.0,
            "tyre_age": float(race_inputs.get("tyre_age") or 0.0),
            "pit_stops": float(race_inputs.get("pit_stops") or 0.0),
            "track_length_km": float(context.track.get("length_km") or 0.0),
            "track_laps": float(context.track.get("laps") or _fallback_laps(context.race)),
            "overtaking_difficulty": num(context.track.get("overtaking_difficulty"), 0.50),
            "qualifying_importance": num(context.track.get("qualifying_importance"), 0.58),
            "tire_stress": num(context.track.get("tire_stress"), 0.50),
            "pit_loss": num(context.track.get("pit_loss"), 23.0),
            "rain_probability": num(context.weather.get("rain_probability"), 0.0),
            "weather_chaos": num(context.weather.get("chaos_score"), 0.0),
            "degradation_rate": num(context.tires.get("degradation_rate"), num(context.track.get("tire_stress"), 0.50)),
        })
    return rows


def _predict_pace_artifact(model: Any, feature_rows: list[dict[str, Any]], artifacts: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str], float | None]:
    sources = {"trained_gbm_pace"}
    try:
        frame = _to_model_frame(feature_rows, artifacts)
        if hasattr(model, "predict_with_quantiles"):
            alphas = tuple(artifacts.get("pace_quantiles") or (0.1, 0.5, 0.9))
            predictions = model.predict_with_quantiles(frame, alphas=alphas)
            rows = {}
            median_index = min(range(len(alphas)), key=lambda idx: abs(float(alphas[idx]) - 0.5))
            low_index = min(range(len(alphas)), key=lambda idx: abs(float(alphas[idx]) - 0.1))
            high_index = min(range(len(alphas)), key=lambda idx: abs(float(alphas[idx]) - 0.9))
            for index, feature in enumerate(feature_rows):
                low = float(predictions[index][low_index])
                median = float(predictions[index][median_index])
                high = float(predictions[index][high_index])
                rows[feature["driver_id"]] = {
                    "pace_mean_seconds": median,
                    "pace_sigma_seconds": max(0.08, abs(high - low) / 2.563),
                    "sources": ["trained_gbm_pace", "trained_gbm_pace_quantiles"],
                    "confidence": float(artifacts.get("pace_model_confidence") or 0.78),
                }
            return rows, sources | {"trained_gbm_pace_quantiles"}, float(artifacts.get("pace_model_confidence") or 0.78)
        if hasattr(model, "predict"):
            predictions = model.predict(frame)
            rows = {
                feature["driver_id"]: {
                    "pace_mean_seconds": float(predictions[index]),
                    "sources": ["trained_gbm_pace"],
                    "confidence": float(artifacts.get("pace_model_confidence") or 0.74),
                }
                for index, feature in enumerate(feature_rows)
            }
            return rows, sources, float(artifacts.get("pace_model_confidence") or 0.74)
    except Exception:
        return {}, {"trained_gbm_pace_unusable"}, None
    return {}, {"trained_gbm_pace_unusable"}, None


def _predict_dnf_artifact(model: Any, feature_rows: list[dict[str, Any]], artifacts: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str], float | None]:
    sources = {"trained_gbm_dnf"}
    try:
        frame = _to_model_frame(feature_rows, artifacts)
        if hasattr(model, "hazard_per_lap"):
            predictions = model.hazard_per_lap(frame)
        elif hasattr(model, "predict_proba"):
            proba = model.predict_proba(frame)
            predictions = [row[1] if len(row) > 1 else row[0] for row in proba]
        elif callable(model):
            predictions = model(frame)
        else:
            return {}, {"trained_dnf_unusable"}, None
        rows = {
            feature["driver_id"]: {
                "dnf_hazard_per_lap": float(predictions[index]),
                "sources": ["trained_gbm_dnf"],
                "confidence": float(artifacts.get("dnf_model_confidence") or 0.72),
            }
            for index, feature in enumerate(feature_rows)
        }
        return rows, sources, float(artifacts.get("dnf_model_confidence") or 0.72)
    except Exception:
        return {}, {"trained_dnf_unusable"}, None


def _to_model_frame(feature_rows: list[dict[str, Any]], artifacts: dict[str, Any]) -> Any:
    columns = artifacts.get("feature_columns")
    numeric_rows = [
        {key: value for key, value in row.items() if key not in {"driver_id", "driver_code"}}
        for row in feature_rows
    ]
    try:
        import pandas as pd

        frame = pd.DataFrame(numeric_rows)
        if columns:
            for column in columns:
                if column not in frame.columns:
                    frame[column] = 0.0
            frame = frame[list(columns)]
        return frame
    except Exception:
        if columns:
            return [{column: row.get(column, 0.0) for column in columns} for row in numeric_rows]
        return numeric_rows


def _initial_state_from_evidence(context: MLRaceInputContext) -> tuple[dict[str, Any], dict[str, Any]]:
    base_lap = _base_lap_seconds(context.evidence, context.track)
    field_strengths = []
    rows: list[dict[str, Any]] = []
    constructors_by_name = {constructor.name.lower(): constructor for constructor in context.constructors}
    max_constructor_points = max((constructor.points for constructor in context.constructors), default=1.0) or 1.0

    for driver in context.drivers:
        feature = context.driver_features.get(driver.id) or {}
        perf = context.performance.get(driver.id) or {}
        constructor = constructors_by_name.get((driver.team or "").lower())
        constructor_feature = context.constructor_features.get((driver.team or "").lower()) or {}
        team_score = num(constructor_feature.get("team_score"), (constructor.points / max_constructor_points) if constructor else 0.50)
        evidence_row = ((context.evidence.get("drivers") or {}).get(driver.id) or {})
        practice = evidence_row.get("practice") or {}
        grid = evidence_row.get("grid") or {}
        race_inputs = evidence_row.get("race_inputs") or {}
        truth_state = ((context.race_truth.get("by_driver_id") or {}).get(driver.id) or {})
        reliability_row = context.reliability.get(driver.id) or {}
        sentiment_component = driver_sentiment_component(driver.id, driver.team, context.sentiment)

        strength = clamp01(
            0.16 * num(perf.get("driver_skill_score"), feature.get("form_score") or 0.50)
            + 0.15 * team_score
            + 0.24 * num(perf.get("performance_score"), feature.get("form_score") or 0.50)
            + 0.14 * num(feature.get("race_pace_score"), perf.get("race_pace_score") or 0.50)
            + 0.12 * num(feature.get("qualifying_pace_score"), perf.get("qualifying_pace_score") or 0.50)
            + 0.10 * num(practice.get("race_evidence_score"), feature.get("race_pace_score") or 0.50)
            + 0.05 * num(reliability_row.get("finish_probability"), feature.get("reliability_score") or 0.75)
            + 0.04 * clamp01(0.50 + num(sentiment_component.get("race_sentiment_delta"), 0.0))
        )
        grid_position = _int(grid.get("grid_position") or feature.get("grid_position"))
        if grid_position:
            grid_score = max(0.0, 1.0 - (grid_position - 1) / max(len(context.drivers), 1))
            grid_weight = 0.10 + 0.18 * float(context.track.get("qualifying_importance") or 0.58)
            strength = clamp01((1.0 - grid_weight) * strength + grid_weight * grid_score)
            if grid.get("pit_lane_start"):
                strength *= 0.76
            elif grid.get("grid_penalty"):
                strength *= max(0.82, 1.0 - min(8, int(grid.get("grid_penalty") or 0)) * 0.025)

        source_mode = str(context.race_truth.get("source_mode") or context.evidence.get("source_mode") or "")
        live_position = _int(truth_state.get("position") or race_inputs.get("position"))
        if live_position and source_mode in {"live", "recorded", "recent"}:
            live_conf = min(0.85, max(float(context.race_truth.get("confidence") or 0.0), float(race_inputs.get("confidence") or 0.0)))
            live_score = max(0.0, 1.0 - (live_position - 1) / max(len(context.drivers), 1))
            strength = clamp01((1.0 - 0.42 * live_conf) * strength + (0.42 * live_conf) * live_score)

        practice_lap = _maybe_num(practice.get("long_run_lap") or practice.get("representative_lap") or practice.get("best_lap"))
        mean_pace = practice_lap if practice_lap else base_lap - (strength - 0.50) * 3.8
        gap = _gap_seconds(truth_state.get("gap_to_leader") or race_inputs.get("gap_to_leader"))
        if gap is not None and source_mode in {"live", "recorded", "recent"}:
            mean_pace += min(1.8, gap / max(8.0, float(context.track.get("laps") or 58.0)) * 0.65)

        stability = num(practice.get("pace_stability"), 0.55)
        fuel_uncertainty = num(practice.get("fuel_uncertainty"), 0.35)
        sigma = max(0.18, min(1.65, 0.52 + (1.0 - stability) * 0.42 + fuel_uncertainty * 0.55))
        sigma *= 1.0 + min(0.40, num(context.weather.get("chaos_score"), 0.0) * 0.55)
        dnf_rate = max(0.0002, min(0.020, num(reliability_row.get("dnf_probability"), 0.12) / max(1, int(context.track.get("laps") or 58))))

        rows.append({
            "driver": driver,
            "strength": strength,
            "mean_pace": mean_pace,
            "sigma": sigma,
            "dnf_rate": dnf_rate,
            "compound": _starting_compound(race_inputs, practice, context.tires),
            "pit_lap": _expected_pit_lap(context.track, context.tires, race_inputs),
            "pit_compound": _pit_compound(_starting_compound(race_inputs, practice, context.tires), context.weather),
        })
        field_strengths.append(strength)

    field_mean_strength = mean(field_strengths) if field_strengths else 0.50
    for row in rows:
        row["mean_pace"] += (field_mean_strength - row["strength"]) * 1.1

    return {
        "driver_codes": [row["driver"].code for row in rows],
        "driver_mean_pace_s": [round(float(row["mean_pace"]), 4) for row in rows],
        "driver_pace_sigma_s": [round(float(row["sigma"]), 4) for row in rows],
        "driver_dnf_rate_per_lap": [round(float(row["dnf_rate"]), 6) for row in rows],
        "driver_starting_compound": [row["compound"] for row in rows],
        "driver_pit_lap": [row["pit_lap"] for row in rows],
        "driver_pit_compound": [row["pit_compound"] for row in rows],
        "total_laps": int(context.track.get("laps") or _fallback_laps(context.race)),
    }, {
        row["driver"].id: {
            "strength": round(row["strength"], 4),
            "mean_pace": round(row["mean_pace"], 4),
            "sigma": round(row["sigma"], 4),
            "dnf_rate_per_lap": round(row["dnf_rate"], 6),
            "starting_compound": row["compound"],
            "pit_lap": row["pit_lap"],
            "pit_compound": row["pit_compound"],
        }
        for row in rows
    }


def _input_confidence(context: MLRaceInputContext, trained: bool) -> float:
    coverage = context.evidence.get("coverage_counts") or {}
    field_size = max(len(context.drivers), 1)
    base = 0.23
    base += min(0.24, float(coverage.get("practice_drivers") or 0) / field_size * 0.24)
    base += min(0.20, float(coverage.get("grid_drivers") or 0) / field_size * 0.20)
    base += min(0.16, float(coverage.get("race_input_drivers") or 0) / field_size * 0.16)
    base += min(0.12, float(context.race_truth.get("confidence") or 0.0) * 0.12)
    base += 0.07 if context.weather and not context.weather.get("missing_data") else 0.0
    base += 0.06 if context.tires and not context.tires.get("missing_data") else 0.0
    base += 0.08 if context.track and not context.track.get("missing_data") else 0.03
    if trained:
        base += 0.08
    return round(max(0.20, min(0.94 if trained else 0.92, base)), 4)


def _evidence_groups(context: MLRaceInputContext) -> list[str]:
    groups = []
    coverage = context.evidence.get("coverage_counts") or {}
    if coverage.get("practice_drivers"):
        groups.append("practice")
    if coverage.get("grid_drivers"):
        groups.append("grid")
    if coverage.get("race_input_drivers"):
        groups.append("race_inputs")
    if context.race_truth.get("source_mode"):
        groups.append("race_truth")
    if context.weather and not context.weather.get("missing_data"):
        groups.append("weather")
    if context.tires and not context.tires.get("missing_data"):
        groups.append("tires")
    if context.track and not context.track.get("missing_data"):
        groups.append("track")
    return sorted(set(groups or ["production_features"]))


def _base_lap_seconds(evidence: dict, track: dict) -> float:
    values = []
    for row in ((evidence.get("drivers") or {}).values()):
        practice = row.get("practice") or {}
        for key in ("representative_lap", "long_run_lap", "best_lap"):
            value = _maybe_num(practice.get(key))
            if value and 45.0 <= value <= 140.0:
                values.append(value)
    if values:
        return mean(values)
    length = _maybe_num(track.get("length_km"))
    if length:
        return max(62.0, min(116.0, length * 16.2))
    return 88.0


def _starting_compound(race_inputs: dict, practice: dict, tires: dict) -> str:
    raw = str(race_inputs.get("compound") or "").upper()
    if raw in {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}:
        return raw
    compounds = [str(item).upper() for item in (practice.get("compounds") or [])]
    for compound in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"):
        if compound in compounds:
            return compound
    compound_set = [str(item).upper() for item in (tires.get("compound_set") or [])]
    if any(item in {"C4", "C5", "SOFT"} for item in compound_set):
        return "SOFT"
    if any(item in {"C1", "HARD"} for item in compound_set):
        return "MEDIUM"
    return "MEDIUM"


def _expected_pit_lap(track: dict, tires: dict, race_inputs: dict) -> int | None:
    if _int(race_inputs.get("pit_stops")) and _int(race_inputs.get("lap")):
        return None
    laps = int(track.get("laps") or 58)
    degradation = num(tires.get("degradation_rate"), num(track.get("tire_stress"), 0.50))
    target = int(laps * (0.33 if degradation >= 0.68 else 0.42 if degradation >= 0.48 else 0.52))
    return max(8, min(laps - 6, target))


def _pit_compound(starting: str, weather: dict) -> str:
    if num(weather.get("rain_probability"), 0.0) >= 0.45:
        return "INTERMEDIATE"
    if starting == "SOFT":
        return "HARD"
    if starting == "HARD":
        return "MEDIUM"
    return "HARD"


def _fallback_laps(race: Race) -> int:
    text = f"{race.name} {race.circuit} {race.country}".lower()
    if "monaco" in text:
        return 78
    if "spa" in text:
        return 44
    if "monza" in text:
        return 53
    return 58


def _gap_seconds(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("+", "")
    if text in {"", "leader", "interval", "lapped"}:
        return None
    if "lap" in text:
        return 90.0
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _maybe_num(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
