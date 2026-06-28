"""Production model wrapper that applies telemetry analysis before simulation."""

from __future__ import annotations

from typing import Any

from sports.f1.ml.simulator.race_sim import simulate_race
from sports.f1.ml.telemetry import (
    apply_telemetry_adjustments,
    build_telemetry_features_from_openf1_session,
    build_telemetry_model_output,
)
from sports.f1.ml.telemetry.policy import telemetry_rollout_policy
from sports.f1.ml.telemetry.types import TelemetryFeaturePayload
from sports.f1.models.f1 import Constructor, Driver, Race, RacePrediction
from sports.f1.predictor.models.configs import RaceModelConfig, get_model_config
from sports.f1.predictor.models.ml_simulator import _build_simulation_context, _map_result_to_prediction, _ml_race


class TelemetrySimulatorRaceModel:
    """Selectable production model backed by telemetry-adjusted ML simulation."""

    def __init__(self, config: RaceModelConfig | None = None, model_id: str | None = None):
        self.config = config or get_model_config(model_id or "telemetry_simulator_v1")
        self.model_id = self.config.model_id
        self.model_version = self.config.model_version

    def predict(
        self,
        race: Race,
        drivers: list[Driver],
        constructors: list[Constructor],
        features: dict,
        sentiment: dict,
    ) -> RacePrediction:
        if not drivers:
            return RacePrediction(model_version=self.model_version, model_id=self.model_id)

        context = _build_simulation_context(race, drivers, constructors, features or {}, sentiment or {})
        telemetry_payload = _telemetry_payload(race, context, features or {})
        stage = _telemetry_stage(features or {})
        telemetry_output = build_telemetry_model_output(
            telemetry_payload,
            stage=stage,
            model_id=self.model_id,
            model_version=self.model_version,
            artifact_path=(features or {}).get("telemetry_artifact_path"),
        )
        policy = telemetry_rollout_policy((features or {}).get("telemetry_policy") or {})
        guard = (telemetry_output.metadata or {}).get("telemetry_leakage_guard_status") or {}
        if guard.get("status") == "blocked":
            adjusted_state, telemetry_meta = context["initial_state"], _telemetry_policy_metadata(
                telemetry_output,
                policy,
                reason="telemetry_leakage_guard_blocked",
            )
            telemetry_meta["telemetry_leakage_guard_status"] = guard
        elif not policy["enabled"]:
            adjusted_state, telemetry_meta = context["initial_state"], _telemetry_policy_metadata(
                telemetry_output,
                policy,
                reason="telemetry_model_disabled",
            )
        elif policy["warn_only"]:
            adjusted_state, telemetry_meta = context["initial_state"], _telemetry_policy_metadata(
                telemetry_output,
                policy,
                reason="telemetry_warn_only",
            )
        else:
            adjusted_state, telemetry_meta = apply_telemetry_adjustments(
                context["initial_state"],
                telemetry_output,
                min_confidence=policy["min_confidence"],
            )
            telemetry_meta.update({
                "telemetry_policy": policy,
                "telemetry_warn_only": False,
                "telemetry_leakage_guard_status": guard,
            })
        context["initial_state"] = adjusted_state
        context["ml_metadata"].update({
            **telemetry_meta,
            "telemetry_probability_deltas": telemetry_output.probability_delta_explanations,
            "telemetry_raw_counts": telemetry_output.metadata.get("raw_counts") or {},
        })
        result = simulate_race(
            race=_ml_race(race, context["track"]),
            initial_state=context["initial_state"],
            models=context["models"],
            config=context["config"],
        )
        prediction = _map_result_to_prediction(
            race=race,
            drivers=drivers,
            constructors=constructors,
            features=context["features"],
            sentiment=sentiment or {},
            performance=context["performance"],
            reliability=context["reliability"],
            result=result,
            model_version=self.model_version,
            model_id=self.model_id,
            confidence=max(float(context["confidence"] or 0.0), float(telemetry_payload.confidence or 0.0)),
            evidence=context["evidence"],
            track=context["track"],
            weather=context["weather"],
            tires=context["tires"],
            ml_metadata=context["ml_metadata"],
        )
        prediction.telemetry_model_used = telemetry_meta.get("telemetry_model_used")
        prediction.telemetry_model_id = telemetry_meta.get("telemetry_model_id")
        prediction.telemetry_model_version = telemetry_meta.get("telemetry_model_version")
        prediction.telemetry_confidence = telemetry_meta.get("telemetry_confidence")
        prediction.telemetry_source_mode = telemetry_meta.get("telemetry_source_mode")
        prediction.telemetry_missing_groups = telemetry_meta.get("telemetry_missing_groups") or []
        prediction.telemetry_fallback_reason = telemetry_meta.get("telemetry_fallback_reason")
        prediction.telemetry_warn_only = telemetry_meta.get("telemetry_warn_only")
        prediction.telemetry_policy = telemetry_meta.get("telemetry_policy") or {}
        prediction.telemetry_leakage_guard_status = telemetry_meta.get("telemetry_leakage_guard_status") or {}
        prediction.telemetry_probability_deltas = telemetry_output.probability_delta_explanations
        return prediction


def _telemetry_payload(race: Race, context: dict[str, Any], features: dict[str, Any]) -> TelemetryFeaturePayload:
    explicit = features.get("telemetry_features")
    if isinstance(explicit, TelemetryFeaturePayload):
        return explicit
    if isinstance(explicit, dict):
        try:
            return TelemetryFeaturePayload.model_validate(explicit)
        except Exception:
            pass
    race_id = f"{race.date.year}-{int(race.round):02d}-{str(context.get('track', {}).get('track_key') or race.circuit_id or race.country or 'F1').upper()}"
    openf1_session = features.get("openf1_session") or {}
    live = bool((features.get("race_truth") or {}).get("source_mode") == "live" or features.get("live_state"))
    return build_telemetry_features_from_openf1_session(
        race_id=race_id,
        session=str(features.get("session") or openf1_session.get("session") or "race"),
        openf1_session=openf1_session,
        live=live,
    )


def _telemetry_stage(features: dict[str, Any]) -> str:
    truth = features.get("race_truth") or {}
    evidence = features.get("weekend_evidence") or {}
    if truth.get("source_mode") == "live" or features.get("live_state"):
        return "live"
    if (evidence.get("grid") or {}).get("available"):
        return "post_qualifying"
    if (evidence.get("practice") or {}).get("available") or features.get("openf1_session"):
        return "practice_available"
    return "pre_weekend"


def _telemetry_policy_metadata(output, policy: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "telemetry_model_used": False,
        "telemetry_model_id": output.model_id if output else None,
        "telemetry_model_version": output.model_version if output else None,
        "telemetry_confidence": output.confidence if output else None,
        "telemetry_source_mode": output.source_mode if output else None,
        "telemetry_missing_groups": output.missing_groups if output else [],
        "telemetry_fallback_reason": reason,
        "telemetry_policy": policy,
        "telemetry_warn_only": bool(policy.get("warn_only")),
        "telemetry_leakage_guard_status": (output.metadata or {}).get("telemetry_leakage_guard_status") if output else {},
    }
