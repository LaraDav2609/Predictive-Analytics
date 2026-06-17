"""Production adapter for the research F1 Monte Carlo simulator.

This model is intentionally registered behind a model id instead of replacing
the heuristic production model. It converts the current production evidence
shape (Weekend Evidence, Race Truth, car/track/weather/tyre features) into the
`sports.f1.ml` simulator's compact initial state, runs the simulator, and maps
the empirical finish distribution back into the existing RacePrediction DTO.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sports.f1.ml.common.types import Race as MLSimRace
from sports.f1.ml.markets.mapper import dnf_probabilities, podium_probabilities, top_k_probabilities, winner_probabilities
from sports.f1.ml.simulator.race_sim import PhysicalParams, SimConfig, SimResult, simulate_race
from sports.f1.models.f1 import Constructor, Driver, DriverRacePrediction, Race, RacePrediction
from sports.f1.predictor.config import DEFAULT_TOTAL_RACES
from sports.f1.predictor.features.performance import build_performance_table
from sports.f1.predictor.features.reliability import ReliabilityFeatureProvider
from sports.f1.predictor.features.sentiment import driver_sentiment_component
from sports.f1.predictor.features.tires import TireFeatureProvider
from sports.f1.predictor.features.track import TrackFeatureProvider
from sports.f1.predictor.features.weather import WeatherFeatureProvider
from sports.f1.predictor.features.weekend import apply_weekend_evidence_adjustments, normalize_weekend_session
from sports.f1.predictor.models.configs import RaceModelConfig, get_model_config
from sports.f1.predictor.models.ml_inputs import MLRaceInputContext, select_ml_input_provider, select_simulator_model_bundle
from sports.f1.predictor.scoring.explanations import explain_driver
from sports.f1.predictor.scoring.normalization import clamp01, num


class MLSimulatorRaceModel:
    """Selectable F1 model backed by the `sports.f1.ml` Monte Carlo simulator."""

    def __init__(self, config: RaceModelConfig | None = None, model_id: str | None = None):
        self.config = config or get_model_config(model_id or "ml_simulator_v1")
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
            return RacePrediction(model_version=self.model_version)

        context = _build_simulation_context(race, drivers, constructors, features, sentiment)
        result = simulate_race(
            race=_ml_race(race, context["track"]),
            initial_state=context["initial_state"],
            models=context["models"],
            config=context["config"],
        )
        return _map_result_to_prediction(
            race=race,
            drivers=drivers,
            constructors=constructors,
            features=context["features"],
            sentiment=sentiment,
            performance=context["performance"],
            reliability=context["reliability"],
            result=result,
            model_version=self.model_version,
            model_id=self.model_id,
            confidence=context["confidence"],
            evidence=context["evidence"],
            track=context["track"],
            weather=context["weather"],
            tires=context["tires"],
            ml_metadata=context["ml_metadata"],
        )


def _build_simulation_context(
    race: Race,
    drivers: list[Driver],
    constructors: list[Constructor],
    features: dict,
    sentiment: dict,
) -> dict[str, Any]:
    features = features or {}
    evidence = features.get("weekend_evidence") or {}
    race_truth = features.get("race_truth") or {}
    session = normalize_weekend_session(evidence.get("session") or race_truth.get("session") or features.get("session") or "race")

    driver_features = apply_weekend_evidence_adjustments(
        drivers,
        features.get("drivers") or {},
        evidence,
        session_stage=session,
    )
    adjusted_features = {**features, "drivers": driver_features}
    constructor_features = features.get("constructors") or {}
    performance = build_performance_table(drivers, constructors, driver_features, constructor_features)
    track = TrackFeatureProvider(features).get_features(race)
    weather = WeatherFeatureProvider(features).get_features(race, features.get("openf1_session") or {}, session=session)
    tires = TireFeatureProvider().get_features(track, features.get("openf1_session") or {}, weather=weather)
    reliability = ReliabilityFeatureProvider(driver_features).get_features(track, weather)
    input_provider = select_ml_input_provider(features)
    model_bundle = select_simulator_model_bundle(features)
    input_payload = input_provider.build(MLRaceInputContext(
        race=race,
        drivers=drivers,
        constructors=constructors,
        performance=performance,
        driver_features=driver_features,
        constructor_features=constructor_features,
        evidence=evidence,
        race_truth=race_truth,
        track=track,
        weather=weather,
        tires=tires,
        reliability=reliability,
        sentiment=sentiment,
    ))
    confidence = input_payload.confidence
    physical = PhysicalParams(
        pit_loss_s=float(track.get("pit_loss") or 23.0),
        dirty_air_threshold_s=1.2 + float(track.get("overtaking_difficulty") or 0.5) * 0.8,
        dirty_air_penalty_per_s_close=0.34 + float(track.get("overtaking_difficulty") or 0.5) * 0.22,
        dirty_air_max_penalty_s=0.70 + float(track.get("overtaking_difficulty") or 0.5) * 0.40,
    )
    physical.deg_per_lap_per_compound.update(_compound_deg_from_tires(tires))
    return {
        "features": adjusted_features,
        "performance": performance,
        "reliability": reliability,
        "track": track,
        "weather": weather,
        "tires": tires,
        "evidence": evidence,
        "initial_state": input_payload.initial_state,
        "models": {
            "adapter": "production_ml_input_provider_v1",
            "inputs": input_payload.driver_inputs,
            "source": input_payload.input_source,
            "provider_sources": input_payload.provider_sources,
            "fallback_reason": input_payload.fallback_reason,
            "trained_artifacts_used": input_payload.trained_artifacts_used,
            "simulator_model_bundle": model_bundle,
        },
        "config": SimConfig(
            n_iterations=int(features.get("ml_simulator_iterations") or 2400),
            seed=_seed_for_race(race),
            enable_physical=True,
            physical=physical,
        ),
        "confidence": confidence,
        "ml_metadata": {
            "ml_input_source": input_payload.input_source,
            "ml_provider_sources": input_payload.provider_sources,
            "ml_fallback_reason": input_payload.fallback_reason,
            "ml_confidence": input_payload.confidence,
            "trained_artifacts_used": input_payload.trained_artifacts_used,
            "evidence_groups_used": input_payload.evidence_groups_used,
            "simulator_iterations": int(features.get("ml_simulator_iterations") or 2400),
            "ml_artifact_id": input_payload.artifact_id,
            "ml_artifact_version": input_payload.artifact_version,
            **(model_bundle.source_summary() if model_bundle is not None else {"ml_model_contract_used": False}),
        },
    }


def _map_result_to_prediction(
    *,
    race: Race,
    drivers: list[Driver],
    constructors: list[Constructor],
    features: dict,
    sentiment: dict,
    performance: dict[str, dict],
    reliability: dict[str, dict],
    result: SimResult,
    model_version: str,
    model_id: str,
    confidence: float,
    evidence: dict,
    track: dict,
    weather: dict,
    tires: dict,
    ml_metadata: dict,
) -> RacePrediction:
    winner = winner_probabilities(result)
    podium = podium_probabilities(result)
    top5 = top_k_probabilities(result, min(5, len(result.driver_codes)))
    dnf = dnf_probabilities(result)
    expected_by_code = {
        code: float(result.finish_positions[:, index].mean())
        for index, code in enumerate(result.driver_codes)
    }
    predicted_order = sorted(drivers, key=lambda driver: expected_by_code.get(driver.code, 99.0))
    predicted_positions = {driver.id: index + 1 for index, driver in enumerate(predicted_order)}
    wdc = _wdc_probabilities(drivers, winner, features)
    driver_features = features.get("drivers") or {}
    predictions: dict[str, DriverRacePrediction] = {}

    for driver in drivers:
        feature = driver_features.get(driver.id) or {}
        perf = performance.get(driver.id) or {}
        rel = reliability.get(driver.id) or {}
        sentiment_component = driver_sentiment_component(driver.id, driver.team, sentiment)
        component = {
            "form_score": num(feature.get("form_score"), perf.get("form_score") or 0.50),
            "team_score": num(perf.get("car_team_score"), perf.get("car_performance_score") or 0.50),
            "driver_skill_score": num(perf.get("driver_skill_score"), feature.get("form_score") or 0.50),
            "car_performance_score": num(perf.get("car_performance_score"), 0.50),
            "performance_score": num(perf.get("performance_score"), 0.50),
            "qualifying_pace_score": num(feature.get("qualifying_pace_score"), perf.get("qualifying_pace_score") or 0.50),
            "race_pace_score": num(feature.get("race_pace_score"), perf.get("race_pace_score") or 0.50),
            "track_fit_score": num(feature.get("track_fit_score"), 0.50),
            "tire_strategy_score": clamp01(0.55 * num(feature.get("race_pace_score"), 0.50) + 0.45 * (1.0 - num(tires.get("degradation_rate"), 0.50))),
            "weather_risk_score": clamp01(1.0 - num(weather.get("chaos_score"), 0.0) * 0.80),
            "reliability_score": num(rel.get("reliability_score"), feature.get("reliability_score") or 0.75),
            "dnf_probability": dnf.get(driver.code, num(rel.get("dnf_probability"), 0.12)),
            **sentiment_component,
        }
        explanation = explain_driver(driver, component)
        explanation.append("ML simulator finish distribution used for win, podium, top-five, DNF, and expected finish.")
        predictions[driver.id] = DriverRacePrediction(
            driver_id=driver.id,
            driver_name=f"{driver.first_name} {driver.last_name}".strip(),
            win_prob=round(winner.get(driver.code, 0.0), 4),
            podium_prob=round(podium.get(driver.code, 0.0), 4),
            top5_prob=round(top5.get(driver.code, 0.0), 4),
            predicted_position=predicted_positions.get(driver.id),
            form_score=round(component["form_score"], 4),
            team_score=round(component["team_score"], 4),
            driver_skill_score=round(component["driver_skill_score"], 4),
            car_performance_score=round(component["car_performance_score"], 4),
            performance_score=round(component["performance_score"], 4),
            qualifying_pace_score=round(component["qualifying_pace_score"], 4),
            race_pace_score=round(component["race_pace_score"], 4),
            track_fit_score=round(component["track_fit_score"], 4),
            tire_strategy_score=round(component["tire_strategy_score"], 4),
            weather_risk_score=round(component["weather_risk_score"], 4),
            sentiment_score=round(component["sentiment_score"], 4),
            sentiment_label=component["sentiment_label"],
            sentiment_mentions=int(component["sentiment_mentions"]),
            personal_news_score=round(component["personal_news_score"], 4),
            personal_news_mentions=int(component["personal_news_mentions"]),
            team_news_score=round(component["team_news_score"], 4),
            team_news_mentions=int(component["team_news_mentions"]),
            overall_news_score=round(component["overall_news_score"], 4),
            news_win_modifier=round(component["news_win_modifier"], 4),
            race_sentiment_impact_score=round(component.get("race_sentiment_impact_score", 0.0), 4),
            race_sentiment_delta=round(component.get("race_sentiment_delta", 0.0), 4),
            race_sentiment_confidence=round(component.get("race_sentiment_confidence", 0.0), 4),
            race_sentiment_articles=int(component.get("race_sentiment_articles") or 0),
            race_sentiment_explanations=component.get("race_sentiment_explanations") or [],
            wdc_prob=round(wdc.get(driver.id, 0.0), 4),
            wdc_modifier=round(component["wdc_modifier"], 4),
            reliability_score=round(component["reliability_score"], 4),
            dnf_prob=round(component["dnf_probability"], 4),
            expected_finish=round(expected_by_code.get(driver.code, float(predicted_positions.get(driver.id) or len(drivers))), 2),
            confidence=confidence,
            recent_summary=feature.get("recent_summary"),
            explanation=explanation,
        )

    return RacePrediction(
        driver_predictions=predictions,
        model_version=model_version,
        generated_at=datetime.now(timezone.utc),
        data_sources=[
            "Weekend Evidence practice/grid/race inputs",
            "Race Truth live/recent/recorded state when available",
            "sports.f1.ml Monte Carlo simulator",
            "Production reliability, tyre, weather, track, sentiment, and car-model features",
            f"ML input source: {ml_metadata.get('ml_input_source') or 'unknown'}",
            f"Model config: {model_id}",
        ],
        confidence=confidence,
        model_id=model_id,
        ml_input_source=ml_metadata.get("ml_input_source"),
        ml_provider_sources=ml_metadata.get("ml_provider_sources") or [],
        ml_fallback_reason=ml_metadata.get("ml_fallback_reason"),
        ml_confidence=ml_metadata.get("ml_confidence"),
        simulator_iterations=ml_metadata.get("simulator_iterations"),
        trained_artifacts_used=ml_metadata.get("trained_artifacts_used"),
        evidence_groups_used=ml_metadata.get("evidence_groups_used") or [],
        ml_artifact_id=ml_metadata.get("ml_artifact_id"),
        ml_artifact_version=ml_metadata.get("ml_artifact_version"),
        ml_model_contract_used=ml_metadata.get("ml_model_contract_used"),
        ml_model_adapters_used=ml_metadata.get("ml_model_adapters_used") or [],
        ml_model_fallback_reason=ml_metadata.get("ml_model_fallback_reason"),
        pace_adapter_source=ml_metadata.get("pace_adapter_source"),
        dnf_adapter_source=ml_metadata.get("dnf_adapter_source"),
        rating_adapter_source=ml_metadata.get("rating_adapter_source"),
    )


def _ml_race(race: Race, track: dict) -> MLSimRace:
    return MLSimRace(
        season=race.date.year,
        round=race.round,
        track_code=str(track.get("track_key") or race.circuit_id or race.country or "F1").upper(),
        name=race.name,
        scheduled_start=race.date,
    )


def _wdc_probabilities(drivers: list[Driver], winner: dict[str, float], features: dict) -> dict[str, float]:
    completed = num(features.get("completed_races"), 0.0)
    total_races = num(features.get("total_races"), DEFAULT_TOTAL_RACES) or DEFAULT_TOTAL_RACES
    progress = max(0.0, min(1.0, completed / total_races))
    max_points = max((driver.points for driver in drivers), default=1.0) or 1.0
    scores = {}
    for driver in drivers:
        standing = (driver.points / max_points) if max_points else 0.0
        future = winner.get(driver.code, 0.0)
        scores[driver.id] = max(0.001, (0.18 + 0.56 * progress) * standing + (0.82 - 0.56 * progress) * future)
    total = sum(scores.values()) or 1.0
    return {driver_id: score / total for driver_id, score in scores.items()}


def _compound_deg_from_tires(tires: dict) -> dict[str, float]:
    degradation = num(tires.get("degradation_rate"), 0.50)
    temp = num(tires.get("track_temperature_effect"), 0.0)
    return {
        "SOFT": max(0.045, min(0.18, 0.070 + degradation * 0.075 + temp * 0.20)),
        "MEDIUM": max(0.030, min(0.13, 0.044 + degradation * 0.052 + temp * 0.15)),
        "HARD": max(0.020, min(0.10, 0.030 + degradation * 0.038 + temp * 0.11)),
        "INTERMEDIATE": max(0.035, min(0.15, 0.055 + degradation * 0.045)),
        "WET": max(0.030, min(0.13, 0.050 + degradation * 0.036)),
    }


def _seed_for_race(race: Race) -> int:
    return 202600 + int(race.round or 0)
