"""Baseline race prediction model preserving the existing scoring behavior."""

from __future__ import annotations

from datetime import datetime, timezone

from models.f1 import Constructor, Driver, DriverRacePrediction, Race, RacePrediction
from f1_predictor.config import DEFAULT_TOTAL_RACES, MODEL_VERSION
from f1_predictor.features.builder import F1FeatureBuilder
from f1_predictor.features.performance import build_performance_table
from f1_predictor.features.sentiment import driver_sentiment_component
from f1_predictor.models.configs import RaceModelConfig, get_model_config
from f1_predictor.scoring.explanations import explain_driver
from f1_predictor.scoring.normalization import clamp01, num


class BaselineRaceModel:
    def __init__(self, model_version: str = MODEL_VERSION, config: RaceModelConfig | None = None, model_id: str | None = None):
        self.config = config or get_model_config(model_id)
        self.model_version = model_version if config is None and model_id is None else self.config.model_version
        self.model_id = self.config.model_id

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

        max_points = max(driver.points for driver in drivers) if drivers else 1.0
        if max_points == 0:
            max_points = 1.0
        max_constructor_points = max((constructor.points for constructor in constructors), default=1.0) or 1.0

        snapshot = F1FeatureBuilder(drivers, constructors, features, sentiment).build(race)
        driver_features = features.get("drivers") or {}
        constructor_features = features.get("constructors") or {}
        completed_races = num(features.get("completed_races"), 0.0)
        total_races = num(features.get("total_races"), DEFAULT_TOTAL_RACES) or DEFAULT_TOTAL_RACES
        season_progress = max(0.0, min(1.0, completed_races / total_races))
        race_standings_influence = min(0.32, 0.08 + 0.34 * season_progress)
        performance_by_driver = build_performance_table(drivers, constructors, driver_features, constructor_features)
        track = snapshot.track or {}
        weather = snapshot.weather or {}
        tires = snapshot.tires or {}
        track_key = track.get("track_key")
        track_history = (((features.get("track_history") or {}).get(track_key) or {}).get("drivers") or {})
        overtaking_difficulty = num(track.get("overtaking_difficulty"), 0.50)
        qualifying_importance = num(track.get("qualifying_importance"), 0.58)
        tire_stress = num(track.get("tire_stress"), 0.50)
        weather_chaos = num(weather.get("chaos_score"), 0.0)
        degradation = num(tires.get("degradation_rate"), tire_stress)

        strengths: dict[str, float] = {}
        wdc_strengths: dict[str, float] = {}
        components: dict[str, dict] = {}
        for driver in drivers:
            points_strength = driver.points / max_points
            position_strength = max(0.0, 1.0 - (driver.position - 1) * 0.08) if driver.position else 0.3
            raw_standing_score = 0.7 * points_strength + 0.3 * position_strength
            standing_score = clamp01(0.50 + (raw_standing_score - 0.50) * race_standings_influence)

            feature = driver_features.get(driver.id) or {}
            form_score = float(feature.get("form_score", standing_score))
            reliability_score = float(feature.get("reliability_score", 0.75))
            performance = performance_by_driver.get(driver.id) or {}
            reliability = (snapshot.reliability or {}).get(driver.id) or {}
            qualifying_pace = num(performance.get("qualifying_pace_score"), feature.get("qualifying_pace_score") or form_score)
            race_pace = num(performance.get("race_pace_score"), feature.get("race_pace_score") or form_score)
            teammate_score = num(performance.get("teammate_score"), feature.get("teammate_score") or 0.50)
            trend_score = num(performance.get("trend_score"), feature.get("trend_score") or 0.50)
            track_history_score = num((track_history.get(driver.id) or {}).get("track_score"), 0.50)
            track_weights = self.config.track_fit_weights
            track_fit_score = clamp01(
                track_weights["race_pace"] * race_pace
                + track_weights["qualifying_pace"] * qualifying_pace * qualifying_importance
                + track_weights["track_history"] * track_history_score
                + track_weights["teammate"] * teammate_score
                + track_weights["overtaking"] * (1.0 - overtaking_difficulty + qualifying_pace * overtaking_difficulty)
            )
            constructor = next((c for c in constructors if c.name.lower() == driver.team.lower()), None)
            constructor_feature = constructor_features.get(driver.team.lower()) or {}
            team_score = (
                float(constructor_feature.get("team_score"))
                if constructor_feature.get("team_score") is not None
                else ((constructor.points / max_constructor_points) if constructor else standing_score)
            )

            sentiment_component = driver_sentiment_component(driver.id, driver.team, sentiment)
            news_win_modifier = sentiment_component["news_win_modifier"]
            wdc_modifier = sentiment_component["wdc_modifier"]
            tire_weights = self.config.tire_strategy_weights
            tire_strategy_score = clamp01(
                tire_weights["race_pace"] * race_pace
                + tire_weights["reliability"] * reliability_score
                + tire_weights["team"] * team_score
                + tire_weights["undercut"] * num(tires.get("undercut_strength"), 0.50)
                + tire_weights["degradation"] * (1.0 - degradation)
                + tire_weights["trend"] * trend_score
            )
            dnf_probability = num(reliability.get("dnf_probability"), max(0.01, min(0.35, 1.0 - reliability_score)))
            weather_risk_score = clamp01(1.0 - weather_chaos * (0.78 + (1.0 - reliability_score) * 0.22))
            reliability_risk_factor = clamp01(1.0 - dnf_probability * self.config.reliability_dnf_multiplier)

            race_weights = self.config.race_weights
            strength = (
                race_weights["standing"] * standing_score
                + race_weights["form"] * form_score
                + race_weights["team"] * team_score
                + race_weights["performance"] * num(performance.get("performance_score"), form_score)
                + race_weights["qualifying_pace"] * qualifying_pace
                + race_weights["race_pace"] * race_pace
                + race_weights["track_fit"] * track_fit_score
                + race_weights["tire_strategy"] * tire_strategy_score
                + race_weights["weather_risk"] * weather_risk_score
            ) * news_win_modifier * reliability_risk_factor
            strengths[driver.id] = max(strength, 0.005)

            wdc_standings_weight = self.config.wdc_standings_base + (self.config.wdc_standings_progress * season_progress)
            wdc_future_weight = 1.0 - wdc_standings_weight
            wdc_weights = self.config.wdc_future_weights
            future_score = (
                wdc_weights["team"] * team_score
                + wdc_weights["form"] * form_score
                + wdc_weights["race_pace"] * race_pace
                + wdc_weights["qualifying_pace"] * qualifying_pace
                + wdc_weights["reliability"] * reliability_score
                + wdc_weights["track_fit"] * track_fit_score
            )
            wdc_strengths[driver.id] = max(
                0.005,
                ((wdc_standings_weight * standing_score) + (wdc_future_weight * future_score)) * wdc_modifier,
            )
            components[driver.id] = {
                "standing_score": standing_score,
                "raw_standing_score": raw_standing_score,
                "standing_influence": race_standings_influence,
                "form_score": form_score,
                "team_score": team_score,
                "driver_skill_score": num(performance.get("driver_skill_score"), form_score),
                "car_performance_score": num(performance.get("car_performance_score"), team_score),
                "performance_score": num(performance.get("performance_score"), form_score),
                "qualifying_pace_score": qualifying_pace,
                "race_pace_score": race_pace,
                "track_fit_score": track_fit_score,
                "tire_strategy_score": tire_strategy_score,
                "weather_risk_score": weather_risk_score,
                "dnf_probability": dnf_probability,
                "reliability_score": reliability_score,
                **sentiment_component,
                "recent_summary": feature.get("recent_summary"),
            }

        strengths = _apply_probability_temperature(strengths, self.config.probability_temperature)
        total = sum(strengths.values()) or 1.0
        wdc_total = sum(wdc_strengths.values()) or 1.0
        predicted_order = sorted(drivers, key=lambda driver: strengths[driver.id], reverse=True)
        predicted_positions = {driver.id: idx + 1 for idx, driver in enumerate(predicted_order)}
        confidence = prediction_confidence(driver_features, (sentiment.get("drivers") or {}))

        predictions: dict[str, DriverRacePrediction] = {}
        for driver in drivers:
            win_prob = strengths[driver.id] / total
            podium_prob = min(0.95, win_prob * 3.0)
            top5_prob = min(0.98, win_prob * 5.0)
            component = components[driver.id]
            predictions[driver.id] = DriverRacePrediction(
                driver_id=driver.id,
                driver_name=f"{driver.first_name} {driver.last_name}",
                win_prob=round(win_prob, 4),
                podium_prob=round(podium_prob, 4),
                top5_prob=round(top5_prob, 4),
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
                sentiment_mentions=component["sentiment_mentions"],
                personal_news_score=round(component["personal_news_score"], 4),
                personal_news_mentions=component["personal_news_mentions"],
                team_news_score=round(component["team_news_score"], 4),
                team_news_mentions=component["team_news_mentions"],
                overall_news_score=round(component["overall_news_score"], 4),
                news_win_modifier=round(component["news_win_modifier"], 4),
                race_sentiment_impact_score=round(component.get("race_sentiment_impact_score", 0.0), 4),
                race_sentiment_delta=round(component.get("race_sentiment_delta", 0.0), 4),
                race_sentiment_confidence=round(component.get("race_sentiment_confidence", 0.0), 4),
                race_sentiment_articles=int(component.get("race_sentiment_articles") or 0),
                race_sentiment_explanations=component.get("race_sentiment_explanations") or [],
                wdc_prob=round(wdc_strengths[driver.id] / wdc_total, 4),
                wdc_modifier=round(component["wdc_modifier"], 4),
                reliability_score=round(component["reliability_score"], 4),
                dnf_prob=round(component["dnf_probability"], 4),
                expected_finish=round(predicted_positions.get(driver.id) or 20, 2),
                confidence=confidence,
                recent_summary=component.get("recent_summary"),
                explanation=explain_driver(driver, component),
            )

        return RacePrediction(
            driver_predictions=predictions,
            model_version=self.model_version,
            generated_at=datetime.now(timezone.utc),
            data_sources=[
                "Jolpica live standings",
                "Jolpica current-season race history",
                "F1/FIA and motorsport RSS news sentiment",
                "Combinatoric driver skill and car performance model",
                f"Model config: {self.model_id}",
            ],
            confidence=confidence,
        )


def prediction_confidence(driver_features: dict, sentiment_drivers: dict) -> float:
    history_part = min(0.45, len(driver_features) / 20 * 0.45)
    sentiment_mentions = sum(1 for item in sentiment_drivers.values() if int((item or {}).get("mentions") or 0) > 0)
    sentiment_part = min(0.25, sentiment_mentions / 20 * 0.25)
    return round(0.30 + history_part + sentiment_part, 2)


def _apply_probability_temperature(strengths: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        temperature = 1.0
    if abs(temperature - 1.0) < 0.0001:
        return strengths
    exponent = 1.0 / temperature
    return {driver_id: max(score, 0.005) ** exponent for driver_id, score in strengths.items()}
