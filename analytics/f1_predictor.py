"""F1 race predictor based on live standings, history, and sentiment."""

import logging
from datetime import datetime, timezone

from models.f1 import Constructor, Driver, DriverRacePrediction, Race, RacePrediction

logger = logging.getLogger(__name__)


class F1Predictor:
    """Predicts F1 race outcomes from standings, recent form, team pace, and sentiment."""

    def __init__(self):
        self._drivers: list[Driver] = []
        self._constructors: list[Constructor] = []
        self._features: dict = {}
        self._sentiment: dict = {}
        self._version = "f1-live-historical-sentiment-v2"

    def load_drivers(
        self,
        drivers: list[Driver],
        constructors: list[Constructor] | None = None,
        features: dict | None = None,
        sentiment: dict | None = None,
    ) -> None:
        self._drivers = sorted(drivers, key=lambda d: d.points, reverse=True)
        self._constructors = constructors or self._constructors
        if features is not None:
            self._features = features
        if sentiment is not None:
            self._sentiment = sentiment
        logger.info(
            "F1 predictor loaded %d drivers, %d constructors, %d feature drivers",
            len(self._drivers),
            len(self._constructors),
            len((self._features.get("drivers") or {})),
        )

    def load_features(self, features: dict) -> None:
        self._features = features or {}

    def load_sentiment(self, sentiment: dict) -> None:
        self._sentiment = sentiment or {}

    def predict_race(self, race: Race) -> RacePrediction:
        """Generate predictions for an upcoming race."""
        if not self._drivers:
            return RacePrediction(model_version=self._version)

        max_points = max(d.points for d in self._drivers) if self._drivers else 1.0
        if max_points == 0:
            max_points = 1.0
        max_constructor_points = max((c.points for c in self._constructors), default=1.0) or 1.0

        driver_features = self._features.get("drivers") or {}
        constructor_features = self._features.get("constructors") or {}
        sentiment_drivers = self._sentiment.get("drivers") or {}
        sentiment_teams = self._sentiment.get("teams") or {}
        global_sentiment = float((self._sentiment.get("composite") or {}).get("Composite") or 0.0)

        strengths: dict[str, float] = {}
        components: dict[str, dict] = {}
        for d in self._drivers:
            points_strength = d.points / max_points
            position_strength = max(0.0, 1.0 - (d.position - 1) * 0.08) if d.position else 0.3
            standing_score = 0.7 * points_strength + 0.3 * position_strength

            feature = driver_features.get(d.id) or {}
            form_score = float(feature.get("form_score", standing_score))
            reliability_score = float(feature.get("reliability_score", 0.75))

            constructor = next((c for c in self._constructors if c.name.lower() == d.team.lower()), None)
            constructor_feature = constructor_features.get(d.team.lower()) or {}
            team_score = (
                float(constructor_feature.get("team_score"))
                if constructor_feature.get("team_score") is not None
                else ((constructor.points / max_constructor_points) if constructor else standing_score)
            )

            driver_sentiment = float((sentiment_drivers.get(d.id) or {}).get("score") or 0.0)
            team_sentiment = float((sentiment_teams.get(d.team.lower()) or {}).get("score") or 0.0)
            sentiment_score = max(-1.0, min(1.0, 0.55 * driver_sentiment + 0.30 * team_sentiment + 0.15 * global_sentiment))
            sentiment_factor = 1.0 + (sentiment_score * 0.08)

            strength = (
                0.38 * standing_score
                + 0.27 * form_score
                + 0.22 * team_score
                + 0.10 * reliability_score
                + 0.03
            ) * sentiment_factor
            strengths[d.id] = max(strength, 0.005)
            components[d.id] = {
                "standing_score": standing_score,
                "form_score": form_score,
                "team_score": team_score,
                "reliability_score": reliability_score,
                "sentiment_score": sentiment_score,
                "recent_summary": feature.get("recent_summary"),
            }

        total = sum(strengths.values())
        if total == 0:
            total = 1.0

        predicted_order = sorted(self._drivers, key=lambda driver: strengths[driver.id], reverse=True)
        predicted_positions = {driver.id: idx + 1 for idx, driver in enumerate(predicted_order)}
        confidence = self._prediction_confidence(driver_features, sentiment_drivers)

        predictions: dict[str, DriverRacePrediction] = {}
        for d in self._drivers:
            win_prob = strengths[d.id] / total
            podium_prob = min(0.95, win_prob * 3.0)
            top5_prob = min(0.98, win_prob * 5.0)
            component = components[d.id]
            explanation = self._explain_driver(d, component)

            predictions[d.id] = DriverRacePrediction(
                driver_id=d.id,
                driver_name=f"{d.first_name} {d.last_name}",
                win_prob=round(win_prob, 4),
                podium_prob=round(podium_prob, 4),
                top5_prob=round(top5_prob, 4),
                predicted_position=predicted_positions.get(d.id),
                form_score=round(component["form_score"], 4),
                team_score=round(component["team_score"], 4),
                sentiment_score=round(component["sentiment_score"], 4),
                reliability_score=round(component["reliability_score"], 4),
                confidence=confidence,
                recent_summary=component.get("recent_summary"),
                explanation=explanation,
            )

        return RacePrediction(
            driver_predictions=predictions,
            model_version=self._version,
            generated_at=datetime.now(timezone.utc),
            data_sources=[
                "Jolpica live standings",
                "Jolpica current-season race history",
                "F1/FIA news sentiment",
            ],
            confidence=confidence,
        )

    def predict_races(self, races: list[Race]) -> list[Race]:
        """Add predictions to all scheduled races."""
        for race in races:
            if race.status == "SCHEDULED":
                race.prediction = self.predict_race(race)
        return races

    @staticmethod
    def _prediction_confidence(driver_features: dict, sentiment_drivers: dict) -> float:
        history_part = min(0.45, len(driver_features) / 20 * 0.45)
        sentiment_part = min(0.25, len(sentiment_drivers) / 20 * 0.25)
        return round(0.30 + history_part + sentiment_part, 2)

    @staticmethod
    def _explain_driver(driver: Driver, component: dict) -> list[str]:
        notes = []
        if component["form_score"] >= 0.72:
            notes.append("strong recent race form")
        elif component["form_score"] <= 0.35:
            notes.append("recent results are limiting the projection")
        if component["team_score"] >= 0.70:
            notes.append(f"{driver.team} has front-running team strength")
        if component["reliability_score"] < 0.65:
            notes.append("reliability risk is pulling the forecast down")
        if component["sentiment_score"] > 0.12:
            notes.append("current news sentiment is positive")
        elif component["sentiment_score"] < -0.12:
            notes.append("current news sentiment is negative")
        return notes[:3]
