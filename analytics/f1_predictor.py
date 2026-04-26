"""F1 race predictor based on live standings, history, and sentiment."""

import logging
from datetime import datetime, timezone
from typing import Any

from models.f1 import Constructor, Driver, DriverRacePrediction, Race, RacePrediction

logger = logging.getLogger(__name__)


class F1Predictor:
    """Predicts F1 race outcomes from standings, recent form, team pace, and sentiment."""

    def __init__(self):
        self._drivers: list[Driver] = []
        self._constructors: list[Constructor] = []
        self._features: dict = {}
        self._sentiment: dict = {}
        self._version = "f1-live-historical-sentiment-v4"

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

    def get_sentiment(self) -> dict:
        return self._sentiment or {}

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
        completed_races = self._num(self._features.get("completed_races"), 0.0)
        total_races = self._num(self._features.get("total_races"), 22.0) or 22.0
        season_progress = max(0.0, min(1.0, completed_races / total_races))
        performance_by_driver = self._build_performance_table(driver_features, constructor_features)

        strengths: dict[str, float] = {}
        wdc_strengths: dict[str, float] = {}
        components: dict[str, dict] = {}
        for d in self._drivers:
            points_strength = d.points / max_points
            position_strength = max(0.0, 1.0 - (d.position - 1) * 0.08) if d.position else 0.3
            standing_score = 0.7 * points_strength + 0.3 * position_strength

            feature = driver_features.get(d.id) or {}
            form_score = float(feature.get("form_score", standing_score))
            reliability_score = float(feature.get("reliability_score", 0.75))
            performance = performance_by_driver.get(d.id) or {}

            constructor = next((c for c in self._constructors if c.name.lower() == d.team.lower()), None)
            constructor_feature = constructor_features.get(d.team.lower()) or {}
            team_score = (
                float(constructor_feature.get("team_score"))
                if constructor_feature.get("team_score") is not None
                else ((constructor.points / max_constructor_points) if constructor else standing_score)
            )

            driver_sentiment_data = sentiment_drivers.get(d.id) or {}
            team_sentiment_data = sentiment_teams.get(d.team.lower()) or {}
            personal_news_score = self._num(
                driver_sentiment_data.get("personal_news_score", driver_sentiment_data.get("personal_score")),
                driver_sentiment_data.get("score", 0.0),
            )
            team_news_score = self._num(
                driver_sentiment_data.get("team_news_score", driver_sentiment_data.get("team_score")),
                team_sentiment_data.get("score", 0.0),
            )
            overall_news_score = self._num(
                driver_sentiment_data.get("overall_news_score", driver_sentiment_data.get("overall_score")),
                global_sentiment,
            )
            sentiment_score = max(
                -1.0,
                min(
                    1.0,
                    self._num(
                        driver_sentiment_data.get("score"),
                        0.55 * personal_news_score + 0.30 * team_news_score + 0.15 * overall_news_score,
                    ),
                ),
            )
            news_win_modifier = max(
                0.90,
                min(1.10, self._num(driver_sentiment_data.get("win_probability_modifier"), 1.0 + sentiment_score * 0.10)),
            )
            wdc_modifier = max(
                0.86,
                min(1.14, self._num(driver_sentiment_data.get("wdc_probability_modifier"), 1.0 + sentiment_score * 0.14)),
            )

            strength = (
                0.22 * standing_score
                + 0.24 * form_score
                + 0.18 * team_score
                + 0.23 * self._num(performance.get("performance_score"), form_score)
                + 0.10 * reliability_score
                + 0.03
            ) * news_win_modifier
            strengths[d.id] = max(strength, 0.005)

            wdc_standings_weight = 0.20 + (0.52 * season_progress)
            wdc_future_weight = 1.0 - wdc_standings_weight
            future_score = 0.38 * team_score + 0.32 * form_score + 0.18 * reliability_score + 0.12 * strengths[d.id]
            wdc_strengths[d.id] = max(
                0.005,
                ((wdc_standings_weight * standing_score) + (wdc_future_weight * future_score)) * wdc_modifier,
            )
            components[d.id] = {
                "standing_score": standing_score,
                "form_score": form_score,
                "team_score": team_score,
                "driver_skill_score": self._num(performance.get("driver_skill_score"), form_score),
                "car_performance_score": self._num(performance.get("car_performance_score"), team_score),
                "performance_score": self._num(performance.get("performance_score"), form_score),
                "reliability_score": reliability_score,
                "sentiment_score": sentiment_score,
                "sentiment_label": driver_sentiment_data.get("label") or _label_from_score(sentiment_score),
                "sentiment_mentions": int(driver_sentiment_data.get("mentions") or 0),
                "personal_news_score": personal_news_score,
                "personal_news_mentions": int(driver_sentiment_data.get("personal_mentions") or 0),
                "team_news_score": team_news_score,
                "team_news_mentions": int(driver_sentiment_data.get("team_mentions") or 0),
                "overall_news_score": overall_news_score,
                "news_win_modifier": news_win_modifier,
                "wdc_modifier": wdc_modifier,
                "source_breakdown": driver_sentiment_data.get("source_breakdown") or {},
                "topic_scores": driver_sentiment_data.get("topic_scores") or {},
                "recent_summary": feature.get("recent_summary"),
            }

        total = sum(strengths.values())
        if total == 0:
            total = 1.0
        wdc_total = sum(wdc_strengths.values()) or 1.0

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
                driver_skill_score=round(component["driver_skill_score"], 4),
                car_performance_score=round(component["car_performance_score"], 4),
                performance_score=round(component["performance_score"], 4),
                sentiment_score=round(component["sentiment_score"], 4),
                sentiment_label=component["sentiment_label"],
                sentiment_mentions=component["sentiment_mentions"],
                personal_news_score=round(component["personal_news_score"], 4),
                team_news_score=round(component["team_news_score"], 4),
                overall_news_score=round(component["overall_news_score"], 4),
                news_win_modifier=round(component["news_win_modifier"], 4),
                wdc_prob=round(wdc_strengths[d.id] / wdc_total, 4),
                wdc_modifier=round(component["wdc_modifier"], 4),
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
                "F1/FIA and motorsport RSS news sentiment",
                "Combinatoric driver skill and car performance model",
            ],
            confidence=confidence,
        )

    def get_performance_intelligence(self) -> dict:
        """Rank drivers by estimated driver skill, car performance, and combined output."""
        driver_features = self._features.get("drivers") or {}
        constructor_features = self._features.get("constructors") or {}
        rows = self._build_performance_table(driver_features, constructor_features)
        ranked = sorted(rows.values(), key=lambda item: item["performance_score"], reverse=True)
        for index, row in enumerate(ranked, start=1):
            row["rank"] = index
        return {
            "ok": True,
            "model_version": self._version,
            "updated_at": self._features.get("updated_at"),
            "completed_races": self._features.get("completed_races", 0),
            "total_races": self._features.get("total_races", 0),
            "drivers": ranked,
            "method": {
                "driver_skill": "recent form, reliability, standings signal, podium/win conversion, and F1 experience",
                "car_performance": "constructor pace from standings, recent team points, and average finish",
                "performance_score": "geometric blend of driver skill and car performance with reliability damping",
            },
        }

    def predict_races(self, races: list[Race]) -> list[Race]:
        """Add predictions to all scheduled races."""
        for race in races:
            if race.status == "SCHEDULED":
                race.prediction = self.predict_race(race)
        return races

    @staticmethod
    def _prediction_confidence(driver_features: dict, sentiment_drivers: dict) -> float:
        history_part = min(0.45, len(driver_features) / 20 * 0.45)
        sentiment_mentions = sum(1 for item in sentiment_drivers.values() if int((item or {}).get("mentions") or 0) > 0)
        sentiment_part = min(0.25, sentiment_mentions / 20 * 0.25)
        return round(0.30 + history_part + sentiment_part, 2)

    def _build_performance_table(self, driver_features: dict, constructor_features: dict) -> dict[str, dict]:
        max_points = max((d.points for d in self._drivers), default=1.0) or 1.0
        max_constructor_points = max((c.points for c in self._constructors), default=1.0) or 1.0
        rows: dict[str, dict] = {}
        for driver in self._drivers:
            feature = driver_features.get(driver.id) or {}
            constructor = next((c for c in self._constructors if c.name.lower() == driver.team.lower()), None)
            constructor_feature = constructor_features.get(driver.team.lower()) or {}

            points_strength = driver.points / max_points
            position_strength = max(0.0, 1.0 - (driver.position - 1) * 0.08) if driver.position else 0.3
            standing_score = 0.64 * points_strength + 0.36 * position_strength
            form_score = self._num(feature.get("form_score"), standing_score)
            reliability_score = self._num(feature.get("reliability_score"), 0.75)
            recent_wins = self._num(feature.get("recent_wins"), 0.0)
            recent_podiums = self._num(feature.get("recent_podiums"), 0.0)
            recent_starts = max(1.0, self._num(feature.get("recent_starts"), 1.0))
            starts = self._num(feature.get("starts"), 0.0)

            conversion_score = min(1.0, (recent_wins * 0.28 + recent_podiums * 0.16) / min(4.0, recent_starts))
            experience_score = min(1.0, starts / 80.0)
            driver_skill_score = self._clamp01(
                0.44 * form_score
                + 0.20 * reliability_score
                + 0.18 * standing_score
                + 0.10 * conversion_score
                + 0.08 * experience_score
            )

            constructor_standing = (constructor.points / max_constructor_points) if constructor else standing_score
            team_score = self._num(constructor_feature.get("team_score"), constructor_standing)
            recent_team_points = self._num(constructor_feature.get("recent_points"), 0.0)
            team_points_score = min(1.0, recent_team_points / 344.0)
            car_performance_score = self._clamp01(
                0.56 * team_score
                + 0.28 * constructor_standing
                + 0.16 * team_points_score
            )

            combined_base = (driver_skill_score ** 0.58) * (max(car_performance_score, 0.01) ** 0.42)
            performance_score = self._clamp01(combined_base * (0.90 + 0.10 * reliability_score))
            rows[driver.id] = {
                "driver_id": driver.id,
                "driver_name": f"{driver.first_name} {driver.last_name}",
                "code": driver.code,
                "team": driver.team,
                "position": driver.position,
                "points": driver.points,
                "driver_skill_score": round(driver_skill_score, 4),
                "car_performance_score": round(car_performance_score, 4),
                "performance_score": round(performance_score, 4),
                "form_score": round(form_score, 4),
                "reliability_score": round(reliability_score, 4),
                "standing_score": round(standing_score, 4),
                "recent_wins": int(recent_wins),
                "recent_podiums": int(recent_podiums),
                "recent_avg_finish": feature.get("recent_avg_finish"),
                "recent_summary": feature.get("recent_summary"),
                "car_recent_points": round(recent_team_points, 2),
                "car_team_score": round(team_score, 4),
            }
        return rows

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

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
            notes.append("driver/team news flow is positive")
        elif component["sentiment_score"] < -0.12:
            notes.append("driver/team news flow is negative")
        elif component.get("sentiment_mentions"):
            notes.append(f"{component['sentiment_mentions']} recent news signals folded in")
        return notes[:3]


def _label_from_score(score: float) -> str:
    if score > 0.08:
        return "Bullish"
    if score < -0.08:
        return "Bearish"
    return "Neutral"
