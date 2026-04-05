"""F1 race predictor based on championship standings and recent performance."""

import logging

from models.f1 import Driver, Race, RacePrediction, DriverRacePrediction

logger = logging.getLogger(__name__)


class F1Predictor:
    """Predicts F1 race outcomes based on driver/constructor strength."""

    def __init__(self):
        self._drivers: list[Driver] = []
        self._version = "f1-points-v1"

    def load_drivers(self, drivers: list[Driver]) -> None:
        self._drivers = sorted(drivers, key=lambda d: d.points, reverse=True)
        logger.info("F1 predictor loaded %d drivers", len(self._drivers))

    def predict_race(self, race: Race) -> RacePrediction:
        """Generate predictions for an upcoming race."""
        if not self._drivers:
            return RacePrediction(model_version=self._version)

        # Compute strength scores from championship position and points
        max_points = max(d.points for d in self._drivers) if self._drivers else 1.0
        if max_points == 0:
            max_points = 1.0

        strengths: dict[str, float] = {}
        for d in self._drivers:
            # Blend of points-based strength + position-based prior
            points_strength = d.points / max_points
            position_strength = max(0.0, 1.0 - (d.position - 1) * 0.08) if d.position else 0.3
            strengths[d.id] = 0.6 * points_strength + 0.4 * position_strength

        # Normalize to probabilities
        total = sum(strengths.values())
        if total == 0:
            total = 1.0

        predictions: dict[str, DriverRacePrediction] = {}
        for d in self._drivers:
            win_prob = strengths[d.id] / total
            # Podium ~ 3x win prob (capped at 0.95)
            podium_prob = min(0.95, win_prob * 3.0)
            # Top 5 ~ 5x win prob (capped at 0.98)
            top5_prob = min(0.98, win_prob * 5.0)

            predictions[d.id] = DriverRacePrediction(
                driver_id=d.id,
                driver_name=f"{d.first_name} {d.last_name}",
                win_prob=round(win_prob, 4),
                podium_prob=round(podium_prob, 4),
                top5_prob=round(top5_prob, 4),
                predicted_position=d.position,
            )

        return RacePrediction(driver_predictions=predictions, model_version=self._version)

    def predict_races(self, races: list[Race]) -> list[Race]:
        """Add predictions to all scheduled races."""
        for race in races:
            if race.status == "SCHEDULED":
                race.prediction = self.predict_race(race)
        return races
