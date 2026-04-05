"""MLB game predictor based on win percentage and run differential."""

import logging
import math

from models.baseball import MLBGame, MLBPrediction, MLBStanding

logger = logging.getLogger(__name__)


class BaseballPredictor:
    """Predicts MLB game outcomes using team strength metrics."""

    def __init__(self):
        self._standings: dict[int, MLBStanding] = {}
        self._version = "mlb-wpct-v1"

    def load_standings(self, standings: list[MLBStanding]) -> None:
        self._standings = {s.team_id: s for s in standings}
        logger.info("Baseball predictor loaded %d team standings", len(self._standings))

    def predict(self, game: MLBGame) -> MLBPrediction:
        home = self._standings.get(game.home_team_id)
        away = self._standings.get(game.away_team_id)

        if not home or not away:
            return MLBPrediction(home_win_prob=0.54, away_win_prob=0.46, confidence=0.2)

        # Pythagorean win expectation using run differential as proxy
        home_strength = self._team_strength(home)
        away_strength = self._team_strength(away)

        # Home advantage is ~54% in MLB
        home_advantage = 0.04

        # Log5 method for head-to-head probability
        p_home = self._log5(home_strength + home_advantage, away_strength)
        p_away = 1.0 - p_home

        # Confidence based on games played
        min_games = min(home.wins + home.losses, away.wins + away.losses)
        if min_games >= 100:
            confidence = 0.75
        elif min_games >= 50:
            confidence = 0.60
        elif min_games >= 20:
            confidence = 0.45
        else:
            confidence = 0.30

        return MLBPrediction(
            home_win_prob=round(p_home, 4),
            away_win_prob=round(p_away, 4),
            confidence=confidence,
            model_version=self._version,
        )

    def predict_games(self, games: list[MLBGame]) -> list[MLBGame]:
        for game in games:
            if game.status == "SCHEDULED":
                game.prediction = self.predict(game)
        return games

    @staticmethod
    def _team_strength(standing: MLBStanding) -> float:
        """Estimate true team strength from win% and run differential."""
        total = standing.wins + standing.losses
        if total == 0:
            return 0.500

        # Blend actual win% with pythagorean expectation
        actual_wpct = standing.win_pct

        # Pythagorean expectation (exponent 1.83 is standard for MLB)
        if standing.run_differential != 0:
            # Approximate runs scored/allowed from differential
            avg_runs = 4.5  # league average runs per game
            rs = avg_runs + standing.run_differential / (2 * total)
            ra = avg_runs - standing.run_differential / (2 * total)
            if rs > 0 and ra > 0:
                pyth = rs ** 1.83 / (rs ** 1.83 + ra ** 1.83)
            else:
                pyth = actual_wpct
        else:
            pyth = actual_wpct

        # 60% actual, 40% pythagorean
        return 0.6 * actual_wpct + 0.4 * pyth

    @staticmethod
    def _log5(p_a: float, p_b: float) -> float:
        """Log5 method: probability team A beats team B given true talent levels."""
        p_a = max(0.01, min(0.99, p_a))
        p_b = max(0.01, min(0.99, p_b))
        return (p_a * (1 - p_b)) / (p_a * (1 - p_b) + p_b * (1 - p_a))
