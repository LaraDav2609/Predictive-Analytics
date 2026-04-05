"""Soccer match predictor using Elo ratings."""

import logging

from analytics.elo import predict_match, initialize_teams
from models.team import Team
from models.match import Match, MatchPrediction

logger = logging.getLogger(__name__)


class SoccerPredictor:
    """Predicts soccer match outcomes using Elo-based model."""

    def __init__(self):
        self._teams: dict[int, Team] = {}
        self._version = "elo-v1"

    def load_teams(self, teams: list[Team]) -> None:
        """Initialize predictor with teams and their Elo ratings."""
        teams = initialize_teams(teams)
        self._teams = {t.id: t for t in teams}
        logger.info("Soccer predictor loaded %d teams", len(self._teams))

    def predict(self, match: Match, neutral_venue: bool = True) -> MatchPrediction:
        """Generate prediction for a match."""
        home = self._teams.get(match.home_team_id)
        away = self._teams.get(match.away_team_id)

        if not home or not away:
            # Unknown teams — return flat prior
            return MatchPrediction(
                home_win_prob=0.35, draw_prob=0.30, away_win_prob=0.35,
                confidence=0.2, model_version=self._version,
            )

        probs = predict_match(home.elo_rating, away.elo_rating, neutral_venue)

        # Confidence based on Elo difference magnitude
        elo_diff = abs(home.elo_rating - away.elo_rating)
        if elo_diff > 300:
            confidence = 0.85
        elif elo_diff > 200:
            confidence = 0.70
        elif elo_diff > 100:
            confidence = 0.55
        else:
            confidence = 0.40

        return MatchPrediction(
            home_win_prob=probs["home_win"],
            draw_prob=probs["draw"],
            away_win_prob=probs["away_win"],
            confidence=confidence,
            model_version=self._version,
        )

    def predict_matches(self, matches: list[Match]) -> list[Match]:
        """Add predictions to all scheduled matches."""
        for match in matches:
            if match.status in ("SCHEDULED", "TIMED"):
                match.prediction = self.predict(match)
        return matches
