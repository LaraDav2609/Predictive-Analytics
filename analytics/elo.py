"""Elo rating system for international soccer."""

import math
from models.team import Team


# K-factors by competition importance
K_FACTORS = {
    "WORLD_CUP": 60,
    "CONTINENTAL": 40,
    "QUALIFIER": 30,
    "FRIENDLY": 20,
}

# Home advantage in Elo points (0 for neutral venues like World Cup)
HOME_ADVANTAGE = 100
NEUTRAL_ADVANTAGE = 0

# Map FIFA ranking to approximate initial Elo
RANKING_TO_ELO = {
    1: 2100, 2: 2050, 3: 2000, 4: 1975, 5: 1950,
    6: 1925, 7: 1900, 8: 1880, 9: 1860, 10: 1840,
}


def init_elo_from_ranking(ranking: int) -> float:
    """Convert FIFA ranking to initial Elo rating."""
    if ranking in RANKING_TO_ELO:
        return float(RANKING_TO_ELO[ranking])
    if ranking <= 20:
        return 1840.0 - (ranking - 10) * 15.0
    if ranking <= 50:
        return 1690.0 - (ranking - 20) * 8.0
    if ranking <= 100:
        return 1450.0 - (ranking - 50) * 5.0
    return max(1200.0, 1200.0 - (ranking - 100) * 2.0)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for team A vs team B."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    actual_score: float,
    k_factor: float = K_FACTORS["WORLD_CUP"],
) -> tuple[float, float]:
    """Update Elo ratings after a match. Returns (new_rating_a, new_rating_b)."""
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1.0 - exp_a
    new_a = rating_a + k_factor * (actual_score - exp_a)
    new_b = rating_b + k_factor * ((1.0 - actual_score) - exp_b)
    return round(new_a, 1), round(new_b, 1)


def predict_match(
    home_elo: float,
    away_elo: float,
    neutral_venue: bool = True,
) -> dict[str, float]:
    """Predict match outcome probabilities from Elo ratings.

    Returns dict with home_win, draw, away_win probabilities.
    """
    advantage = NEUTRAL_ADVANTAGE if neutral_venue else HOME_ADVANTAGE
    adjusted_home = home_elo + advantage

    exp_home = expected_score(adjusted_home, away_elo)

    # Convert expected score to three-way probabilities
    # Using a simple model: draw probability peaks when teams are equal
    elo_diff = adjusted_home - away_elo
    draw_base = 0.26  # base draw probability
    draw_factor = max(0.0, draw_base - abs(elo_diff) / 2000.0)

    home_win = exp_home * (1.0 - draw_factor)
    away_win = (1.0 - exp_home) * (1.0 - draw_factor)
    draw = 1.0 - home_win - away_win

    # Normalize
    total = home_win + draw + away_win
    return {
        "home_win": round(home_win / total, 4),
        "draw": round(draw / total, 4),
        "away_win": round(away_win / total, 4),
    }


def initialize_teams(teams: list[Team]) -> list[Team]:
    """Set Elo ratings based on FIFA rankings."""
    for team in teams:
        if team.fifa_ranking:
            team.elo_rating = init_elo_from_ranking(team.fifa_ranking)
    return teams
