"""Baseball (MLB) data models."""

from datetime import datetime
from pydantic import BaseModel


class MLBTeam(BaseModel):
    id: int
    name: str
    abbreviation: str
    league: str  # "AL" or "NL"
    division: str  # "East", "Central", "West"
    wins: int = 0
    losses: int = 0
    streak: str = ""
    run_differential: int = 0

    @property
    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0


class MLBGame(BaseModel):
    id: int
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    home_abbrev: str = ""
    away_abbrev: str = ""
    date: datetime
    venue: str | None = None
    status: str = "SCHEDULED"  # SCHEDULED, LIVE, FINAL
    home_score: int | None = None
    away_score: int | None = None
    home_pitcher: str | None = None
    away_pitcher: str | None = None
    prediction: "MLBPrediction | None" = None


class MLBPrediction(BaseModel):
    home_win_prob: float
    away_win_prob: float
    confidence: float
    model_version: str = "mlb-wpct-v1"


class MLBStanding(BaseModel):
    team_id: int
    team_name: str
    abbreviation: str
    league: str
    division: str
    wins: int
    losses: int
    win_pct: float
    games_back: str
    streak: str
    last_10: str = ""
    run_differential: int = 0
