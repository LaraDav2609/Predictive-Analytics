from datetime import datetime
from pydantic import BaseModel


class MatchPrediction(BaseModel):
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    confidence: float
    model_version: str = "elo-v1"


class MatchResult(BaseModel):
    home_score: int
    away_score: int
    winner: str | None = None  # "HOME", "AWAY", "DRAW"


class Match(BaseModel):
    id: int
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    home_crest: str | None = None
    away_crest: str | None = None
    date: datetime
    competition: str
    stage: str | None = None
    group: str | None = None
    venue: str | None = None
    status: str = "SCHEDULED"  # SCHEDULED, LIVE, FINISHED, POSTPONED
    result: MatchResult | None = None
    prediction: MatchPrediction | None = None
