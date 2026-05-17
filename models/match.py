from datetime import datetime
from typing import Optional

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
    winner: Optional[str] = None  # "HOME", "AWAY", "DRAW"


class Match(BaseModel):
    id: int
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    home_crest: Optional[str] = None
    away_crest: Optional[str] = None
    date: datetime
    competition: str
    stage: Optional[str] = None
    group: Optional[str] = None
    venue: Optional[str] = None
    status: str = "SCHEDULED"  # SCHEDULED, LIVE, FINISHED, POSTPONED
    result: Optional[MatchResult] = None
    prediction: Optional[MatchPrediction] = None
