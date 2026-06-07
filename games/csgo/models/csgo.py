"""CSGO / CS2 esports data models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class CsgoTeam(BaseModel):
    id: int
    name: str
    abbreviation: str
    region: str = ""             # "EU", "NA", "SA", ...
    world_rank: Optional[int] = None
    rating: float = 1500.0       # Elo-style rating used by the baseline model


class CsgoPrediction(BaseModel):
    team1_win_prob: float
    team2_win_prob: float
    confidence: float
    model_version: str = "csgo-elo-v1"


class CsgoMatch(BaseModel):
    id: str                      # e.g. "2026-iem-katowice-navi-faze"
    team1: str
    team2: str
    team1_id: int
    team2_id: int
    team1_abbrev: str = ""
    team2_abbrev: str = ""
    date: datetime
    event: Optional[str] = None  # "IEM Katowice 2026"
    best_of: int = 3             # bo1 / bo3 / bo5
    status: str = "SCHEDULED"    # SCHEDULED, LIVE, FINAL
    team1_score: Optional[int] = None
    team2_score: Optional[int] = None
    prediction: Optional[CsgoPrediction] = None
