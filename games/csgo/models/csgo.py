"""CSGO / CS2 esports data models.

Carries stable *identity* (source ids, aliases, event slug, rosters) and *results*
(map scores, winner) so the same shapes work for the stub and for real feeds
(PandaScore / HLTV / GRID), and so markets can be matched reliably.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class CsgoTeam(BaseModel):
    id: int
    name: str
    abbreviation: str
    region: str = ""                 # "EU", "NA", "SA", ...
    world_rank: Optional[int] = None
    rating: float = 1500.0           # Elo-style rating used by the baseline model
    rating_deviation: float = 350.0  # Glicko RD (uncertainty); high until rated
    aliases: list[str] = []          # name variants for market matching
    source_ids: dict[str, str] = {}  # {"pandascore": "123", "hltv": "4608"}
    roster: list[int] = []           # current player ids (stand-in detection later)


class CsgoPrediction(BaseModel):
    team1_win_prob: float
    team2_win_prob: float
    confidence: float
    model_version: str = "csgo-elo-v1"
    # Optional secondary markets (populated by the calibrated model when data allows).
    map1_team1_win_prob: Optional[float] = None
    over_2_5_maps_prob: Optional[float] = None
    feature_provenance: dict[str, str] = {}


class MapScore(BaseModel):
    """One map within a series."""
    order: Optional[int] = None        # 1, 2, 3 ... within the series
    map_name: Optional[str] = None     # "Mirage", "Inferno", ...
    team1_rounds: Optional[int] = None
    team2_rounds: Optional[int] = None
    winner_id: Optional[int] = None
    status: str = "FINISHED"           # NOT_STARTED, RUNNING, FINISHED


class CsgoMatch(BaseModel):
    id: str                            # canonical id (e.g. PandaScore match id as str)
    team1: str
    team2: str
    team1_id: int
    team2_id: int
    team1_abbrev: str = ""
    team2_abbrev: str = ""
    team1_aliases: list[str] = []
    team2_aliases: list[str] = []
    date: datetime                     # scheduled / begin time (UTC)
    event: Optional[str] = None        # "IEM Katowice 2026"
    event_slug: Optional[str] = None   # "iem-katowice-2026"
    best_of: int = 3                   # bo1 / bo3 / bo5
    status: str = "SCHEDULED"          # SCHEDULED, LIVE, FINAL
    team1_score: Optional[int] = None  # series score (maps won)
    team2_score: Optional[int] = None
    winner_id: Optional[int] = None
    winner_code: Optional[str] = None  # abbreviation of the winning team
    map_scores: list[MapScore] = []
    team1_roster: list[int] = []       # player ids in this match (stand-in detection)
    team2_roster: list[int] = []
    source_ids: dict[str, str] = {}    # {"pandascore": "1001"}
    prediction: Optional[CsgoPrediction] = None
