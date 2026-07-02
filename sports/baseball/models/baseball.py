"""Baseball (MLB) data models."""

from datetime import datetime
from typing import Optional, Union

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
    venue: Optional[str] = None
    status: str = "SCHEDULED"  # SCHEDULED, LIVE, FINAL
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    home_pitcher: Optional[str] = None
    away_pitcher: Optional[str] = None
    home_pitcher_id: Optional[int] = None
    away_pitcher_id: Optional[int] = None
    prediction: Optional["MLBPrediction"] = None


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


class MLBHistoricalTeamSeason(BaseModel):
    team_id: int
    season: Optional[int] = None
    name: str
    location_name: str = ""
    franchise_name: str = ""
    club_name: str = ""
    abbreviation: str = ""
    league: str = ""
    division: str = ""
    venue: str = ""
    first_year_of_play: str = ""
    active: bool = True


class MLBHistoricalRosterPlayer(BaseModel):
    player_id: int
    full_name: str
    jersey_number: Optional[str] = None
    position: str = ""
    status: str = ""
    status_description: str = ""
    batting_side: Optional[str] = None
    pitching_hand: Optional[str] = None
    birth_date: Optional[str] = None


class MLBHistoricalTeamRoster(BaseModel):
    team_id: int
    season: int
    roster_type: str
    team_name: str = ""
    players: list[MLBHistoricalRosterPlayer]


class MLBStatSplit(BaseModel):
    season: Optional[str] = None
    stat_type: str = ""
    group: str = ""
    team_id: Optional[int] = None
    team_name: Optional[str] = None
    league_name: Optional[str] = None
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    stat: dict[str, Optional[Union[str, int, float]]]


class MLBHistoricalPlayerProfile(BaseModel):
    player_id: int
    full_name: str
    primary_number: Optional[str] = None
    birth_date: Optional[str] = None
    current_age: Optional[int] = None
    birth_city: Optional[str] = None
    birth_state_province: Optional[str] = None
    birth_country: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[int] = None
    active: Optional[bool] = None
    primary_position: Optional[str] = None
    batting_side: Optional[str] = None
    pitching_hand: Optional[str] = None
    current_team_id: Optional[int] = None
    current_team_name: Optional[str] = None


class MLBHistoricalPlayerStats(BaseModel):
    player: MLBHistoricalPlayerProfile
    groups: list[str]
    stat_type: str
    splits: list[MLBStatSplit]
