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


class MLBHistoricalTeamSeason(BaseModel):
    team_id: int
    season: int | None = None
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
    jersey_number: str | None = None
    position: str = ""
    status: str = ""
    status_description: str = ""
    batting_side: str | None = None
    pitching_hand: str | None = None
    birth_date: str | None = None


class MLBHistoricalTeamRoster(BaseModel):
    team_id: int
    season: int
    roster_type: str
    team_name: str = ""
    players: list[MLBHistoricalRosterPlayer]


class MLBStatSplit(BaseModel):
    season: str | None = None
    stat_type: str = ""
    group: str = ""
    team_id: int | None = None
    team_name: str | None = None
    league_name: str | None = None
    player_id: int | None = None
    player_name: str | None = None
    stat: dict[str, str | int | float | None]


class MLBHistoricalPlayerProfile(BaseModel):
    player_id: int
    full_name: str
    primary_number: str | None = None
    birth_date: str | None = None
    current_age: int | None = None
    birth_city: str | None = None
    birth_state_province: str | None = None
    birth_country: str | None = None
    height: str | None = None
    weight: int | None = None
    active: bool | None = None
    primary_position: str | None = None
    batting_side: str | None = None
    pitching_hand: str | None = None
    current_team_id: int | None = None
    current_team_name: str | None = None


class MLBHistoricalPlayerStats(BaseModel):
    player: MLBHistoricalPlayerProfile
    groups: list[str]
    stat_type: str
    splits: list[MLBStatSplit]
