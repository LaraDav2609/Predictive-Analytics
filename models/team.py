from typing import Optional

from pydantic import BaseModel


class TeamStats(BaseModel):
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.played if self.played > 0 else 0.0


class Team(BaseModel):
    id: int
    name: str
    short_name: str
    country_code: str
    crest_url: Optional[str] = None
    fifa_ranking: Optional[int] = None
    elo_rating: float = 1500.0
    group: Optional[str] = None
    stats: TeamStats = TeamStats()
    recent_form: list[str] = []  # e.g. ["W", "W", "D", "L", "W"]
