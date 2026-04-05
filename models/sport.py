from enum import Enum
from pydantic import BaseModel


class Sport(str, Enum):
    SOCCER = "soccer"
    FORMULA_ONE = "f1"
    BASKETBALL = "basketball"
    BASEBALL = "baseball"
    TENNIS = "tennis"
    AMERICAN_FOOTBALL = "nfl"
    CRICKET = "cricket"


class Competition(BaseModel):
    id: str
    name: str
    sport: Sport
    season: str
    country: str | None = None
    api_id: str | None = None


class SportInfo(BaseModel):
    sport: Sport
    label: str
    icon: str
    available: bool
    competitions: list[Competition] = []
    description: str = ""
