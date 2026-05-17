from enum import Enum
from typing import Optional

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
    country: Optional[str] = None
    api_id: Optional[str] = None


class SportInfo(BaseModel):
    sport: Sport
    label: str
    icon: str
    available: bool
    href: str = ""
    competitions: list[Competition] = []
    description: str = ""
