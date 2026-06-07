from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Sport(str, Enum):
    # Only the actively-supported sports are listed. Add new members here
    # (and a matching sports/<sport>/ package) to extend the platform.
    FORMULA_ONE = "f1"
    BASEBALL = "baseball"


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
