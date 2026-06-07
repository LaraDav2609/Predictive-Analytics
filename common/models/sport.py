from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Category(str, Enum):
    """Top-level grouping a prediction domain belongs to."""
    SPORT = "sport"
    GAME = "game"


class Sport(str, Enum):
    # Identifies a prediction domain (sport or game). Add a member here plus a
    # matching sports/<sport>/ or games/<game>/ package to extend the platform.
    # (Name kept as `Sport` for back-compat; it really means "domain".)
    FORMULA_ONE = "f1"
    BASEBALL = "baseball"
    CSGO = "csgo"


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
    category: Category = Category.SPORT
    href: str = ""
    competitions: list[Competition] = []
    description: str = ""
