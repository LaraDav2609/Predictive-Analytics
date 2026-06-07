"""Domain-agnostic prediction-market types shared across all sports and games.

`OutcomeProbability` is the contract every domain (f1, baseball, csgo, ...) emits:
a calibrated probability for one selection in one market of one event. The
`domain` + `entity_id` + `entity_code` triple lets the shared Redis bridge route
it without any domain-specific knowledge — see `common.ml.bridge.outcome_publisher`.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class OutcomeProbability(BaseModel):
    """One calibrated probability, ready to compare against a market price.

    Examples
    --------
    F1:       domain="f1",   entity_id="2026-01-BAHRAIN",     entity_code="VER", market="winner"
    Baseball: domain="baseball", entity_id="2026-04-12-NYY-BOS", entity_code="NYY", market="winner"
    CSGO:     domain="csgo", entity_id="2026-IEM-KATOWICE-NAVI-FAZE", entity_code="NAVI", market="winner"
    """

    domain: str                 # "f1", "baseball", "csgo", ...
    entity_id: str              # event id (race / game / match)
    entity_code: str            # selection code (driver / team)
    market: str                 # "winner", "map1_winner", "over_2.5_maps", ...
    probability: float = Field(ge=0.0, le=1.0)
    knowable_as_of: datetime    # point-in-time stamp for leak-free backtests
    model_version: str
