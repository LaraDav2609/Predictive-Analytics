"""Compatibility wrapper for F1 session simulation."""

from __future__ import annotations

from typing import Any

from models.f1 import Constructor, Driver, Race
from f1_predictor.simulation.session_projection import build_session_projection


def build_session_simulation(
    race: Race,
    drivers: list[Driver],
    constructors: list[Constructor],
    prediction: dict[str, Any],
    features: dict[str, Any],
    qualifying: list[dict],
    sprint: list[dict],
    results: list[dict],
    session: str = "race",
    live: bool = False,
) -> dict[str, Any]:
    return build_session_projection(
        race=race,
        drivers=drivers,
        constructors=constructors,
        prediction=prediction,
        features=features,
        qualifying=qualifying,
        sprint=sprint,
        results=results,
        session=session,
        live=live,
    )
