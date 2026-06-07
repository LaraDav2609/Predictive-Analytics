"""Small numeric helpers shared by F1 feature and model modules."""

from __future__ import annotations

import math
from typing import Any

from sports.f1.models.f1 import Driver


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def label_from_score(score: float) -> str:
    if score > 0.08:
        return "Bullish"
    if score < -0.08:
        return "Bearish"
    return "Neutral"


def prior_score(prior: float, drivers: list[Driver]) -> float:
    """Convert a model win probability into a bounded strength component.

    The old implementation multiplied by grid size and capped at 1.0, which
    made several front runners indistinguishable and let a championship-led
    baseline dominate downstream session models. A uniform prior maps to 0.50;
    very strong priors rise gradually instead of saturating.
    """

    field_size = max(len(drivers), 1)
    ratio = max(0.0, float(prior or 0.0)) * field_size
    return max(0.12, min(0.88, 0.50 + math.tanh((ratio - 1.0) * 0.55) * 0.32))


def position_score(position: Any, default: float) -> float:
    try:
        value = int(position)
    except (TypeError, ValueError):
        return default
    return max(0.04, min(1.0, (22 - value) / 21))


def sentiment_factor(sentiment: float) -> float:
    return max(0.0, min(1.0, 0.5 + sentiment / 2))
