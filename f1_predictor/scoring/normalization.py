"""Small numeric helpers shared by F1 feature and model modules."""

from __future__ import annotations

from typing import Any

from models.f1 import Driver


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
    return max(0.02, min(1.0, prior * max(len(drivers), 1)))


def position_score(position: Any, default: float) -> float:
    try:
        value = int(position)
    except (TypeError, ValueError):
        return default
    return max(0.04, min(1.0, (22 - value) / 21))


def sentiment_factor(sentiment: float) -> float:
    return max(0.0, min(1.0, 0.5 + sentiment / 2))
