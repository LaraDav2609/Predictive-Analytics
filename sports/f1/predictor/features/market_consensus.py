"""Market-implied consensus as a bounded, governed prediction feature.

Converts venue-implied per-driver win probabilities (Kalshi / Polymarket, supplied
by the dashboard) into:
  - a per-driver bounded strength modifier, capped at the registered 0.045 max
    influence (see ``source_registry`` ``market_consensus``), so the market
    INFORMS the model without dominating it;
  - a constructor-level signal usable by the car-model feature;
  - a market leader + confidence for the existing market-vs-model disagreement
    governance (``data_quality.disagreement``).

Caps + source/explanation metadata keep this compliant with the hard rule that
every probability influence is sourced, capped, and explained.
"""
from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any

# Registered influence cap for market_consensus (source_registry.SOURCE_POLICIES).
MARKET_MAX_INFLUENCE = 0.045


def build_market_consensus(
    driver_implied: dict[str, float],
    *,
    driver_team: dict[str, str] | None = None,
    confidence: float = 0.58,
) -> dict[str, Any]:
    """Build the governed market-consensus signal from per-driver implied win probs.

    ``driver_implied``: ``{driver_id: implied_win_probability}`` in 0..1.
    ``driver_team``:    optional ``{driver_id: team_key}`` to roll up a constructor signal.
    """
    clean = {
        str(k): max(0.0, min(1.0, float(v)))
        for k, v in (driver_implied or {}).items()
        if v is not None
    }
    if not clean:
        return {
            "drivers": {}, "constructors": {}, "field_mean": 0.0, "field_std": 0.0,
            "top_driver_id": None, "confidence": 0.0, "source": "market_consensus",
            "max_influence": MARKET_MAX_INFLUENCE, "explanation": "no market consensus supplied",
        }

    values = list(clean.values())
    field_mean = mean(values)
    field_std = pstdev(values) if len(values) > 1 else 0.0
    top_driver_id = max(clean, key=clean.get)

    constructors: dict[str, float] = {}
    if driver_team:
        by_team: dict[str, list[float]] = {}
        for did, prob in clean.items():
            team = driver_team.get(did)
            if team:
                by_team.setdefault(str(team), []).append(prob)
        top = max(values) or 1.0
        for team, probs in by_team.items():
            constructors[team] = round(min(1.0, mean(probs) / top), 4)

    return {
        "drivers": clean,
        "constructors": constructors,
        "field_mean": round(field_mean, 6),
        "field_std": round(field_std, 6),
        "top_driver_id": top_driver_id,
        "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
        "source": "market_consensus",
        "max_influence": MARKET_MAX_INFLUENCE,
        "explanation": f"Market consensus from {len(clean)} drivers; capped at {MARKET_MAX_INFLUENCE} influence.",
    }


def market_strength_modifier(
    driver_id: Any,
    market_signals: dict[str, Any] | None,
    cap: float = MARKET_MAX_INFLUENCE,
) -> float:
    """Bounded multiplier in ``[1-cap, 1+cap]`` nudging a driver's strength toward the
    market-implied probability. Returns ``1.0`` (no effect) when market data is absent —
    so predictions are unchanged unless market consensus is explicitly supplied."""
    if not market_signals:
        return 1.0
    implied = (market_signals.get("drivers") or {}).get(str(driver_id))
    if implied is None:
        return 1.0
    mean_p = float(market_signals.get("field_mean") or implied)
    std = float(market_signals.get("field_std") or 0.0) or 0.1
    z = (float(implied) - mean_p) / std
    return max(1.0 - cap, min(1.0 + cap, 1.0 + cap * math.tanh(z)))
