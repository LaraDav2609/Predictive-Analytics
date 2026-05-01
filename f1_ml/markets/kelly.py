"""Kelly-criterion position sizing — fractional Kelly with shrinkage.

Full Kelly is bankroll-optimal under perfect probability estimates. We're never
that confident, so size at ¼-Kelly to buffer against parameter uncertainty.
With heavy shrinkage we lose ~75% of growth rate but cut drawdown variance
dramatically — the right trade for trading on a model that's still being
calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KellySize:
    fraction_of_bankroll: float
    notional_usd: float


def kelly_fraction(model_prob: float, market_implied_prob: float) -> float:
    """f* = (p - q*odds) / odds, simplified for binary. Returns full-Kelly fraction."""
    if market_implied_prob <= 0 or market_implied_prob >= 1:
        return 0.0
    odds = (1 - market_implied_prob) / market_implied_prob
    return max(0.0, (model_prob * (odds + 1) - 1) / odds)


def shrunk_kelly(model_prob: float, market_implied_prob: float, shrinkage: float = 0.25) -> float:
    return shrinkage * kelly_fraction(model_prob, market_implied_prob)


def size_position(
    model_prob: float,
    market_implied_prob: float,
    bankroll_usd: float,
    shrinkage: float = 0.25,
    max_per_market_pct: float = 0.05,
) -> KellySize:
    f = min(shrunk_kelly(model_prob, market_implied_prob, shrinkage), max_per_market_pct)
    return KellySize(fraction_of_bankroll=f, notional_usd=f * bankroll_usd)
