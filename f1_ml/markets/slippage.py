"""Slippage / fill model for Polymarket and Kalshi.

A model edge of 200 bps disappears fast against thin orderbooks. Realistic
backtests must penalize trades for the depth they consume.

Polymarket: AMM-style + CLOB on some markets. Walks the book; depth often thin
on niche F1 markets.
Kalshi: pure CLOB. Limit orders may not fill; size by fill probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OrderbookLevel:
    price: float       # 0..1
    size_units: float  # contracts available at this price


@dataclass
class FillEstimate:
    fill_units: float       # how many of `target_units` actually filled
    avg_price: float        # volume-weighted average fill price (0..1)
    slippage_bps: float     # vs. top-of-book mid (signed: positive = adverse)


def walk_book(
    levels: list[OrderbookLevel],
    target_units: float,
    side: str = "BUY_YES",
) -> FillEstimate:
    """Simulate market-order fill against a CLOB.

    For BUY orders, levels are sorted ascending by price (best ask first).
    For SELL orders, sorted descending (best bid first). Caller is
    responsible for the right ordering — we walk the list as-is.
    """
    if target_units <= 0 or not levels:
        return FillEstimate(fill_units=0.0, avg_price=0.0, slippage_bps=0.0)

    # Top of book is the first level; we measure slippage against it.
    top_price = levels[0].price
    remaining = target_units
    cost = 0.0
    filled = 0.0
    for lvl in levels:
        if remaining <= 0:
            break
        take = min(lvl.size_units, remaining)
        cost += take * lvl.price
        filled += take
        remaining -= take

    if filled == 0:
        return FillEstimate(fill_units=0.0, avg_price=0.0, slippage_bps=0.0)

    avg_price = cost / filled
    # For BUY: slippage = (avg_price - top_price) / top_price (positive = paid more).
    # For SELL: slippage = (top_price - avg_price) / top_price.
    raw_slip = avg_price - top_price if side.upper().startswith("BUY") else top_price - avg_price
    slippage_bps = (raw_slip / top_price) * 10_000 if top_price > 0 else 0.0
    return FillEstimate(
        fill_units=filled,
        avg_price=avg_price,
        slippage_bps=slippage_bps,
    )


def kalshi_limit_fill_prob(distance_from_mid_bps: float, recent_volatility: float) -> float:
    """Empirical model: P(fill within trading window) ≈ exp(-d^2 / (2σ²)).

    `distance_from_mid_bps` is signed; for limit orders posted on the *passive*
    side we expect d < 0 (more aggressive) → near 1.0. Limit orders posted
    *farther* from mid (d > 0) need bigger price moves to fill → exponentially
    decaying probability.

    `recent_volatility` is in bps (e.g., 200 = 2% expected price range over
    the trading window).
    """
    if recent_volatility <= 0:
        return 0.0 if distance_from_mid_bps > 0 else 1.0
    # Aggressive limits (d < 0) clamp to high probability.
    if distance_from_mid_bps <= 0:
        return min(1.0, 0.99)
    z = distance_from_mid_bps / recent_volatility
    return float(math.exp(-(z ** 2) / 2))
