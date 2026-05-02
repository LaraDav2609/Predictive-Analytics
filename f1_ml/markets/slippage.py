"""Slippage / fill model for Polymarket and Kalshi.

A model edge of 200 bps disappears fast against thin orderbooks. Realistic
backtests must penalize trades for the depth they consume.

Polymarket: AMM-style + CLOB on some markets. Walks the book; depth often thin
on niche F1 markets.
Kalshi: pure CLOB. Limit orders may not fill; size by fill probability.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrderbookLevel:
    price: float
    size_units: float


@dataclass
class FillEstimate:
    fill_units: float
    avg_price: float
    slippage_bps: float


def walk_book(
    levels: list[OrderbookLevel],
    target_units: float,
    side: str,  # "BUY_YES" | "BUY_NO"
) -> FillEstimate:
    """Simulate market-order fill against a CLOB."""
    raise NotImplementedError("iterate levels; sum until target_units exhausted; weighted avg price")


def kalshi_limit_fill_prob(distance_from_mid_bps: float, recent_volatility: float) -> float:
    """Empirical model: probability a limit order fills within trading window."""
    raise NotImplementedError
