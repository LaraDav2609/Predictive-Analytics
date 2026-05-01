"""Edge calculation — model_prob vs. market_implied_prob, after fees / spread.

Inputs:
  - model_prob: from markets.mapper
  - market: best-bid / best-offer from the .NET market clients (read off Redis)
  - fees: per-venue (Polymarket fee, Kalshi maker/taker)

Output: edge_bps and direction (BUY YES / BUY NO).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketQuote:
    venue: str             # "polymarket" | "kalshi"
    market_id: str
    best_bid: float        # implied prob to BUY NO at
    best_ask: float        # implied prob to BUY YES at
    fee_bps: float


@dataclass
class EdgeOpportunity:
    market_id: str
    direction: str         # "YES" | "NO"
    model_prob: float
    market_implied_prob: float
    edge_bps: float
    fee_bps: float


def compute_edge(model_prob: float, quote: MarketQuote, min_edge_bps: float = 200.0) -> EdgeOpportunity | None:
    """Return None if no side has edge above the threshold."""
    raise NotImplementedError(
        "compare model_prob to ask (YES side) and 1 - bid (NO side); subtract fee"
    )
