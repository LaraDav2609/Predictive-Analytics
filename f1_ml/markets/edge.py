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
    best_bid: float        # implied prob a YES contract trades at; 1-best_bid = NO ask
    best_ask: float        # implied prob a YES contract is offered at
    fee_bps: float


@dataclass
class EdgeOpportunity:
    market_id: str
    direction: str         # "YES" | "NO"
    model_prob: float
    market_implied_prob: float
    edge_bps: float
    fee_bps: float


def compute_edge(
    model_prob: float,
    quote: MarketQuote,
    min_edge_bps: float = 200.0,
) -> EdgeOpportunity | None:
    """Return the better of (BUY YES, BUY NO) edges if it clears `min_edge_bps`,
    otherwise None.

    For BUY YES: edge = model_prob - best_ask (you pay best_ask, win at 1)
    For BUY NO:  edge = (1 - model_prob) - (1 - best_bid) = best_bid - model_prob
                  (you pay 1 - best_bid for a NO contract, win at 1)

    Both are then expressed in basis points after subtracting the venue fee.
    """
    if not (0.0 < model_prob < 1.0):
        return None
    if not (0.0 <= quote.best_bid <= 1.0 and 0.0 <= quote.best_ask <= 1.0):
        return None
    if quote.best_ask < quote.best_bid:
        # Crossed book → ill-formed quote; refuse.
        return None

    yes_edge_bps = (model_prob - quote.best_ask) * 10_000 - quote.fee_bps
    no_edge_bps = (quote.best_bid - model_prob) * 10_000 - quote.fee_bps

    if yes_edge_bps >= no_edge_bps and yes_edge_bps >= min_edge_bps:
        return EdgeOpportunity(
            market_id=quote.market_id,
            direction="YES",
            model_prob=model_prob,
            market_implied_prob=quote.best_ask,
            edge_bps=yes_edge_bps,
            fee_bps=quote.fee_bps,
        )
    if no_edge_bps > yes_edge_bps and no_edge_bps >= min_edge_bps:
        return EdgeOpportunity(
            market_id=quote.market_id,
            direction="NO",
            model_prob=model_prob,
            market_implied_prob=quote.best_bid,
            edge_bps=no_edge_bps,
            fee_bps=quote.fee_bps,
        )
    return None
