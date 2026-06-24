"""F1 betting-edge service — the loop-closer that joins model probabilities to
venue market quotes and sizes a stake, reusing the shared betting core
(`common.ml.markets.{edge, kelly, implied}`). The F1 analogue of
`games.csgo.analytics.csgo_predictor.edge_and_stake`, batched over a race's quotes.

ANALYSIS-ONLY: produces edges + recommended fractional-Kelly stakes for review.
It does NOT place orders. The F1 model is still uncalibrated heuristic Monte-Carlo
(see `/api/f1/pipeline/health`), so keep this paper / gated until calibration is
proven.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from common.ml.markets.edge import MarketQuote, compute_edge
from common.ml.markets.implied import implied_from_quote
from common.ml.markets.kelly import size_position

# Model market types the F1 simulator publishes (see markets.mapper / the bridge).
SUPPORTED_MARKETS = ("winner", "podium", "top5", "points", "dnf")


class F1EdgeQuoteIn(BaseModel):
    """One venue quote (YES side in probability units, 0..1) to price against the model."""
    market_id: str
    venue: str = "kalshi"                                  # "kalshi" | "polymarket"
    market: str                                            # one of SUPPORTED_MARKETS
    entity_code: str                                       # driver code, e.g. "VER"
    yes_bid: float = Field(ge=0.0, le=1.0)
    yes_ask: float = Field(ge=0.0, le=1.0)
    no_bid: float | None = Field(default=None, ge=0.0, le=1.0)
    no_ask: float | None = Field(default=None, ge=0.0, le=1.0)
    fee_bps: float = 0.0


class F1EdgeRequest(BaseModel):
    """POST body for /races/{round}/edges: a bankroll + a batch of venue quotes."""
    bankroll_usd: float = 1000.0
    min_edge_bps: float = 200.0
    shrinkage: float = 0.25
    max_per_market_pct: float = 0.05
    quotes: list[F1EdgeQuoteIn] = Field(default_factory=list)


@dataclass
class RaceQuote:
    market_id: str
    venue: str
    market: str
    entity_code: str
    yes_bid: float
    yes_ask: float
    no_bid: float | None
    no_ask: float | None
    fee_bps: float

    @classmethod
    def from_input(cls, q: F1EdgeQuoteIn) -> "RaceQuote":
        return cls(q.market_id, q.venue, q.market, q.entity_code.upper(),
                   q.yes_bid, q.yes_ask, q.no_bid, q.no_ask, q.fee_bps)


def probability_map_from_records(records) -> dict[tuple[str, str], float]:
    """Build {(entity_code, market): probability} from OutcomeProbability records."""
    return {(r.entity_code, r.market): r.probability for r in records}


def compute_race_edges(
    prob_map: dict[tuple[str, str], float],
    quotes: list[RaceQuote],
    *,
    bankroll_usd: float = 1000.0,
    min_edge_bps: float = 200.0,
    shrinkage: float = 0.25,
    max_per_market_pct: float = 0.05,
) -> list[dict]:
    """For each quote, look up the model probability for (entity_code, market),
    de-vig the quote, compute the edge in bps and — if it clears `min_edge_bps` —
    a fractional-Kelly stake. Returns one row per quote, sorted by edge desc.

    Mirrors `compute_edge`'s framing: BUY YES edge = model_prob - best_ask;
    BUY NO edge = best_bid - model_prob (both minus the venue fee)."""
    rows: list[dict] = []
    for q in quotes:
        model_prob = prob_map.get((q.entity_code, q.market))
        base = {
            "market_id": q.market_id, "venue": q.venue, "market": q.market,
            "entity_code": q.entity_code, "yes_bid": q.yes_bid, "yes_ask": q.yes_ask,
            "fee_bps": q.fee_bps,
        }
        if model_prob is None:
            rows.append({**base, "ok": False, "reason": "no_model_probability",
                         "model_prob": None, "edge_bps": None, "tradeable": False,
                         "direction": None, "stake_fraction": 0.0, "stake_usd": 0.0})
            continue

        implied = implied_from_quote(q.yes_bid, q.yes_ask, q.no_bid, q.no_ask)
        opp = compute_edge(
            model_prob,
            MarketQuote(venue=q.venue, market_id=q.market_id,
                        best_bid=q.yes_bid, best_ask=q.yes_ask, fee_bps=q.fee_bps),
            min_edge_bps=min_edge_bps,
        )
        # Raw (pre-threshold) edges, reported for transparency even when not tradeable.
        yes_edge_bps = (model_prob - q.yes_ask) * 10_000 - q.fee_bps
        no_edge_bps = (q.yes_bid - model_prob) * 10_000 - q.fee_bps
        best_edge = max(yes_edge_bps, no_edge_bps)

        row = {
            **base, "ok": True,
            "model_prob": round(model_prob, 4),
            "fair_yes": round(implied.fair_yes, 4) if implied.valid else None,
            "overround_bps": round(implied.overround * 10_000, 1) if implied.valid else None,
            "spread_bps": round(implied.spread * 10_000, 1) if implied.valid else None,
            "yes_edge_bps": round(yes_edge_bps, 1),
            "no_edge_bps": round(no_edge_bps, 1),
            "best_direction": "YES" if yes_edge_bps >= no_edge_bps else "NO",
            "best_edge_bps": round(best_edge, 1),
            "tradeable": opp is not None,
        }
        if opp is not None:
            # Size against the chosen side's price (NO = the complement quote).
            if opp.direction == "YES":
                stake = size_position(model_prob, q.yes_ask, bankroll_usd, shrinkage, max_per_market_pct)
            else:
                stake = size_position(1.0 - model_prob, 1.0 - q.yes_bid, bankroll_usd, shrinkage, max_per_market_pct)
            row.update({
                "direction": opp.direction,
                "edge_bps": round(opp.edge_bps, 1),
                "market_implied_prob": round(opp.market_implied_prob, 4),
                "stake_fraction": round(stake.fraction_of_bankroll, 4),
                "stake_usd": round(stake.notional_usd, 2),
            })
        else:
            row.update({"direction": None, "edge_bps": round(best_edge, 1),
                        "stake_fraction": 0.0, "stake_usd": 0.0})
        rows.append(row)

    rows.sort(key=lambda r: (r.get("edge_bps") if r.get("edge_bps") is not None else -1e9), reverse=True)
    return rows
