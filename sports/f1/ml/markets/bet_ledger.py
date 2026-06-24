"""Model-vs-market bet-ledger backtest — evaluates DECISIONS, not just accuracy.

Given a sequence of decisions (model probability, market quote, realized outcome),
this settles each bet, accumulates a bankroll, and reports the metrics that tell you
whether the edge is real: realized P&L, ROI on staked capital, max drawdown, hit
rate, closing-line value (CLV), and calibration-by-edge-bucket.

Reuses the shared betting core (``common.ml.markets`` edge / kelly / implied), so the
per-decision math is identical to the live ``edge_service``. Paper-only — the F1
model is uncalibrated heuristic Monte-Carlo, so this backtest is the instrument that
would *prove or disprove* the edge, not a license to trade.
"""
from __future__ import annotations

from typing import Any

from common.ml.markets.edge import MarketQuote, compute_edge
from common.ml.markets.implied import implied_from_quote, line_move
from common.ml.markets.kelly import size_position


def _edge_bucket(edge_bps: float) -> str:
    e = float(edge_bps)
    if e < 400:
        return "200-400"
    if e < 600:
        return "400-600"
    if e < 800:
        return "600-800"
    return "800+"


def _settle_pnl(price: float, stake: float, won: bool) -> float:
    """Binary-contract P&L: pay ``price`` per contract, each pays $1 on a win."""
    price = max(0.01, min(0.99, float(price)))
    return stake * (1.0 - price) / price if won else -stake


def run_bet_ledger(
    decisions: list[dict[str, Any]],
    *,
    bankroll_usd: float = 1000.0,
    min_edge_bps: float = 200.0,
    shrinkage: float = 0.25,
    max_per_market_pct: float = 0.05,
    default_fee_bps: float = 0.0,
) -> dict[str, Any]:
    """Run the ledger over decisions, each ``{model_prob, yes_bid, yes_ask, outcome,
    fee_bps?, close_yes?, label?}`` where ``outcome`` is 1 if YES occurred."""
    bankroll = float(bankroll_usd)
    start = bankroll
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    total_staked = 0.0
    wins = 0
    skipped = 0
    clv_values: list[float] = []
    buckets: dict[str, dict[str, int]] = {}
    bets: list[dict[str, Any]] = []

    for dec in decisions or []:
        try:
            mp = float(dec["model_prob"])
            yb = float(dec["yes_bid"])
            ya = float(dec["yes_ask"])
            yes_won = bool(int(dec["outcome"]))
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        fee = float(dec.get("fee_bps", default_fee_bps))
        opp = compute_edge(mp, MarketQuote("synthetic", str(dec.get("label", "m")), yb, ya, fee),
                           min_edge_bps=min_edge_bps)
        if opp is None:
            skipped += 1
            continue

        if opp.direction == "YES":
            price = max(0.01, min(0.99, ya))
            stake = size_position(mp, price, bankroll, shrinkage, max_per_market_pct).notional_usd
            won = yes_won
        else:
            price = max(0.01, min(0.99, 1.0 - yb))
            stake = size_position(1.0 - mp, price, bankroll, shrinkage, max_per_market_pct).notional_usd
            won = not yes_won
        if stake <= 0:
            skipped += 1
            continue

        pnl = _settle_pnl(price, stake, won)
        bankroll += pnl
        cum_pnl += pnl
        total_staked += stake
        if won:
            wins += 1
        peak = max(peak, cum_pnl)
        max_dd = max(max_dd, peak - cum_pnl)

        close = dec.get("close_yes")
        if close is not None:
            entry = implied_from_quote(yb, ya)
            if entry.valid:
                move = line_move(entry.fair_yes, max(0.0, min(1.0, float(close))))
                clv_values.append(move if opp.direction == "YES" else -move)

        bkt = buckets.setdefault(_edge_bucket(opp.edge_bps), {"bets": 0, "wins": 0})
        bkt["bets"] += 1
        bkt["wins"] += 1 if won else 0

        bets.append({
            "label": dec.get("label"),
            "direction": opp.direction,
            "edge_bps": round(opp.edge_bps, 1),
            "price": round(price, 4),
            "stake_usd": round(stake, 2),
            "won": won,
            "pnl_usd": round(pnl, 2),
            "bankroll_usd": round(bankroll, 2),
        })

    n = len(bets)
    return {
        "start_bankroll": round(start, 2),
        "end_bankroll": round(bankroll, 2),
        "realized_pnl": round(cum_pnl, 2),
        "total_staked": round(total_staked, 2),
        "roi": round(cum_pnl / total_staked, 4) if total_staked else 0.0,
        "bets_placed": n,
        "skipped": skipped,
        "wins": wins,
        "hit_rate": round(wins / n, 4) if n else 0.0,
        "max_drawdown": round(max_dd, 2),
        "avg_clv": round(sum(clv_values) / len(clv_values), 4) if clv_values else None,
        "calibration_by_edge_bucket": {
            k: {"bets": v["bets"], "win_rate": round(v["wins"] / v["bets"], 4) if v["bets"] else 0.0}
            for k, v in sorted(buckets.items())
        },
        "bets": bets,
        "disclaimer": "Paper-only. Model is uncalibrated heuristic Monte-Carlo; this proves/disproves edge, not a license to trade.",
    }
