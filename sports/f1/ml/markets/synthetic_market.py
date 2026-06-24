"""Synthetic prediction-market generator for retrospective bet-ledger backtests.

There is no stored history of real F1 Kalshi/Polymarket prices, so to exercise the
ledger over past races we derive a synthetic price from a reference probability
perturbed by a bias + caller-supplied noise + an overround (vig) + a bid/ask spread.

IMPORTANT: a market derived from the model is partly CIRCULAR — it tests the ledger
plumbing and the edge math, but cannot validate real edge. Use the forward
F1OddsSnapshotService (real history) for genuine model-vs-market evaluation.
"""
from __future__ import annotations

from typing import Any


def synthetic_quote(
    reference_prob: float,
    *,
    bias: float = 0.0,
    noise: float = 0.0,
    overround: float = 0.04,
    spread: float = 0.02,
) -> dict[str, float]:
    """A YES bid/ask around a perturbed reference probability. Deterministic given
    inputs (pass ``noise`` from a seeded RNG keyed by race/driver for reproducibility)."""
    mid = max(0.02, min(0.98, float(reference_prob) + float(bias) + float(noise)))
    half = (float(spread) + float(overround)) / 2.0
    yes_bid = max(0.01, round(mid - half, 4))
    yes_ask = min(0.99, round(mid + half, 4))
    if yes_ask <= yes_bid:
        yes_ask = min(0.99, round(yes_bid + 0.01, 4))
    return {"yes_bid": yes_bid, "yes_ask": yes_ask, "mid": round(mid, 4)}


def build_synthetic_decisions(
    samples: list[dict[str, Any]],
    *,
    market_bias: float = 0.0,
    overround: float = 0.04,
    spread: float = 0.02,
) -> list[dict[str, Any]]:
    """Turn ``(model_prob, outcome[, market_ref, noise, close_yes])`` samples into
    ledger decisions with synthetic quotes. ``market_ref`` defaults to ``model_prob``
    (most circular); pass a separate reference to reduce circularity."""
    decisions = []
    for s in samples or []:
        mp = float(s.get("model_prob") or 0.0)
        ref = float(s.get("market_ref", mp))
        q = synthetic_quote(ref, bias=market_bias, noise=float(s.get("noise", 0.0)),
                            overround=overround, spread=spread)
        decisions.append({
            "model_prob": mp,
            "yes_bid": q["yes_bid"],
            "yes_ask": q["yes_ask"],
            "outcome": int(s.get("outcome", 0)),
            "close_yes": s.get("close_yes"),
            "label": s.get("label"),
        })
    return decisions
