"""Walk-forward, leak-free CS2 backtest + calibration metrics.

Replays finished matches in chronological order. At each match the Glicko ratings
and feature history reflect ONLY prior matches (no lookahead); the ensemble model
predicts, then the actual result updates the ratings for future matches. Produces
Brier / log loss / reliability buckets / accuracy with favorite-vs-underdog and
BO1/BO3/BO5 splits, and an optional betting simulation (ROI) when market prices
are supplied. Reuses common.ml.calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from common.ml.calibration import brier_score, log_loss, reliability_curve
from common.ml.markets.edge import MarketQuote, compute_edge
from common.ml.markets.kelly import size_position
from games.csgo.analytics.model import CsgoEnsembleModel
from games.csgo.analytics.ratings import Glicko2, Rating
from games.csgo.features import FeatureExtractor, _winner
from games.csgo.models.csgo import CsgoMatch


@dataclass
class BacktestRecord:
    match_id: str
    team1: str
    team2: str
    prob: float       # model P(team1 wins the series)
    actual: float     # 1 if team1 won else 0
    best_of: int
    event: str = ""
    team1_score: int = 0
    team2_score: int = 0


def _row(r: "BacktestRecord") -> dict:
    """Per-match row in the dashboard Backtest-tab contract."""
    p = min(1 - 1e-12, max(1e-12, r.prob))
    a = r.actual
    return {
        "team1": r.team1, "team2": r.team2, "event": r.event,
        "winner": r.team1 if a == 1.0 else r.team2,
        "team1_win_probability": round(r.prob, 4),
        "team1_score": r.team1_score, "team2_score": r.team2_score,
        "brier": round((p - a) ** 2, 4),
        "log_loss": round(-(a * math.log(p) + (1 - a) * math.log(1 - p)), 4),
    }


def _split(probs: np.ndarray, actual: np.ndarray) -> dict:
    if len(probs) == 0:
        return {"n": 0}
    return {
        "n": int(len(probs)),
        "brier": round(brier_score(probs, actual), 4),
        "accuracy": round(float(np.mean((probs >= 0.5) == (actual == 1))), 4),
    }


def _simulate_bet(match_id: str, model_prob: float, actual: float,
                  market_prob: float, min_edge_bps: float, bankroll: float) -> Optional[dict]:
    """Flat binary settlement using the shared edge + Kelly core. market_prob is
    the venue's implied P(team1) (already no-vig)."""
    quote = MarketQuote(venue="sim", market_id=match_id, best_bid=market_prob,
                        best_ask=market_prob, fee_bps=0.0)
    edge = compute_edge(model_prob, quote, min_edge_bps=min_edge_bps)
    if edge is None:
        return None
    stake = size_position(model_prob, market_prob, bankroll_usd=bankroll).notional_usd
    if stake <= 0:
        return None
    price = market_prob if edge.direction == "YES" else (1.0 - market_prob)
    if price <= 0 or price >= 1:
        return None
    team1_won = actual == 1.0
    won = (edge.direction == "YES" and team1_won) or (edge.direction == "NO" and not team1_won)
    pnl = stake * ((1.0 - price) / price) if won else -stake
    return {"stake": stake, "pnl": pnl, "won": won}


def run_backtest(
    matches: list[CsgoMatch],
    model: CsgoEnsembleModel | None = None,
    min_history: int = 20,
    teams_by_id: dict | None = None,
    market_prob_for: Callable[[CsgoMatch], Optional[float]] | None = None,
    min_edge_bps: float = 200.0,
    bankroll: float = 1000.0,
) -> dict:
    model = model or CsgoEnsembleModel()
    finished = sorted([m for m in matches if _winner(m) is not None], key=lambda x: x.date)

    engine = Glicko2()
    ratings: dict[int, Rating] = {}
    map_ratings: dict[tuple[int, str], Rating] = {}
    history: list[CsgoMatch] = []
    records: list[BacktestRecord] = []
    bets: list[dict] = []

    for m in finished:
        # Predict using ONLY prior information.
        if len(history) >= min_history and m.team1_id in ratings and m.team2_id in ratings:
            feats = FeatureExtractor(history, dict(ratings), teams_by_id or {},
                                     map_ratings=dict(map_ratings)).extract(m)
            out = model.predict(feats)
            actual = 1.0 if _winner(m) == m.team1_id else 0.0
            records.append(BacktestRecord(
                m.id, m.team1, m.team2, out.winner_prob, actual, m.best_of,
                event=m.event or "",
                team1_score=m.team1_score if isinstance(m.team1_score, int) else 0,
                team2_score=m.team2_score if isinstance(m.team2_score, int) else 0,
            ))
            if market_prob_for is not None:
                mk = market_prob_for(m)
                if mk is not None:
                    bet = _simulate_bet(m.id, out.winner_prob, actual, mk, min_edge_bps, bankroll)
                    if bet:
                        bets.append(bet)

        # Then fold the result in (leak-free).
        r1 = ratings.get(m.team1_id, Rating())
        r2 = ratings.get(m.team2_id, Rating())
        s1 = 1.0 if _winner(m) == m.team1_id else 0.0
        ratings[m.team1_id] = engine.update(r1, r2, s1)
        ratings[m.team2_id] = engine.update(r2, r1, 1.0 - s1)
        for ms in m.map_scores:
            if ms.winner_id is None or not ms.map_name:
                continue
            k1, k2 = (m.team1_id, ms.map_name), (m.team2_id, ms.map_name)
            mr1 = map_ratings.get(k1, Rating())
            mr2 = map_ratings.get(k2, Rating())
            ms1 = 1.0 if ms.winner_id == m.team1_id else 0.0
            map_ratings[k1] = engine.update(mr1, mr2, ms1)
            map_ratings[k2] = engine.update(mr2, mr1, 1.0 - ms1)
        history.append(m)

    return _metrics(records, bets)


def _metrics(records: list[BacktestRecord], bets: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"scored": 0, "insufficient_data": True,
                "note": "Need finished matches with history; check the data provider."}

    probs = np.array([r.prob for r in records], dtype=float)
    actual = np.array([r.actual for r in records], dtype=float)

    brier = brier_score(probs, actual)
    base = brier_score(np.full(n, actual.mean()), actual)
    mp, mo, cnt = reliability_curve(probs, actual, n_bins=10)
    reliability = [
        {"bin": round((i + 0.5) / 10.0, 2), "mean_pred": round(float(mp[i]), 4),
         "mean_obs": round(float(mo[i]), 4), "count": int(cnt[i])}
        for i in range(10) if cnt[i] > 0
    ]

    confidence = np.maximum(probs, 1.0 - probs)
    fav_mask = confidence >= 0.65

    by_bo = {}
    for bo in (1, 3, 5):
        mask = np.array([r.best_of == bo for r in records])
        if mask.any():
            by_bo[f"bo{bo}"] = _split(probs[mask], actual[mask])

    result = {
        "scored": n,
        "brier": round(brier, 4),
        "log_loss": round(log_loss(probs, actual), 4),
        "base_rate_brier": round(base, 4),
        "brier_skill_score": round(1.0 - brier / base, 4) if base > 0 else 0.0,
        "accuracy": round(float(np.mean((probs >= 0.5) == (actual == 1))), 4),
        "reliability": reliability,
        "favorite": _split(probs[fav_mask], actual[fav_mask]),
        "underdog": _split(probs[~fav_mask], actual[~fav_mask]),
        "by_best_of": by_bo,
        # Dashboard Backtest-tab contract:
        "matches": n,                       # "Scored Matches" KPI (count)
        "rows": [_row(r) for r in records[-50:]],
    }

    if bets:
        staked = sum(b["stake"] for b in bets)
        pnl = sum(b["pnl"] for b in bets)
        wins = sum(1 for b in bets if b["won"])
        result["betting"] = {
            "bets": len(bets),
            "staked_usd": round(staked, 2),
            "pnl_usd": round(pnl, 2),
            "roi_pct": round(100.0 * pnl / staked, 2) if staked > 0 else 0.0,
            "win_rate": round(wins / len(bets), 4),
        }
    return result
