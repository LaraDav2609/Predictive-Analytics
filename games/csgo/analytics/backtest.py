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
from games.csgo.live import _race
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
        "match_id": r.match_id,
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


def _fold_result(engine: Glicko2, ratings: dict, map_ratings: dict, m: CsgoMatch) -> None:
    """Update global + per-map Glicko ratings with match m's result. The single leak-free
    'reveal the outcome' step shared by the backtest, the replay, and calibrator fitting."""
    r1 = ratings.get(m.team1_id, Rating())
    r2 = ratings.get(m.team2_id, Rating())
    s1 = 1.0 if _winner(m) == m.team1_id else 0.0
    ratings[m.team1_id] = engine.update(r1, r2, s1)
    ratings[m.team2_id] = engine.update(r2, r1, 1.0 - s1)
    for ms in m.map_scores:
        if ms.winner_id is None or not ms.map_name:
            continue
        k1, k2 = (m.team1_id, ms.map_name), (m.team2_id, ms.map_name)
        mr1, mr2 = map_ratings.get(k1, Rating()), map_ratings.get(k2, Rating())
        ms1 = 1.0 if ms.winner_id == m.team1_id else 0.0
        map_ratings[k1] = engine.update(mr1, mr2, ms1)
        map_ratings[k2] = engine.update(mr2, mr1, 1.0 - ms1)


def fit_calibrator(matches: list[CsgoMatch], teams_by_id: dict | None = None, min_history: int = 20):
    """Fit a Platt scaler on the ensemble's walk-forward predictions over history, for
    applying to FUTURE live predictions (the calibrator learns the model's miscalibration;
    fit on the past, applied going forward). Returns None when there's too little history
    (<30 scored) or sklearn is unavailable, so the caller stays uncalibrated."""
    model = CsgoEnsembleModel()   # raw, to collect uncalibrated predictions
    finished = sorted([m for m in matches if _winner(m) is not None], key=lambda x: x.date)

    engine = Glicko2()
    ratings: dict[int, Rating] = {}
    map_ratings: dict[tuple[int, str], Rating] = {}
    history: list[CsgoMatch] = []
    probs: list[float] = []
    actuals: list[float] = []
    for m in finished:
        if len(history) >= min_history and m.team1_id in ratings and m.team2_id in ratings:
            feats = FeatureExtractor(history, dict(ratings), teams_by_id or {},
                                     map_ratings=dict(map_ratings)).extract(m)
            probs.append(model.predict(feats).winner_prob)
            actuals.append(1.0 if _winner(m) == m.team1_id else 0.0)
        _fold_result(engine, ratings, map_ratings, m)
        history.append(m)

    if len(probs) < 30:
        return None
    try:
        from common.ml.calibration import PlattScaler
    except Exception:
        return None
    try:
        return PlattScaler().fit(np.array(probs), np.array(actuals))
    except Exception:
        return None


def run_backtest(
    matches: list[CsgoMatch],
    model: CsgoEnsembleModel | None = None,
    min_history: int = 20,
    teams_by_id: dict | None = None,
    market_prob_for: Callable[[CsgoMatch], Optional[float]] | None = None,
    min_edge_bps: float = 200.0,
    bankroll: float = 1000.0,
    calibrate: bool = False,
    calibration_fraction: float = 0.4,
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
        _fold_result(engine, ratings, map_ratings, m)
        history.append(m)

    result = _metrics(records, bets)
    if calibrate:
        result["calibration"] = _calibration_report(records, calibration_fraction)
    return result


def _calibration_report(records: list[BacktestRecord], fraction: float) -> dict:
    """Fit a Platt scaler on an EARLIER window and score it on the later holdout —
    leak-free in time. Reports raw vs calibrated Brier/log-loss on the holdout."""
    n = len(records)
    if n < 30:
        return {"available": False, "reason": "need >=30 scored matches"}
    split = max(10, int(n * fraction))
    if split >= n - 5:
        return {"available": False, "reason": "insufficient holdout"}
    try:
        from common.ml.calibration import PlattScaler
    except Exception as exc:  # sklearn not installed
        return {"available": False, "reason": f"calibrator unavailable: {exc}"}

    train, holdout = records[:split], records[split:]
    tp = np.array([r.prob for r in train]); ty = np.array([r.actual for r in train])
    hp = np.array([r.prob for r in holdout]); hy = np.array([r.actual for r in holdout])
    try:
        calibrated = PlattScaler().fit(tp, ty).transform(hp)
    except Exception as exc:
        return {"available": False, "reason": f"fit failed: {exc}"}

    return {
        "available": True,
        "method": "platt",
        "train_n": len(train),
        "holdout_n": len(holdout),
        "raw": {"brier": round(brier_score(hp, hy), 4), "log_loss": round(log_loss(hp, hy), 4)},
        "calibrated": {"brier": round(brier_score(calibrated, hy), 4),
                       "log_loss": round(log_loss(calibrated, hy), 4)},
    }


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


def _series_from_maps(w1: int, w2: int, maps_needed: int, per_map: float) -> float:
    """Series win prob for team1 given maps won so far (w1-w2), from a per-map prob.
    Equals _series_win_prob at 0-0, and clamps to a clinch (1/0) when decided."""
    m1, m2 = maps_needed - w1, maps_needed - w2
    if m1 <= 0:
        return 1.0
    if m2 <= 0:
        return 0.0
    return _race(m1, m2, per_map)


def replay_match(
    match_id: str,
    matches: list[CsgoMatch],
    model: CsgoEnsembleModel | None = None,
    teams_by_id: dict | None = None,
    min_history: int = 20,
) -> dict:
    """Single-game replay for one finished match: the leak-free pre-game prediction
    (ratings built from ONLY earlier matches), the actual result, and the map-by-map
    series win-probability trajectory the live model would have shown — anchored so the
    'Pre-game' point equals the model's series call and the curve clinches at 1/0."""
    model = model or CsgoEnsembleModel()
    finished = sorted([m for m in matches if _winner(m) is not None], key=lambda x: x.date)
    target = next((m for m in finished if str(m.id) == str(match_id)), None)
    if target is None:
        return {"ok": False, "reason": "match_not_found_or_unfinished", "match_id": str(match_id)}

    # Build ratings from matches strictly before the target (no lookahead).
    engine = Glicko2()
    ratings: dict[int, Rating] = {}
    map_ratings: dict[tuple[int, str], Rating] = {}
    history: list[CsgoMatch] = []
    for m in finished:
        if str(m.id) == str(target.id):
            break
        _fold_result(engine, ratings, map_ratings, m)
        history.append(m)

    leak_free = (len(history) >= min_history
                 and target.team1_id in ratings and target.team2_id in ratings)
    if leak_free:
        feats = FeatureExtractor(history, dict(ratings), teams_by_id or {},
                                 map_ratings=dict(map_ratings)).extract(target)
        out = model.predict(feats)
        pregame, per_map, confidence = out.winner_prob, out.map1_team1_prob, out.confidence
    else:
        pregame, per_map, confidence = 0.5, 0.5, 0.0   # too little prior history for a real read

    actual = 1.0 if _winner(target) == target.team1_id else 0.0
    maps_needed = target.best_of // 2 + 1

    # Map-by-map trajectory: Pre-game point + one point per decided map, stop at clinch.
    ordered = sorted(target.map_scores, key=lambda x: (x.order or 0))
    # Anchor the first point to the model's exact series call (0-0 == pre-game prob).
    trajectory = [{"label": "Pre-game", "maps": "0-0", "team1_series_prob": round(pregame, 4)}]
    maps_detail: list[dict] = []
    w1 = w2 = 0
    for i, ms in enumerate(ordered, start=1):
        if ms.winner_id == target.team1_id:
            w1 += 1
        elif ms.winner_id == target.team2_id:
            w2 += 1
        else:
            continue
        map_winner = (target.team1_abbrev or target.team1) if ms.winner_id == target.team1_id \
            else (target.team2_abbrev or target.team2)
        maps_detail.append({"order": i, "map_name": ms.map_name,
                            "team1_rounds": ms.team1_rounds, "team2_rounds": ms.team2_rounds,
                            "winner": map_winner})
        label = f"Map {i}" + (f" · {ms.map_name}" if ms.map_name else "")
        trajectory.append({"label": label, "map_name": ms.map_name,
                           "team1_rounds": ms.team1_rounds, "team2_rounds": ms.team2_rounds,
                           "maps": f"{w1}-{w2}",
                           "team1_series_prob": round(_series_from_maps(w1, w2, maps_needed, per_map), 4)})
        if w1 >= maps_needed or w2 >= maps_needed:
            break

    p = min(1 - 1e-12, max(1e-12, pregame))
    return {
        "ok": True,
        "match_id": str(target.id),
        "team1": target.team1, "team2": target.team2,
        "team1_abbrev": target.team1_abbrev or target.team1,
        "team2_abbrev": target.team2_abbrev or target.team2,
        "event": target.event or "",
        "best_of": target.best_of,
        "date": target.date.isoformat() if target.date else None,
        "leak_free": leak_free,
        "history_used": len(history),
        "pregame_team1_prob": round(pregame, 4),
        "pregame_per_map_prob": round(per_map, 4),
        "confidence": round(confidence, 4),
        "winner": target.team1 if actual == 1.0 else target.team2,
        "winner_is_team1": actual == 1.0,
        "team1_score": target.team1_score if isinstance(target.team1_score, int) else 0,
        "team2_score": target.team2_score if isinstance(target.team2_score, int) else 0,
        "maps": maps_detail,
        "brier": round((p - actual) ** 2, 4),
        "log_loss": round(-(actual * math.log(p) + (1 - actual) * math.log(1 - p)), 4),
        "correct": (pregame >= 0.5) == (actual == 1.0),
        "trajectory": trajectory,
    }
