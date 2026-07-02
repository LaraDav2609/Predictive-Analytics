"""Leak-free, walk-forward MLB backtest + calibration metrics.

Replays finished games in date order. Each game is predicted from a running
``TeamState`` built ONLY from prior games (no lookahead); then the result is
revealed and folded in for future games. Produces Brier / log loss / reliability /
accuracy and a Brier-skill score vs the home-base-rate baseline, plus an optional
Platt calibration report (fit on an earlier window, scored on a later holdout).
Mirrors the CS2 / F1 backtests and reuses ``common.ml.calibration``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from common.ml.calibration import brier_score, log_loss, reliability_curve
from sports.baseball.analytics.game_model import TeamState, predict_game


@dataclass
class BacktestRecord:
    game_id: int
    date: str
    home: str
    away: str
    prob_home: float       # model P(home wins)
    actual: float          # 1 if home won else 0
    home_score: int = 0
    away_score: int = 0
    components: dict = field(default_factory=dict)


def _finished(games) -> list:
    out = [g for g in games
           if getattr(g, "home_score", None) is not None
           and getattr(g, "away_score", None) is not None
           and g.home_score != g.away_score]        # MLB has no ties
    return sorted(out, key=lambda g: g.date)


def run_backtest(games, *, min_games: int = 10, calibrate: bool = True, calibration_fraction: float = 0.4) -> dict:
    finished = _finished(games)
    states: dict[int, TeamState] = {}
    records: list[BacktestRecord] = []

    for g in finished:
        home = states.setdefault(g.home_team_id, TeamState())
        away = states.setdefault(g.away_team_id, TeamState())
        pred = predict_game(home, away, home_pitcher=g.home_pitcher, away_pitcher=g.away_pitcher, min_games=min_games)
        if pred["leak_free"]:
            actual = 1.0 if g.home_score > g.away_score else 0.0
            records.append(BacktestRecord(
                game_id=g.id, date=g.date.isoformat() if hasattr(g.date, "isoformat") else str(g.date),
                home=g.home_team, away=g.away_team, prob_home=pred["home_win_prob"], actual=actual,
                home_score=int(g.home_score), away_score=int(g.away_score), components=pred["components"],
            ))
        home.record_game(g.home_score, g.away_score)
        away.record_game(g.away_score, g.home_score)

    return _metrics(records, calibrate, calibration_fraction)


def _row(r: BacktestRecord) -> dict:
    p = min(1 - 1e-12, max(1e-12, r.prob_home))
    return {
        "game_id": r.game_id, "date": r.date, "home": r.home, "away": r.away,
        "home_win_probability": round(r.prob_home, 4),
        "winner": r.home if r.actual == 1.0 else r.away,
        "score": f"{r.home_score}-{r.away_score}",
        "brier": round((p - r.actual) ** 2, 4),
        "correct": (r.prob_home >= 0.5) == (r.actual == 1.0),
    }


def _metrics(records: list[BacktestRecord], calibrate: bool, calibration_fraction: float) -> dict:
    n = len(records)
    if n == 0:
        return {"scored": 0, "insufficient_data": True,
                "note": "Need finished games with prior history; check the data provider / season range."}

    probs = np.array([r.prob_home for r in records], dtype=float)
    actual = np.array([r.actual for r in records], dtype=float)

    brier = brier_score(probs, actual)
    base = brier_score(np.full(n, actual.mean()), actual)      # home-base-rate baseline
    mp, mo, cnt = reliability_curve(probs, actual, n_bins=10)
    reliability = [
        {"bin": round((i + 0.5) / 10.0, 2), "mean_pred": round(float(mp[i]), 4),
         "mean_obs": round(float(mo[i]), 4), "count": int(cnt[i])}
        for i in range(10) if cnt[i] > 0
    ]
    confidence = np.maximum(probs, 1.0 - probs)
    fav = confidence >= 0.58

    result = {
        "scored": n,
        "home_win_rate": round(float(actual.mean()), 4),
        "brier": round(brier, 4),
        "log_loss": round(log_loss(probs, actual), 4),
        "base_rate_brier": round(base, 4),
        "brier_skill_score": round(1.0 - brier / base, 4) if base > 0 else 0.0,
        "accuracy": round(float(np.mean((probs >= 0.5) == (actual == 1))), 4),
        "home_pick_accuracy": round(float(actual.mean()), 4),   # accuracy of "always pick home"
        "reliability": reliability,
        "favorite": _split(probs[fav], actual[fav]),
        "underdog": _split(probs[~fav], actual[~fav]),
        "matches": n,
        "rows": [_row(r) for r in records[-60:]],
    }
    if calibrate:
        result["calibration"] = _calibration_report(records, calibration_fraction)
    return result


def _split(probs: np.ndarray, actual: np.ndarray) -> dict:
    if len(probs) == 0:
        return {"n": 0}
    return {"n": int(len(probs)),
            "brier": round(brier_score(probs, actual), 4),
            "accuracy": round(float(np.mean((probs >= 0.5) == (actual == 1))), 4)}


def _calibration_report(records: list[BacktestRecord], fraction: float) -> dict:
    """Fit a Platt scaler on an EARLIER window, score it on the later holdout —
    leak-free in time. Reports raw vs calibrated Brier/log-loss on the holdout."""
    n = len(records)
    if n < 50:
        return {"available": False, "reason": "need >=50 scored games"}
    split = max(20, int(n * fraction))
    if split >= n - 10:
        return {"available": False, "reason": "insufficient holdout"}
    try:
        from common.ml.calibration import PlattScaler
    except Exception as exc:
        return {"available": False, "reason": f"calibrator unavailable: {exc}"}

    train, holdout = records[:split], records[split:]
    tp = np.array([r.prob_home for r in train]); ty = np.array([r.actual for r in train])
    hp = np.array([r.prob_home for r in holdout]); hy = np.array([r.actual for r in holdout])
    try:
        calibrated = PlattScaler().fit(tp, ty).transform(hp)
    except Exception as exc:
        return {"available": False, "reason": f"fit failed: {exc}"}
    return {
        "available": True, "method": "platt", "train_n": len(train), "holdout_n": len(holdout),
        "raw": {"brier": round(brier_score(hp, hy), 4), "log_loss": round(log_loss(hp, hy), 4)},
        "calibrated": {"brier": round(brier_score(calibrated, hy), 4), "log_loss": round(log_loss(calibrated, hy), 4)},
    }
