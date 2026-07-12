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
    features: dict = field(default_factory=dict)   # raw pre-weight signals for the blend


def _finished(games) -> list:
    out = [g for g in games
           if getattr(g, "home_score", None) is not None
           and getattr(g, "away_score", None) is not None
           and g.home_score != g.away_score]        # MLB has no ties
    return sorted(out, key=lambda g: g.date)


def team_states_before(games, before) -> dict[int, TeamState]:
    """Build each team's running state from only the games played strictly before
    ``before`` (leak-free) — used to produce a live/pre-game analysis for a match."""
    states: dict[int, TeamState] = {}
    for g in _finished(games):
        if g.date >= before:
            break
        states.setdefault(g.home_team_id, TeamState()).record_game(g.home_score, g.away_score)
        states.setdefault(g.away_team_id, TeamState()).record_game(g.away_score, g.home_score)
    return states


def replay(games, *, min_games: int = 10, pitcher_era: dict | None = None,
           rate_pitcher=None, run_env: bool = False, weather_by_game: dict | None = None,
           fatigue: bool = False, fatigue_by_game: dict | None = None,
           bullpen_by_game: dict | None = None,
           lineup_by_game: dict | None = None) -> list[BacktestRecord]:
    """Leak-free date-ordered replay → the scored per-game records. Shared by the
    backtest metrics and the serve-calibrator fit so both see the exact same
    (probability, outcome) pairs from a prior-only ``TeamState``.

    ``run_env`` overlays each game's static home-park run factor (free, leak-free)
    onto the model so the backtest can MEASURE whether the run-environment signal
    helps. ``weather_by_game`` optionally supplies {game_id: weather_dict} (from
    ``WeatherClient``) to add first-pitch wind/temperature on top of the park factor.
    ``fatigue`` (or an explicit ``fatigue_by_game`` map) overlays schedule-only rest /
    game-density signals (leak-free) so their effect can be measured too.
    ``bullpen_by_game`` optionally supplies {game_id: bullpen_dict} (from
    ``bullpen_fatigue.bullpen_by_game``) to overlay REAL leak-free bullpen workload. All
    are off by default so serve behaviour is unchanged."""
    finished = _finished(games)
    states: dict[int, TeamState] = {}
    records: list[BacktestRecord] = []
    park_factor = None
    if run_env:
        from sports.baseball.data.weather_client import park_factor as _pf
        park_factor = _pf
    if fatigue and fatigue_by_game is None:
        from sports.baseball.analytics.schedule_fatigue import fatigue_by_game as _fbg
        fatigue_by_game = _fbg(games)

    for g in finished:
        home = states.setdefault(g.home_team_id, TeamState())
        away = states.setdefault(g.away_team_id, TeamState())
        if rate_pitcher is not None:
            h_era = rate_pitcher(g.home_pitcher_id, g.date)
            a_era = rate_pitcher(g.away_pitcher_id, g.date)
        elif pitcher_era:
            h_era = pitcher_era.get(g.home_pitcher_id)
            a_era = pitcher_era.get(g.away_pitcher_id)
        else:
            h_era = a_era = None
        pf = park_factor(getattr(g, "venue", None) or g.home_team) if park_factor else None
        wx = weather_by_game.get(g.id) if weather_by_game else None
        fx = fatigue_by_game.get(g.id) if fatigue_by_game else None
        bx = bullpen_by_game.get(g.id) if bullpen_by_game else None
        lx = lineup_by_game.get(g.id) if lineup_by_game else None
        pred = predict_game(home, away, home_pitcher=g.home_pitcher, away_pitcher=g.away_pitcher,
                            home_pitcher_era=h_era, away_pitcher_era=a_era,
                            park_factor=pf, weather=wx, fatigue=fx, bullpen=bx, lineup=lx,
                            min_games=min_games)
        if pred["leak_free"]:
            actual = 1.0 if g.home_score > g.away_score else 0.0
            records.append(BacktestRecord(
                game_id=g.id, date=g.date.isoformat() if hasattr(g.date, "isoformat") else str(g.date),
                home=g.home_team, away=g.away_team, prob_home=pred["home_win_prob"], actual=actual,
                home_score=int(g.home_score), away_score=int(g.away_score), components=pred["components"],
                features=pred.get("features", {}),
            ))
        home.record_game(g.home_score, g.away_score)
        away.record_game(g.away_score, g.home_score)
    return records


def run_backtest(games, *, min_games: int = 10, calibrate: bool = True,
                 calibration_fraction: float = 0.4, pitcher_era: dict | None = None,
                 rate_pitcher=None, run_env: bool = False, weather_by_game: dict | None = None,
                 fatigue: bool = False, fatigue_by_game: dict | None = None,
                 bullpen_by_game: dict | None = None,
                 lineup_by_game: dict | None = None) -> dict:
    """``rate_pitcher(pitcher_id, game_date) -> float | None`` supplies a date-aware
    (leak-free) starter rating; ``pitcher_era`` is the static per-id fallback.
    ``run_env=True`` overlays the static park factor so the effect on Brier-skill is
    measurable (compare a run with and without it). ``fatigue=True`` (or a
    ``fatigue_by_game`` map) overlays the schedule-only rest/density signal likewise.
    ``bullpen_by_game`` overlays the REAL leak-free bullpen workload signal so its effect
    on Brier / Brier-skill can be measured head-to-head with a run without it."""
    records = replay(games, min_games=min_games, pitcher_era=pitcher_era, rate_pitcher=rate_pitcher,
                     run_env=run_env, weather_by_game=weather_by_game,
                     fatigue=fatigue, fatigue_by_game=fatigue_by_game,
                     bullpen_by_game=bullpen_by_game, lineup_by_game=lineup_by_game)
    return _metrics(records, calibrate, calibration_fraction)


def fit_serve_calibrator(games, *, min_games: int = 10, pitcher_era: dict | None = None,
                         rate_pitcher=None, holdout_fraction: float = 0.4) -> dict:
    """Fit a Platt scaler for use at SERVE time. Two reads come back:

    - ``params`` (a, b) fit on ALL scored games in the range — the calibrator we
      actually deploy (use every past game to calibrate the next one).
    - ``holdout`` — an honest earlier-window-fit / later-window-score check proving
      the calibration helps out-of-sample before we enable it.
    """
    from common.ml.calibration import PlattScaler

    records = replay(games, min_games=min_games, pitcher_era=pitcher_era, rate_pitcher=rate_pitcher)
    n = len(records)
    if n < 50:
        return {"fitted": False, "reason": f"need >=50 scored games, have {n}"}
    p = np.array([r.prob_home for r in records]); y = np.array([r.actual for r in records])
    scaler = PlattScaler().fit(p, y)
    fit_brier_raw = brier_score(p, y)
    fit_brier_cal = brier_score(scaler.transform(p), y)
    return {
        "fitted": True, "method": "platt",
        "params": {"a": round(scaler.a, 6), "b": round(scaler.b, 6)},
        "n": n,
        "fit_brier_raw": round(float(fit_brier_raw), 4),
        "fit_brier_cal": round(float(fit_brier_cal), 4),
        "holdout": _calibration_report(records, holdout_fraction),
    }


def _feature_matrix(records, feature_order):
    X = np.array([[float(r.features.get(f, 0.0)) for f in feature_order] for r in records], dtype=float)
    y = np.array([r.actual for r in records], dtype=float)
    return X, y


def fit_blend(games, *, min_games: int = 15, pitcher_era: dict | None = None,
              rate_pitcher=None, holdout_fraction: float = 0.4,
              run_env: bool = False, weather_by_game: dict | None = None,
              fatigue: bool = False, fatigue_by_game: dict | None = None,
              bullpen_by_game: dict | None = None,
              lineup_by_game: dict | None = None) -> dict:
    """Fit a logistic BLEND over the raw signals and compare it head-to-head with the
    hand-tuned model on an honest earlier→later holdout. Returns learned weights plus
    the holdout Brier / log-loss / Brier-skill for both models, so the caller can gate
    on ``blend beats hand-tuned`` before enabling it at serve.

    ``run_env`` / ``weather_by_game`` / ``fatigue`` / ``fatigue_by_game`` /
    ``bullpen_by_game`` turn on the corresponding overlays during the replay so their raw
    signals are non-constant and the blend can actually learn a weight for them (off by
    default → the original four-feature fit)."""
    from sklearn.linear_model import LogisticRegression
    from sports.baseball.analytics.blend_model import FEATURE_ORDER, apply_blend

    records = replay(games, min_games=min_games, pitcher_era=pitcher_era, rate_pitcher=rate_pitcher,
                     run_env=run_env, weather_by_game=weather_by_game,
                     fatigue=fatigue, fatigue_by_game=fatigue_by_game,
                     bullpen_by_game=bullpen_by_game, lineup_by_game=lineup_by_game)
    n = len(records)
    if n < 100:
        return {"fitted": False, "reason": f"need >=100 scored games, have {n}"}

    X, y = _feature_matrix(records, FEATURE_ORDER)
    # Deployed weights: fit on ALL games (use every past game to weight the next).
    full = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(X, y)
    weights = {"intercept": float(full.intercept_[0]),
               "coef": {f: float(c) for f, c in zip(FEATURE_ORDER, full.coef_[0])}}

    # Honest holdout: fit on earlier window, score both models on the later window.
    split = max(50, int(n * holdout_fraction))
    holdout: dict = {"available": False, "reason": "insufficient holdout"}
    if split < n - 30:
        Xtr, ytr = X[:split], y[:split]
        Xho = X[split:]
        hy = y[split:]
        hand = np.array([r.prob_home for r in records[split:]])
        if len(np.unique(ytr)) >= 2:
            wtr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit(Xtr, ytr)
            w_ho = {"intercept": float(wtr.intercept_[0]),
                    "coef": {f: float(c) for f, c in zip(FEATURE_ORDER, wtr.coef_[0])}}
            bp = np.array([apply_blend({f: Xho[i, j] for j, f in enumerate(FEATURE_ORDER)}, w_ho)["value"]
                           for i in range(len(Xho))])
            base = brier_score(np.full(len(hy), hy.mean()), hy)
            holdout = {
                "available": True, "holdout_n": int(len(hy)), "train_n": int(split),
                "base_rate_brier": round(base, 4),
                "hand_tuned": {"brier": round(brier_score(hand, hy), 4),
                               "log_loss": round(log_loss(hand, hy), 4),
                               "brier_skill": round(1.0 - brier_score(hand, hy) / base, 4) if base > 0 else 0.0},
                "blend": {"brier": round(brier_score(bp, hy), 4),
                          "log_loss": round(log_loss(bp, hy), 4),
                          "brier_skill": round(1.0 - brier_score(bp, hy) / base, 4) if base > 0 else 0.0},
            }
            holdout["blend_beats_hand_tuned"] = bool(holdout["blend"]["brier"] < holdout["hand_tuned"]["brier"])

    return {"fitted": True, "model_version": "mlb-blend-v1", "n": n,
            "feature_order": FEATURE_ORDER, "weights": weights, "holdout": holdout}


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
        scaler = PlattScaler().fit(tp, ty)
        calibrated = scaler.transform(hp)
    except Exception as exc:
        return {"available": False, "reason": f"fit failed: {exc}"}
    return {
        "available": True, "method": "platt", "train_n": len(train), "holdout_n": len(holdout),
        "params": {"a": round(scaler.a, 6), "b": round(scaler.b, 6)},
        "raw": {"brier": round(brier_score(hp, hy), 4), "log_loss": round(log_loss(hp, hy), 4)},
        "calibrated": {"brier": round(brier_score(calibrated, hy), 4), "log_loss": round(log_loss(calibrated, hy), 4)},
    }
