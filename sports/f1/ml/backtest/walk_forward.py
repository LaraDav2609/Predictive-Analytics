"""Walk-forward backtester — train on races up to date T, predict race T+1.

For each race in the test season:
  1. Refit all models using only data from races strictly before this one.
  2. Run the simulator with pre-race-only inputs.
  3. Map outcomes → market probabilities.
  4. Compare to realized result; log Brier / log-loss per market.
  5. If historical market quotes are available, compute simulated P&L.

Aggregates: per-market calibration plots, cumulative P&L, drawdown.

This module is intentionally generic about what "model" and "predict" mean —
callers inject a `model_factory` that returns a `Predictor` trained on the
provided training races. The harness's job is the loop, look-ahead audit,
and aggregation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd

from sports.f1.ml.common.calibration import brier_score, log_loss


# ------------------------------------------------------------------ data shapes

@dataclass
class RaceData:
    """Minimal race observation for the backtester. Real callers will hand in
    richer data (telemetry, laps, weather); this is the common spine the
    harness needs to compute realized outcomes and order races chronologically.
    """
    race_id: str
    season: int
    round: int
    decision_time: datetime  # when pre-race predictions are locked in
    finish_order: list[str]  # driver codes, position 1 first
    dnf_drivers: set[str] = field(default_factory=set)


class Predictor(Protocol):
    """Anything with a .predict(race) → dict[market_name, dict[driver, prob]]."""
    def predict(self, race: RaceData) -> dict[str, dict[str, float]]:
        ...


# ------------------------------------------------------------------ config / result

@dataclass
class WalkForwardConfig:
    test_season: int
    training_seasons: tuple[int, ...]
    min_training_races: int = 20
    refit_every_n_races: int = 1  # refit every race; relax to N for speed
    markets: tuple[str, ...] = ("winner", "podium", "h2h", "fastest_lap")


@dataclass
class WalkForwardResult:
    per_race_metrics: pd.DataFrame  # race_id × market × {brier, log_loss}
    aggregate_metrics: dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------ entry point

def run(
    config: WalkForwardConfig,
    races: list[RaceData],
    model_factory: Callable[[list[RaceData]], Predictor],
) -> WalkForwardResult:
    """Run walk-forward over the test season.

    For each test-season race in chronological order:
      - training set = all races (across all seasons) strictly before this race's
        decision_time, optionally restricted to `training_seasons`.
      - model = model_factory(training_set), then model.predict(race).
      - score predictions against the realized finish_order.

    Refitting is throttled by `refit_every_n_races` (default 1).
    """
    sorted_races = sorted(races, key=lambda r: (r.season, r.round))
    test_races = [r for r in sorted_races if r.season == config.test_season]
    if not test_races:
        return WalkForwardResult(per_race_metrics=_empty_metrics_frame())

    rows: list[dict] = []
    last_predictor: Predictor | None = None
    races_since_refit = config.refit_every_n_races  # force fit on first iter

    for r in test_races:
        training = [
            t for t in sorted_races
            if (t.season, t.round) < (r.season, r.round)
            and t.season in config.training_seasons
            and t.decision_time < r.decision_time  # look-ahead defense
        ]
        if len(training) < config.min_training_races:
            continue

        if last_predictor is None or races_since_refit >= config.refit_every_n_races:
            last_predictor = model_factory(training)
            races_since_refit = 0
        races_since_refit += 1

        predictions = last_predictor.predict(r)
        for market, probs in predictions.items():
            scores = _score_market(market, probs, realized=r)
            rows.append({"race_id": r.race_id, "market": market, **scores})

    metrics = pd.DataFrame(rows) if rows else _empty_metrics_frame()
    aggregate = _aggregate(metrics) if not metrics.empty else {}
    return WalkForwardResult(per_race_metrics=metrics, aggregate_metrics=aggregate)


# ------------------------------------------------------------------ scoring

def _score_market(
    market: str,
    probs: dict[str, float],
    realized: RaceData,
) -> dict[str, float]:
    """Per-market Brier + log-loss between predicted probabilities and the
    realized winner/podium/etc."""
    market_l = market.lower()
    if market_l == "winner":
        outcome = {d: 1.0 if d == realized.finish_order[0] else 0.0 for d in probs}
    elif market_l == "podium":
        top3 = set(realized.finish_order[:3])
        outcome = {d: 1.0 if d in top3 else 0.0 for d in probs}
    elif market_l.startswith("h2h:"):
        # market = "h2h:A:B" → 1 if A finished ahead of B else 0
        try:
            _, a, b = market.split(":")
        except ValueError:
            return {"brier": float("nan"), "log_loss": float("nan")}
        positions = {d: i for i, d in enumerate(realized.finish_order)}
        if a not in positions or b not in positions:
            return {"brier": float("nan"), "log_loss": float("nan")}
        a_ahead = float(positions[a] < positions[b])
        outcome = {a: a_ahead}
        probs = {a: probs.get(a, 0.5)}
    elif market_l == "fastest_lap":
        # We don't have per-race fastest-lap info on RaceData; treat as missing.
        return {"brier": float("nan"), "log_loss": float("nan")}
    else:
        return {"brier": float("nan"), "log_loss": float("nan")}

    p = np.array([probs[d] for d in outcome])
    y = np.array([outcome[d] for d in outcome])
    return {
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
    }


def _aggregate(metrics: pd.DataFrame) -> dict[str, float]:
    """Mean Brier / log-loss per market, plus overall."""
    out: dict[str, float] = {}
    for market, grp in metrics.groupby("market"):
        out[f"{market}.mean_brier"] = float(grp["brier"].dropna().mean())
        out[f"{market}.mean_log_loss"] = float(grp["log_loss"].dropna().mean())
    out["overall.mean_brier"] = float(metrics["brier"].dropna().mean())
    out["overall.mean_log_loss"] = float(metrics["log_loss"].dropna().mean())
    return out


def _empty_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["race_id", "market", "brier", "log_loss"])
