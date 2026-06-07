"""Tests for sports.f1.ml.backtest — walk_forward + in_race_replay."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sports.f1.ml.backtest.in_race_replay import InRaceReplayConfig, run as run_replay
from sports.f1.ml.backtest.walk_forward import (
    RaceData,
    WalkForwardConfig,
    run as run_walk_forward,
)


# --------------------------------------------------------------- walk_forward

def _make_races(seasons: list[int], rounds_per_season: int = 5) -> list[RaceData]:
    out = []
    for s in seasons:
        for r in range(1, rounds_per_season + 1):
            out.append(RaceData(
                race_id=f"{s}_{r}",
                season=s,
                round=r,
                decision_time=datetime(s, r, 1, 12, 0, 0),
                finish_order=["VER", "HAM", "LEC", "NOR", "PIA"],
                dnf_drivers=set(),
            ))
    return out


class _ConstantPredictor:
    """Always picks VER as the winner, equal mass elsewhere."""
    def predict(self, race: RaceData) -> dict[str, dict[str, float]]:
        drivers = race.finish_order
        return {
            "winner": {d: (0.6 if d == "VER" else 0.4 / (len(drivers) - 1)) for d in drivers},
            "podium": {d: 0.6 if d in {"VER", "HAM", "LEC"} else 0.1 for d in drivers},
        }


def test_walk_forward_runs_through_test_season():
    races = _make_races([2022, 2023, 2024], rounds_per_season=5)
    config = WalkForwardConfig(
        test_season=2024,
        training_seasons=(2022, 2023, 2024),
        min_training_races=5,
    )
    result = run_walk_forward(config, races, model_factory=lambda _: _ConstantPredictor())
    assert not result.per_race_metrics.empty
    # Winner market is scored every test race.
    n_winner = (result.per_race_metrics["market"] == "winner").sum()
    assert n_winner == 5


def test_walk_forward_aggregate_metrics_populated():
    races = _make_races([2022, 2023, 2024], rounds_per_season=5)
    config = WalkForwardConfig(
        test_season=2024,
        training_seasons=(2022, 2023, 2024),
        min_training_races=5,
    )
    result = run_walk_forward(config, races, model_factory=lambda _: _ConstantPredictor())
    assert "winner.mean_brier" in result.aggregate_metrics
    assert "podium.mean_brier" in result.aggregate_metrics
    assert "overall.mean_brier" in result.aggregate_metrics


def test_walk_forward_skips_when_training_too_small():
    """Only one race available (in 2024); min_training_races=20 ⇒ no metrics."""
    races = _make_races([2024], rounds_per_season=2)
    config = WalkForwardConfig(
        test_season=2024,
        training_seasons=(2024,),
        min_training_races=20,
    )
    result = run_walk_forward(config, races, model_factory=lambda _: _ConstantPredictor())
    assert result.per_race_metrics.empty


def test_walk_forward_no_test_season_returns_empty():
    races = _make_races([2022, 2023], rounds_per_season=5)
    config = WalkForwardConfig(
        test_season=2024,
        training_seasons=(2022, 2023),
        min_training_races=5,
    )
    result = run_walk_forward(config, races, model_factory=lambda _: _ConstantPredictor())
    assert result.per_race_metrics.empty


def test_walk_forward_training_excludes_future_races():
    """Track which training sets the model_factory sees and assert chronology."""
    races = _make_races([2023, 2024], rounds_per_season=4)
    seen_train_sizes: list[int] = []

    def factory(training: list[RaceData]) -> _ConstantPredictor:
        seen_train_sizes.append(len(training))
        return _ConstantPredictor()

    config = WalkForwardConfig(
        test_season=2024,
        training_seasons=(2023, 2024),
        min_training_races=4,
    )
    run_walk_forward(config, races, model_factory=factory)
    # 1st test race (2024 round 1): training = 4 (2023 entire season)
    # 2nd: 5; 3rd: 6; 4th: 7
    assert seen_train_sizes == [4, 5, 6, 7]


def test_walk_forward_h2h_market_scored():
    races = _make_races([2023, 2024], rounds_per_season=5)

    class _H2HPredictor:
        def predict(self, race: RaceData) -> dict[str, dict[str, float]]:
            return {"h2h:VER:HAM": {"VER": 0.7}}

    config = WalkForwardConfig(
        test_season=2024,
        training_seasons=(2023, 2024),
        min_training_races=4,
    )
    result = run_walk_forward(config, races, model_factory=lambda _: _H2HPredictor())
    h2h_rows = result.per_race_metrics[result.per_race_metrics["market"] == "h2h:VER:HAM"]
    assert len(h2h_rows) == 5
    # VER beats HAM in every synthetic race ⇒ Brier ≈ (0.7 - 1)² = 0.09
    assert h2h_rows["brier"].mean() == pytest_approx(0.09, abs=1e-9)


def test_walk_forward_brier_zero_for_perfect_prediction():
    races = _make_races([2023, 2024], rounds_per_season=5)

    class _Oracle:
        def predict(self, race: RaceData) -> dict[str, dict[str, float]]:
            winner = race.finish_order[0]
            return {"winner": {d: 1.0 if d == winner else 0.0 for d in race.finish_order}}

    config = WalkForwardConfig(
        test_season=2024,
        training_seasons=(2023, 2024),
        min_training_races=4,
    )
    result = run_walk_forward(config, races, model_factory=lambda _: _Oracle())
    assert result.aggregate_metrics["winner.mean_brier"] == pytest_approx(0.0, abs=1e-9)


# --------------------------------------------------------- in_race_replay

def _make_lap_data(n_laps: int = 10, drivers=("VER", "HAM", "LEC")) -> pd.DataFrame:
    rows = []
    for lap in range(1, n_laps + 1):
        for d in drivers:
            rows.append({"lap_number": lap, "driver_code": d, "lap_time_s": 80.0})
    return pd.DataFrame(rows)


def test_in_race_replay_emits_one_prediction_per_update_lap():
    laps = _make_lap_data(n_laps=10)

    def predict_fn(lap: int, history: pd.DataFrame) -> dict[str, dict[str, float]]:
        return {"winner": {"VER": 0.5, "HAM": 0.3, "LEC": 0.2}}

    config = InRaceReplayConfig(race_id="test", update_every_lap=1)
    result = run_replay(config, laps, predict_fn)
    # 10 laps × 3 drivers × 1 market = 30 prediction rows.
    assert len(result.per_lap_predictions) == 30


def test_in_race_replay_update_every_n_laps_throttles():
    laps = _make_lap_data(n_laps=12)
    calls: list[int] = []

    def predict_fn(lap: int, history: pd.DataFrame) -> dict[str, dict[str, float]]:
        calls.append(lap)
        return {"winner": {"VER": 1.0}}

    config = InRaceReplayConfig(race_id="test", update_every_lap=3)
    run_replay(config, laps, predict_fn)
    # First lap = 1 (offset 0); then 4, 7, 10.
    assert calls == [1, 4, 7, 10]


def test_in_race_replay_handles_empty_laps():
    result = run_replay(
        InRaceReplayConfig(race_id="test"),
        pd.DataFrame(columns=["lap_number", "driver_code"]),
        predict_fn=lambda lap, hist: {},
    )
    assert result.per_lap_predictions.empty
    assert result.simulated_pnl == 0.0


def test_in_race_replay_pnl_positive_when_model_beats_market():
    """Model says VER is 0.9 to win; market is 0.3 (cheap); VER actually wins ⇒ +PnL."""
    laps = _make_lap_data(n_laps=2)

    def predict_fn(lap: int, history: pd.DataFrame) -> dict[str, dict[str, float]]:
        return {"winner": {"VER": 0.9, "HAM": 0.05, "LEC": 0.05}}

    quotes = pd.DataFrame([
        {"lap_number": 1, "market": "winner", "side": "YES", "price": 0.3},
        {"lap_number": 2, "market": "winner", "side": "YES", "price": 0.3},
    ])
    config = InRaceReplayConfig(race_id="test", bankroll=1000.0, kelly_fraction=0.25)
    result = run_replay(config, laps, predict_fn, market_quote_history=quotes,
                        finish_order=["VER", "HAM", "LEC"])
    assert result.simulated_pnl > 0.0


def test_in_race_replay_pnl_negative_when_model_wrong():
    """Model says VER is 0.9; price 0.3 (cheap); VER loses ⇒ -PnL."""
    laps = _make_lap_data(n_laps=2)

    def predict_fn(lap: int, history: pd.DataFrame) -> dict[str, dict[str, float]]:
        return {"winner": {"VER": 0.9, "HAM": 0.05, "LEC": 0.05}}

    quotes = pd.DataFrame([
        {"lap_number": 1, "market": "winner", "side": "YES", "price": 0.3},
    ])
    config = InRaceReplayConfig(race_id="test", bankroll=1000.0, kelly_fraction=0.25)
    result = run_replay(config, laps, predict_fn, market_quote_history=quotes,
                        finish_order=["HAM", "VER", "LEC"])  # HAM wins, not VER
    assert result.simulated_pnl < 0.0


def test_in_race_replay_ignores_no_edge_quotes():
    """If the best model prob is within the edge buffer of the market price,
    the replay should not fire any orders."""
    laps = _make_lap_data(n_laps=2)

    def predict_fn(lap: int, history: pd.DataFrame) -> dict[str, dict[str, float]]:
        # Top model prob = 0.31; market price 0.30 ⇒ edge 0.01 < buffer 0.02
        return {"winner": {"VER": 0.31, "HAM": 0.30, "LEC": 0.29}}

    quotes = pd.DataFrame([
        {"lap_number": 1, "market": "winner", "side": "YES", "price": 0.30},
    ])
    result = run_replay(
        InRaceReplayConfig(race_id="test"),
        laps, predict_fn,
        market_quote_history=quotes,
        finish_order=["VER", "HAM", "LEC"],
    )
    assert result.fills.empty


# Reuse pytest.approx without importing pytest at the top of every helper
def pytest_approx(value, abs=None, rel=None):
    import pytest
    return pytest.approx(value, abs=abs, rel=rel)
