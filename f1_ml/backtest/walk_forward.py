"""Walk-forward backtester — train on races up to date T, predict race T+1.

For each race in the test season:
  1. Refit all models using only data from races strictly before this one.
  2. Run the simulator with pre-race-only inputs.
  3. Map outcomes → market probabilities.
  4. Compare to realized result; log Brier / log-loss per market.
  5. If historical market quotes are available, compute simulated P&L.

Aggregates: per-market calibration plots, cumulative P&L, drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class WalkForwardConfig:
    test_season: int
    training_seasons: tuple[int, ...]
    min_training_races: int = 20
    refit_every_n_races: int = 1  # refit every race; relax to N for speed
    markets: tuple[str, ...] = ("winner", "podium", "h2h", "fastest_lap")


@dataclass
class WalkForwardResult:
    per_race_metrics: pd.DataFrame  # race_id × market × {brier, log_loss, pnl}
    aggregate_metrics: dict[str, float] = field(default_factory=dict)


def run(config: WalkForwardConfig, provider, registry_names: dict[str, str]) -> WalkForwardResult:
    raise NotImplementedError(
        "loop races chronologically; refit; predict; score; aggregate"
    )
