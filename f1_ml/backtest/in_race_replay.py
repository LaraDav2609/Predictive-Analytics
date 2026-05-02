"""In-race replay backtester — for v2, simulates lap-by-lap live trading.

For a historical race, stream laps in order; at each lap N:
  - Update online_bayes / Kalman with newly visible lap.
  - Re-run simulator with current state (faster MC: 10k iters).
  - Map outcomes → market probabilities.
  - Compare to historical market quotes at that wall-clock; compute hypothetical
    P&L given Kelly-sized fills (using slippage model).

Validates that v2's lap-by-lap probability updates actually generate edge
beyond the static pre-race model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class InRaceReplayConfig:
    race_id: str
    sim_iterations_per_update: int = 10_000
    update_every_lap: int = 1
    markets: tuple[str, ...] = ("winner", "podium", "h2h")


@dataclass
class InRaceReplayResult:
    per_lap_predictions: pd.DataFrame  # lap × driver × market × prob
    simulated_pnl: float
    fills: pd.DataFrame                 # lap × market × side × size × fill_price


def run(config: InRaceReplayConfig, provider, market_quote_history: pd.DataFrame) -> InRaceReplayResult:
    raise NotImplementedError
