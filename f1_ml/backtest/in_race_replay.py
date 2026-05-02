"""In-race replay backtester — for v2, simulates lap-by-lap live trading.

For a historical race, stream laps in order; at each lap N:
  - Update online_bayes / Kalman with newly visible lap.
  - Re-run simulator with current state (faster MC: 10k iters).
  - Map outcomes → market probabilities.
  - Compare to historical market quotes at that wall-clock; compute hypothetical
    P&L given Kelly-sized fills (using slippage model).

Validates that v2's lap-by-lap probability updates actually generate edge
beyond the static pre-race model.

The replay is generic about the lap-to-prediction function: callers inject a
`predict_fn(lap_number, history_so_far) -> dict[market, dict[driver, prob]]`
that wraps whatever inference path (Kalman + simulator, GBM-only, etc.).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class InRaceReplayConfig:
    race_id: str
    sim_iterations_per_update: int = 10_000
    update_every_lap: int = 1
    markets: tuple[str, ...] = ("winner", "podium", "h2h")
    bankroll: float = 1000.0
    kelly_fraction: float = 0.25  # quarter-Kelly default


@dataclass
class InRaceReplayResult:
    per_lap_predictions: pd.DataFrame  # lap × driver × market × prob
    simulated_pnl: float
    fills: pd.DataFrame                 # lap × market × side × size × fill_price


def run(
    config: InRaceReplayConfig,
    laps: pd.DataFrame,
    predict_fn: Callable[[int, pd.DataFrame], dict[str, dict[str, float]]],
    market_quote_history: pd.DataFrame | None = None,
    finish_order: list[str] | None = None,
) -> InRaceReplayResult:
    """Stream laps in order, call `predict_fn` at each update interval.

    `laps` columns expected: lap_number, driver_code, ... (rest opaque).
    `market_quote_history` (optional) columns: lap_number, market, side, price
        — used to compute hypothetical P&L. side in {"YES","NO"}, price in [0,1].
    `finish_order` (optional): realized finishing order; used to settle YES bets
        for evaluation when market_quote_history is provided.

    Returns the per-lap predictions, simulated P&L, and the list of theoretical
    fills.
    """
    if laps.empty:
        return InRaceReplayResult(
            per_lap_predictions=_empty_predictions_frame(),
            simulated_pnl=0.0,
            fills=_empty_fills_frame(),
        )

    if "lap_number" not in laps.columns:
        raise KeyError("laps DataFrame must have a 'lap_number' column")

    pred_rows: list[dict] = []
    fill_rows: list[dict] = []
    bankroll = float(config.bankroll)
    sorted_laps = sorted(laps["lap_number"].unique())

    for n in sorted_laps:
        if (n - sorted_laps[0]) % config.update_every_lap != 0:
            continue
        history = laps[laps["lap_number"] <= n].copy()
        predictions = predict_fn(int(n), history)
        for market, probs in predictions.items():
            for driver, p in probs.items():
                pred_rows.append({
                    "lap_number": int(n),
                    "market": market,
                    "driver_code": driver,
                    "probability": float(p),
                })

        if market_quote_history is not None and finish_order is not None:
            new_fills = _simulate_fills(
                lap=int(n),
                predictions=predictions,
                quotes=market_quote_history,
                finish_order=finish_order,
                bankroll=bankroll,
                kelly_fraction=config.kelly_fraction,
            )
            fill_rows.extend(new_fills)

    fills = pd.DataFrame(fill_rows) if fill_rows else _empty_fills_frame()
    pnl = float(fills["realized_pnl"].sum()) if "realized_pnl" in fills.columns else 0.0
    return InRaceReplayResult(
        per_lap_predictions=pd.DataFrame(pred_rows) if pred_rows else _empty_predictions_frame(),
        simulated_pnl=pnl,
        fills=fills,
    )


def _simulate_fills(
    lap: int,
    predictions: dict[str, dict[str, float]],
    quotes: pd.DataFrame,
    finish_order: list[str],
    bankroll: float,
    kelly_fraction: float,
) -> list[dict]:
    """Naive fill simulator: at each lap with available quotes, take YES on any
    market where model_prob > ask + 0.02 edge buffer; size at fractional Kelly.

    Realized PnL settles on the final outcome (winner or podium membership).
    """
    out: list[dict] = []
    lap_quotes = quotes[quotes["lap_number"] == lap]
    if lap_quotes.empty:
        return out

    winner = finish_order[0] if finish_order else None
    top3 = set(finish_order[:3]) if finish_order else set()

    for _, row in lap_quotes.iterrows():
        market = row.get("market")
        side = row.get("side", "YES")
        price = float(row.get("price", 0.5))
        if market not in predictions or side != "YES":
            continue
        # Pick the highest-probability driver in this market for the fill.
        per_driver = predictions[market]
        if not per_driver:
            continue
        driver, prob = max(per_driver.items(), key=lambda kv: kv[1])
        edge = prob - price
        if edge <= 0.02:
            continue
        # Fractional Kelly on a binary outcome at price `price`.
        b = (1.0 - price) / max(price, 1e-9)
        f_star = (b * prob - (1 - prob)) / b
        size = max(0.0, f_star) * kelly_fraction * bankroll

        market_l = market.lower()
        if market_l == "winner":
            won = (driver == winner)
        elif market_l == "podium":
            won = (driver in top3)
        else:
            continue

        payoff = (1.0 - price) * size if won else -price * size
        out.append({
            "lap_number": lap,
            "market": market,
            "driver_code": driver,
            "side": side,
            "fill_price": price,
            "size": size,
            "edge": edge,
            "won": bool(won),
            "realized_pnl": payoff,
        })
    return out


def _empty_predictions_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["lap_number", "market", "driver_code", "probability"])


def _empty_fills_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "lap_number", "market", "driver_code", "side", "fill_price",
        "size", "edge", "won", "realized_pnl",
    ])
