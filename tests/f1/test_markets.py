"""Tests for sports.f1.ml.markets — mapper, edge, slippage, kelly.

Most kelly + brier coverage already lives in test_smoke.py; this file fills in
the rest of the markets surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from common.ml.markets.edge import EdgeOpportunity, MarketQuote, compute_edge
from common.ml.markets.slippage import FillEstimate, OrderbookLevel, kalshi_limit_fill_prob, walk_book


# ----------------------------------------------------------------- edge.compute_edge

def test_compute_edge_returns_none_when_below_threshold():
    quote = MarketQuote(venue="kalshi", market_id="X", best_bid=0.40, best_ask=0.42, fee_bps=200)
    # model_prob = 0.43 → YES edge = (0.43 - 0.42) * 10_000 - 200 = 100 - 200 = -100 bps
    opp = compute_edge(model_prob=0.43, quote=quote, min_edge_bps=200)
    assert opp is None


def test_compute_edge_returns_yes_when_model_above_ask():
    quote = MarketQuote(venue="kalshi", market_id="X", best_bid=0.40, best_ask=0.42, fee_bps=200)
    # model_prob = 0.55 → YES edge = (0.55 - 0.42) * 10_000 - 200 = 1300 - 200 = 1100 bps
    opp = compute_edge(model_prob=0.55, quote=quote, min_edge_bps=200)
    assert opp is not None
    assert opp.direction == "YES"
    assert opp.edge_bps == pytest.approx(1100, abs=1)
    assert opp.market_implied_prob == 0.42


def test_compute_edge_returns_no_when_model_below_bid():
    quote = MarketQuote(venue="kalshi", market_id="X", best_bid=0.40, best_ask=0.42, fee_bps=200)
    # model_prob = 0.20 → NO edge = (0.40 - 0.20) * 10_000 - 200 = 2000 - 200 = 1800 bps
    opp = compute_edge(model_prob=0.20, quote=quote, min_edge_bps=200)
    assert opp is not None
    assert opp.direction == "NO"
    assert opp.edge_bps == pytest.approx(1800, abs=1)
    assert opp.market_implied_prob == 0.40


def test_compute_edge_picks_better_side():
    """When both sides have positive edge (would only happen with crossed book or
    unusual quote), pick the larger one."""
    # Asymmetric: huge YES edge.
    quote = MarketQuote(venue="kalshi", market_id="X", best_bid=0.30, best_ask=0.35, fee_bps=100)
    opp = compute_edge(model_prob=0.80, quote=quote, min_edge_bps=200)
    assert opp is not None
    assert opp.direction == "YES"


def test_compute_edge_rejects_crossed_book():
    quote = MarketQuote(venue="kalshi", market_id="X", best_bid=0.50, best_ask=0.45, fee_bps=200)
    opp = compute_edge(model_prob=0.60, quote=quote)
    assert opp is None


def test_compute_edge_rejects_invalid_model_prob():
    quote = MarketQuote(venue="kalshi", market_id="X", best_bid=0.40, best_ask=0.42, fee_bps=200)
    assert compute_edge(model_prob=0.0, quote=quote) is None
    assert compute_edge(model_prob=1.0, quote=quote) is None
    assert compute_edge(model_prob=-0.1, quote=quote) is None
    assert compute_edge(model_prob=1.5, quote=quote) is None


def test_compute_edge_subtracts_fee():
    quote_lo = MarketQuote(venue="kalshi", market_id="X", best_bid=0.40, best_ask=0.42, fee_bps=50)
    quote_hi = MarketQuote(venue="kalshi", market_id="X", best_bid=0.40, best_ask=0.42, fee_bps=400)
    opp_lo = compute_edge(model_prob=0.55, quote=quote_lo, min_edge_bps=0)
    opp_hi = compute_edge(model_prob=0.55, quote=quote_hi, min_edge_bps=0)
    assert opp_lo is not None and opp_hi is not None
    # Higher fee → lower edge.
    assert opp_lo.edge_bps - opp_hi.edge_bps == pytest.approx(350, abs=1)


# -------------------------------------------------------------- slippage.walk_book

def test_walk_book_empty_target():
    levels = [OrderbookLevel(price=0.42, size_units=100)]
    result = walk_book(levels, target_units=0)
    assert result.fill_units == 0.0


def test_walk_book_empty_book():
    result = walk_book([], target_units=10)
    assert result.fill_units == 0.0


def test_walk_book_consumes_top_level_when_fits():
    levels = [
        OrderbookLevel(price=0.42, size_units=100),
        OrderbookLevel(price=0.43, size_units=200),
    ]
    result = walk_book(levels, target_units=50, side="BUY_YES")
    assert result.fill_units == 50
    assert result.avg_price == pytest.approx(0.42)
    assert result.slippage_bps == pytest.approx(0.0, abs=1)


def test_walk_book_walks_through_levels():
    levels = [
        OrderbookLevel(price=0.42, size_units=10),
        OrderbookLevel(price=0.45, size_units=20),
        OrderbookLevel(price=0.50, size_units=100),
    ]
    # Need 25 → 10 @ 0.42 + 15 @ 0.45 = 4.20 + 6.75 = 10.95 → avg = 10.95/25 = 0.438
    result = walk_book(levels, target_units=25, side="BUY_YES")
    assert result.fill_units == 25
    assert result.avg_price == pytest.approx(10.95 / 25, abs=1e-6)
    # Slippage = (0.438 - 0.42) / 0.42 = ~0.0429 = 428.6 bps
    assert result.slippage_bps == pytest.approx(428.6, abs=2)


def test_walk_book_partial_fill_when_book_too_thin():
    levels = [OrderbookLevel(price=0.42, size_units=10)]
    result = walk_book(levels, target_units=100)
    assert result.fill_units == 10
    assert result.avg_price == pytest.approx(0.42)


def test_walk_book_sell_slippage_sign():
    """For SELL, walking down the bids should produce positive slippage too."""
    bids = [
        OrderbookLevel(price=0.50, size_units=10),
        OrderbookLevel(price=0.48, size_units=20),
    ]
    # Sell 25 → 10 @ 0.50 + 15 @ 0.48 = 5.00 + 7.20 = 12.20 → avg = 0.488
    # Slippage for SELL = (top_price - avg) / top_price = (0.50-0.488)/0.50 = 0.024 = 240 bps
    result = walk_book(bids, target_units=25, side="SELL_YES")
    assert result.fill_units == 25
    assert result.slippage_bps == pytest.approx(240, abs=2)


# ----------------------------------------------------- slippage.kalshi_limit_fill_prob

def test_kalshi_limit_fill_prob_aggressive_at_or_through_mid():
    # Negative distance = posting more aggressively than mid → near-certain fill.
    assert kalshi_limit_fill_prob(distance_from_mid_bps=-50, recent_volatility=200) >= 0.95
    assert kalshi_limit_fill_prob(distance_from_mid_bps=0, recent_volatility=200) >= 0.95


def test_kalshi_limit_fill_prob_decays_with_distance():
    p_close = kalshi_limit_fill_prob(distance_from_mid_bps=50, recent_volatility=200)
    p_mid = kalshi_limit_fill_prob(distance_from_mid_bps=200, recent_volatility=200)
    p_far = kalshi_limit_fill_prob(distance_from_mid_bps=600, recent_volatility=200)
    assert p_close > p_mid > p_far
    assert 0 <= p_far <= 0.05  # 3-sigma+ should be very small


def test_kalshi_limit_fill_prob_handles_zero_volatility():
    assert kalshi_limit_fill_prob(distance_from_mid_bps=0, recent_volatility=0) == 1.0
    assert kalshi_limit_fill_prob(distance_from_mid_bps=10, recent_volatility=0) == 0.0


# ----------------------------------------------------- mapper extra cases

def test_mapper_top_k_probabilities():
    """top_k subsumes podium when k=3."""
    from sports.f1.ml.markets.mapper import podium_probabilities, top_k_probabilities
    from sports.f1.ml.providers.synthetic_provider import SyntheticDriver, SyntheticProvider, SyntheticRaceConfig
    from sports.f1.ml.simulator.race_sim import SimConfig, simulate_race

    cfg = SyntheticRaceConfig(
        n_laps=10,
        drivers=[SyntheticDriver(code=f"D{i}", team_code="T", true_pace_s=80.0 + i * 0.1) for i in range(5)],
    )
    provider = SyntheticProvider(cfg)
    drivers = list(provider.driver_pace_table().keys())
    state = {
        "driver_codes": drivers,
        "driver_mean_pace_s": [80.0 + i * 0.1 for i in range(5)],
        "driver_pace_sigma_s": [0.2] * 5,
        "driver_dnf_rate_per_lap": [0.0] * 5,
        "total_laps": 10,
    }
    result = simulate_race(provider.list_races(2026)[0], state, {}, SimConfig(n_iterations=500, seed=0))

    podium = podium_probabilities(result)
    top3 = top_k_probabilities(result, k=3)
    # Podium = top-3.
    for d in drivers:
        assert podium[d] == pytest.approx(top3[d], abs=1e-9)


def test_mapper_h2h_consistency():
    """P(A beats B) + P(B beats A) ≤ 1 (ties are possible only with DNFs)."""
    from sports.f1.ml.markets.mapper import h2h_probability
    from sports.f1.ml.providers.synthetic_provider import SyntheticDriver, SyntheticProvider, SyntheticRaceConfig
    from sports.f1.ml.simulator.race_sim import SimConfig, simulate_race

    cfg = SyntheticRaceConfig(
        n_laps=10,
        drivers=[SyntheticDriver(code="A", team_code="T", true_pace_s=80.0),
                 SyntheticDriver(code="B", team_code="T", true_pace_s=80.5)],
    )
    provider = SyntheticProvider(cfg)
    state = {
        "driver_codes": ["A", "B"],
        "driver_mean_pace_s": [80.0, 80.5],
        "driver_pace_sigma_s": [0.1, 0.1],
        "driver_dnf_rate_per_lap": [0.0, 0.0],
        "total_laps": 10,
    }
    result = simulate_race(provider.list_races(2026)[0], state, {}, SimConfig(n_iterations=500, seed=0))
    p_ab = h2h_probability(result, "A", "B")
    p_ba = h2h_probability(result, "B", "A")
    # No DNFs / ties → exactly 1.
    assert p_ab + p_ba == pytest.approx(1.0, abs=1e-9)
    # A is faster → wins more.
    assert p_ab > 0.95
