"""Tests for market-implied (no-vig) probability."""

import pytest

from common.ml.markets.implied import (
    implied_from_cents,
    implied_from_quote,
    line_move,
    no_vig_two_way,
)


def test_binary_one_sided_overround_equals_spread():
    mi = implied_from_quote(0.52, 0.56)
    assert mi.valid
    assert mi.mid == pytest.approx(0.54)
    assert mi.spread == pytest.approx(0.04)
    # implied_no derived as 1 - yes_bid = 0.48; fair = 0.56 / (0.56 + 0.48)
    assert mi.fair_yes == pytest.approx(0.56 / 1.04, rel=1e-6)
    assert mi.overround == pytest.approx(0.04)          # vig == spread for a one-sided binary
    assert mi.fair_yes + mi.fair_no == pytest.approx(1.0)


def test_two_way_independent_quotes():
    mi = implied_from_quote(0.52, 0.56, no_ask=0.50)
    assert mi.valid
    assert mi.implied_no == pytest.approx(0.50)
    assert mi.fair_yes == pytest.approx(0.56 / 1.06, rel=1e-6)
    assert mi.overround == pytest.approx(0.06)


def test_crossed_and_out_of_range_are_invalid():
    assert not implied_from_quote(0.60, 0.50).valid     # crossed book
    assert not implied_from_quote(None, 0.50).valid
    assert not implied_from_quote(0.50, 1.5).valid


def test_cents_helper_matches_prob_helper():
    a = implied_from_cents(52, 56)
    b = implied_from_quote(0.52, 0.56)
    assert a.fair_yes == pytest.approx(b.fair_yes)
    assert a.overround == pytest.approx(b.overround)


def test_no_vig_two_way_normalizes_to_one():
    q_yes, q_no = no_vig_two_way(0.55, 0.50)
    assert q_yes + q_no == pytest.approx(1.0)
    assert q_yes == pytest.approx(0.55 / 1.05, rel=1e-6)


def test_line_move_signed():
    assert line_move(0.50, 0.58) == pytest.approx(0.08)
    assert line_move(0.50, 0.42) == pytest.approx(-0.08)
