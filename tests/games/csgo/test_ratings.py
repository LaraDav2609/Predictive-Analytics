"""Tests for Glicko-2 CS2 ratings."""

from datetime import datetime, timedelta, timezone

import pytest

from games.csgo.analytics.ratings import (
    Glicko2,
    Rating,
    confidence_from_rd,
    rate_matches,
    win_probability,
)
from games.csgo.models.csgo import CsgoMatch


def _match(i: int, t1: int, t2: int, winner: int, day: int) -> CsgoMatch:
    return CsgoMatch(
        id=str(i), team1=f"T{t1}", team2=f"T{t2}", team1_id=t1, team2_id=t2,
        date=datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(days=day),
        winner_id=winner, best_of=3,
    )


def test_win_probability_is_symmetric():
    p = win_probability(1700, 50, 1500, 50)
    assert p > 0.5
    assert win_probability(1500, 50, 1700, 50) == pytest.approx(1.0 - p, abs=1e-9)


def test_confidence_rises_as_rd_falls():
    assert confidence_from_rd(50, 50) > confidence_from_rd(350, 350)
    assert confidence_from_rd(350, 350) == pytest.approx(0.5)


def test_a_win_raises_rating_and_reduces_uncertainty():
    g = Glicko2()
    after = g.update(Rating(), Rating(), score=1.0)
    assert after.rating > 1500.0
    assert after.rd < 350.0


def test_rate_matches_ranks_stronger_teams_higher():
    matches, day = [], 0
    for _ in range(6):
        matches.append(_match(len(matches), 1, 2, winner=1, day=day)); day += 1
        matches.append(_match(len(matches), 1, 3, winner=1, day=day)); day += 1
        matches.append(_match(len(matches), 2, 3, winner=2, day=day)); day += 1
    r = rate_matches(matches)
    assert r[1].rating > r[2].rating > r[3].rating
    assert r[1].rd < 350.0  # repeated games establish the rating
