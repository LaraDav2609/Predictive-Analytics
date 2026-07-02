"""Tests for within-season pitcher form (FIP, leak-free, shrunk)."""
from sports.baseball.analytics.pitcher_form import (
    FIP_CONST, LEAGUE_FIP, REG_IP, parse_ip, rating_asof, starter_ids,
)


def test_parse_ip_baseball_notation():
    assert parse_ip("5.0") == 5.0
    assert abs(parse_ip("5.2") - (5 + 2 / 3)) < 1e-9   # 5 and 2/3 innings
    assert parse_ip(None) == 0.0


def test_rating_uses_only_starts_before_the_date():
    starts = [
        {"date": "2023-04-01", "ip": 6.0, "hr": 0, "bb": 1, "hbp": 0, "k": 8},   # great
        {"date": "2023-04-08", "ip": 6.0, "hr": 0, "bb": 1, "hbp": 0, "k": 8},
        {"date": "2023-04-15", "ip": 2.0, "hr": 3, "bb": 4, "hbp": 0, "k": 0},   # disaster (future)
    ]
    # As of 2023-04-15 only the two good starts count -> strong (low) rating.
    good = rating_asof(starts, "2023-04-15", prior=4.50)
    # As of season end all three count -> worse (higher) rating.
    withbad = rating_asof(starts, "2023-12-01", prior=4.50)
    assert good < withbad                               # the future disaster is not leaked in


def test_rating_returns_anchor_without_history():
    assert rating_asof([], "2023-04-01", prior=3.20) == 3.20      # no starts -> prior
    assert rating_asof(None, "2023-04-01", prior=None) == LEAGUE_FIP   # no prior -> league


def test_rating_regresses_toward_prior():
    one_elite_start = [{"date": "2023-04-01", "ip": 7.0, "hr": 0, "bb": 0, "hbp": 0, "k": 12}]
    r = rating_asof(one_elite_start, "2023-05-01", prior=5.00)
    # 7 IP is small vs REG_IP=45, so the rating stays much closer to the 5.00 prior
    # than to the elite single-start FIP.
    assert r > 4.0


def test_starter_ids_collects_probable_pitchers():
    class G:
        def __init__(self, h, a):
            self.home_pitcher_id, self.away_pitcher_id = h, a
    ids = starter_ids([G(1, 2), G(2, 3), G(None, 4)])
    assert ids == {1, 2, 3, 4}
