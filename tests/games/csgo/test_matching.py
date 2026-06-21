"""Tests for CS2 match↔market matching (confidence + reason codes)."""

from datetime import timedelta

from games.csgo.data.csgo_client import StubCsgoClient
from games.csgo.matching import MarketRef, best_market, score_match


def _navi_faze():
    client = StubCsgoClient()
    return next(m for m in client.get_matches()
                if m.team1_abbrev == "NAVI" and m.team2_abbrev == "FAZE")


def test_both_teams_event_and_time_strong_match():
    m = _navi_faze()
    mk = MarketRef(venue="kalshi", title="Will Natus Vincere beat FaZe Clan?",
                   event="IEM Katowice 2026", ticker="KXCS-NAVI-FAZE", start_time=m.date)
    res = score_match(m, mk)
    assert res.is_match
    assert res.confidence >= 0.8
    assert {"BOTH_TEAMS", "EVENT_MATCH", "TIME_WINDOW"} <= set(res.reasons)


def test_alias_handling_navi_apostrophe():
    m = _navi_faze()
    res = score_match(m, MarketRef(title="Na'Vi vs FaZe", start_time=m.date))
    assert res.is_match
    assert "BOTH_TEAMS" in res.reasons


def test_one_team_outright_is_rejected():
    m = _navi_faze()
    res = score_match(m, MarketRef(title="Natus Vincere to win IEM Katowice",
                                   event="IEM Katowice 2026"))
    assert not res.is_match
    assert res.reasons == ["ONE_TEAM_ONLY"]


def test_no_teams_no_match():
    m = _navi_faze()
    res = score_match(m, MarketRef(title="Team Vitality vs G2 Esports"))
    assert not res.is_match
    assert res.reasons == ["NO_TEAMS"]


def test_time_mismatch_penalizes():
    m = _navi_faze()
    far = m.date + timedelta(days=5)
    res = score_match(m, MarketRef(title="Natus Vincere vs FaZe Clan", start_time=far))
    assert "TIME_MISMATCH" in res.reasons
    assert not res.is_match  # 0.6 - 0.3 = 0.3 < threshold


def test_best_market_picks_highest_confidence():
    m = _navi_faze()
    markets = [
        MarketRef(title="Natus Vincere to win the event"),            # one-team → rejected
        MarketRef(title="Na'Vi vs FaZe", event="IEM Katowice 2026", start_time=m.date),  # strong
        MarketRef(title="Vitality vs G2"),                            # no teams
    ]
    best, res = best_market(m, markets)
    assert best is not None
    assert res.is_match and "BOTH_TEAMS" in res.reasons
