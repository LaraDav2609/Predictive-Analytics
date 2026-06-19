"""Tests for the walk-forward CS2 backtest + dashboard contract."""

from games.csgo.analytics.backtest import run_backtest
from games.csgo.data.csgo_client import StubCsgoClient


def test_backtest_on_stub_history_produces_calibration_metrics():
    past = StubCsgoClient().get_past_matches()
    assert len(past) > 50
    res = run_backtest(past, min_history=20)
    assert res["matches"] > 0
    assert res["scored"] == res["matches"]
    assert 0.0 < res["brier"] < 0.35          # better than a coin flip
    assert "log_loss" in res and "reliability" in res and "by_best_of" in res
    assert res["matches"] < len(past)          # leak-free: early matches go unscored


def test_backtest_rows_match_dashboard_tab_contract():
    res = run_backtest(StubCsgoClient().get_past_matches(), min_history=20)
    assert isinstance(res["matches"], int)
    assert isinstance(res["rows"], list) and res["rows"]
    row = res["rows"][0]
    for key in ("team1", "team2", "winner", "team1_win_probability",
                "team1_score", "team2_score", "brier", "log_loss"):
        assert key in row


def test_backtest_betting_sim_when_prices_supplied():
    past = StubCsgoClient().get_past_matches()
    res = run_backtest(past, min_history=20, market_prob_for=lambda m: 0.5, min_edge_bps=100)
    assert "betting" in res
    bet = res["betting"]
    assert bet["bets"] > 0
    assert "roi_pct" in bet and "win_rate" in bet


def test_backtest_insufficient_data_is_graceful():
    res = run_backtest([])
    assert res.get("scored") == 0
    assert res.get("insufficient_data") is True
    assert "matches" not in res                # falsy → tab shows the friendly message
