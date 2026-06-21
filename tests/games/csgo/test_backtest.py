"""Tests for the walk-forward CS2 backtest + dashboard contract."""

from games.csgo.analytics.backtest import replay_match, run_backtest
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
    for key in ("match_id", "team1", "team2", "winner", "team1_win_probability",
                "team1_score", "team2_score", "brier", "log_loss"):
        assert key in row


def test_replay_match_anchors_at_pregame_and_clinches():
    client = StubCsgoClient()
    past = client.get_past_matches()
    teams = {t.id: t for t in client.get_teams()}
    mid = run_backtest(past, teams_by_id=teams)["rows"][-1]["match_id"]

    r = replay_match(mid, past, teams_by_id=teams)
    assert r["ok"] and r["leak_free"]
    traj = r["trajectory"]
    # First point is the model's exact pre-game series call.
    assert traj[0]["label"] == "Pre-game"
    assert traj[0]["team1_series_prob"] == r["pregame_team1_prob"]
    # Probabilities stay in range; the series clinches at 1 or 0 by the last decided map.
    assert all(0.0 <= p["team1_series_prob"] <= 1.0 for p in traj)
    assert traj[-1]["team1_series_prob"] in (0.0, 1.0)
    # The clinch direction agrees with who actually won.
    assert (traj[-1]["team1_series_prob"] == 1.0) == r["winner_is_team1"]
    assert r["correct"] == ((r["pregame_team1_prob"] >= 0.5) == r["winner_is_team1"])


def test_replay_match_unknown_id_is_graceful():
    r = replay_match("nope", StubCsgoClient().get_past_matches())
    assert r["ok"] is False and r["reason"] == "match_not_found_or_unfinished"


def test_fit_calibrator_returns_scaler_on_history_and_none_on_too_little():
    from games.csgo.analytics.backtest import fit_calibrator
    past = StubCsgoClient().get_past_matches()
    assert fit_calibrator(past, min_history=10) is not None   # enough scored → usable scaler
    assert fit_calibrator(past[:5]) is None                   # too little → caller stays raw


def test_pipeline_applies_calibration_to_live_predictions():
    from games.csgo.analytics.pipeline import CsgoModelPipeline
    c = StubCsgoClient()
    p = CsgoModelPipeline()
    p.fit(c.get_past_matches(), c.get_teams())
    assert p.model.calibrator is not None
    scheduled = [m for m in c.get_matches() if m.status == "SCHEDULED"]
    assert scheduled, "stub should expose upcoming matches"
    pred = p.predict(scheduled[0])
    assert pred.feature_provenance.get("calibrated") == "yes"
    assert 0.0 <= pred.team1_win_prob <= 1.0


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


def test_backtest_calibration_block():
    res = run_backtest(StubCsgoClient().get_past_matches(), min_history=10, calibrate=True)
    cal = res.get("calibration")
    assert cal is not None
    # Robust to whether sklearn is installed in the env.
    if cal.get("available"):
        assert "raw" in cal and "calibrated" in cal
        assert cal["holdout_n"] > 0
        assert cal["raw"]["brier"] >= 0 and cal["calibrated"]["brier"] >= 0
    else:
        assert "reason" in cal
