"""Tests for the decomposable MLB game model + leak-free backtest."""
import datetime as dt

from sports.baseball.analytics.backtest import run_backtest
from sports.baseball.analytics.game_model import TeamState, log5, predict_game
from sports.baseball.models.baseball import MLBGame


def _game(gid, home_id, away_id, hs, as_, day):
    return MLBGame(
        id=gid, home_team=f"H{home_id}", away_team=f"A{away_id}",
        home_team_id=home_id, away_team_id=away_id,
        date=dt.datetime(2023, 4, 1) + dt.timedelta(days=day),
        status="FINAL", home_score=hs, away_score=as_,
    )


def test_pythag_and_log5():
    strong = TeamState(runs_for=500, runs_against=400)
    assert strong.pythag_wpct() > 0.5
    assert abs(log5(0.5, 0.5) - 0.5) < 1e-9
    assert log5(0.6, 0.4) > 0.5


def test_components_sum_to_probability():
    home = TeamState(runs_for=500, runs_against=400, wins=50, losses=30, last10=[1] * 7 + [0] * 3)
    away = TeamState(runs_for=400, runs_against=460, wins=35, losses=45, last10=[0] * 6 + [1] * 4)
    out = predict_game(home, away, min_games=10)
    total = out["baseline"] + sum(c["contribution"] for c in out["components"].values())
    assert abs(total - out["home_win_prob"]) < 1e-6      # additive waterfall reconstructs the prob
    assert out["home_win_prob"] > 0.5                    # stronger home team is favored
    assert out["components"]["home_field"]["contribution"] > 0
    assert out["leak_free"] is True


def test_pitcher_era_shifts_probability():
    home = TeamState(runs_for=400, runs_against=400, wins=40, losses=40)
    away = TeamState(runs_for=400, runs_against=400, wins=40, losses=40)   # even teams
    # home starter much better (lower ERA) -> favours home
    out = predict_game(home, away, home_pitcher_era=2.80, away_pitcher_era=5.20, min_games=10)
    sp = out["components"]["starting_pitcher"]
    assert sp["modeled"] is True
    assert sp["contribution"] > 0                      # better home starter helps home
    assert sp["home_prior_era"] == 2.80
    # symmetric: worse home starter hurts home
    out2 = predict_game(home, away, home_pitcher_era=5.20, away_pitcher_era=2.80, min_games=10)
    assert out2["components"]["starting_pitcher"]["contribution"] < 0


def test_pitcher_neutral_when_era_missing():
    out = predict_game(TeamState(wins=40, losses=40, runs_for=400, runs_against=400),
                       TeamState(wins=40, losses=40, runs_for=400, runs_against=400),
                       home_pitcher_era=3.0, away_pitcher_era=None, min_games=10)
    sp = out["components"]["starting_pitcher"]
    assert sp["modeled"] is False and sp["contribution"] == 0.0


def test_confidence_zero_without_history():
    out = predict_game(TeamState(), TeamState(), min_games=10)
    assert out["confidence"] == 0.0
    assert out["leak_free"] is False
    assert out["home_win_prob"] > 0.5                    # only the home-field edge, near 0.535


def test_team_state_record_game():
    s = TeamState()
    s.record_game(5, 3)
    s.record_game(2, 4)
    assert (s.wins, s.losses) == (1, 1)
    assert s.runs_for == 7 and s.runs_against == 7
    assert s.last10 == [1, 0]


def test_backtest_is_leak_free_and_scores():
    # Strong teams (1, 3) beat weak teams (2, 4) whether home or away; the running
    # state is built only from prior games, so early games (no history) are skipped.
    games, day = [], 0
    for i in range(40):
        games.append(_game(1000 + i, 1, 2, 6, 2, day)); day += 1   # strong home beats weak
        games.append(_game(2000 + i, 4, 3, 2, 6, day)); day += 1   # strong away (3) wins at weak home (4)
    res = run_backtest(games, min_games=5, calibrate=False)
    assert res["scored"] > 0
    assert 0.0 <= res["brier"] <= 1.0
    assert "reliability" in res and "favorite" in res
    assert res["rows"], "should surface per-game rows for the dashboard"


def test_backtest_empty_is_graceful():
    res = run_backtest([], min_games=10)
    assert res["scored"] == 0 and res.get("insufficient_data")
