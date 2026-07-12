"""Tests for the schedule fatigue / rest feature: leak-free aggregation from the
schedule alone, the decomposable predict_game component, and backward compatibility
when no fatigue data is supplied."""
from datetime import datetime, timezone

from sports.baseball.analytics.game_model import TeamState, predict_game
from sports.baseball.analytics.schedule_fatigue import (
    FATIGUE_WINDOW_DAYS, MAX_REST_DAYS, fatigue_by_game,
)


class _G:
    def __init__(self, gid, home_id, away_id, y, m, d):
        self.id = gid
        self.home_team_id = home_id
        self.away_team_id = away_id
        self.date = datetime(y, m, d, 23, 0, tzinfo=timezone.utc)


def _state(rf, ra, wins, losses, last10=None):
    return TeamState(runs_for=rf, runs_against=ra, wins=wins, losses=losses,
                     last10=last10 or [])


# ─────────────────────────── leak-free aggregation ───────────────────────────
def test_first_game_has_no_rest_and_zero_density():
    games = [_G(1, 10, 20, 2023, 4, 1)]
    out = fatigue_by_game(games)
    assert out[1]["home_days_rest"] is None            # no prior game → unknown rest
    assert out[1]["away_days_rest"] is None
    assert out[1]["home_games_last_n"] == 0
    assert out[1]["away_games_last_n"] == 0


def test_days_rest_and_density_accumulate_leak_free():
    # Team 10 plays Apr 1, Apr 2, then Apr 5 (2 days later). Team 20 plays only Apr 1.
    games = [
        _G(1, 10, 20, 2023, 4, 1),
        _G(2, 10, 30, 2023, 4, 2),
        _G(3, 10, 40, 2023, 4, 5),
    ]
    out = fatigue_by_game(games)
    # Game 2: team 10 played yesterday → 1 day rest; one prior game in the window.
    assert out[2]["home_days_rest"] == 1
    assert out[2]["home_games_last_n"] == 1
    # Game 3: team 10 last played Apr 2 → 3 days rest; two prior games within 7 days.
    assert out[3]["home_days_rest"] == 3
    assert out[3]["home_games_last_n"] == 2


def test_days_rest_capped():
    # A huge gap is capped so it can't dominate the signal.
    games = [_G(1, 10, 20, 2023, 4, 1), _G(2, 10, 30, 2023, 6, 1)]
    out = fatigue_by_game(games)
    assert out[2]["home_days_rest"] == MAX_REST_DAYS


def test_density_window_excludes_old_games():
    # Games spaced > FATIGUE_WINDOW_DAYS apart never count toward each other's density.
    games = [_G(1, 10, 20, 2023, 4, 1),
             _G(2, 10, 30, 2023, 4, 1 + FATIGUE_WINDOW_DAYS + 3)]
    out = fatigue_by_game(games)
    assert out[2]["home_games_last_n"] == 0            # first game fell outside the window


def test_fatigue_by_game_skips_malformed_games():
    class _Bad:
        id = None
        home_team_id = 1
        away_team_id = 2
        date = None
    out = fatigue_by_game([_Bad()])
    assert out == {}                                   # no crash, just skipped


# ─────────────────────────── predict_game integration ───────────────────────────
def test_predict_game_unchanged_without_fatigue():
    home = _state(300, 250, 40, 30, [1, 1, 0, 1, 1, 0, 1, 1, 0, 1])
    away = _state(250, 300, 30, 40, [0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    pred = predict_game(home, away, min_games=10)
    fat = pred["components"]["schedule_fatigue"]
    assert fat["modeled"] is False
    assert fat["contribution"] == 0.0
    assert pred["features"]["fatigue_present"] == 0.0
    # Decomposition still reconciles with the new (neutral) component included.
    c = pred["components"]
    total = (pred["baseline"] + c["team_strength"]["contribution"]
             + c["home_field"]["contribution"] + c["recent_form"]["contribution"]
             + c["starting_pitcher"]["contribution"] + c["run_environment"]["contribution"]
             + c["schedule_fatigue"]["contribution"])
    assert abs(total - pred["home_win_prob"]) < 1e-9


def test_predict_game_fatigue_fresher_home_nudges_up():
    home = _state(300, 270, 40, 30)
    away = _state(270, 300, 30, 40)
    base = predict_game(home, away, min_games=10)
    # Home very rested (5 days) + light week; away tired (0 days rest, dense week).
    fresh = predict_game(home, away, min_games=10,
                         fatigue={"home_days_rest": 5, "away_days_rest": 0,
                                  "home_games_last_n": 1, "away_games_last_n": 6})
    fat = fresh["components"]["schedule_fatigue"]
    assert fat["modeled"] is True
    assert fresh["home_win_prob"] > base["home_win_prob"]        # fresher home → up
    assert fresh["features"]["rest_gap"] == 5.0                  # 5 - 0
    assert fresh["features"]["density_gap"] == 5.0               # away 6 - home 1
    assert fresh["features"]["fatigue_present"] == 1.0


def test_predict_game_fatigue_bounded_and_reconciles():
    home = _state(320, 240, 45, 25)
    away = _state(240, 320, 25, 45)
    base = predict_game(home, away, min_games=10)
    extreme = predict_game(home, away, min_games=10,
                           fatigue={"home_days_rest": 6, "away_days_rest": 0,
                                    "home_games_last_n": 0, "away_games_last_n": 10})
    # Contribution is capped — never a runaway edge.
    assert abs(extreme["components"]["schedule_fatigue"]["contribution"]) <= 0.015 + 1e-9
    assert abs(extreme["home_win_prob"] - base["home_win_prob"]) <= 0.02
    c = extreme["components"]
    total = (extreme["baseline"] + c["team_strength"]["contribution"]
             + c["home_field"]["contribution"] + c["recent_form"]["contribution"]
             + c["starting_pitcher"]["contribution"] + c["run_environment"]["contribution"]
             + c["schedule_fatigue"]["contribution"])
    assert abs(total - extreme["home_win_prob"]) < 1e-9


def test_predict_game_fatigue_partial_data_safe():
    # Only one side's rest known → rest_gap stays neutral (no crash, no half-signal).
    home = _state(300, 270, 40, 30)
    away = _state(270, 300, 30, 40)
    pred = predict_game(home, away, min_games=10,
                        fatigue={"home_days_rest": 4, "away_days_rest": None,
                                 "home_games_last_n": 2, "away_games_last_n": 5})
    assert pred["components"]["schedule_fatigue"]["modeled"] is True
    assert pred["features"]["rest_gap"] == 0.0          # incomplete rest → neutral
    assert pred["features"]["density_gap"] == 3.0       # density still usable (5 - 2)
