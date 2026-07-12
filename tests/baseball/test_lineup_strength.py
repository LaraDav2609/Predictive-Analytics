"""Tests for the confirmed-lineups feature: mean prior-season wOBA of the 9 starters
with league-average fallback and empty-order handling, the leak-free / null-safe
per-game pass, the boxscore battingOrder parse + cache-refresh-on-old-shape, and the
decomposable predict_game component (backward-compat when no lineup is supplied)."""
import json
import os
from datetime import datetime, timezone

from sports.baseball.analytics.lineup_strength import (
    LEAGUE_AVG_WOBA, lineup_by_game, lineup_quality,
)
from sports.baseball.analytics.game_model import TeamState, predict_game
from sports.baseball.data import boxscore_client
from sports.baseball.data.boxscore_client import parse_boxscore


class _G:
    def __init__(self, gid, home_id, away_id, y, m, d):
        self.id = gid
        self.home_team_id = home_id
        self.away_team_id = away_id
        self.date = datetime(y, m, d, 23, 0, tzinfo=timezone.utc)


def _state(rf, ra, wins, losses, last10=None):
    return TeamState(runs_for=rf, runs_against=ra, wins=wins, losses=losses,
                     last10=last10 or [])


# ─────────────────────────── lineup_quality ───────────────────────────
def test_lineup_quality_mean_of_known_bats():
    woba = {1: 0.30, 2: 0.40, 3: 0.35}
    # Mean of the three known wOBAs.
    assert lineup_quality([1, 2, 3], woba) == (0.30 + 0.40 + 0.35) / 3


def test_lineup_quality_league_avg_fallback_for_missing():
    woba = {1: 0.40}
    # Two of three ids unknown → filled with league_avg.
    got = lineup_quality([1, 2, 3], woba, league_avg=0.320)
    assert got == (0.40 + 0.320 + 0.320) / 3


def test_lineup_quality_empty_and_null_safe():
    assert lineup_quality([], {1: 0.4}) is None
    assert lineup_quality(None, {1: 0.4}) is None
    assert lineup_quality("notalist", {1: 0.4}) is None
    # A None map → every id falls back to league_avg (no crash).
    assert lineup_quality([1, 2], None, league_avg=0.320) == 0.320


def test_lineup_quality_default_league_avg_constant():
    # All-unknown lineup collapses to exactly the league-average constant.
    assert lineup_quality([99, 98, 97], {}) == LEAGUE_AVG_WOBA


# ─────────────────────────── lineup_by_game (leak-free, null-safe) ───────────────────────────
def _usage(entries):
    """{gid: {"home": {"lineup":[...]}, "away": {"lineup":[...]}}} builder."""
    out = {}
    for gid, home_line, away_line in entries:
        out[gid] = {
            "home": {"pitches": 0, "relievers": 0, "lineup": home_line},
            "away": {"pitches": 0, "relievers": 0, "lineup": away_line},
        }
    return out


def test_lineup_by_game_scores_confirmed_lineups():
    woba = {1: 0.40, 2: 0.40, 3: 0.40, 11: 0.30, 12: 0.30, 13: 0.30}
    games = [_G(1, 10, 20, 2023, 4, 1)]
    usage = _usage([(1, [1, 2, 3], [11, 12, 13])])
    out = lineup_by_game(games, usage, woba)
    assert abs(out[1]["home_lineup_quality"] - 0.40) < 1e-9
    assert abs(out[1]["away_lineup_quality"] - 0.30) < 1e-9
    assert out[1]["home_lineup_size"] == 3


def test_lineup_by_game_missing_lineup_is_none():
    # A game with no usage entry → both qualities None (leak-free, just no signal).
    games = [_G(1, 10, 20, 2023, 4, 1)]
    out = lineup_by_game(games, {}, {1: 0.4})
    assert out[1]["home_lineup_quality"] is None
    assert out[1]["away_lineup_quality"] is None
    assert out[1]["home_lineup_size"] == 0


def test_lineup_by_game_tolerates_old_shape_usage():
    # A usage entry WITHOUT a lineup key (old bullpen-only shape) → None quality, no crash.
    games = [_G(1, 10, 20, 2023, 4, 1)]
    usage = {1: {"home": {"pitches": 60, "relievers": 3}, "away": {"pitches": 62, "relievers": 2}}}
    out = lineup_by_game(games, usage, {1: 0.4})
    assert out[1]["home_lineup_quality"] is None
    assert out[1]["away_lineup_quality"] is None


def test_lineup_by_game_skips_games_without_id():
    class _Bad:
        id = None
        home_team_id = 1
        away_team_id = 2
        date = None
    assert lineup_by_game([_Bad()], {}, {}) == {}


def test_lineup_by_game_is_leak_free_uses_prior_anchor_only():
    # The anchor map is the PRIOR season's wOBA; scoring never touches this game's outcome
    # or date. Two identical lineups on different dates score identically — no time drift.
    woba = {1: 0.35, 2: 0.35}
    games = [_G(1, 10, 20, 2023, 4, 1), _G(2, 10, 20, 2023, 9, 1)]
    usage = _usage([(1, [1, 2], [1, 2]), (2, [1, 2], [1, 2])])
    out = lineup_by_game(games, usage, woba)
    assert out[1]["home_lineup_quality"] == out[2]["home_lineup_quality"]
    assert abs(out[1]["home_lineup_quality"] - 0.35) < 1e-9


# ─────────────────────────── boxscore battingOrder parse ───────────────────────────
def _box_with_lineup(home_order, away_order):
    def side(order):
        return {"pitchers": [], "players": {}, "battingOrder": order}
    return {"teams": {"home": side(home_order), "away": side(away_order)}}


def test_parse_boxscore_captures_batting_order():
    data = _box_with_lineup([657041, 671277, 600869], [642086, 608841])
    out = parse_boxscore(data)
    assert out["home"]["lineup"] == [657041, 671277, 600869]
    assert out["away"]["lineup"] == [642086, 608841]
    # Bullpen fields still present and backward-compatible.
    assert out["home"]["pitches"] == 0 and out["home"]["relievers"] == 0


def test_parse_boxscore_lineup_null_safe_and_dedup():
    # Missing battingOrder → empty lineup; duplicate ids de-duplicated in order.
    data = {"teams": {"home": {"pitchers": [], "players": {}},
                      "away": {"pitchers": [], "players": {}, "battingOrder": [5, 5, 6, "x", 6]}}}
    out = parse_boxscore(data)
    assert out["home"]["lineup"] == []
    assert out["away"]["lineup"] == [5, 6]


def test_parse_boxscore_keeps_reliever_and_lineup_together():
    # A full-ish payload: relievers counted AND lineup captured on the same team block.
    data = {"teams": {
        "home": {"pitchers": [1, 2],
                 "players": {"ID1": {"stats": {"pitching": {"gamesStarted": 1, "pitchesThrown": 90}}},
                             "ID2": {"stats": {"pitching": {"gamesStarted": 0, "pitchesThrown": 20}}}},
                 "battingOrder": [100, 101, 102]},
        "away": {"pitchers": [], "players": {}, "battingOrder": [200]}}}
    out = parse_boxscore(data)
    assert out["home"] == {"pitches": 20, "relievers": 1, "lineup": [100, 101, 102]}
    assert out["away"] == {"pitches": 0, "relievers": 0, "lineup": [200]}


# ─────────────────────────── cache refresh on old shape ───────────────────────────
def test_load_cached_refreshes_old_shape(tmp_path, monkeypatch):
    # Write an OLD-shape cache file (no "lineup" key) and confirm _load_cached returns None
    # so the caller re-fetches; then a NEW-shape file (with lineup) loads normally.
    monkeypatch.setattr(boxscore_client, "_CACHE_DIR", str(tmp_path))

    old_path = os.path.join(str(tmp_path), "111.json")
    with open(old_path, "w", encoding="utf-8") as fh:
        json.dump({"home": {"pitches": 60, "relievers": 3},
                   "away": {"pitches": 62, "relievers": 2}}, fh)
    assert boxscore_client._load_cached(111) is None      # old shape → force re-fetch

    new_path = os.path.join(str(tmp_path), "222.json")
    new_val = {"home": {"pitches": 60, "relievers": 3, "lineup": [1, 2, 3]},
               "away": {"pitches": 62, "relievers": 2, "lineup": [4, 5, 6]}}
    with open(new_path, "w", encoding="utf-8") as fh:
        json.dump(new_val, fh)
    assert boxscore_client._load_cached(222) == new_val   # current shape → served from cache


def test_has_lineup_detector():
    assert boxscore_client._has_lineup(
        {"home": {"lineup": []}, "away": {"lineup": []}}) is True
    assert boxscore_client._has_lineup(
        {"home": {"pitches": 1}, "away": {"lineup": []}}) is False
    assert boxscore_client._has_lineup(None) is False


# ─────────────────────────── predict_game integration ───────────────────────────
def test_predict_game_unchanged_without_lineup():
    home = _state(300, 250, 40, 30, [1, 1, 0, 1, 1, 0, 1, 1, 0, 1])
    away = _state(250, 300, 30, 40, [0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    pred = predict_game(home, away, min_games=10)
    cl = pred["components"]["confirmed_lineups"]
    assert cl["modeled"] is False
    assert cl["contribution"] == 0.0
    assert pred["features"]["lineup_present"] == 0.0
    assert pred["features"]["lineup_gap"] == 0.0
    # Decomposition still reconciles with the new (neutral) component included.
    c = pred["components"]
    total = (pred["baseline"] + c["team_strength"]["contribution"]
             + c["home_field"]["contribution"] + c["recent_form"]["contribution"]
             + c["starting_pitcher"]["contribution"] + c["run_environment"]["contribution"]
             + c["schedule_fatigue"]["contribution"] + c["bullpen_fatigue"]["contribution"]
             + c["confirmed_lineups"]["contribution"])
    assert abs(total - pred["home_win_prob"]) < 1e-9


def test_predict_game_better_home_lineup_nudges_up():
    home = _state(300, 270, 40, 30)
    away = _state(270, 300, 30, 40)
    base = predict_game(home, away, min_games=10)
    better = predict_game(home, away, min_games=10,
                          lineup={"home_lineup_quality": 0.360, "away_lineup_quality": 0.300})
    cl = better["components"]["confirmed_lineups"]
    assert cl["modeled"] is True
    assert better["home_win_prob"] > base["home_win_prob"]           # better home bats → up
    assert round(better["features"]["lineup_gap"], 4) == 0.06        # 0.360 - 0.300
    assert better["features"]["lineup_present"] == 1.0


def test_predict_game_lineup_bounded_and_reconciles():
    home = _state(320, 240, 45, 25)
    away = _state(240, 320, 25, 45)
    base = predict_game(home, away, min_games=10)
    extreme = predict_game(home, away, min_games=10,
                           lineup={"home_lineup_quality": 0.500, "away_lineup_quality": 0.150})
    assert abs(extreme["components"]["confirmed_lineups"]["contribution"]) <= 0.03 + 1e-9
    c = extreme["components"]
    total = (extreme["baseline"] + c["team_strength"]["contribution"]
             + c["home_field"]["contribution"] + c["recent_form"]["contribution"]
             + c["starting_pitcher"]["contribution"] + c["run_environment"]["contribution"]
             + c["schedule_fatigue"]["contribution"] + c["bullpen_fatigue"]["contribution"]
             + c["confirmed_lineups"]["contribution"])
    assert abs(total - extreme["home_win_prob"]) < 1e-9


def test_predict_game_lineup_partial_data_safe():
    # Only one side's quality known → stays neutral (no crash, no half-signal).
    home = _state(300, 270, 40, 30)
    away = _state(270, 300, 30, 40)
    pred = predict_game(home, away, min_games=10,
                        lineup={"home_lineup_quality": 0.360, "away_lineup_quality": None})
    assert pred["components"]["confirmed_lineups"]["modeled"] is False
    assert pred["features"]["lineup_gap"] == 0.0
    assert pred["features"]["lineup_present"] == 0.0
