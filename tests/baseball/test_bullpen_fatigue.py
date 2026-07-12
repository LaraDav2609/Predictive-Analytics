"""Tests for the bullpen-fatigue feature: null-safe boxscore parsing, leak-free
accumulation of prior reliever workload (only games strictly before), the decomposable
predict_game component, and backward compatibility when no bullpen usage is supplied."""
from datetime import datetime, timezone

from sports.baseball.analytics.bullpen_fatigue import (
    LOAD_SCALE, LOOKBACK_DAYS, bullpen_by_game,
)
from sports.baseball.analytics.game_model import TeamState, predict_game
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


def _box(home_pitchers, away_pitchers):
    """Minimal boxscore payload. Each side is a list of (pitcher_id, gamesStarted,
    pitchesThrown) tuples in appearance order."""
    def side(pitchers):
        players = {}
        ids = []
        for pid, gs, pitches in pitchers:
            ids.append(pid)
            stat = {"gamesStarted": gs}
            if pitches is not None:
                stat["pitchesThrown"] = pitches
            players[f"ID{pid}"] = {"stats": {"pitching": stat}}
        return {"pitchers": ids, "players": players}
    return {"teams": {"home": side(home_pitchers), "away": side(away_pitchers)}}


# ─────────────────────────── boxscore parsing (null-safety) ───────────────────────────
def test_parse_boxscore_counts_only_relievers():
    # Starter (gs=1, 85 pitches) + 2 relievers (34, 23). Only relievers counted.
    data = _box(
        home_pitchers=[(1, 1, 85), (2, 0, 34), (3, 0, 23)],
        away_pitchers=[(4, 1, 60), (5, 0, 40)],
    )
    out = parse_boxscore(data)
    # Reliever counts unchanged; the shape now also carries a (here empty) lineup list.
    assert out["home"]["pitches"] == 57 and out["home"]["relievers"] == 2   # 34 + 23, starter excluded
    assert out["away"]["pitches"] == 40 and out["away"]["relievers"] == 1
    assert out["home"]["lineup"] == [] and out["away"]["lineup"] == []


def test_parse_boxscore_null_safe():
    assert parse_boxscore(None) is None
    assert parse_boxscore({}) is None
    assert parse_boxscore({"teams": {"home": {}}}) is None    # missing away
    # Missing player entry / missing pitching stat → skipped, no crash.
    data = {"teams": {"home": {"pitchers": [1, 2], "players": {"ID1": {}}},
                      "away": {"pitchers": [], "players": {}}}}
    out = parse_boxscore(data)
    assert out["home"] == {"pitches": 0, "relievers": 0, "lineup": []}
    assert out["away"] == {"pitches": 0, "relievers": 0, "lineup": []}


def test_parse_boxscore_falls_back_to_battersfaced():
    # No pitchesThrown/numberOfPitches → battersFaced is the workload proxy.
    data = {"teams": {
        "home": {"pitchers": [1, 2],
                 "players": {"ID1": {"stats": {"pitching": {"gamesStarted": 1, "battersFaced": 25}}},
                             "ID2": {"stats": {"pitching": {"gamesStarted": 0, "battersFaced": 6}}}}},
        "away": {"pitchers": [], "players": {}}}}
    out = parse_boxscore(data)
    assert out["home"] == {"pitches": 6, "relievers": 1, "lineup": []}   # reliever's 6 BF, starter excluded


# ─────────────────────────── leak-free accumulation ───────────────────────────
def _usage(entries):
    """{gid: {"home": {pitches,relievers}, "away": {...}}} builder."""
    return {gid: {"home": {"pitches": h, "relievers": 3}, "away": {"pitches": a, "relievers": 3}}
            for gid, h, a in entries}


def test_first_game_has_no_bullpen_load():
    games = [_G(1, 10, 20, 2023, 4, 1)]
    usage = _usage([(1, 120, 100)])
    out = bullpen_by_game(games, usage)
    assert out[1]["home_bullpen_load"] is None            # no prior game → unknown
    assert out[1]["away_bullpen_load"] is None
    assert out[1]["home_relievers_used"] == 0


def test_bullpen_load_accumulates_leak_free():
    # Team 10 plays Apr 1 (pen 150), Apr 2 (pen 90), then Apr 4.
    games = [
        _G(1, 10, 20, 2023, 4, 1),
        _G(2, 10, 30, 2023, 4, 2),
        _G(3, 10, 40, 2023, 4, 4),
    ]
    usage = _usage([(1, 150, 60), (2, 90, 60), (3, 0, 0)])
    out = bullpen_by_game(games, usage)
    # Game 2: only Apr 1's pen (150) is prior → 150/scale.
    assert out[2]["home_bullpen_load"] == 150.0 / LOAD_SCALE
    assert out[2]["home_relievers_used"] == 1
    # Game 3 (Apr 4): prior games Apr 1 (150) + Apr 2 (90) both within 3-day lookback.
    assert out[3]["home_bullpen_load"] == (150.0 + 90.0) / LOAD_SCALE
    assert out[3]["home_relievers_used"] == 2


def test_bullpen_lookback_excludes_old_games():
    # A game outside the LOOKBACK_DAYS window never counts toward current load.
    games = [_G(1, 10, 20, 2023, 4, 1),
             _G(2, 10, 30, 2023, 4, 1 + LOOKBACK_DAYS + 2)]
    usage = _usage([(1, 150, 60), (2, 0, 0)])
    out = bullpen_by_game(games, usage)
    assert out[2]["home_bullpen_load"] is None            # prior game fell outside window


def test_bullpen_by_game_skips_malformed_games():
    class _Bad:
        id = None
        home_team_id = 1
        away_team_id = 2
        date = None
    assert bullpen_by_game([_Bad()], {}) == {}


def test_bullpen_by_game_tolerates_missing_usage():
    # No usage at all → every load is None (leak-free, just no signal), no crash.
    games = [_G(1, 10, 20, 2023, 4, 1), _G(2, 10, 30, 2023, 4, 2)]
    out = bullpen_by_game(games, {})
    assert out[2]["home_bullpen_load"] is None


# ─────────────────────────── predict_game integration ───────────────────────────
def test_predict_game_unchanged_without_bullpen():
    home = _state(300, 250, 40, 30, [1, 1, 0, 1, 1, 0, 1, 1, 0, 1])
    away = _state(250, 300, 30, 40, [0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    pred = predict_game(home, away, min_games=10)
    bp = pred["components"]["bullpen_fatigue"]
    assert bp["modeled"] is False
    assert bp["contribution"] == 0.0
    assert pred["features"]["bullpen_present"] == 0.0
    assert pred["features"]["bullpen_gap"] == 0.0
    # Decomposition still reconciles with the new (neutral) component included.
    c = pred["components"]
    total = (pred["baseline"] + c["team_strength"]["contribution"]
             + c["home_field"]["contribution"] + c["recent_form"]["contribution"]
             + c["starting_pitcher"]["contribution"] + c["run_environment"]["contribution"]
             + c["schedule_fatigue"]["contribution"] + c["bullpen_fatigue"]["contribution"])
    assert abs(total - pred["home_win_prob"]) < 1e-9


def test_predict_game_taxed_home_bullpen_nudges_down():
    home = _state(300, 270, 40, 30)
    away = _state(270, 300, 30, 40)
    base = predict_game(home, away, min_games=10)
    # Home pen heavily used (load 2.0), away pen fresh (0.2) → home should drop.
    taxed = predict_game(home, away, min_games=10,
                         bullpen={"home_bullpen_load": 2.0, "away_bullpen_load": 0.2,
                                  "home_relievers_used": 5, "away_relievers_used": 1})
    bp = taxed["components"]["bullpen_fatigue"]
    assert bp["modeled"] is True
    assert taxed["home_win_prob"] < base["home_win_prob"]         # taxed home → down
    assert taxed["features"]["bullpen_gap"] == 1.8                # 2.0 - 0.2
    assert taxed["features"]["bullpen_present"] == 1.0


def test_predict_game_bullpen_bounded_and_reconciles():
    home = _state(320, 240, 45, 25)
    away = _state(240, 320, 25, 45)
    base = predict_game(home, away, min_games=10)
    extreme = predict_game(home, away, min_games=10,
                           bullpen={"home_bullpen_load": 10.0, "away_bullpen_load": 0.0,
                                    "home_relievers_used": 8, "away_relievers_used": 0})
    assert abs(extreme["components"]["bullpen_fatigue"]["contribution"]) <= 0.015 + 1e-9
    assert abs(extreme["home_win_prob"] - base["home_win_prob"]) <= 0.02
    c = extreme["components"]
    total = (extreme["baseline"] + c["team_strength"]["contribution"]
             + c["home_field"]["contribution"] + c["recent_form"]["contribution"]
             + c["starting_pitcher"]["contribution"] + c["run_environment"]["contribution"]
             + c["schedule_fatigue"]["contribution"] + c["bullpen_fatigue"]["contribution"])
    assert abs(total - extreme["home_win_prob"]) < 1e-9


def test_predict_game_bullpen_partial_data_safe():
    # Only one side's load known → stays neutral (no crash, no half-signal).
    home = _state(300, 270, 40, 30)
    away = _state(270, 300, 30, 40)
    pred = predict_game(home, away, min_games=10,
                        bullpen={"home_bullpen_load": 1.5, "away_bullpen_load": None,
                                 "home_relievers_used": 4, "away_relievers_used": 0})
    assert pred["components"]["bullpen_fatigue"]["modeled"] is False
    assert pred["features"]["bullpen_gap"] == 0.0
    assert pred["features"]["bullpen_present"] == 0.0
