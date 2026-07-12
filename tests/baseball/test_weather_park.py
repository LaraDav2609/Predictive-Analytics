"""Tests for the run-environment signal: static park factors, weather null-safety,
and that predict_game stays backward-compatible when no park/weather is supplied."""
import asyncio
from datetime import datetime, timezone

from sports.baseball.analytics.game_model import TeamState, predict_game
from sports.baseball.data import weather_client as wx
from sports.baseball.data.weather_client import (
    NEUTRAL_PARK_FACTOR, PARK_FACTORS, VENUES, WeatherClient,
    _lookup_hour, _parse_hour_grid, park_factor, season_weather_by_game,
    venue_info, wind_out_component,
)


def _state(rf, ra, wins, losses, last10=None):
    return TeamState(runs_for=rf, runs_against=ra, wins=wins, losses=losses,
                     last10=last10 or [])


# ─────────────────────────── park factors ───────────────────────────
def test_park_factor_lookup_by_team_and_venue():
    assert park_factor("Colorado Rockies") > 1.10          # Coors is a hitter park
    assert park_factor("Coors Field") > 1.10
    assert park_factor("San Diego Padres") < 0.95          # Petco is a pitcher park
    assert park_factor("Petco Park") < 0.95
    assert park_factor("Padres") < 0.95                    # nickname resolves too


def test_park_factor_unknown_is_neutral():
    assert park_factor("Nonexistent Park") == NEUTRAL_PARK_FACTOR
    assert park_factor("") == NEUTRAL_PARK_FACTOR
    assert park_factor(None) == NEUTRAL_PARK_FACTOR


def test_all_30_teams_have_a_venue_and_factor():
    assert len(VENUES) == 30
    for team in VENUES:
        assert team in PARK_FACTORS                        # every venue has a factor
        info = venue_info(team)
        assert info is not None
        assert 0 <= info["cf_azimuth"] <= 360
        assert isinstance(info["lat"], (int, float))
        assert isinstance(info["lon"], (int, float))


def test_venue_info_unknown_is_none():
    assert venue_info("Not A Team") is None


# ─────────────────────────── wind vector ───────────────────────────
def test_wind_out_component_signs():
    # CF azimuth 0 (due north). Wind FROM the south (180) blows toward north → out.
    out = wind_out_component(10.0, 180.0, 0.0)
    assert out is not None and out > 9.0                   # ~+10 mph out to CF
    # Wind FROM the north (0) blows toward south → in.
    into = wind_out_component(10.0, 0.0, 0.0)
    assert into is not None and into < -9.0
    # Crosswind (from east, 90) is ~0 out-component.
    cross = wind_out_component(10.0, 90.0, 0.0)
    assert cross is not None and abs(cross) < 1.0


def test_wind_out_component_null_safe():
    assert wind_out_component(None, 180, 0) is None
    assert wind_out_component(10, None, 0) is None
    assert wind_out_component(10, 180, None) is None


# ─────────────────────────── predict_game backward compat ───────────────────────────
def test_predict_game_unchanged_without_env():
    # No park_factor / weather → identical to the pre-feature behaviour: the
    # run_environment component is present but neutral (0 contribution, not modeled).
    home = _state(300, 250, 40, 30, [1, 1, 0, 1, 1, 0, 1, 1, 0, 1])
    away = _state(250, 300, 30, 40, [0, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    pred = predict_game(home, away, min_games=10)
    env = pred["components"]["run_environment"]
    assert env["modeled"] is False
    assert env["contribution"] == 0.0
    assert pred["features"]["env_present"] == 0.0
    # Named components still sum (with baseline) to the raw home prob.
    c = pred["components"]
    total = (pred["baseline"] + c["team_strength"]["contribution"]
             + c["home_field"]["contribution"] + c["recent_form"]["contribution"]
             + c["starting_pitcher"]["contribution"] + env["contribution"])
    assert abs(total - pred["home_win_prob"]) < 1e-9


def test_predict_game_env_present_when_park_supplied():
    home = _state(320, 240, 45, 25, [1] * 8 + [0, 0])      # strong home favourite
    away = _state(240, 320, 25, 45, [0] * 8 + [1, 1])
    base = predict_game(home, away, min_games=10)
    hot = predict_game(home, away, park_factor=1.15, min_games=10)   # Coors-like
    env = hot["components"]["run_environment"]
    assert env["modeled"] is True
    assert env["park_factor"] == 1.15
    assert hot["features"]["env_present"] == 1.0
    # Hot run environment regresses the favourite toward the coin flip (small).
    assert hot["home_win_prob"] < base["home_win_prob"]
    assert abs(hot["home_win_prob"] - base["home_win_prob"]) <= 0.02 + 1e-9
    # Decomposition still reconciles.
    c = hot["components"]
    total = (hot["baseline"] + c["team_strength"]["contribution"]
             + c["home_field"]["contribution"] + c["recent_form"]["contribution"]
             + c["starting_pitcher"]["contribution"] + c["run_environment"]["contribution"])
    assert abs(total - hot["home_win_prob"]) < 1e-9


def test_predict_game_weather_none_is_safe():
    home = _state(300, 260, 40, 30)
    away = _state(260, 300, 30, 40)
    # Explicit weather=None must behave like no weather (no crash, neutral env).
    pred = predict_game(home, away, weather=None, min_games=10)
    assert pred["components"]["run_environment"]["modeled"] is False


def test_predict_game_with_weather_dict():
    home = _state(320, 240, 45, 25, [1] * 8 + [0, 0])
    away = _state(240, 320, 25, 45, [0] * 8 + [1, 1])
    weather = {"wind_speed_mph": 15.0, "wind_direction_deg": 180.0, "cf_azimuth": 0,
               "temperature_f": 85.0, "domed": False}
    pred = predict_game(home, away, park_factor=1.02, weather=weather, min_games=10)
    env = pred["components"]["run_environment"]
    assert env["modeled"] is True
    assert env["wind_out_mph"] is not None and env["wind_out_mph"] > 9.0
    assert pred["features"]["wind_out_mph"] > 9.0


def test_predict_game_domed_ignores_wind():
    home = _state(320, 240, 45, 25)
    away = _state(240, 320, 25, 45)
    weather = {"wind_speed_mph": 20.0, "wind_direction_deg": 180.0, "cf_azimuth": 0,
               "temperature_f": 72.0, "domed": True}
    pred = predict_game(home, away, park_factor=1.02, weather=weather, min_games=10)
    # Domed → wind ignored (wind_out stays None), only the park factor applies.
    assert pred["components"]["run_environment"]["wind_out_mph"] is None


def test_weather_client_instantiates():
    # Construction must not touch the network.
    wc = WeatherClient()
    assert wc is not None


# ─────────────── per-venue seasonal aggregation (efficient backtest path) ───────────────
def test_parse_hour_grid_keys_to_hour_and_is_null_safe():
    hourly = {
        "time": ["2023-04-01T18:00", "2023-04-01T19:00", "2023-04-01T20:00"],
        "temperature_2m": [60.0, 62.0, None],           # a missing value must not crash
        "wind_speed_10m": [8.0, 9.0, 10.0],
        "wind_direction_10m": [180.0, 200.0, 210.0],
    }
    grid = _parse_hour_grid(hourly)
    assert set(grid) == {"2023-04-01T18", "2023-04-01T19", "2023-04-01T20"}
    assert grid["2023-04-01T19"]["temperature_f"] == 62.0
    assert grid["2023-04-01T20"]["temperature_f"] is None      # null preserved, not error


def test_parse_hour_grid_empty_and_ragged():
    assert _parse_hour_grid({}) == {}
    # Ragged arrays (wind shorter than time) → _at returns None, no IndexError.
    grid = _parse_hour_grid({"time": ["2023-04-01T19:00"], "temperature_2m": [70.0],
                             "wind_speed_10m": [], "wind_direction_10m": []})
    assert grid["2023-04-01T19"]["wind_speed_mph"] is None


def test_lookup_hour_exact_and_nearest():
    grid = {"2023-06-15T23": {"temperature_f": 75.0, "wind_speed_mph": 5.0, "wind_direction_deg": 90.0}}
    # Exact UTC-hour match.
    dt = datetime(2023, 6, 15, 23, 10, tzinfo=timezone.utc)
    assert _lookup_hour(grid, dt)["temperature_f"] == 75.0
    # One hour off, same day → nearest-hour fallback within ±3h.
    dt2 = datetime(2023, 6, 15, 22, 5, tzinfo=timezone.utc)
    assert _lookup_hour(grid, dt2)["temperature_f"] == 75.0
    # Far away (next day) → nothing.
    dt3 = datetime(2023, 6, 16, 12, 0, tzinfo=timezone.utc)
    assert _lookup_hour(grid, dt3) is None


def test_lookup_hour_null_safe():
    assert _lookup_hour({}, datetime(2023, 6, 15, 23, tzinfo=timezone.utc)) is None
    assert _lookup_hour({"2023-06-15T23": {}}, None) is None
    # Naive datetime is treated as UTC (no crash).
    grid = {"2023-06-15T23": {"temperature_f": 70.0}}
    assert _lookup_hour(grid, datetime(2023, 6, 15, 23))["temperature_f"] == 70.0


class _FakeGame:
    def __init__(self, gid, home_team, home_team_id, away_team_id, dt):
        self.id = gid
        self.home_team = home_team
        self.away_team = "Somebody"
        self.home_team_id = home_team_id
        self.away_team_id = away_team_id
        self.venue = None
        self.date = dt


def test_season_weather_by_game_aggregates_per_venue(monkeypatch):
    # Two Rockies home games; the fetch is stubbed so no network is touched and we can
    # assert it is called exactly ONCE per venue (the efficiency guarantee), not per game.
    calls = {"n": 0}

    async def fake_fetch(canon, info, season, *, timeout, use_cache):
        calls["n"] += 1
        return {
            "2023-07-01T01": {"temperature_f": 85.0, "wind_speed_mph": 12.0, "wind_direction_deg": 180.0},
            "2023-07-02T01": {"temperature_f": 80.0, "wind_speed_mph": 6.0, "wind_direction_deg": 90.0},
        }

    monkeypatch.setattr(wx, "_fetch_venue_season", fake_fetch)
    games = [
        _FakeGame(1, "Colorado Rockies", 115, 999, datetime(2023, 7, 1, 1, 10, tzinfo=timezone.utc)),
        _FakeGame(2, "Colorado Rockies", 115, 998, datetime(2023, 7, 2, 1, 5, tzinfo=timezone.utc)),
    ]
    out = asyncio.run(season_weather_by_game(games, 2023))
    assert calls["n"] == 1                                  # ONE venue fetch, not two games
    assert set(out) == {1, 2}
    assert out[1]["temperature_f"] == 85.0
    assert out[1]["venue"] == "Coors Field"
    assert out[1]["cf_azimuth"] == VENUES["colorado rockies"]["cf_azimuth"]
    assert out[1]["domed"] is False


def test_season_weather_by_game_unknown_venue_skipped(monkeypatch):
    async def fake_fetch(*a, **k):
        return {"2023-07-01T01": {"temperature_f": 70.0, "wind_speed_mph": 5.0, "wind_direction_deg": 0.0}}

    monkeypatch.setattr(wx, "_fetch_venue_season", fake_fetch)
    games = [_FakeGame(1, "Not A Real Team", 1, 2, datetime(2023, 7, 1, 1, tzinfo=timezone.utc))]
    out = asyncio.run(season_weather_by_game(games, 2023))
    assert out == {}                                        # unknown venue → no entry, no crash


def test_season_weather_by_game_failed_fetch_degrades(monkeypatch):
    async def fake_fetch(*a, **k):
        return {}                                           # simulate a fetch failure

    monkeypatch.setattr(wx, "_fetch_venue_season", fake_fetch)
    games = [_FakeGame(1, "Colorado Rockies", 115, 999, datetime(2023, 7, 1, 1, tzinfo=timezone.utc))]
    out = asyncio.run(season_weather_by_game(games, 2023))
    assert out == {}                                        # no weather → park-only fallback
