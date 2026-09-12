from sports.basketball.analytics.game_model import predict_game, recent_form
from sports.basketball.api.basketball_routes import _player
from datetime import datetime, timezone


def _event(event_id, when, team_id, own, other, completed=True):
    return {"id": event_id, "date": when, "competitions": [{"status": {"type": {"completed": completed}}, "competitors": [
        {"team": {"id": team_id}, "score": {"value": own}},
        {"team": {"id": "opp"}, "score": {"value": other}},
    ]}]}


def _stats(points, rebounds, turnovers, fgm, fga, threes):
    values = {"avgPoints": points, "avgRebounds": rebounds, "avgTurnovers": turnovers,
              "avgFieldGoalsMade": fgm, "avgFieldGoalsAttempted": fga,
              "avgThreePointFieldGoalsMade": threes, "avgFreeThrowsAttempted": 20,
              "avgOffensiveRebounds": 10}
    return {"results": {"stats": {"categories": [{"stats": [
        {"name": key, "value": value} for key, value in values.items()
    ]}]}}}


def test_recent_form_excludes_target_and_future_games():
    schedule = {"events": [
        _event("past", "2026-01-01T00:00Z", "home", 100, 80),
        _event("target", "2026-01-10T00:00Z", "home", 40, 20, False),
        _event("future", "2026-01-20T00:00Z", "home", 50, 120),
    ]}
    form = recent_form(schedule, "home", datetime(2026, 1, 10, tzinfo=timezone.utc))
    assert form["games"] == 1
    assert form["win_pct"] == 1.0
    assert form["avg_margin"] == 20.0


def test_multi_factor_model_favors_stronger_recent_team():
    home_schedule = {"events": [_event(str(i), f"2026-01-{i + 1:02d}T00:00Z", "home", 100, 85) for i in range(1, 9)]}
    away_schedule = {"events": [_event(str(i), f"2026-01-{i + 1:02d}T00:00Z", "away", 82, 96) for i in range(1, 9)]}
    result = predict_game(
        home={"id": "home", "record": "18-6"}, away={"id": "away", "record": "8-16"},
        home_schedule=home_schedule, away_schedule=away_schedule,
        home_stats=_stats(102, 42, 11, 38, 80, 12), away_stats=_stats(86, 34, 17, 30, 80, 7),
        game_date="2026-02-01T00:00Z", home_edge=0.04,
    )
    assert result["home_win_probability"] > 0.70
    assert result["model_version"] == "bb-form-v1"
    assert result["reliability"] > 0.5


def test_player_accepts_string_status_from_scoreboard_leaders():
    player = _player({
        "id": "123",
        "displayName": "Example Player",
        "status": "Active",
        "headshot": None,
        "position": None,
    })

    assert player["status"] == "Active"
    assert player["headshot"] is None
    assert player["position"] is None
