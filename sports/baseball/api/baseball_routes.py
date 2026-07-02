"""MLB Baseball API routes."""

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException

from sports.baseball.data.mlb_client import MLBClient
from sports.baseball.analytics.baseball_predictor import BaseballPredictor

router = APIRouter(prefix="/baseball", tags=["baseball"])

client: Optional[MLBClient] = None
predictor: Optional[BaseballPredictor] = None


def init(mc: MLBClient, bp: BaseballPredictor):
    global client, predictor
    client, predictor = mc, bp


_era_cache: dict[int, dict[int, float]] = {}


async def _prior_season_era_map(season: int, min_ip: float = 20.0) -> dict[int, float]:
    """Prior-season starter ERA keyed by player id — leak-free for ``season`` (the
    previous season is fully known before this one starts). Filtered to >= min_ip
    innings so tiny relief samples don't add noise. Cached per season."""
    prior = int(season) - 1
    if prior in _era_cache:
        return _era_cache[prior]
    raw = await client.fetch_pitching_season_stats(prior)
    era_map = {pid: s["era"] for pid, s in raw.items() if s.get("ip") and s["ip"] >= min_ip}
    _era_cache[prior] = era_map
    return era_map


@router.get("/teams")
async def get_teams():
    teams = client.get_teams()
    return {"ok": True, "teams": [t.model_dump() for t in teams]}


@router.get("/standings")
async def get_standings():
    standings = client.get_standings()
    return {"ok": True, "standings": [s.model_dump() for s in standings]}


@router.get("/schedule")
async def get_schedule(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    if start_date or end_date:
        sd = start_date or date.today()
        ed = end_date or (sd + timedelta(days=7))
        games = await client.fetch_schedule_range(sd, ed)
    else:
        games = client.get_schedule()
    if predictor:
        games = predictor.predict_games(games)
    return {
        "ok": True,
        "games": [g.model_dump(mode="json") for g in games],
        "start_date": (start_date.isoformat() if start_date else None),
        "end_date": (end_date.isoformat() if end_date else None),
    }


@router.get("/games/{game_pk}")
async def get_game(game_pk: int):
    game = await client.fetch_game(game_pk)
    if not game:
        raise HTTPException(status_code=404, detail=f"Game {game_pk} not found")
    games = [game]
    if predictor:
        games = predictor.predict_games(games)
    return {"ok": True, "game": games[0].model_dump(mode="json")}


@router.get("/games/{game_pk}/analysis")
async def game_analysis(game_pk: int):
    """Decomposable pre-game read for one game: the named analysis components
    (team strength, home field, recent form, starting pitcher) and how they add up
    to the home win probability — the baseball equivalent of the F1 'why this pick'
    breakdown. Team state is rebuilt leak-free from the season's prior games."""
    from sports.baseball.analytics.game_model import TeamState, predict_game
    from sports.baseball.analytics.backtest import team_states_before

    game = await client.fetch_game(game_pk)
    if not game:
        raise HTTPException(status_code=404, detail=f"Game {game_pk} not found")
    season = game.date.year
    games = await client.fetch_schedule_range(date(season, 3, 1), game.date.date() + timedelta(days=1))
    states = team_states_before(games, game.date)
    home_state = states.get(game.home_team_id, TeamState())
    away_state = states.get(game.away_team_id, TeamState())
    era_map = await _prior_season_era_map(season)
    pred = predict_game(
        home_state, away_state,
        home_pitcher=game.home_pitcher, away_pitcher=game.away_pitcher,
        home_pitcher_era=era_map.get(game.home_pitcher_id), away_pitcher_era=era_map.get(game.away_pitcher_id),
    )
    return {
        "ok": True,
        "game": {
            "id": game.id, "date": game.date.isoformat(),
            "home_team": game.home_team, "away_team": game.away_team,
            "home_team_id": game.home_team_id, "away_team_id": game.away_team_id,
            "home_pitcher": game.home_pitcher, "away_pitcher": game.away_pitcher,
            "venue": game.venue, "status": game.status,
            "home_score": game.home_score, "away_score": game.away_score,
        },
        "season": season,
        **pred,
    }


@router.get("/backtest")
async def backtest(season: int = 2023, min_games: int = 15, calibrate: bool = True):
    """Leak-free, walk-forward backtest of the model over a full season — Brier /
    log loss / reliability / accuracy, Brier-skill vs the home-base-rate baseline,
    and a Platt calibration report. The go/no-go signal is whether Brier skill > 0
    (does the model beat just predicting the home base rate)."""
    from sports.baseball.analytics.backtest import run_backtest as _run

    games = await client.fetch_schedule_range(date(season, 3, 1), date(season, 11, 1))
    era_map = await _prior_season_era_map(season)               # prior season = leak-free
    result = _run(games, min_games=min_games, calibrate=calibrate, pitcher_era=era_map)
    result["pitcher_prior_season"] = season - 1
    result["pitchers_rated"] = len(era_map)
    skill = result.get("brier_skill_score", 0.0)
    result["gate"] = {
        "beats_base_rate": bool(skill > 0),
        "brier_skill_score": skill,
        "verdict": (
            "Model beats the home base rate out-of-sample (positive Brier skill)."
            if skill > 0 else
            "Model ranks teams (accuracy > base rate) but does not yet beat base-rate "
            "probability precision — the starting-pitcher lever is the next step."
        ),
    }
    return {"ok": True, "season": season, **result}


@router.get("/pipeline/health")
async def pipeline_health():
    """Data-source + model health for the baseball pipeline monitor (mirrors the F1
    pipeline monitor). All sources are free."""
    teams = client.get_teams()
    standings = client.get_standings()
    schedule = client.get_schedule()
    reachable = len(teams) > 0

    sources = [
        {
            "name": "MLB StatsAPI", "url": "statsapi.mlb.com", "kind": "primary",
            "status": "ok" if reachable else "down", "free": True,
            "detail": f"{len(teams)} teams · {len(standings)} standings · {len(schedule)} scheduled games",
        },
        {
            "name": "pybaseball (FanGraphs / Baseball-Reference / Statcast)", "url": "pybaseball",
            "kind": "advanced", "status": "optional", "free": True,
            "detail": "Advanced metrics for the player pages; loaded on demand",
        },
    ]
    return {
        "ok": True,
        "reachable": reachable,
        "sources": sources,
        "counts": {"teams": len(teams), "standings": len(standings), "scheduled_games": len(schedule)},
        "model": {
            "version": "mlb-decomp-v1",
            "components": [
                {"name": "team_strength", "modeled": True},
                {"name": "home_field", "modeled": True},
                {"name": "recent_form", "modeled": True},
                {"name": "starting_pitcher", "modeled": False},
            ],
            "calibration": "Platt (fit in backtest; not yet applied at serve)",
        },
        "backtest": {"available": True, "endpoint": "/api/baseball/backtest", "kind": "leak-free walk-forward"},
        "notes": [
            "Data is free and real-time (MLB StatsAPI); no paid feed required.",
            "Model ranks teams (accuracy > base rate) but Brier skill ~0 — the starting-pitcher model is the next lever.",
        ],
    }


@router.post("/refresh")
async def refresh():
    await client.refresh()
    if predictor:
        predictor.load_standings(client.get_standings())
    return {
        "ok": True,
        "teams": len(client.get_teams()),
        "standings": len(client.get_standings()),
        "games": len(client.get_schedule()),
    }
