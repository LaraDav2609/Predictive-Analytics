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
_logs_cache: dict[int, dict[int, list]] = {}


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


async def _season_pitcher_logs(season: int, games) -> dict[int, list]:
    """All starters' game logs for a season, keyed by id (cached). ~one StatsAPI
    call per starter; the whole map is built once per season."""
    if season in _logs_cache:
        return _logs_cache[season]
    from sports.baseball.analytics.pitcher_form import fetch_pitcher_logs, starter_ids
    logs = await fetch_pitcher_logs(client, starter_ids(games), season)
    _logs_cache[season] = logs
    return logs


async def _pitcher_rating_fn(season: int, games):
    """Return a leak-free ``rate(pitcher_id, game_date) -> float | None`` using
    within-season FIP form regressed toward the pitcher's prior-season ERA."""
    from sports.baseball.analytics.pitcher_form import rating_asof
    prior = await _prior_season_era_map(season)
    logs = await _season_pitcher_logs(season, games)

    def _rate(pitcher_id, game_date):
        if not pitcher_id:
            return None
        return rating_asof(logs.get(int(pitcher_id)), game_date, prior.get(int(pitcher_id)))

    return _rate, len(logs)


def _apply_serve_calibration(pred: dict) -> dict:
    """Apply the enabled serve calibrator to a prediction dict IN PLACE. Keeps the
    additive waterfall honest: the named components still sum to the raw model
    probability, and calibration is surfaced as a final ``calibration`` component
    (adjustment = calibrated − model) plus a top-level ``calibration`` block. When
    no calibrator is enabled the numbers are untouched (applied=False)."""
    from sports.baseball.analytics.serve_calibration import calibrate

    raw = pred["home_win_prob"]
    cal = calibrate(raw)
    pred["calibration"] = cal
    pred["home_win_prob_model"] = round(raw, 4)
    if cal["applied"]:
        pred["home_win_prob"] = cal["value"]
        pred["away_win_prob"] = round(1.0 - cal["value"], 4)
        pred.setdefault("components", {})["calibration"] = {
            "contribution": cal["delta"],
            "applied": True,
            "method": cal["method"],
            "detail": "Platt calibration fit on past seasons — nudges the raw model "
                      "toward historically accurate probabilities",
        }
    return pred


@router.get("/calibration")
async def get_calibration():
    """Current serve-calibrator state (params, provenance, enabled flag)."""
    from sports.baseball.analytics.serve_calibration import state
    return {"ok": True, **state()}


@router.post("/calibration/fit")
async def fit_calibration(season: int = 2023, min_games: int = 15, enable: bool = False):
    """Fit a Platt serve-calibrator on a full season (leak-free walk-forward
    predictions), persist it, and optionally enable it. Returns the fit summary
    incl. an honest earlier-window→later-window holdout check."""
    from sports.baseball.analytics.backtest import fit_serve_calibrator
    from sports.baseball.analytics.serve_calibration import save

    games = await client.fetch_schedule_range(date(season, 3, 1), date(season, 11, 1))
    rate, rated = await _pitcher_rating_fn(season, games)
    fit = fit_serve_calibrator(games, min_games=min_games, rate_pitcher=rate)
    if not fit.get("fitted"):
        return {"ok": False, **fit}
    artifact = {
        "method": fit["method"],
        "params": fit["params"],
        "enabled": bool(enable),
        "fit_season": season,
        "fit_n": fit["n"],
        "pitchers_rated": rated,
        "fit_brier_raw": fit["fit_brier_raw"],
        "fit_brier_cal": fit["fit_brier_cal"],
        "holdout": fit.get("holdout"),
    }
    state = save(artifact)
    return {"ok": True, "fit": fit, "state": state}


@router.post("/calibration/enable")
async def enable_calibration(on: bool = True):
    """Enable/disable the fitted serve-calibrator without refitting."""
    from sports.baseball.analytics.serve_calibration import set_enabled
    return {"ok": True, **set_enabled(on)}


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
    from sports.baseball.analytics.pitcher_form import fetch_pitcher_logs, rating_asof
    prior = await _prior_season_era_map(season)
    ids = {i for i in (game.home_pitcher_id, game.away_pitcher_id) if i}
    logs = await fetch_pitcher_logs(client, ids, season)
    h_rating = rating_asof(logs.get(game.home_pitcher_id), game.date, prior.get(game.home_pitcher_id)) if game.home_pitcher_id else None
    a_rating = rating_asof(logs.get(game.away_pitcher_id), game.date, prior.get(game.away_pitcher_id)) if game.away_pitcher_id else None
    pred = predict_game(
        home_state, away_state,
        home_pitcher=game.home_pitcher, away_pitcher=game.away_pitcher,
        home_pitcher_era=h_rating, away_pitcher_era=a_rating,
    )
    _apply_serve_calibration(pred)
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
    rate, rated = await _pitcher_rating_fn(season, games)       # leak-free within-season FIP
    result = _run(games, min_games=min_games, calibrate=calibrate, rate_pitcher=rate)
    result["pitcher_signal"] = "within-season FIP (shrunk toward prior-season ERA)"
    result["pitchers_rated"] = rated
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
    from sports.baseball.analytics.serve_calibration import state as calib_state
    teams = client.get_teams()
    standings = client.get_standings()
    schedule = client.get_schedule()
    reachable = len(teams) > 0
    calib = calib_state()

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
            "version": "mlb-decomp-v2",
            "components": [
                {"name": "team_strength", "modeled": True},
                {"name": "home_field", "modeled": True},
                {"name": "recent_form", "modeled": True},
                {"name": "starting_pitcher", "modeled": True},
            ],
            "calibration": (
                f"Platt applied at serve (fit {calib.get('fit_season')}, n={calib.get('fit_n')})"
                if calib.get("enabled") else
                "Platt fitted but disabled at serve" if calib.get("fitted") else
                "Platt available in backtest; not yet fitted for serve"
            ),
            "calibrator": calib,
        },
        "backtest": {"available": True, "endpoint": "/api/baseball/backtest", "kind": "leak-free walk-forward"},
        "notes": [
            "Data is free and real-time (MLB StatsAPI); no paid feed required.",
            "Starting pitcher now modeled via within-season FIP (shrunk toward prior-season ERA).",
            ("Serve probabilities are Platt-calibrated." if calib.get("enabled")
             else "Fit a serve calibrator via POST /api/baseball/calibration/fit?enable=true."),
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
