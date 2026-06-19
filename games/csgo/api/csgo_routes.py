"""CSGO esports API routes."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from games.csgo.data.csgo_client import CsgoDataClient
from games.csgo.analytics.csgo_predictor import CsgoPredictor
from games.csgo.live import LiveMatchState, update_live_probability

router = APIRouter(prefix="/csgo", tags=["csgo"])

client: Optional[CsgoDataClient] = None
predictor: Optional[CsgoPredictor] = None


def init(cc: CsgoDataClient, cp: CsgoPredictor):
    global client, predictor
    client, predictor = cc, cp


@router.get("/teams")
async def get_teams():
    return {"ok": True, "teams": [t.model_dump() for t in client.get_teams()]}


@router.get("/matches")
async def get_matches():
    matches = client.get_matches()
    if predictor:
        matches = predictor.predict_matches(matches)
    return {"ok": True, "matches": [m.model_dump(mode="json") for m in matches]}


@router.get("/matches/{match_id}")
async def get_match(match_id: str):
    match = client.get_match(match_id)
    if not match:
        raise HTTPException(status_code=404, detail=f"Match {match_id} not found")
    matches = [match]
    if predictor:
        matches = predictor.predict_matches(matches)
    return {"ok": True, "match": matches[0].model_dump(mode="json")}


@router.get("/teams/{team_id}")
async def get_team(team_id: int):
    team = client.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
    return {"ok": True, "team": team.model_dump(),
            "players": [p.model_dump() for p in client.get_players(team_id)]}


@router.get("/teams/{team_id}/players")
async def get_team_players(team_id: int):
    return {"ok": True, "players": [p.model_dump() for p in client.get_players(team_id)]}


@router.get("/players/{player_id}")
async def get_player(player_id: int):
    player = client.get_player(player_id)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player {player_id} not found")
    return {"ok": True, "player": player.model_dump()}


@router.post("/refresh")
async def refresh():
    await client.refresh()
    if predictor:
        if hasattr(predictor, "fit"):
            predictor.fit(client.get_past_matches(), client.get_teams())   # full ensemble pipeline
        else:
            predictor.load_teams(client.get_teams())
    return {"ok": True, "teams": len(client.get_teams()), "matches": len(client.get_matches())}


@router.get("/backtest")
async def backtest(calibrate: bool = False):
    """Walk-forward, leak-free backtest of the pre-game model over finished matches.

    Returns the dashboard Backtest-tab contract (matches/brier/log_loss/rows) plus
    richer calibration metrics (reliability buckets, favorite/underdog + BO splits).
    Pass ?calibrate=true to also fit a Platt scaler on an earlier window and report
    raw-vs-calibrated Brier/log-loss on the held-out remainder.
    """
    from games.csgo.analytics.backtest import run_backtest

    past = client.get_past_matches()
    teams_by_id = {t.id: t for t in client.get_teams()}
    result = run_backtest(past, teams_by_id=teams_by_id, calibrate=calibrate)
    return {"ok": True, **result}


class LiveUpdateRequest(BaseModel):
    pregame_team1_prob: float
    state: LiveMatchState
    match_id: str = ""
    pregame_per_map_prob: Optional[float] = None


@router.post("/live/update")
async def live_update(req: LiveUpdateRequest):
    """Update a live win probability from current in-game state (transparent heuristic)."""
    live = update_live_probability(req.pregame_team1_prob, req.state, req.pregame_per_map_prob)
    return {"ok": True, "match_id": req.match_id, "live": live.model_dump()}
