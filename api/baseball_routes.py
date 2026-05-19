"""MLB Baseball API routes."""

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException

from data.mlb_client import MLBClient
from analytics.baseball_predictor import BaseballPredictor

router = APIRouter(prefix="/baseball", tags=["baseball"])

client: Optional[MLBClient] = None
predictor: Optional[BaseballPredictor] = None


def init(mc: MLBClient, bp: BaseballPredictor):
    global client, predictor
    client, predictor = mc, bp


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
