"""Soccer API routes."""

from typing import Optional

from fastapi import APIRouter, Query

from data.football_data_client import FootballDataClient
from analytics.soccer_predictor import SoccerPredictor

router = APIRouter(prefix="/soccer", tags=["soccer"])

# Shared instances (initialized in server.py lifespan)
client: Optional[FootballDataClient] = None
predictor: Optional[SoccerPredictor] = None


def init(fc: FootballDataClient, sp: SoccerPredictor):
    global client, predictor
    client, predictor = fc, sp


@router.get("/teams")
async def get_teams():
    teams = client.get_teams()
    return {"ok": True, "teams": [t.model_dump() for t in teams]}


@router.get("/matches")
async def get_matches(filter: str = Query("all", pattern="^(all|upcoming|past)$")):
    matches = client.get_matches(filter_status=filter if filter != "all" else None)
    # Attach predictions to scheduled matches
    if predictor:
        matches = predictor.predict_matches(matches)
    return {"ok": True, "matches": [m.model_dump(mode="json") for m in matches]}


@router.get("/matches/{match_id}")
async def get_match(match_id: int):
    match = client.get_match_by_id(match_id)
    if not match:
        return {"ok": False, "reason": "Match not found"}
    if match.status in ("SCHEDULED", "TIMED") and predictor:
        match.prediction = predictor.predict(match)
    return {"ok": True, "match": match.model_dump(mode="json")}


@router.get("/standings")
async def get_standings():
    standings = client.get_standings()
    return {"ok": True, "standings": standings}


@router.post("/refresh")
async def refresh():
    await client.refresh()
    if predictor:
        predictor.load_teams(client.get_teams())
    return {"ok": True, "teams": len(client.get_teams()), "matches": len(client.get_matches())}
