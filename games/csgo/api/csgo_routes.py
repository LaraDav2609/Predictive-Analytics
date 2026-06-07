"""CSGO esports API routes."""

from typing import Optional

from fastapi import APIRouter, HTTPException

from games.csgo.data.csgo_client import CsgoDataClient
from games.csgo.analytics.csgo_predictor import CsgoPredictor

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


@router.post("/refresh")
async def refresh():
    await client.refresh()
    if predictor:
        predictor.load_teams(client.get_teams())
    return {"ok": True, "teams": len(client.get_teams()), "matches": len(client.get_matches())}
