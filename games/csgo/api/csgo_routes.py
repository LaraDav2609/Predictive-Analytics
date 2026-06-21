"""CSGO esports API routes."""

from typing import Optional

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from games.csgo.data.csgo_client import CsgoDataClient
from games.csgo.analytics.csgo_predictor import CsgoPredictor
from games.csgo.gsi import gsi_to_live_state
from games.csgo.identity import normalize
from games.csgo.live import LiveMatchState, publish_live, update_live_probability

router = APIRouter(prefix="/csgo", tags=["csgo"])

client: Optional[CsgoDataClient] = None
predictor: Optional[CsgoPredictor] = None

# Live (GSI) state — _publisher lights up the csgo:prob:* bridge when set by the server.
_publisher = None
_gsi_latest: dict = {}


def init(cc: CsgoDataClient, cp: CsgoPredictor):
    global client, predictor
    client, predictor = cc, cp


def set_publisher(pub) -> None:
    """Server hook: pass an OutcomePublisher to push live GSI updates to Redis."""
    global _publisher
    _publisher = pub


def _resolve_pregame(ct_name, t_name):
    """Anchor the live state to a known upcoming match (canonical team1 + its pre-game prob)."""
    names = {normalize(ct_name or ""), normalize(t_name or "")}
    if client is not None and predictor is not None and "" not in names:
        for m in client.get_matches():
            if {normalize(m.team1), normalize(m.team2)} == names:
                pred = m.prediction
                if pred is None and hasattr(predictor, "predict"):
                    try:
                        pred = predictor.predict(m)
                    except Exception:
                        pred = None
                if pred is not None:
                    return (pred.team1_win_prob, m.id, m.team1,
                            m.team1_abbrev or m.team1, m.team2_abbrev or m.team2)
    t1, t2 = (ct_name or "CT"), (t_name or "T")
    return (0.5, f"gsi-{normalize(t1)}-{normalize(t2)}", t1, t1, t2)


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
    return {"ok": True,
            "provider": getattr(client, "provider_name", "stub"),
            "synthetic": getattr(client, "is_synthetic", True),
            **result}


@router.get("/source")
async def source():
    """Active data provenance — which provider is live and whether it's synthetic — so the
    dashboard can flag stub numbers instead of letting them pass for real results."""
    return {"ok": True,
            "provider": getattr(client, "provider_name", "stub"),
            "synthetic": getattr(client, "is_synthetic", True),
            "history_matches": len(client.get_past_matches()),
            "teams": len(client.get_teams())}


@router.get("/replay/{match_id}")
async def replay(match_id: str):
    """Single-game replay (click-through from the Backtest tab): the leak-free pre-game
    prediction for one finished match, the actual result, and the map-by-map series
    win-probability trajectory the live model would have shown."""
    from games.csgo.analytics.backtest import replay_match

    past = client.get_past_matches()
    teams_by_id = {t.id: t for t in client.get_teams()}
    return replay_match(match_id, past, teams_by_id=teams_by_id)


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


@router.post("/gsi")
async def gsi(payload: dict = Body(...)):
    """Ingest a CS2 Game State Integration payload → live win probability.

    Point CS2's gamestate_integration_*.cfg at this endpoint. Resolves the canonical
    team1 (and its pre-game prob) from the upcoming-match prediction when possible,
    folds in live state, stores the latest, and publishes to csgo:prob:* if a
    publisher is configured (Redis optional).
    """
    mp = payload.get("map") or {}
    ct_name = (mp.get("team_ct") or {}).get("name")
    t_name = (mp.get("team_t") or {}).get("name")
    pregame, match_id, team1_name, code1, code2 = _resolve_pregame(ct_name, t_name)
    state, name1, name2 = gsi_to_live_state(payload, team1_name=team1_name)
    live = update_live_probability(pregame, state)

    global _gsi_latest
    _gsi_latest = {
        "match_id": match_id, "team1": name1, "team2": name2,
        "pregame_team1_prob": round(pregame, 4),
        "live": live.model_dump(), "state": state.model_dump(),
    }
    if _publisher is not None:
        try:
            publish_live(match_id, code1, code2, live, _publisher)
        except Exception:  # Redis unavailable — still serve the HTTP response
            pass
    return {"ok": True, **_gsi_latest}


@router.get("/gsi/live")
async def gsi_live():
    if not _gsi_latest:
        return {"ok": True, "live": None, "note": "no GSI updates received yet"}
    return {"ok": True, **_gsi_latest}
