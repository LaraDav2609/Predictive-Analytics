"""Historical MLB routes backed by the free MLB Stats API."""

from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from sports.baseball.data.mlb_historical_client import DEFAULT_STAT_GROUPS, MLBHistoricalClient
from sports.baseball.data.pybaseball_client import PybaseballClient

router = APIRouter(prefix="/baseball/history", tags=["baseball-history"])

client: Optional[MLBHistoricalClient] = None
advanced_client: Optional[PybaseballClient] = None


def init(historical_client: MLBHistoricalClient, pybaseball_client: Optional[PybaseballClient] = None):
    global client, advanced_client
    client = historical_client
    advanced_client = pybaseball_client


def _parse_groups(group: Optional[str]) -> list[str]:
    if not group:
        return DEFAULT_STAT_GROUPS
    return [part.strip() for part in group.split(",") if part.strip()]


@router.get("/teams/{team_id}")
async def get_team_history(
    team_id: int,
    start_season: Optional[int] = None,
    end_season: Optional[int] = None,
):
    history = await client.get_team_history(team_id, start_season, end_season)
    return {
        "ok": True,
        "team_id": team_id,
        "start_season": start_season,
        "end_season": end_season,
        "seasons": [season.model_dump() for season in history],
    }


@router.get("/teams/{team_id}/roster")
async def get_team_roster(
    team_id: int,
    season: int,
    roster_type: str = Query(default="fullSeason"),
):
    roster = await client.get_team_roster(team_id, season, roster_type)
    return {"ok": True, **roster.model_dump()}


@router.get("/teams/{team_id}/stats")
async def get_team_stats(
    team_id: int,
    season: int,
    group: Optional[str] = Query(default=None, description="Comma-separated: hitting,pitching,fielding"),
    stats_type: str = Query(default="season"),
):
    stat_groups = _parse_groups(group)
    stats = await client.get_team_stats(team_id, season, stat_groups, stats_type)
    return {
        "ok": True,
        "team_id": team_id,
        "season": season,
        "groups": stat_groups,
        "stats_type": stats_type,
        "splits": [split.model_dump() for split in stats],
    }


@router.get("/players/{player_id}")
async def get_player_profile(
    player_id: int,
    season: Optional[int] = None,
):
    try:
        player = await client.get_player_profile(player_id, season)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "player": player.model_dump()}


@router.get("/players/{player_id}/stats")
async def get_player_stats(
    player_id: int,
    group: Optional[str] = Query(default=None, description="Comma-separated: hitting,pitching,fielding"),
    stat_type: str = Query(default="yearByYear"),
    season: Optional[int] = None,
):
    stat_groups = _parse_groups(group)
    try:
        stats = await client.get_player_year_by_year_stats(player_id, stat_groups, stat_type, season)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, **stats.model_dump()}


@router.get("/players/{player_id}/advanced")
async def get_player_advanced(
    player_id: int,
    years: int = Query(default=5, ge=1, le=15, description="How many recent seasons to fetch"),
):
    """Advanced FanGraphs metrics via pybaseball: WAR, wRC+, wOBA, FIP, etc.

    Returns empty arrays if pybaseball isn't installed or the player isn't on FanGraphs.
    """
    if advanced_client is None or not advanced_client.is_available():
        return {
            "ok": True,
            "available": False,
            "reason": "pybaseball not installed",
            "hitting": [],
            "pitching": [],
        }
    current = date.today().year
    year_list = list(range(current - years + 1, current + 1))
    hitting = advanced_client.get_advanced_batting(player_id, year_list)
    pitching = advanced_client.get_advanced_pitching(player_id, year_list)
    return {
        "ok": True,
        "available": True,
        "player_id": player_id,
        "years": year_list,
        "hitting": hitting,
        "pitching": pitching,
    }
