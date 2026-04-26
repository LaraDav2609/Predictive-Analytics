"""Historical MLB routes backed by the free MLB Stats API."""

from fastapi import APIRouter, HTTPException, Query

from data.mlb_historical_client import DEFAULT_STAT_GROUPS, MLBHistoricalClient

router = APIRouter(prefix="/baseball/history", tags=["baseball-history"])

client: MLBHistoricalClient | None = None


def init(historical_client: MLBHistoricalClient):
    global client
    client = historical_client


def _parse_groups(group: str | None) -> list[str]:
    if not group:
        return DEFAULT_STAT_GROUPS
    return [part.strip() for part in group.split(",") if part.strip()]


@router.get("/teams/{team_id}")
async def get_team_history(
    team_id: int,
    start_season: int | None = None,
    end_season: int | None = None,
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
    group: str | None = Query(default=None, description="Comma-separated: hitting,pitching,fielding"),
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
    season: int | None = None,
):
    try:
        player = await client.get_player_profile(player_id, season)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "player": player.model_dump()}


@router.get("/players/{player_id}/stats")
async def get_player_stats(
    player_id: int,
    group: str | None = Query(default=None, description="Comma-separated: hitting,pitching,fielding"),
    stat_type: str = Query(default="yearByYear"),
    season: int | None = None,
):
    stat_groups = _parse_groups(group)
    try:
        stats = await client.get_player_year_by_year_stats(player_id, stat_groups, stat_type, season)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, **stats.model_dump()}
