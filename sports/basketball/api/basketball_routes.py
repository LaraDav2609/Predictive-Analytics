"""League-neutral basketball endpoints backed by ESPN's public score feeds."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
import asyncio
import unicodedata
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Query
from sports.basketball.analytics.game_model import predict_game

router = APIRouter(prefix="/basketball", tags=["basketball"])

LEAGUES: dict[str, dict[str, Any]] = {
    "nba": {
        "name": "NBA", "full_name": "National Basketball Association",
        "provider_slug": "nba", "period_label": "Quarter", "game_minutes": 48,
        "accent": "#ff6b24", "home_edge": 0.04,
    },
    "wnba": {
        "name": "WNBA", "full_name": "Women's National Basketball Association",
        "provider_slug": "wnba", "period_label": "Quarter", "game_minutes": 40,
        "accent": "#35d0ba", "home_edge": 0.045,
    },
    "ncaam": {
        "name": "NCAA Men", "full_name": "NCAA Division I Men's Basketball",
        "provider_slug": "mens-college-basketball", "period_label": "Half", "game_minutes": 40,
        "accent": "#4f8cff", "home_edge": 0.065,
    },
    "ncaaw": {
        "name": "NCAA Women", "full_name": "NCAA Division I Women's Basketball",
        "provider_slug": "womens-college-basketball", "period_label": "Quarter", "game_minutes": 40,
        "accent": "#b877ff", "home_edge": 0.06,
    },
    "summer": {
        "name": "Summer League", "full_name": "NBA Las Vegas Summer League",
        "provider_slug": "nba-summer-las-vegas", "period_label": "Quarter", "game_minutes": 40,
        "accent": "#ffd166", "home_edge": 0.0,
    },
}

BASE = "https://site.api.espn.com/apis"
GLEAGUE_STATS_BASE = "https://stats.gleague.nba.com/stats"
_gleague_directory_cache: list[dict[str, Any]] | None = None
_gleague_directory_lock = asyncio.Lock()


def _league_or_404(league: str) -> dict[str, Any]:
    config = LEAGUES.get(league.lower())
    if config is None:
        raise HTTPException(404, f"Unknown basketball league '{league}'")
    return config


async def _get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    # Wikimedia requires an identifying User-Agent; this value also remains
    # acceptable to ESPN and the other public JSON providers used here.
    headers = {"User-Agent": "PredictiveBasketball/1.0 (https://github.com/activetrader/Predictive-Analytics)"}
    async with httpx.AsyncClient(timeout=20, headers=headers, follow_redirects=True) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def _get_gleague_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://gleague.nba.com/",
        "Origin": "https://gleague.nba.com",
    }
    async with httpx.AsyncClient(timeout=25, headers=headers, follow_redirects=True) as client:
        response = await client.get(f"{GLEAGUE_STATS_BASE}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()


def _nba_result_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result_sets = payload.get("resultSets") or []
    if not result_sets:
        return []
    result = result_sets[0]
    headers = result.get("headers") or []
    return [dict(zip(headers, row)) for row in result.get("rowSet") or []]


async def _gleague_directory() -> list[dict[str, Any]]:
    global _gleague_directory_cache
    if _gleague_directory_cache is not None:
        return _gleague_directory_cache
    async with _gleague_directory_lock:
        if _gleague_directory_cache is None:
            payload = await _get_gleague_json("commonallplayers", {
                "LeagueID": "20", "Season": "2025-26", "IsOnlyCurrentSeason": "0",
            })
            _gleague_directory_cache = _nba_result_rows(payload)
    return _gleague_directory_cache


def _team(raw: dict[str, Any]) -> dict[str, Any]:
    logos = raw.get("logos") or []
    logo = raw.get("logo")
    if isinstance(logo, dict):
        logo = logo.get("href")
    if not logo and logos:
        logo = logos[0].get("href")
    return {
        "id": raw.get("id"),
        "name": raw.get("displayName") or raw.get("name"),
        "short_name": raw.get("shortDisplayName") or raw.get("name"),
        "abbreviation": raw.get("abbreviation"),
        "location": raw.get("location"),
        "color": raw.get("color"),
        "alternate_color": raw.get("alternateColor"),
        "logo": logo,
    }


def _player(raw: dict[str, Any]) -> dict[str, Any]:
    headshot = raw.get("headshot") or {}
    position = raw.get("position") or {}
    experience = raw.get("experience") or {}
    college = raw.get("college") or {}
    birth_place = raw.get("birthPlace") or {}
    return {
        "id": raw.get("id"),
        "name": raw.get("displayName") or raw.get("fullName"),
        "short_name": raw.get("shortName"),
        "first_name": raw.get("firstName"),
        "last_name": raw.get("lastName"),
        "headshot": headshot.get("href"),
        "jersey": raw.get("jersey"),
        "position": position.get("displayName") or position.get("name"),
        "position_abbreviation": position.get("abbreviation"),
        "height": raw.get("displayHeight"),
        "weight": raw.get("displayWeight"),
        "age": raw.get("age"),
        "experience_years": experience.get("years"),
        "college": college.get("name"),
        "birth_country": birth_place.get("country"),
        "status": (raw.get("status") or {}).get("name"),
    }


def _same_person(left: Any, right: Any) -> bool:
    def normalized(value: Any) -> str:
        text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
        return " ".join(text.lower().split())
    return normalized(left) == normalized(right)


async def _enrich_player_from_core(player: dict[str, Any], league: str) -> dict[str, Any]:
    """Fill roster omissions using ESPN's identity-linked athlete records."""
    if player.get("headshot") and player.get("college"):
        return player
    if league.lower() in {"wnba", "ncaaw"}:
        slugs = ["womens-college-basketball", "wnba"]
    else:
        slugs = ["mens-college-basketball", "nba"]
    enriched = dict(player)
    for slug in slugs:
        url = f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/{slug}/athletes/{player.get('id')}"
        try:
            profile = await _get_json(url, {"lang": "en", "region": "us"})
        except httpx.HTTPError:
            continue
        if not _same_person(profile.get("fullName") or profile.get("displayName"), player.get("name")):
            continue
        birth_place = profile.get("birthPlace") or {}
        if not enriched.get("birth_country") and birth_place.get("country"):
            enriched["birth_country"] = birth_place["country"]
        headshot = profile.get("headshot") or {}
        if not enriched.get("headshot") and headshot.get("href"):
            enriched["headshot"] = headshot["href"]
            enriched["headshot_source"] = f"ESPN {slug} athlete profile"
        team_ref = (profile.get("team") or {}).get("$ref")
        if not enriched.get("college") and team_ref and "college-basketball" in slug:
            try:
                college_team = await _get_json(team_ref.replace("http://", "https://"))
                enriched["college"] = college_team.get("displayName") or college_team.get("name")
                if enriched.get("college"):
                    enriched["college_source"] = f"ESPN {slug} athlete profile"
            except httpx.HTTPError:
                pass
        if enriched.get("headshot") and enriched.get("college"):
            break
    return enriched


async def _enrich_player_from_gleague(player: dict[str, Any]) -> dict[str, Any]:
    """Fill gaps from the official NBA G League identity and profile feeds."""
    if player.get("headshot") and player.get("college"):
        return player
    try:
        directory = await _gleague_directory()
        match = next((row for row in directory
                      if _same_person(row.get("DISPLAY_FIRST_LAST"), player.get("name"))), None)
        if not match:
            return player
        person_id = match.get("PERSON_ID")
        payload = await _get_gleague_json("commonplayerinfo", {
            "LeagueID": "20", "PlayerID": str(person_id),
        })
        profile = next(iter(_nba_result_rows(payload)), {})
        if not _same_person(profile.get("DISPLAY_FIRST_LAST"), player.get("name")):
            return player
        enriched = dict(player)
        if not enriched.get("headshot") and person_id:
            enriched["headshot"] = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{person_id}.png"
            enriched["headshot_source"] = "NBA G League player profile"
        if not enriched.get("college") and profile.get("SCHOOL"):
            enriched["college"] = profile["SCHOOL"]
            enriched["college_source"] = "NBA G League player profile"
        if not enriched.get("birth_country") and profile.get("COUNTRY"):
            enriched["birth_country"] = profile["COUNTRY"]
        return enriched
    except (httpx.HTTPError, KeyError, TypeError, ValueError):
        return player


async def _enrich_player_from_alternatives(player: dict[str, Any]) -> dict[str, Any]:
    """Use exact-name basketball sources for omissions and expanded school names."""
    if player.get("headshot") and player.get("college") and " " in str(player.get("college")):
        return player
    alternative = await _alternative_player_data(str(player.get("name") or ""))
    enriched = dict(player)
    if not enriched.get("headshot") and alternative.get("photo"):
        enriched["headshot"] = alternative["photo"]
        enriched["headshot_source"] = alternative.get("source")
    alternative_college = alternative.get("college")
    current_college = str(enriched.get("college") or "")
    if alternative_college and (not current_college or current_college.lower() in alternative_college.lower()):
        enriched["college"] = alternative_college
        enriched["college_source"] = alternative.get("source")
    if not enriched.get("birth_country") and alternative.get("birth_country"):
        enriched["birth_country"] = alternative["birth_country"]
    return enriched


async def _alternative_player_data(name: str) -> dict[str, Any]:
    """Use exact basketball identities only; never accept a same-name athlete."""
    photo = college = birth_country = source = None
    try:
        payload = await _get_json("https://www.thesportsdb.com/api/v1/json/123/searchplayers.php", {"p": name})
        match = next((p for p in payload.get("player") or []
                      if _same_person(p.get("strPlayer"), name)
                      and str(p.get("strSport") or "").lower() == "basketball"), None)
        if match:
            photo = match.get("strCutout") or match.get("strThumb")
            college = match.get("strCollege")
            if photo or college:
                source = "TheSportsDB"
    except httpx.HTTPError:
        pass
    try:
        search = await _get_json("https://www.wikidata.org/w/api.php", {
            "action": "wbsearchentities", "search": name, "language": "en", "format": "json", "limit": 8,
        })
        entity_match = next((item for item in search.get("search") or []
                             if _same_person(item.get("label"), name)
                             and "basketball player" in str(item.get("description") or "").lower()), None)
        if entity_match:
            entity_id = entity_match["id"]
            detail = await _get_json("https://www.wikidata.org/w/api.php", {
                "action": "wbgetentities", "ids": entity_id, "props": "claims", "format": "json",
            })
            claims = ((detail.get("entities") or {}).get(entity_id) or {}).get("claims") or {}
            image_name = (((claims.get("P18") or [{}])[0].get("mainsnak") or {}).get("datavalue") or {}).get("value")
            if not photo and image_name:
                photo = f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{quote(str(image_name))}?width=700"
                source = "Wikimedia Commons"
            education_ids = [((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value", {}).get("id")
                             for claim in claims.get("P69") or []]
            education_ids = [item for item in education_ids if item]
            if not college and education_ids:
                labels = await _get_json("https://www.wikidata.org/w/api.php", {
                    "action": "wbgetentities", "ids": "|".join(education_ids), "props": "labels",
                    "languages": "en", "format": "json",
                })
                college_names = [((labels.get("entities") or {}).get(item) or {}).get("labels", {}).get("en", {}).get("value")
                                 for item in education_ids]
                college = ", ".join(item for item in college_names if item) or None
                if college and not source:
                    source = "Wikidata"
            country_ids = [((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value", {}).get("id")
                           for claim in claims.get("P27") or []]
            country_ids = [item for item in country_ids if item]
            if country_ids:
                labels = await _get_json("https://www.wikidata.org/w/api.php", {
                    "action": "wbgetentities", "ids": "|".join(country_ids), "props": "labels",
                    "languages": "en", "format": "json",
                })
                country_names = [((labels.get("entities") or {}).get(item) or {}).get("labels", {}).get("en", {}).get("value")
                                 for item in country_ids]
                birth_country = ", ".join(item for item in country_names if item) or None
    except (httpx.HTTPError, TypeError, ValueError):
        pass
    return {"photo": photo, "college": college, "birth_country": birth_country, "source": source}


def _record_value(competitor: dict[str, Any]) -> tuple[str | None, float | None]:
    records = competitor.get("records") or []
    summary = records[0].get("summary") if records else None
    if not summary or "-" not in summary:
        return summary, None
    try:
        wins, losses = (int(part) for part in summary.split("-")[:2])
        total = wins + losses
        return summary, wins / total if total else None
    except (TypeError, ValueError):
        return summary, None


def _prediction(home: dict[str, Any], away: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    _, home_pct = _record_value(home)
    _, away_pct = _record_value(away)
    if home_pct is None or away_pct is None:
        base = 0.5 + float(config["home_edge"])
        confidence = "low"
    else:
        base = 0.5 + (home_pct - away_pct) * 0.38 + float(config["home_edge"])
        confidence = "medium" if abs(home_pct - away_pct) >= 0.08 else "low"
    home_probability = min(0.88, max(0.12, base))
    return {
        "home_win_probability": round(home_probability, 4),
        "away_win_probability": round(1 - home_probability, 4),
        "pick": "home" if home_probability >= 0.5 else "away",
        "confidence": confidence,
        "model": "record-strength + league home-court baseline",
    }


def _normalize_event(event: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    competition = (event.get("competitions") or [{}])[0]
    competitors = competition.get("competitors") or []
    home = next((item for item in competitors if item.get("homeAway") == "home"), {})
    away = next((item for item in competitors if item.get("homeAway") == "away"), {})
    home_record, _ = _record_value(home)
    away_record, _ = _record_value(away)
    status = event.get("status") or {}
    status_type = status.get("type") or {}
    venue = competition.get("venue") or {}
    broadcasts = competition.get("broadcasts") or []
    return {
        "id": event.get("id"),
        "name": event.get("shortName") or event.get("name"),
        "date": event.get("date"),
        "status": status_type.get("state", "pre"),
        "status_detail": status_type.get("shortDetail") or status_type.get("detail"),
        "period": status.get("period"),
        "clock": status.get("displayClock"),
        "home": {**_team(home.get("team") or {}), "score": home.get("score"), "record": home_record},
        "away": {**_team(away.get("team") or {}), "score": away.get("score"), "record": away_record},
        "venue": venue.get("fullName"),
        "neutral_site": bool(competition.get("neutralSite")),
        "broadcast": ", ".join((broadcasts[0].get("names") or [])) if broadcasts else None,
        "prediction": _prediction(home, away, config),
    }


def _flatten_standings(node: dict[str, Any], groups: list[dict[str, Any]]) -> None:
    standings = node.get("standings") or {}
    entries = standings.get("entries") or []
    if entries:
        rows = []
        for entry in entries:
            stats = {stat.get("name"): stat.get("displayValue") for stat in entry.get("stats") or []}
            rows.append({
                "team": _team(entry.get("team") or {}),
                "wins": stats.get("wins"), "losses": stats.get("losses"),
                "win_percent": stats.get("winPercent"), "games_behind": stats.get("gamesBehind"),
                "streak": stats.get("streak"), "point_differential": stats.get("differential"),
                "points_for": stats.get("avgPointsFor"), "points_against": stats.get("avgPointsAgainst"),
            })
        groups.append({"name": node.get("name") or standings.get("name") or "Standings", "entries": rows})
    for child in node.get("children") or []:
        _flatten_standings(child, groups)


def _team_group_map(node: dict[str, Any], result: dict[str, str]) -> None:
    """Map team IDs to the nearest published standings group (conference/league)."""
    entries = (node.get("standings") or {}).get("entries") or []
    group_name = node.get("name") or (node.get("standings") or {}).get("name")
    if entries and group_name:
        for entry in entries:
            team_id = str((entry.get("team") or {}).get("id") or "")
            if team_id:
                result.setdefault(team_id, group_name)
    for child in node.get("children") or []:
        _team_group_map(child, result)


@router.get("/leagues")
async def leagues():
    return {"ok": True, "leagues": [{"id": key, **value} for key, value in LEAGUES.items()]}


@router.get("/{league}/scoreboard")
async def scoreboard(
    league: str,
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
):
    config = _league_or_404(league)
    start = start_date or date.today()
    end = end_date or (start + timedelta(days=7))
    dates = start.strftime("%Y%m%d") if start == end else f"{start:%Y%m%d}-{end:%Y%m%d}"
    url = f"{BASE}/site/v2/sports/basketball/{config['provider_slug']}/scoreboard"
    try:
        payload = await _get_json(url, {"dates": dates, "limit": 300})
    except httpx.HTTPStatusError as exc:
        # ESPN returns 404 for a valid league when a requested offseason range has
        # no scoreboard resource. Treat that as an empty slate, not a dead API.
        if exc.response.status_code != 404:
            raise
        payload = {"events": []}
    games = [_normalize_event(event, config) for event in payload.get("events") or []]
    return {"ok": True, "league": league, "start_date": str(start), "end_date": str(end), "games": games}


@router.get("/{league}/games/{game_id}")
async def game_detail(league: str, game_id: str):
    config = _league_or_404(league)
    url = f"{BASE}/site/v2/sports/basketball/{config['provider_slug']}/summary"
    payload = await _get_json(url, {"event": game_id})
    competition = ((payload.get("header") or {}).get("competitions") or [{}])[0]
    competitors = competition.get("competitors") or []
    home_raw = next((item for item in competitors if item.get("homeAway") == "home"), {})
    away_raw = next((item for item in competitors if item.get("homeAway") == "away"), {})

    def detail_team(raw: dict[str, Any]) -> dict[str, Any]:
        team = _team(raw.get("team") or {})
        records = raw.get("record") or raw.get("records") or []
        record = records[0].get("displayValue") or records[0].get("summary") if records else None
        return {**team, "score": raw.get("score"), "record": record,
                "linescores": [line.get("displayValue") for line in raw.get("linescores") or []]}

    # Adapt the summary's singular `record` field for the shared predictor.
    pred_home = {**home_raw, "records": [{"summary": detail_team(home_raw).get("record")}]}
    pred_away = {**away_raw, "records": [{"summary": detail_team(away_raw).get("record")}]}
    status = competition.get("status") or {}
    status_type = status.get("type") or {}
    venue = ((payload.get("gameInfo") or {}).get("venue") or {})

    game_date = competition.get("date")
    parsed_game_date = date.today().year
    try:
        parsed_game_date = int(str(game_date)[:4])
    except (TypeError, ValueError):
        pass
    home_id = str((home_raw.get("team") or {}).get("id") or "")
    away_id = str((away_raw.get("team") or {}).get("id") or "")
    detail_base = f"{BASE}/site/v2/sports/basketball/{config['provider_slug']}/teams"
    context_results = await asyncio.gather(
        _get_json(f"{detail_base}/{home_id}/schedule", {"season": parsed_game_date}),
        _get_json(f"{detail_base}/{away_id}/schedule", {"season": parsed_game_date}),
        _get_json(f"{detail_base}/{home_id}/statistics", {"season": parsed_game_date}),
        _get_json(f"{detail_base}/{away_id}/statistics", {"season": parsed_game_date}),
        return_exceptions=True,
    )
    home_schedule, away_schedule, home_season_stats, away_season_stats = [
        {} if isinstance(item, Exception) else item for item in context_results
    ]
    # ESPN's season aggregate is current rather than point-in-time. It is valid
    # for a future pregame prediction, but would leak information into a replay
    # of an in-progress/completed game. Recent schedule features are always cut
    # off at the target tip time and remain safe in every state.
    is_pregame = status_type.get("state") == "pre"
    enhanced_prediction = predict_game(
        home=detail_team(home_raw), away=detail_team(away_raw),
        home_schedule=home_schedule, away_schedule=away_schedule,
        home_stats=home_season_stats if is_pregame else {},
        away_stats=away_season_stats if is_pregame else {},
        game_date=game_date, home_edge=float(config["home_edge"]),
        neutral_site=bool(competition.get("neutralSite")),
    )

    team_stats = []
    for item in (payload.get("boxscore") or {}).get("teams") or []:
        stats = {stat.get("name"): stat.get("displayValue") for stat in item.get("statistics") or []}
        team_stats.append({"team": _team(item.get("team") or {}), "stats": stats})

    leaders = []
    for team_group in payload.get("leaders") or []:
        for category in team_group.get("leaders") or []:
            leader = (category.get("leaders") or [{}])[0]
            athlete = leader.get("athlete") or {}
            leaders.append({
                "team": _team(team_group.get("team") or {}),
                "category": category.get("displayName") or category.get("name"),
                "value": leader.get("displayValue"),
                "summary": leader.get("summary"),
                "player": _player(athlete),
            })

    odds = (payload.get("pickcenter") or payload.get("odds") or [])
    return {
        "ok": True, "league": league, "id": game_id, "date": competition.get("date"),
        "status": status_type.get("state"), "status_detail": status_type.get("shortDetail") or status_type.get("detail"),
        "period": status.get("period"), "clock": status.get("displayClock"),
        "home": detail_team(home_raw), "away": detail_team(away_raw),
        "venue": venue.get("fullName"), "location": (venue.get("address") or {}),
        "prediction": enhanced_prediction,
        "team_stats": team_stats, "leaders": leaders, "odds": odds,
    }


@router.get("/{league}/teams")
async def teams(league: str):
    config = _league_or_404(league)
    url = f"{BASE}/site/v2/sports/basketball/{config['provider_slug']}/teams"
    payload = await _get_json(url, {"limit": 500})
    sports = payload.get("sports") or []
    league_data = ((sports[0].get("leagues") or [{}])[0]) if sports else {}
    result = [_team(item.get("team") or {}) for item in league_data.get("teams") or []]

    # The teams feed is flat. Join it to the standings hierarchy so clients can
    # offer conference/group filters without hard-coding memberships.
    group_map: dict[str, str] = {}
    try:
        standings_url = f"{BASE}/v2/sports/basketball/{config['provider_slug']}/standings"
        standings_payload = await _get_json(standings_url, {"season": date.today().year})
        _team_group_map(standings_payload, group_map)
    except (httpx.HTTPError, KeyError, TypeError):
        pass

    for team in result:
        team["group"] = group_map.get(str(team.get("id") or ""), "Other")
    groups = sorted({team["group"] for team in result if team["group"] != "Other"})
    return {"ok": True, "league": league, "groups": groups, "teams": result}


@router.get("/{league}/teams/{team_id}/roster")
async def roster(league: str, team_id: str):
    config = _league_or_404(league)
    url = f"{BASE}/site/v2/sports/basketball/{config['provider_slug']}/teams/{team_id}/roster"
    payload = await _get_json(url)
    team = _team(payload.get("team") or {})
    players = [_player(raw) for raw in payload.get("athletes") or []]
    players = list(await asyncio.gather(*(_enrich_player_from_core(player, league) for player in players)))
    players = list(await asyncio.gather(*(_enrich_player_from_gleague(player) for player in players)))
    players = list(await asyncio.gather(*(_enrich_player_from_alternatives(player) for player in players)))
    players.sort(key=lambda p: (p.get("position") or "", p.get("name") or ""))
    return {"ok": True, "league": league, "team": team, "season": payload.get("season"), "players": players}


def _shooting_pct(value: Any) -> str | None:
    try:
        made, attempted = (float(part) for part in str(value).split("-", 1))
        return f"{made / attempted * 100:.1f}" if attempted else "0.0"
    except (TypeError, ValueError):
        return None


async def _boxscore_player_games(config: dict[str, Any], player_id: str, team_id: str,
                                 season: int) -> list[dict[str, Any]]:
    """Rebuild a player log from team schedules and event box scores."""
    if not team_id:
        return []
    slug = config["provider_slug"]
    schedule_url = f"{BASE}/site/v2/sports/basketball/{slug}/teams/{team_id}/schedule"
    try:
        schedule = await _get_json(schedule_url, {"season": season})
    except httpx.HTTPError:
        return []
    event_ids = [str(event.get("id")) for event in schedule.get("events") or []
                 if event.get("id") and (event.get("competitions") or [{}])[0]
                 .get("status", {}).get("type", {}).get("completed")]
    summaries = await asyncio.gather(*(
        _get_json(f"{BASE}/site/v2/sports/basketball/{slug}/summary", {"event": event_id})
        for event_id in event_ids[-24:]
    ), return_exceptions=True)
    rows: list[dict[str, Any]] = []
    for event_id, summary in zip(event_ids[-24:], summaries):
        if isinstance(summary, Exception):
            continue
        competition = ((summary.get("header") or {}).get("competitions") or [{}])[0]
        competitors = competition.get("competitors") or []
        own = next((x for x in competitors if str((x.get("team") or {}).get("id")) == str(team_id)), None)
        opponent = next((x for x in competitors if str((x.get("team") or {}).get("id")) != str(team_id)), None)
        if own is None or opponent is None:
            continue
        athlete_row: dict[str, Any] | None = None
        labels: list[str] = []
        for team_box in (summary.get("boxscore") or {}).get("players") or []:
            if str((team_box.get("team") or {}).get("id")) != str(team_id):
                continue
            for block in team_box.get("statistics") or []:
                candidate = next((x for x in block.get("athletes") or []
                                  if str((x.get("athlete") or {}).get("id")) == str(player_id)), None)
                if candidate is not None:
                    athlete_row, labels = candidate, block.get("labels") or []
                    break
        if not athlete_row or athlete_row.get("didNotPlay") or not athlete_row.get("stats"):
            continue
        raw = dict(zip(labels, athlete_row.get("stats") or []))
        stats = {
            "minutes": raw.get("MIN"), "points": raw.get("PTS"),
            "fieldGoalsMade-fieldGoalsAttempted": raw.get("FG"),
            "fieldGoalPct": _shooting_pct(raw.get("FG")),
            "threePointFieldGoalsMade-threePointFieldGoalsAttempted": raw.get("3PT"),
            "threePointPct": _shooting_pct(raw.get("3PT")),
            "freeThrowsMade-freeThrowsAttempted": raw.get("FT"),
            "freeThrowPct": _shooting_pct(raw.get("FT")),
            "totalRebounds": raw.get("REB"), "assists": raw.get("AST"),
            "turnovers": raw.get("TO"), "steals": raw.get("STL"), "blocks": raw.get("BLK"),
            "offensiveRebounds": raw.get("OREB"), "defensiveRebounds": raw.get("DREB"),
            "fouls": raw.get("PF"), "plusMinus": raw.get("+/-"),
        }
        rows.append({
            "event_id": event_id, "date": competition.get("date"),
            "opponent": _team(opponent.get("team") or {}),
            "home_away": "vs" if own.get("homeAway") == "home" else "@",
            "result": "W" if own.get("winner") else "L",
            "score": f"{own.get('score', '—')}-{opponent.get('score', '—')}",
            "season_type": "Summer League", "stage": "Box score fallback", "stats": stats,
        })
    rows.sort(key=lambda row: row.get("date") or "", reverse=True)
    return rows


@router.get("/{league}/players/{player_id}/gamelog")
async def player_gamelog(league: str, player_id: str, season: int | None = None,
                         team_id: str | None = None):
    config = _league_or_404(league)
    selected_season = season or date.today().year
    provider_slugs = [config["provider_slug"]]
    # Summer League roster entries often point to an NBA athlete profile even
    # when the summer-specific athlete endpoint has not been published yet.
    if league.lower() == "summer":
        provider_slugs.append("nba")
    payload: dict[str, Any] | None = None
    for provider_slug in provider_slugs:
        url = f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/{provider_slug}/athletes/{player_id}/gamelog"
        try:
            payload = await _get_json(url, {"season": selected_season})
            break
        except httpx.HTTPStatusError:
            # ESPN uses both 404 and 500 for athlete records that have not been
            # published in a competition yet. Try the next compatible feed.
            continue
    names = payload.get("names") or [] if payload else []
    labels = payload.get("labels") or [] if payload else []
    events = payload.get("events") or {} if payload else {}
    rows: list[dict[str, Any]] = []
    for season_type in (payload.get("seasonTypes") or []) if payload else []:
        for category in season_type.get("categories") or []:
            if category.get("type") != "event":
                continue
            for item in category.get("events") or []:
                event = events.get(str(item.get("eventId"))) or {}
                stats = item.get("stats") or []
                stat_values = {name: stats[index] if index < len(stats) else None for index, name in enumerate(names)}
                rows.append({
                    "event_id": item.get("eventId"),
                    "date": event.get("gameDate"),
                    "opponent": _team(event.get("opponent") or {}),
                    "home_away": event.get("atVs"),
                    "result": event.get("gameResult"),
                    "score": event.get("score"),
                    "season_type": season_type.get("displayName"),
                    "stage": category.get("displayName"),
                    "stats": stat_values,
                })
    rows.sort(key=lambda row: row.get("date") or "", reverse=True)
    source = "athlete_gamelog"
    if not rows and team_id:
        rows = await _boxscore_player_games(config, player_id, team_id, selected_season)
        source = "team_boxscores" if rows else "unavailable"
    return {"ok": True, "league": league, "player_id": player_id, "season": selected_season,
            "labels": labels, "names": names, "games": rows, "available": bool(rows), "source": source}


@router.get("/{league}/players/{player_id}/alternative-photo")
async def alternative_player_photo(league: str, player_id: str, name: str):
    _league_or_404(league)
    alternative = await _alternative_player_data(name)
    return {"ok": True, "player_id": player_id, **alternative}


@router.get("/{league}/standings")
async def standings(league: str, season: int | None = None):
    config = _league_or_404(league)
    url = f"{BASE}/v2/sports/basketball/{config['provider_slug']}/standings"
    payload = await _get_json(url, {"season": season or date.today().year})
    groups: list[dict[str, Any]] = []
    _flatten_standings(payload, groups)
    return {"ok": True, "league": league, "season": season or date.today().year, "groups": groups}


@router.get("/{league}/health")
async def health(league: str):
    config = _league_or_404(league)
    return {"ok": True, "league": league, "provider": "ESPN public score feed", "rules": config}
