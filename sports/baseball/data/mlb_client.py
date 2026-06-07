"""Client for MLB Stats API (free, no auth required)."""

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Optional

import httpx

from common.data.base_client import SportsDataClient
from common.models.sport import Sport, Competition
from sports.baseball.models.baseball import MLBTeam, MLBGame, MLBStanding

logger = logging.getLogger(__name__)

BASE_URL = "https://statsapi.mlb.com/api/v1"


class MLBClient(SportsDataClient):
    def __init__(self):
        self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)
        self._teams: list[MLBTeam] = []
        self._standings: list[MLBStanding] = []
        self._schedule: list[MLBGame] = []
        self._loaded = False

    def get_sport(self) -> Sport:
        return Sport.BASEBALL

    async def get_competitions(self) -> list[Competition]:
        return [Competition(
            id="MLB-2026",
            name="MLB 2026 Season",
            sport=Sport.BASEBALL,
            season="2026",
            country="USA",
        )]

    def is_available(self) -> bool:
        return True

    async def refresh(self) -> None:
        await self._fetch_teams()
        await self._fetch_standings()
        await self._fetch_schedule()
        self._loaded = True
        logger.info("MLB data refreshed: %d teams, %d standings, %d games",
                     len(self._teams), len(self._standings), len(self._schedule))

    async def _fetch_teams(self) -> None:
        try:
            resp = await self._client.get("/teams", params={"sportId": 1})
            resp.raise_for_status()
            data = resp.json()
            self._teams = []
            for t in data.get("teams", []):
                league = t.get("league", {})
                division = t.get("division", {})
                league_id = league.get("id", 0)
                league_abbr = "AL" if league_id == 103 else "NL" if league_id == 104 else league.get("abbreviation", "")
                league_name = league.get("name", "")
                div_name = division.get("name", "").replace(league_name + " ", "") if league_name else division.get("name", "")
                self._teams.append(MLBTeam(
                    id=t["id"],
                    name=t.get("name", ""),
                    abbreviation=t.get("abbreviation", ""),
                    league=league_abbr,
                    division=div_name,
                ))
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("Failed to fetch MLB teams: %s", e)
            if not self._teams:
                self._teams = self._get_fallback_teams()

    async def _fetch_standings(self) -> None:
        try:
            resp = await self._client.get("/standings", params={
                "leagueId": "103,104",  # AL=103, NL=104
                "season": date.today().year,
                "standingsTypes": "regularSeason",
            })
            resp.raise_for_status()
            data = resp.json()
            self._standings = []
            # MLB Stats API division IDs → short name. The /standings response
            # sometimes returns an empty division.name, so fall back to the ID.
            div_short_by_id = {
                200: "West", 201: "East", 202: "Central",   # AL
                203: "West", 204: "East", 205: "Central",   # NL
            }
            for record in data.get("records", []):
                division = record.get("division", {})
                div_name = division.get("name", "")
                league = record.get("league", {})
                league_id = league.get("id", 0)
                league_abbr = "AL" if league_id == 103 else "NL" if league_id == 104 else league.get("abbreviation", "")
                # Extract short division name; prefer ID lookup when name is empty
                short_div = div_name.replace("American League ", "").replace("National League ", "")
                if not short_div:
                    short_div = div_short_by_id.get(division.get("id", 0), "")
                for entry in record.get("teamRecords", []):
                    team = entry.get("team", {})
                    streak_obj = entry.get("streak", {})
                    streak_str = streak_obj.get("streakCode", "") if streak_obj else ""
                    # last 10
                    records_split = entry.get("records", {}).get("splitRecords", [])
                    last_10 = ""
                    for sr in records_split:
                        if sr.get("type") == "lastTen":
                            last_10 = f"{sr.get('wins', 0)}-{sr.get('losses', 0)}"
                            break
                    self._standings.append(MLBStanding(
                        team_id=team.get("id", 0),
                        team_name=team.get("name", ""),
                        abbreviation=self._get_team_abbrev(team.get("id", 0)),
                        league=league_abbr,
                        division=short_div,
                        wins=entry.get("wins", 0),
                        losses=entry.get("losses", 0),
                        win_pct=float(entry.get("winningPercentage", "0.000")),
                        games_back=str(entry.get("gamesBack", "-")),
                        streak=streak_str,
                        last_10=last_10,
                        run_differential=entry.get("runDifferential", 0),
                    ))
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch MLB standings: %s", e)

    def _parse_schedule_games(self, data, fallback_date: date) -> list[MLBGame]:
        games: list[MLBGame] = []
        for day in data.get("dates", []):
            for g in day.get("games", []):
                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})
                home_team = home.get("team", {})
                away_team = away.get("team", {})

                status_code = g.get("status", {}).get("abstractGameState", "Preview")
                status_map = {"Preview": "SCHEDULED", "Live": "LIVE", "Final": "FINAL"}
                status = status_map.get(status_code, "SCHEDULED")

                home_pitcher = home.get("probablePitcher", {}).get("fullName")
                away_pitcher = away.get("probablePitcher", {}).get("fullName")

                games.append(MLBGame(
                    id=g.get("gamePk", 0),
                    home_team=home_team.get("name", ""),
                    away_team=away_team.get("name", ""),
                    home_team_id=home_team.get("id", 0),
                    away_team_id=away_team.get("id", 0),
                    home_abbrev=home_team.get("abbreviation", ""),
                    away_abbrev=away_team.get("abbreviation", ""),
                    date=datetime.fromisoformat(g.get("gameDate", fallback_date.isoformat()).replace("Z", "+00:00")),
                    venue=g.get("venue", {}).get("name"),
                    status=status,
                    home_score=home.get("score"),
                    away_score=away.get("score"),
                    home_pitcher=home_pitcher,
                    away_pitcher=away_pitcher,
                ))
        return games

    async def fetch_game(self, game_pk: int) -> Optional[MLBGame]:
        """Fetch a single game by gamePk."""
        try:
            resp = await self._client.get("/schedule", params={
                "sportId": 1,
                "gamePk": game_pk,
                "hydrate": "probablePitcher,team,linescore",
            })
            resp.raise_for_status()
            games = self._parse_schedule_games(resp.json(), date.today())
            return games[0] if games else None
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch MLB game %s: %s", game_pk, e)
            return None

    async def fetch_schedule_range(self, start_date: date, end_date: date) -> list[MLBGame]:
        """Fetch a custom date range without affecting the cached default schedule."""
        try:
            resp = await self._client.get("/schedule", params={
                "sportId": 1,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "hydrate": "probablePitcher,team",
            })
            resp.raise_for_status()
            return self._parse_schedule_games(resp.json(), start_date)
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch MLB schedule %s..%s: %s", start_date, end_date, e)
            return []

    async def _fetch_schedule(self) -> None:
        today = date.today()
        self._schedule = await self.fetch_schedule_range(today, today + timedelta(days=7))

    def _get_team_abbrev(self, team_id: int) -> str:
        for t in self._teams:
            if t.id == team_id:
                return t.abbreviation
        return ""

    def get_teams(self) -> list[MLBTeam]:
        return self._teams

    def get_standings(self) -> list[MLBStanding]:
        return self._standings

    def get_schedule(self) -> list[MLBGame]:
        return self._schedule

    def get_team_by_id(self, team_id: int) -> Optional[MLBTeam]:
        return next((t for t in self._teams if t.id == team_id), None)

    @staticmethod
    def _get_fallback_teams() -> list[MLBTeam]:
        teams = [
            (147, "New York Yankees", "NYY", "AL", "East"),
            (111, "Boston Red Sox", "BOS", "AL", "East"),
            (141, "Toronto Blue Jays", "TOR", "AL", "East"),
            (110, "Baltimore Orioles", "BAL", "AL", "East"),
            (139, "Tampa Bay Rays", "TB", "AL", "East"),
            (114, "Cleveland Guardians", "CLE", "AL", "Central"),
            (118, "Kansas City Royals", "KC", "AL", "Central"),
            (116, "Detroit Tigers", "DET", "AL", "Central"),
            (142, "Minnesota Twins", "MIN", "AL", "Central"),
            (145, "Chicago White Sox", "CWS", "AL", "Central"),
            (117, "Houston Astros", "HOU", "AL", "West"),
            (133, "Oakland Athletics", "OAK", "AL", "West"),
            (136, "Seattle Mariners", "SEA", "AL", "West"),
            (108, "Los Angeles Angels", "LAA", "AL", "West"),
            (140, "Texas Rangers", "TEX", "AL", "West"),
            (119, "Los Angeles Dodgers", "LAD", "NL", "East"),
            (143, "Philadelphia Phillies", "PHI", "NL", "East"),
            (121, "New York Mets", "NYM", "NL", "East"),
            (144, "Atlanta Braves", "ATL", "NL", "East"),
            (120, "Washington Nationals", "WSH", "NL", "East"),
            (112, "Chicago Cubs", "CHC", "NL", "Central"),
            (158, "Milwaukee Brewers", "MIL", "NL", "Central"),
            (138, "St. Louis Cardinals", "STL", "NL", "Central"),
            (134, "Pittsburgh Pirates", "PIT", "NL", "Central"),
            (113, "Cincinnati Reds", "CIN", "NL", "Central"),
            (137, "San Francisco Giants", "SF", "NL", "West"),
            (135, "San Diego Padres", "SD", "NL", "West"),
            (109, "Arizona Diamondbacks", "ARI", "NL", "West"),
            (115, "Colorado Rockies", "COL", "NL", "West"),
            (146, "Miami Marlins", "MIA", "NL", "West"),
        ]
        return [MLBTeam(id=tid, name=n, abbreviation=a, league=l, division=d) for tid, n, a, l, d in teams]

    async def close(self):
        await self._client.aclose()
