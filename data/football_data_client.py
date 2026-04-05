"""Client for football-data.org v4 API (free tier, no auth for basic requests)."""

import logging
from datetime import datetime

import httpx

from data.base_client import SportsDataClient
from models.sport import Sport, Competition
from models.team import Team, TeamStats
from models.match import Match, MatchResult, MatchPrediction

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"
WC_CODE = "WC"  # FIFA World Cup competition code


class FootballDataClient(SportsDataClient):
    def __init__(self, api_key: str | None = None):
        headers = {"Accept": "application/json"}
        if api_key:
            headers["X-Auth-Token"] = api_key
        self._client = httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=30.0)
        self._teams: list[Team] = []
        self._matches: list[Match] = []
        self._standings: list[dict] = []
        self._loaded = False

    def get_sport(self) -> Sport:
        return Sport.SOCCER

    async def get_competitions(self) -> list[Competition]:
        return [Competition(
            id="WC2026",
            name="FIFA World Cup 2026",
            sport=Sport.SOCCER,
            season="2026",
            country="International",
            api_id=WC_CODE,
        )]

    def is_available(self) -> bool:
        return True

    async def refresh(self) -> None:
        """Fetch all World Cup data from football-data.org."""
        await self._fetch_teams()
        await self._fetch_matches()
        await self._fetch_standings()
        self._loaded = True
        logger.info("Football data refreshed: %d teams, %d matches", len(self._teams), len(self._matches))

    async def _fetch_teams(self) -> None:
        try:
            resp = await self._client.get(f"/competitions/{WC_CODE}/teams")
            resp.raise_for_status()
            data = resp.json()
            self._teams = []
            for t in data.get("teams", []):
                team = Team(
                    id=t["id"],
                    name=t.get("name", ""),
                    short_name=t.get("shortName", t.get("tla", "")),
                    country_code=t.get("tla", ""),
                    crest_url=t.get("crest"),
                )
                self._teams.append(team)
        except httpx.HTTPStatusError as e:
            logger.warning("Failed to fetch WC teams: %s", e)
            if not self._teams:
                self._teams = self._get_fallback_teams()
        except httpx.HTTPError as e:
            logger.warning("HTTP error fetching teams: %s", e)
            if not self._teams:
                self._teams = self._get_fallback_teams()

    async def _fetch_matches(self) -> None:
        try:
            resp = await self._client.get(f"/competitions/{WC_CODE}/matches")
            resp.raise_for_status()
            data = resp.json()
            self._matches = []
            for m in data.get("matches", []):
                home = m.get("homeTeam", {})
                away = m.get("awayTeam", {})
                score = m.get("score", {})
                ft = score.get("fullTime", {})

                result = None
                if m.get("status") == "FINISHED" and ft.get("home") is not None:
                    h, a = ft["home"], ft["away"]
                    winner = "DRAW" if h == a else ("HOME" if h > a else "AWAY")
                    result = MatchResult(home_score=h, away_score=a, winner=winner)

                match = Match(
                    id=m["id"],
                    home_team=home.get("name", "TBD"),
                    away_team=away.get("name", "TBD"),
                    home_team_id=home.get("id", 0),
                    away_team_id=away.get("id", 0),
                    home_crest=home.get("crest"),
                    away_crest=away.get("crest"),
                    date=datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")),
                    competition="FIFA World Cup 2026",
                    stage=m.get("stage"),
                    group=m.get("group"),
                    status=m.get("status", "SCHEDULED"),
                    result=result,
                )
                self._matches.append(match)
        except httpx.HTTPStatusError as e:
            logger.warning("Failed to fetch WC matches: %s", e)
            if not self._matches:
                self._matches = self._get_fallback_matches()
        except httpx.HTTPError as e:
            logger.warning("HTTP error fetching matches: %s", e)
            if not self._matches:
                self._matches = self._get_fallback_matches()

    async def _fetch_standings(self) -> None:
        try:
            resp = await self._client.get(f"/competitions/{WC_CODE}/standings")
            resp.raise_for_status()
            data = resp.json()
            self._standings = []
            for standing in data.get("standings", []):
                group_name = standing.get("group", "")
                for row in standing.get("table", []):
                    team = row.get("team", {})
                    self._standings.append({
                        "group": group_name,
                        "position": row.get("position", 0),
                        "team_id": team.get("id", 0),
                        "team_name": team.get("name", ""),
                        "team_crest": team.get("crest", ""),
                        "played": row.get("playedGames", 0),
                        "won": row.get("won", 0),
                        "draw": row.get("draw", 0),
                        "lost": row.get("lost", 0),
                        "goals_for": row.get("goalsFor", 0),
                        "goals_against": row.get("goalsAgainst", 0),
                        "goal_difference": row.get("goalDifference", 0),
                        "points": row.get("points", 0),
                    })
        except (httpx.HTTPStatusError, httpx.HTTPError) as e:
            logger.warning("Failed to fetch standings: %s", e)

    def get_teams(self) -> list[Team]:
        return self._teams

    def get_matches(self, filter_status: str | None = None) -> list[Match]:
        if filter_status == "upcoming":
            return [m for m in self._matches if m.status in ("SCHEDULED", "TIMED")]
        elif filter_status == "past":
            return [m for m in self._matches if m.status == "FINISHED"]
        return self._matches

    def get_match_by_id(self, match_id: int) -> Match | None:
        return next((m for m in self._matches if m.id == match_id), None)

    def get_standings(self) -> list[dict]:
        return self._standings

    def get_team_by_id(self, team_id: int) -> Team | None:
        return next((t for t in self._teams if t.id == team_id), None)

    @staticmethod
    def _get_fallback_teams() -> list[Team]:
        """Fallback WC 2026 teams when API is unavailable (top 16 by FIFA ranking)."""
        fallback = [
            (1, "Argentina", "ARG", 1), (2, "France", "FRA", 2),
            (3, "Brazil", "BRA", 3), (4, "England", "ENG", 4),
            (5, "Belgium", "BEL", 5), (6, "Portugal", "POR", 6),
            (7, "Netherlands", "NED", 7), (8, "Spain", "ESP", 8),
            (9, "Germany", "GER", 9), (10, "Italy", "ITA", 10),
            (11, "Croatia", "CRO", 11), (12, "Uruguay", "URU", 12),
            (13, "Colombia", "COL", 13), (14, "Mexico", "MEX", 14),
            (15, "USA", "USA", 15), (16, "Japan", "JPN", 16),
            (17, "Morocco", "MAR", 17), (18, "Senegal", "SEN", 18),
            (19, "Canada", "CAN", 19), (20, "Switzerland", "SUI", 20),
            (21, "Denmark", "DEN", 21), (22, "Australia", "AUS", 22),
            (23, "South Korea", "KOR", 23), (24, "Ecuador", "ECU", 24),
        ]
        return [
            Team(id=tid, name=name, short_name=code, country_code=code, fifa_ranking=rank)
            for tid, name, code, rank in fallback
        ]

    @staticmethod
    def _get_fallback_matches() -> list[Match]:
        """Fallback with sample group stage matches."""
        from datetime import timezone
        base = datetime(2026, 6, 11, 18, 0, tzinfo=timezone.utc)
        samples = [
            (9001, "Mexico", "USA", 14, 15, "GROUP_A"),
            (9002, "Canada", "Morocco", 19, 17, "GROUP_A"),
            (9003, "Argentina", "Japan", 1, 16, "GROUP_B"),
            (9004, "France", "Brazil", 2, 3, "GROUP_C"),
            (9005, "Germany", "Spain", 9, 8, "GROUP_D"),
            (9006, "England", "Portugal", 4, 6, "GROUP_E"),
        ]
        matches = []
        for i, (mid, h, a, hid, aid, group) in enumerate(samples):
            matches.append(Match(
                id=mid, home_team=h, away_team=a,
                home_team_id=hid, away_team_id=aid,
                date=base.replace(day=11 + i),
                competition="FIFA World Cup 2026",
                stage="GROUP_STAGE", group=group,
            ))
        return matches

    async def close(self):
        await self._client.aclose()
