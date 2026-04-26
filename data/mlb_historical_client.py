"""Client for historical MLB team and player data using the free MLB Stats API."""

import logging

import httpx

from models.baseball import (
    MLBHistoricalPlayerProfile,
    MLBHistoricalPlayerStats,
    MLBHistoricalRosterPlayer,
    MLBHistoricalTeamRoster,
    MLBHistoricalTeamSeason,
    MLBStatSplit,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://statsapi.mlb.com/api/v1"
DEFAULT_STAT_GROUPS = ["hitting", "pitching", "fielding"]


class MLBHistoricalClient:
    """Fetches historical team and player data from the MLB Stats API."""

    def __init__(self):
        self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)

    async def get_team_history(
        self,
        team_id: int,
        start_season: int | None = None,
        end_season: int | None = None,
    ) -> list[MLBHistoricalTeamSeason]:
        params: dict[str, str | int] = {"teamIds": team_id}
        if start_season is not None:
            params["startSeason"] = start_season
        if end_season is not None:
            params["endSeason"] = end_season

        data = await self._get_json("/teams/history", params=params)
        return [self._parse_team_history_entry(team) for team in data.get("teams", [])]

    async def get_team_roster(
        self,
        team_id: int,
        season: int,
        roster_type: str = "fullSeason",
    ) -> MLBHistoricalTeamRoster:
        params = {
            "season": season,
            "rosterType": roster_type,
            "hydrate": "person",
        }
        data = await self._get_json(f"/teams/{team_id}/roster", params=params)
        roster = data.get("roster", [])
        team_name = data.get("team", {}).get("name", "")
        players = [self._parse_roster_player(player) for player in roster]
        return MLBHistoricalTeamRoster(
            team_id=team_id,
            season=season,
            roster_type=roster_type,
            team_name=team_name,
            players=players,
        )

    async def get_team_stats(
        self,
        team_id: int,
        season: int,
        groups: list[str] | None = None,
        stats_type: str = "season",
    ) -> list[MLBStatSplit]:
        groups = groups or DEFAULT_STAT_GROUPS
        all_splits: list[MLBStatSplit] = []
        for group in groups:
            params = {
                "season": season,
                "group": group,
                "stats": stats_type,
            }
            data = await self._get_json(f"/teams/{team_id}/stats", params=params)
            all_splits.extend(self._parse_stat_sets(data.get("stats", [])))
        return all_splits

    async def get_player_profile(
        self,
        player_id: int,
        season: int | None = None,
    ) -> MLBHistoricalPlayerProfile:
        params: dict[str, str | int] = {}
        if season is not None:
            params["season"] = season
        data = await self._get_json(f"/people/{player_id}", params=params)
        people = data.get("people", [])
        if not people:
            raise ValueError(f"No player found for id {player_id}")
        return self._parse_player_profile(people[0])

    async def get_player_year_by_year_stats(
        self,
        player_id: int,
        groups: list[str] | None = None,
        stat_type: str = "yearByYear",
        season: int | None = None,
    ) -> MLBHistoricalPlayerStats:
        groups = groups or DEFAULT_STAT_GROUPS
        hydrate = f"stats(group=[{','.join(groups)}],type=[{stat_type}])"
        params: dict[str, str | int] = {"hydrate": hydrate}
        if season is not None:
            params["season"] = season

        data = await self._get_json(f"/people/{player_id}", params=params)
        people = data.get("people", [])
        if not people:
            raise ValueError(f"No player found for id {player_id}")

        person = people[0]
        splits = self._parse_stat_sets(person.get("stats", []))
        return MLBHistoricalPlayerStats(
            player=self._parse_player_profile(person),
            groups=groups,
            stat_type=stat_type,
            splits=splits,
        )

    async def _get_json(self, path: str, params: dict[str, str | int] | None = None) -> dict:
        try:
            resp = await self._client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("MLB historical request failed for %s: %s", path, exc)
            if exc.response.status_code == 404:
                return {}
            raise

    @staticmethod
    def _parse_team_history_entry(team: dict) -> MLBHistoricalTeamSeason:
        league = team.get("league", {})
        division = team.get("division", {})
        venue = team.get("venue", {})
        return MLBHistoricalTeamSeason(
            team_id=team.get("id", 0),
            season=team.get("season"),
            name=team.get("name", ""),
            location_name=team.get("locationName", ""),
            franchise_name=team.get("franchiseName", ""),
            club_name=team.get("clubName", ""),
            abbreviation=team.get("abbreviation", ""),
            league=league.get("abbreviation", "") or league.get("name", ""),
            division=division.get("nameShort", "") or division.get("name", ""),
            venue=venue.get("name", ""),
            first_year_of_play=team.get("firstYearOfPlay", ""),
            active=team.get("active", True),
        )

    @staticmethod
    def _parse_roster_player(entry: dict) -> MLBHistoricalRosterPlayer:
        person = entry.get("person", {})
        position = entry.get("position", {})
        status = entry.get("status", {})
        return MLBHistoricalRosterPlayer(
            player_id=person.get("id", 0),
            full_name=person.get("fullName", ""),
            jersey_number=entry.get("jerseyNumber"),
            position=position.get("abbreviation", "") or position.get("name", ""),
            status=status.get("code", ""),
            status_description=status.get("description", ""),
            batting_side=person.get("batSide", {}).get("code"),
            pitching_hand=person.get("pitchHand", {}).get("code"),
            birth_date=person.get("birthDate"),
        )

    @staticmethod
    def _parse_player_profile(person: dict) -> MLBHistoricalPlayerProfile:
        current_team = person.get("currentTeam", {})
        return MLBHistoricalPlayerProfile(
            player_id=person.get("id", 0),
            full_name=person.get("fullName", ""),
            primary_number=person.get("primaryNumber"),
            birth_date=person.get("birthDate"),
            current_age=person.get("currentAge"),
            birth_city=person.get("birthCity"),
            birth_state_province=person.get("birthStateProvince"),
            birth_country=person.get("birthCountry"),
            height=person.get("height"),
            weight=person.get("weight"),
            active=person.get("active"),
            primary_position=person.get("primaryPosition", {}).get("abbreviation"),
            batting_side=person.get("batSide", {}).get("code"),
            pitching_hand=person.get("pitchHand", {}).get("code"),
            current_team_id=current_team.get("id"),
            current_team_name=current_team.get("name"),
        )

    @staticmethod
    def _parse_stat_sets(stat_sets: list[dict]) -> list[MLBStatSplit]:
        splits: list[MLBStatSplit] = []
        for stat_set in stat_sets:
            stat_type = stat_set.get("type", {}).get("displayName", "")
            group = stat_set.get("group", {}).get("displayName", "")
            for split in stat_set.get("splits", []):
                team = split.get("team", {})
                league = split.get("league", {})
                player = split.get("player", {})
                splits.append(
                    MLBStatSplit(
                        season=split.get("season"),
                        stat_type=stat_type,
                        group=group,
                        team_id=team.get("id"),
                        team_name=team.get("name"),
                        league_name=league.get("name"),
                        player_id=player.get("id"),
                        player_name=player.get("fullName"),
                        stat=split.get("stat", {}),
                    )
                )
        return splits

    async def close(self):
        await self._client.aclose()
