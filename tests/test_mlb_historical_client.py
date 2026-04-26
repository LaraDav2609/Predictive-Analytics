import unittest
from unittest.mock import AsyncMock

from data.mlb_historical_client import DEFAULT_STAT_GROUPS, MLBHistoricalClient


class MLBHistoricalClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = MLBHistoricalClient()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_get_team_history_parses_team_seasons(self):
        self.client._get_json = AsyncMock(return_value={
            "teams": [
                {
                    "id": 147,
                    "season": 2024,
                    "name": "New York Yankees",
                    "locationName": "New York",
                    "franchiseName": "Yankees",
                    "clubName": "Yankees",
                    "abbreviation": "NYY",
                    "league": {"abbreviation": "AL"},
                    "division": {"nameShort": "AL East"},
                    "venue": {"name": "Yankee Stadium"},
                    "firstYearOfPlay": "1903",
                    "active": True,
                }
            ]
        })

        seasons = await self.client.get_team_history(147, 2024, 2024)

        self.assertEqual(1, len(seasons))
        self.assertEqual(147, seasons[0].team_id)
        self.assertEqual(2024, seasons[0].season)
        self.assertEqual("New York Yankees", seasons[0].name)
        self.assertEqual("AL", seasons[0].league)
        self.client._get_json.assert_awaited_once_with(
            "/teams/history",
            params={"teamIds": 147, "startSeason": 2024, "endSeason": 2024},
        )

    async def test_get_team_roster_parses_players(self):
        self.client._get_json = AsyncMock(return_value={
            "team": {"name": "Los Angeles Dodgers"},
            "roster": [
                {
                    "jerseyNumber": "17",
                    "position": {"abbreviation": "DH"},
                    "status": {"code": "A", "description": "Active"},
                    "person": {
                        "id": 660271,
                        "fullName": "Shohei Ohtani",
                        "batSide": {"code": "L"},
                        "pitchHand": {"code": "R"},
                        "birthDate": "1994-07-05",
                    },
                }
            ],
        })

        roster = await self.client.get_team_roster(119, 2024, "fullSeason")

        self.assertEqual(119, roster.team_id)
        self.assertEqual(2024, roster.season)
        self.assertEqual("Los Angeles Dodgers", roster.team_name)
        self.assertEqual(1, len(roster.players))
        self.assertEqual("Shohei Ohtani", roster.players[0].full_name)
        self.assertEqual("DH", roster.players[0].position)

    async def test_get_team_stats_fetches_each_group_and_combines_splits(self):
        self.client._get_json = AsyncMock(side_effect=[
            {
                "stats": [
                    {
                        "type": {"displayName": "season"},
                        "group": {"displayName": "hitting"},
                        "splits": [
                            {
                                "season": "2024",
                                "team": {"id": 147, "name": "New York Yankees"},
                                "league": {"name": "American League"},
                                "stat": {"runs": 815},
                            }
                        ],
                    }
                ]
            },
            {
                "stats": [
                    {
                        "type": {"displayName": "season"},
                        "group": {"displayName": "pitching"},
                        "splits": [
                            {
                                "season": "2024",
                                "team": {"id": 147, "name": "New York Yankees"},
                                "league": {"name": "American League"},
                                "stat": {"era": "3.74"},
                            }
                        ],
                    }
                ]
            },
        ])

        splits = await self.client.get_team_stats(147, 2024, ["hitting", "pitching"], "season")

        self.assertEqual(2, len(splits))
        self.assertEqual("hitting", splits[0].group)
        self.assertEqual({"runs": 815}, splits[0].stat)
        self.assertEqual("pitching", splits[1].group)
        self.assertEqual({"era": "3.74"}, splits[1].stat)
        self.assertEqual(2, self.client._get_json.await_count)

    async def test_get_player_profile_raises_for_missing_player(self):
        self.client._get_json = AsyncMock(return_value={"people": []})

        with self.assertRaisesRegex(ValueError, "No player found"):
            await self.client.get_player_profile(999999)

    async def test_get_player_year_by_year_stats_parses_profile_and_splits(self):
        self.client._get_json = AsyncMock(return_value={
            "people": [
                {
                    "id": 660271,
                    "fullName": "Shohei Ohtani",
                    "primaryNumber": "17",
                    "birthDate": "1994-07-05",
                    "currentAge": 30,
                    "birthCity": "Oshu",
                    "birthCountry": "Japan",
                    "height": "6' 4\"",
                    "weight": 210,
                    "active": True,
                    "primaryPosition": {"abbreviation": "DH"},
                    "batSide": {"code": "L"},
                    "pitchHand": {"code": "R"},
                    "currentTeam": {"id": 119, "name": "Los Angeles Dodgers"},
                    "stats": [
                        {
                            "type": {"displayName": "yearByYear"},
                            "group": {"displayName": "hitting"},
                            "splits": [
                                {
                                    "season": "2024",
                                    "team": {"id": 119, "name": "Los Angeles Dodgers"},
                                    "league": {"name": "National League"},
                                    "player": {"id": 660271, "fullName": "Shohei Ohtani"},
                                    "stat": {"homeRuns": 54, "avg": ".310"},
                                }
                            ],
                        }
                    ],
                }
            ]
        })

        stats = await self.client.get_player_year_by_year_stats(660271, ["hitting"], "yearByYear")

        self.assertEqual("Shohei Ohtani", stats.player.full_name)
        self.assertEqual(["hitting"], stats.groups)
        self.assertEqual("yearByYear", stats.stat_type)
        self.assertEqual(1, len(stats.splits))
        self.assertEqual("2024", stats.splits[0].season)
        self.assertEqual({"homeRuns": 54, "avg": ".310"}, stats.splits[0].stat)

    async def test_default_stat_groups_are_used_for_player_stats(self):
        self.client._get_json = AsyncMock(return_value={
            "people": [
                {
                    "id": 1,
                    "fullName": "Player One",
                    "stats": [],
                }
            ]
        })

        await self.client.get_player_year_by_year_stats(1)

        _, kwargs = self.client._get_json.await_args
        self.assertEqual(
            f"stats(group=[{','.join(DEFAULT_STAT_GROUPS)}],type=[yearByYear])",
            kwargs["params"]["hydrate"],
        )


if __name__ == "__main__":
    unittest.main()
