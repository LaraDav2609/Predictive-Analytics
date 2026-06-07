import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException

from sports.baseball.api import baseball_history_routes


class BaseballHistoryRoutesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncMock()
        baseball_history_routes.init(self.client)

    async def test_parse_groups_returns_defaults_when_missing(self):
        self.assertEqual(
            ["hitting", "pitching", "fielding"],
            baseball_history_routes._parse_groups(None),
        )

    async def test_get_team_history_returns_serialized_response(self):
        season = SimpleNamespace(model_dump=lambda: {"team_id": 147, "season": 2024})
        self.client.get_team_history.return_value = [season]

        response = await baseball_history_routes.get_team_history(147, 2024, 2024)

        self.assertTrue(response["ok"])
        self.assertEqual(147, response["team_id"])
        self.assertEqual([{"team_id": 147, "season": 2024}], response["seasons"])
        self.client.get_team_history.assert_awaited_once_with(147, 2024, 2024)

    async def test_get_team_roster_returns_model_dump(self):
        roster = SimpleNamespace(model_dump=lambda: {"team_id": 147, "season": 2024, "players": []})
        self.client.get_team_roster.return_value = roster

        response = await baseball_history_routes.get_team_roster(147, 2024, "fullSeason")

        self.assertEqual({"ok": True, "team_id": 147, "season": 2024, "players": []}, response)

    async def test_get_team_stats_uses_parsed_groups(self):
        split = SimpleNamespace(model_dump=lambda: {"group": "hitting", "stat": {"runs": 10}})
        self.client.get_team_stats.return_value = [split]

        response = await baseball_history_routes.get_team_stats(147, 2024, "hitting,pitching", "season")

        self.assertEqual(["hitting", "pitching"], response["groups"])
        self.assertEqual([{"group": "hitting", "stat": {"runs": 10}}], response["splits"])
        self.client.get_team_stats.assert_awaited_once_with(147, 2024, ["hitting", "pitching"], "season")

    async def test_get_player_profile_maps_missing_player_to_404(self):
        self.client.get_player_profile.side_effect = ValueError("No player found for id 1")

        with self.assertRaises(HTTPException) as context:
            await baseball_history_routes.get_player_profile(1)

        self.assertEqual(404, context.exception.status_code)
        self.assertIn("No player found", context.exception.detail)

    async def test_get_player_stats_returns_serialized_response(self):
        stats = SimpleNamespace(model_dump=lambda: {"player": {"player_id": 660271}, "splits": []})
        self.client.get_player_year_by_year_stats.return_value = stats

        response = await baseball_history_routes.get_player_stats(660271, "hitting", "yearByYear", None)

        self.assertTrue(response["ok"])
        self.assertEqual({"player_id": 660271}, response["player"])
        self.client.get_player_year_by_year_stats.assert_awaited_once_with(
            660271,
            ["hitting"],
            "yearByYear",
            None,
        )


if __name__ == "__main__":
    unittest.main()
