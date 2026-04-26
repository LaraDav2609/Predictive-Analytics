import unittest

from api.server import app


class ApiServerHistoryRegistrationTests(unittest.TestCase):
    def test_history_routes_are_registered(self):
        paths = {route.path for route in app.routes}

        self.assertIn("/api/baseball/history/teams/{team_id}", paths)
        self.assertIn("/api/baseball/history/teams/{team_id}/roster", paths)
        self.assertIn("/api/baseball/history/teams/{team_id}/stats", paths)
        self.assertIn("/api/baseball/history/players/{player_id}", paths)
        self.assertIn("/api/baseball/history/players/{player_id}/stats", paths)


if __name__ == "__main__":
    unittest.main()
