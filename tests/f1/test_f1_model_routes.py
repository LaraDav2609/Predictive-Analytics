import unittest

from sports.f1.api import f1_routes
from sports.f1.analytics.f1_predictor import F1Predictor
from sports.f1.models.f1 import Constructor, Driver, Race


class F1ModelRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = _FakeF1Client()
        self.predictor = F1Predictor()
        self.predictor.load_drivers(self.client.get_drivers(), self.client.get_constructors(), self.client.features, sentiment={})
        f1_routes.client = self.client
        f1_routes.predictor = self.predictor
        f1_routes.openf1 = None
        f1_routes.live_engine = None

    async def test_models_endpoint_lists_registry(self):
        result = await f1_routes.get_f1_models()

        self.assertTrue(result["ok"])
        self.assertEqual("production_v1", result["default_model_id"])
        self.assertGreaterEqual(len(result["models"]), 3)

    async def test_health_endpoint_reports_missing_sources(self):
        result = await f1_routes.get_f1_health()

        self.assertTrue(result["ok"])
        self.assertIn("sentiment", result["fallback_heavy"])
        self.assertIn("live", result["fallback_heavy"])
        self.assertEqual("unavailable", result["sources"]["live"]["latest_mode"])
        self.assertEqual(2, result["drivers"])

    async def test_existing_race_prediction_shape_still_exists(self):
        race = self.client.get_race_by_round(1)
        race.prediction = self.predictor.predict_race(race)
        prediction = race.prediction.model_dump(mode="json")

        first = next(iter(prediction["driver_predictions"].values()))
        self.assertIn("win_prob", first)
        self.assertIn("podium_prob", first)
        self.assertIn("track_fit_score", first)

    async def test_session_result_uses_official_qualifying_rows(self):
        result = await f1_routes.get_race_session_result(1, "qualifying")

        self.assertTrue(result["ok"])
        self.assertTrue(result["available"])
        self.assertEqual("qualifying", result["session"]["code"])
        self.assertEqual(2, result["result_count"])
        self.assertEqual("VER", result["results"][0]["driver_code"])
        self.assertEqual("1:29.100", result["results"][0]["best_time"])

    async def test_session_result_uses_openf1_practice_rows_when_available(self):
        f1_routes.openf1 = _FakeOpenF1()

        result = await f1_routes.get_race_session_result(1, "fp1")

        self.assertTrue(result["ok"])
        self.assertTrue(result["available"])
        self.assertEqual("fp1", result["session"]["code"])
        self.assertEqual("LEC", result["results"][0]["driver_code"])
        self.assertEqual("1:30.900", result["results"][0]["best_time"])
        self.assertEqual(7, result["results"][0]["laps"])

    async def test_session_result_reports_unavailable_when_no_rows_exist(self):
        result = await f1_routes.get_race_session_result(1, "fp2")

        self.assertTrue(result["ok"])
        self.assertFalse(result["available"])
        self.assertEqual("fp2", result["session"]["code"])
        self.assertEqual([], result["results"])


class _FakeF1Client:
    season = 2026

    def __init__(self):
        self.features = {
            "completed_races": 1,
            "total_races": 2,
            "drivers": {
                "max": {"form_score": 0.85, "reliability_score": 0.92, "recent_starts": 1, "starts": 100},
                "charles": {"form_score": 0.72, "reliability_score": 0.88, "recent_starts": 1, "starts": 100},
            },
            "constructors": {
                "red bull racing": {"team_score": 0.86, "recent_points": 25},
                "ferrari": {"team_score": 0.78, "recent_points": 18},
            },
        }
        self.races = [
            Race(round=1, name="Bahrain Grand Prix", circuit="Bahrain International Circuit", country="Bahrain", date="2026-03-01T14:00:00Z"),
        ]

    def get_drivers(self):
        return [
            Driver(id="max", number=1, code="VER", first_name="Max", last_name="Verstappen", nationality="Dutch", team="Red Bull Racing", points=25, wins=1, position=1),
            Driver(id="charles", number=16, code="LEC", first_name="Charles", last_name="Leclerc", nationality="Monegasque", team="Ferrari", points=18, wins=0, position=2),
        ]

    def get_constructors(self):
        return [
            Constructor(id="red_bull", name="Red Bull Racing", nationality="Austrian", points=25, wins=1, position=1),
            Constructor(id="ferrari", name="Ferrari", nationality="Italian", points=18, wins=0, position=2),
        ]

    def get_races(self):
        return self.races

    def get_race_by_round(self, round_num):
        return self.races[0] if round_num == 1 else None

    async def get_race_profile(self, round_num, predictor=None):
        race = self.get_race_by_round(round_num)
        return {
            "ok": True,
            "race": race.model_dump(mode="json"),
            "sessions": [
                {"code": "fp1", "name": "Practice 1", "status": "completed", "date": "2026-02-27T12:00:00Z", "note": "Practice"},
                {"code": "fp2", "name": "Practice 2", "status": "completed", "date": "2026-02-27T16:00:00Z", "note": "Practice"},
                {"code": "qualifying", "name": "Qualifying", "status": "completed", "date": "2026-02-28T15:00:00Z", "note": "Grid order"},
                {"code": "race", "name": "Race", "status": "scheduled", "date": "2026-03-01T14:00:00Z", "note": "Grand Prix"},
            ],
            "results": [],
            "qualifying": [
                {"position": 1, "driver_id": "max", "driver_code": "VER", "driver_name": "Max Verstappen", "team": "Red Bull Racing", "q1": "1:30.000", "q2": "1:29.500", "q3": "1:29.100"},
                {"position": 2, "driver_id": "charles", "driver_code": "LEC", "driver_name": "Charles Leclerc", "team": "Ferrari", "q1": "1:30.100", "q2": "1:29.700", "q3": "1:29.300"},
            ],
            "sprint": [],
            "context": {"has_sprint": False, "has_results": False, "has_qualifying": True},
        }


class _FakeOpenF1:
    async def get_session_features(self, race, session="race", drivers=None, live=False):
        if session != "fp1":
            return {"ok": False, "source": "openf1", "reason": "openf1_session_unavailable"}
        return {
            "ok": True,
            "source": "openf1",
            "session": "practice1",
            "laps": {
                "drivers": {
                    "1": {"driver_number": 1, "driver_code": "VER", "laps": 5, "best_lap": 91.2, "representative_lap": 91.4, "median_lap": 91.7, "compounds": ["SOFT"]},
                    "16": {"driver_number": 16, "driver_code": "LEC", "laps": 7, "best_lap": 90.9, "representative_lap": 91.1, "median_lap": 91.5, "compounds": ["MEDIUM"]},
                }
            },
            "positions": {"drivers": {}},
            "intervals": {"drivers": {}},
            "stints": {"drivers": {}},
            "pits": {"drivers": {}},
            "raw_counts": {"laps": 12},
        }


if __name__ == "__main__":
    unittest.main()
