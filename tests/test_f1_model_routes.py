import unittest

from api import f1_routes
from analytics.f1_predictor import F1Predictor
from models.f1 import Constructor, Driver, Race


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


if __name__ == "__main__":
    unittest.main()
