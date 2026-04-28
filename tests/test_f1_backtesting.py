import unittest

from f1_predictor.backtesting.metrics import evaluate_race, summarize_races
from f1_predictor.backtesting.replay import RaceReplayBuilder
from f1_predictor.backtesting.service import F1BacktestService
from f1_predictor.models.configs import get_model_config
from f1_predictor.models.registry import F1ModelRegistry
from models.f1 import DriverRacePrediction, RacePrediction


class F1BacktestingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.races = [_race(1, "Opening GP", "max", "charles"), _race(2, "Second GP", "charles", "max")]

    def test_replay_removes_future_races_from_features(self):
        replay = RaceReplayBuilder().build(2025, self.races, 2)

        self.assertEqual(1, replay.features["completed_races"])
        self.assertEqual(1, replay.features["drivers"]["max"]["starts"])
        self.assertEqual(1, replay.features["drivers"]["charles"]["starts"])
        self.assertEqual(25.0, replay.drivers[0].points)
        self.assertEqual("charles", replay.actual_winner)

    def test_metrics_calculate_accuracy_and_probability_quality(self):
        prediction = RacePrediction(
            driver_predictions={
                "max": DriverRacePrediction(
                    driver_id="max",
                    driver_name="Max Verstappen",
                    win_prob=0.70,
                    podium_prob=0.90,
                    top5_prob=0.95,
                    predicted_position=1,
                    expected_finish=1.3,
                ),
                "charles": DriverRacePrediction(
                    driver_id="charles",
                    driver_name="Charles Leclerc",
                    win_prob=0.30,
                    podium_prob=0.80,
                    top5_prob=0.90,
                    predicted_position=2,
                    expected_finish=2.1,
                ),
            }
        )
        metrics = evaluate_race(prediction, _parse_actual(self.races[0]))

        self.assertTrue(metrics["winner_hit"])
        self.assertEqual(2, metrics["podium_hits"])
        self.assertLess(metrics["log_loss"], 0.4)

    async def test_service_runs_without_sentiment_or_external_calls(self):
        service = F1BacktestService(_FakeClient(self.races))

        result = await service.backtest_season(2025, include_races=True)

        self.assertTrue(result["ok"])
        self.assertEqual(2, result["race_count"])
        self.assertIn("recommended_weights", result)
        self.assertEqual(2, len(result["races"]))

    async def test_model_compare_returns_best_model(self):
        service = F1BacktestService(_FakeClient(self.races))

        result = await service.compare_season(2025)

        self.assertTrue(result["ok"])
        self.assertIn("best_model_id", result)
        self.assertGreaterEqual(len(result["models"]), 3)
        self.assertTrue(all("model_id" in model for model in result["models"]))

    def test_model_config_falls_back_to_production(self):
        config = get_model_config("does_not_exist")

        self.assertEqual("production_v1", config.model_id)
        self.assertTrue(any(model["model_id"] == "conservative_v1" for model in F1ModelRegistry.list_models()))

    async def test_backtest_accepts_selected_model_id(self):
        service = F1BacktestService(_FakeClient(self.races))

        result = await service.backtest_season(2025, model_id="conservative_v1")

        self.assertTrue(result["ok"])
        self.assertEqual("conservative_v1", result["model_id"])

    async def test_current_partial_season_is_rejected_without_flag(self):
        service = F1BacktestService(_FakeClient(self.races, season=2025))

        result = await service.backtest_season(2025)

        self.assertFalse(result["ok"])
        self.assertTrue(result["partial"])

    def test_summary_aggregates_calibration_buckets(self):
        prediction = RacePrediction(
            driver_predictions={
                "max": DriverRacePrediction(driver_id="max", driver_name="Max Verstappen", win_prob=0.70, podium_prob=0.9, top5_prob=0.95),
                "charles": DriverRacePrediction(driver_id="charles", driver_name="Charles Leclerc", win_prob=0.30, podium_prob=0.8, top5_prob=0.9),
            }
        )
        metrics = evaluate_race(prediction, _parse_actual(self.races[0]))
        summary = summarize_races([{
            "metrics": metrics,
            "actual_winner": "max",
            "probability_distribution": [
                {"driver_id": "max", "win_probability": 0.70},
                {"driver_id": "charles", "win_probability": 0.30},
            ],
        }])

        self.assertEqual(1, summary["race_count"])
        self.assertTrue(summary["calibration_buckets"])


class _FakeClient:
    def __init__(self, races, season=2026):
        self.season = season
        self._races = races

    async def get_historical_race_results(self, season):
        return self._races


def _race(round_num, race_name, winner_id, runner_up_id):
    drivers = {
        "max": {
            "driverId": "max",
            "permanentNumber": "1",
            "code": "VER",
            "givenName": "Max",
            "familyName": "Verstappen",
            "nationality": "Dutch",
        },
        "charles": {
            "driverId": "charles",
            "permanentNumber": "16",
            "code": "LEC",
            "givenName": "Charles",
            "familyName": "Leclerc",
            "nationality": "Monegasque",
        },
    }
    constructors = {
        "max": {"constructorId": "red_bull", "name": "Red Bull Racing", "nationality": "Austrian"},
        "charles": {"constructorId": "ferrari", "name": "Ferrari", "nationality": "Italian"},
    }
    order = [winner_id, runner_up_id]
    return {
        "season": "2025",
        "round": str(round_num),
        "raceName": race_name,
        "date": f"2025-03-{round_num + 1:02d}",
        "time": "14:00:00Z",
        "Circuit": {
            "circuitId": "bahrain",
            "circuitName": "Bahrain International Circuit",
            "Location": {"country": "Bahrain", "locality": "Sakhir", "lat": "26.0325", "long": "50.5106"},
        },
        "Results": [
            {
                "position": str(index + 1),
                "grid": str(index + 1),
                "points": "25" if index == 0 else "18",
                "status": "Finished",
                "Driver": drivers[driver_id],
                "Constructor": constructors[driver_id],
            }
            for index, driver_id in enumerate(order)
        ],
    }


def _parse_actual(raw):
    rows = []
    for item in raw["Results"]:
        driver = item["Driver"]
        rows.append({
            "driver_id": driver["driverId"],
            "position": int(item["position"]),
            "points": float(item["points"]),
        })
    return rows


if __name__ == "__main__":
    unittest.main()
