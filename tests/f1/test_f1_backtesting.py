import unittest

from sports.f1.predictor.backtesting.metrics import evaluate_race, summarize_races
from sports.f1.predictor.backtesting.replay import RaceReplayBuilder
from sports.f1.predictor.backtesting.service import F1BacktestService
from sports.f1.predictor.models.configs import get_model_config
from sports.f1.predictor.models.registry import F1ModelRegistry
from sports.f1.models.f1 import DriverRacePrediction, RacePrediction


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

    def test_replay_stage_controls_current_weekend_evidence(self):
        races = [
            self.races[0],
            _race(
                2,
                "Second GP",
                "charles",
                "max",
                qualifying_order=["max", "charles"],
                practice_laps={"max": 71.0, "charles": 72.5},
            ),
        ]

        pre_weekend = RaceReplayBuilder().build(2025, races, 2, stage="pre_weekend")
        post_quali = RaceReplayBuilder().build(2025, races, 2, stage="post_qualifying")
        practice = RaceReplayBuilder().build(2025, races, 2, stage="practice")

        self.assertNotIn("qualifying_source", pre_weekend.features["drivers"]["max"])
        self.assertNotIn("practice_pace_score", pre_weekend.features["drivers"]["max"])
        self.assertEqual("target_qualifying_results", post_quali.features["drivers"]["max"]["qualifying_source"])
        self.assertEqual(1, post_quali.features["drivers"]["max"]["grid_position"])
        self.assertTrue(post_quali.features["backtest"]["target_qualifying_used"])
        self.assertIn("practice_pace_score", practice.features["drivers"]["max"])
        self.assertTrue(practice.features["backtest"]["target_practice_used"])

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

    async def test_backtest_accepts_stage_and_reports_replay_evidence(self):
        races = [
            self.races[0],
            _race(2, "Second GP", "charles", "max", qualifying_order=["max", "charles"]),
        ]
        service = F1BacktestService(_FakeClient(races))

        result = await service.backtest_race(2025, 2, stage="post_quali")

        self.assertTrue(result["ok"])
        self.assertEqual("post_qualifying", result["stage"])
        self.assertEqual("post_qualifying", result["replay_evidence"]["stage"])
        self.assertTrue(result["replay_evidence"]["backtest"]["target_qualifying_used"])

    async def test_backtest_scores_session_projection_probabilities(self):
        service = F1BacktestService(_FakeClient(self.races))

        result = await service.backtest_race(2025, 2, stage="pre_weekend")

        self.assertTrue(result["ok"])
        self.assertIn("probability_audit", result)
        self.assertIn("calibration_profile", result)
        self.assertIn("session-simulator", result["model_version"])
        self.assertTrue(result["probability_distribution"])
        self.assertIn("grid_position", next(iter(result["component_scores"].values())))

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
        self.assertTrue(summary["top_pick_calibration"])
        self.assertTrue(summary["stage_calibration"])
        self.assertEqual("unknown", summary["stage_calibration"][0]["stage"])

    def test_summary_reports_stage_specific_calibration(self):
        rows = [
            {
                "stage": "pre_weekend",
                "actual_winner": "max",
                "probability_distribution": [{"driver_id": "max", "win_probability": 0.40}],
                "metrics": {"winner_hit": True, "top_probability": 0.40, "winner_probability": 0.40, "log_loss": 0.9},
            },
            {
                "stage": "post_qualifying",
                "actual_winner": "charles",
                "probability_distribution": [{"driver_id": "max", "win_probability": 0.60}],
                "metrics": {"winner_hit": False, "top_probability": 0.60, "winner_probability": 0.20, "log_loss": 1.6},
            },
        ]

        summary = summarize_races(rows)
        by_stage = {item["stage"]: item for item in summary["stage_calibration"]}

        self.assertIn("pre_weekend", by_stage)
        self.assertIn("post_qualifying", by_stage)
        self.assertEqual(1.0, by_stage["pre_weekend"]["winner_accuracy"])
        self.assertEqual(0.0, by_stage["post_qualifying"]["winner_accuracy"])


class _FakeClient:
    def __init__(self, races, season=2026):
        self.season = season
        self._races = races

    async def get_historical_race_results(self, season):
        return self._races

    async def get_historical_qualifying_results(self, season):
        return [
            {"round": race.get("round"), "QualifyingResults": race.get("QualifyingResults") or []}
            for race in self._races
            if race.get("QualifyingResults")
        ]


def _race(round_num, race_name, winner_id, runner_up_id, qualifying_order=None, practice_laps=None):
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
    raw = {
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
    if qualifying_order:
        raw["QualifyingResults"] = [
            {
                "position": str(index + 1),
                "Driver": drivers[driver_id],
                "Constructor": constructors[driver_id],
                "Q1": "1:20.000",
                "Q2": "1:19.500",
                "Q3": f"1:18.{index:03d}",
            }
            for index, driver_id in enumerate(qualifying_order)
        ]
    if practice_laps:
        raw["PracticeResults"] = [
            {
                "driver_id": driver_id,
                "best_lap": lap,
                "representative_lap": lap + 0.4,
                "laps": 18,
            }
            for driver_id, lap in practice_laps.items()
        ]
    return raw


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
