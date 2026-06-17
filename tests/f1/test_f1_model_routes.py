import unittest
from tempfile import TemporaryDirectory

from common.ml.bridge.outcome_publisher import InMemoryOutcomePublisher
from sports.f1.api import f1_routes
from sports.f1.predictor.backtesting.evidence_cache import WeekendEvidenceCacheWriter
from sports.f1.analytics.f1_predictor import F1Predictor
from sports.f1.models.f1 import Constructor, Driver, Race


class F1ModelRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = _FakeF1Client()
        self.predictor = F1Predictor()
        self.sentiment = {
            "drivers": {"max": {"score": 0.0}, "charles": {"score": 0.0}},
            "teams": {"red bull racing": {"score": 0.0}, "ferrari": {"score": 0.0}},
            "items": [],
        }
        self.predictor.load_drivers(self.client.get_drivers(), self.client.get_constructors(), self.client.features, sentiment=self.sentiment)
        f1_routes.client = self.client
        f1_routes.predictor = self.predictor
        f1_routes.openf1 = None
        f1_routes.live_engine = None
        f1_routes.live_recorder = None
        f1_routes.storage = None
        f1_routes._outcome_publisher_override = None
        f1_routes._last_bridge_publish_status = {"ok": None, "reason": "not_requested", "record_count": 0}
        f1_routes._clear_static_response_caches()

    async def test_models_endpoint_lists_registry(self):
        result = await f1_routes.get_f1_models()

        self.assertTrue(result["ok"])
        self.assertEqual("production_v1", result["default_model_id"])
        self.assertGreaterEqual(len(result["models"]), 3)

    async def test_health_endpoint_reports_missing_sources(self):
        self.predictor.load_drivers(self.client.get_drivers(), self.client.get_constructors(), self.client.features, sentiment={})
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

    async def test_simulation_defaults_to_production_model(self):
        result = await f1_routes.get_race_simulation(1)

        self.assertTrue(result["ok"])
        self.assertEqual("production_v1", result["model_id"])
        self.assertEqual("production_v1", result["selected_model_id"])
        self.assertGreater(result["bridge_record_count"], 0)

    async def test_simulation_accepts_ml_simulator_model_id(self):
        result = await f1_routes.get_race_simulation(1, model_id="ml_simulator_v1")

        self.assertTrue(result["ok"])
        self.assertEqual("ml_simulator_v1", result["model_id"])
        self.assertEqual("ml_simulator_v1", result["selected_model_id"])
        self.assertIn("ml_input_source", result)

    async def test_simulation_rejects_unknown_model_id(self):
        result = await f1_routes.get_race_simulation(1, model_id="not_real")

        self.assertFalse(result["ok"])
        self.assertEqual("invalid_model_id", result["code"])
        self.assertIn("ml_simulator_v1", result["available_models"])

    async def test_simulation_cache_is_model_specific(self):
        first = await f1_routes.get_race_simulation(1, model_id="production_v1")
        second = await f1_routes.get_race_simulation(1, model_id="ml_simulator_v1")
        first_cached = await f1_routes.get_race_simulation(1, model_id="production_v1")
        second_cached = await f1_routes.get_race_simulation(1, model_id="ml_simulator_v1")

        self.assertEqual("production_v1", first["model_id"])
        self.assertEqual("ml_simulator_v1", second["model_id"])
        self.assertTrue(first_cached["response_cache"]["hit"])
        self.assertTrue(second_cached["response_cache"]["hit"])
        self.assertEqual("production_v1", first_cached["model_id"])
        self.assertEqual("ml_simulator_v1", second_cached["model_id"])

    async def test_probability_audit_passes_model_id(self):
        result = await f1_routes.get_race_probability_audit(1, model_id="ml_simulator_v1")

        self.assertTrue(result["ok"])
        self.assertEqual("ml_simulator_v1", result["model_id"])
        self.assertIn("probabilities", result)

    async def test_live_probabilities_accept_ml_simulator_model_id(self):
        result = await f1_routes.get_f1_live_probabilities(1, model_id="ml_simulator_v1")

        self.assertTrue(result["ok"])
        self.assertEqual("ml_simulator_v1", result["model_id"])
        self.assertEqual("ml_simulator_v1", result["selected_model_id"])
        self.assertTrue(result["ml_live_runner_used"])
        self.assertIn("ml_live_runner", result)
        self.assertIn("probabilities", result)

    async def test_simulation_can_publish_outcome_probability_records(self):
        publisher = InMemoryOutcomePublisher()
        f1_routes._outcome_publisher_override = publisher

        result = await f1_routes.get_race_simulation(1, model_id="ml_simulator_v1", publish=True)

        self.assertTrue(result["ok"])
        self.assertTrue(result["bridge_publish"]["ok"])
        self.assertGreater(result["bridge_publish"]["record_count"], 0)
        self.assertTrue(publisher.channel_messages)

    async def test_bridge_publish_failure_does_not_crash_simulation(self):
        f1_routes._outcome_publisher_override = _FailingOutcomePublisher()

        result = await f1_routes.get_race_simulation(1, publish=True)

        self.assertTrue(result["ok"])
        self.assertFalse(result["bridge_publish"]["ok"])
        self.assertIn("RuntimeError", result["bridge_publish"]["reason"])

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

    async def test_deep_backtest_endpoint_reports_stage_matrix(self):
        result = await f1_routes.get_f1_deep_backtest(
            start_season=2025,
            end_season=2025,
            stages="pre_weekend,post_qualifying",
            model_ids="production_v1",
            include_races=False,
            allow_partial=True,
            cache_policy="refresh",
        )

        self.assertTrue(result["ok"])
        self.assertEqual("deep_backtest", result["mode"])
        self.assertEqual(["pre_weekend", "post_qualifying"], result["requested"]["stages"])
        self.assertEqual("refresh", result["requested"]["cache_policy"])
        self.assertEqual("refresh", result["load_strategy"]["cache_policy"])
        self.assertTrue(result["stage_matrix"])
        self.assertIn("compact_report", result)
        self.assertIn("coverage", result)
        self.assertTrue(result["recommendations"])
        self.assertTrue(result["load_strategy"]["upstream_calls_minimized"])
        self.assertEqual(1, self.client.historical_race_loads)

    async def test_backtest_evidence_cache_endpoint_builds_diagnostic_payload(self):
        f1_routes.openf1 = _FakeOpenF1()

        result = await f1_routes.build_f1_backtest_evidence_cache(
            1,
            season=2026,
            sessions="fp1,race",
            write=False,
        )

        self.assertTrue(result["ok"])
        self.assertFalse(result["write"])
        self.assertTrue(result["available"])
        self.assertEqual(2, result["coverage"]["practice_rows"])
        self.assertEqual(2, result["coverage"]["qualifying_rows"])
        self.assertTrue(result["openf1_available"])

    async def test_backtest_evidence_cache_endpoint_supports_historical_season(self):
        f1_routes.openf1 = _FakeOpenF1()

        result = await f1_routes.build_f1_backtest_evidence_cache(
            2,
            season=2025,
            sessions="fp1",
            write=False,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["historical_mode"])
        self.assertTrue(result["available"])
        self.assertEqual(2, result["coverage"]["practice_rows"])
        self.assertEqual(2, result["coverage"]["qualifying_rows"])
        self.assertEqual(0, result["coverage"]["race_input_drivers"])

    async def test_backtest_evidence_cache_profile_only_skips_openf1(self):
        f1_routes.openf1 = _ExplodingOpenF1()

        result = await f1_routes.build_f1_backtest_evidence_cache(
            2,
            season=2025,
            sessions="fp1,race",
            write=False,
            openf1_mode="profile_only",
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["historical_mode"])
        self.assertEqual("profile_only", result["openf1_mode"])
        self.assertFalse(result["openf1_available"])
        self.assertTrue(result["available"])
        self.assertGreaterEqual(result["coverage"]["practice_rows"], 0)
        self.assertEqual(2, result["coverage"]["qualifying_rows"])
        self.assertEqual(2, result["coverage"]["grid_rows"])

    async def test_backtest_evidence_cache_batch_limits_provider_pressure(self):
        f1_routes.openf1 = _FakeOpenF1()

        result = await f1_routes.build_f1_backtest_evidence_cache_batch(
            season=2025,
            sessions="fp1",
            write=False,
            start_round=1,
            end_round=2,
            max_rounds=1,
            delay_seconds=0,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["historical_mode"])
        self.assertEqual(1, result["processed"])
        self.assertEqual(1, result["result_count"])
        self.assertEqual([1, 2], result["candidate_rounds"])
        self.assertEqual(2, result["coverage"]["practice_rows"])
        self.assertIn("batch_size_limited_to_reduce_provider_pressure", result["limitations"])

    async def test_backtest_evidence_cache_batch_refreshes_existing_files_missing_practice(self):
        f1_routes.openf1 = _FakeOpenF1()
        with TemporaryDirectory() as tmp:
            writer = WeekendEvidenceCacheWriter(cache_dir=tmp)
            path = writer.path_for(2025, 1)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"coverage":{"practice_rows":0},"PracticeResults":[]}', encoding="utf-8")
            original_writer = f1_routes.WeekendEvidenceCacheWriter
            f1_routes.WeekendEvidenceCacheWriter = lambda: writer
            try:
                result = await f1_routes.build_f1_backtest_evidence_cache_batch(
                    season=2025,
                    sessions="fp1",
                    write=True,
                    start_round=1,
                    end_round=1,
                    max_rounds=1,
                    skip_existing=True,
                    refresh_missing_practice=True,
                    delay_seconds=0,
                )
            finally:
                f1_routes.WeekendEvidenceCacheWriter = original_writer

        self.assertTrue(result["ok"])
        self.assertEqual(1, result["processed"])
        self.assertEqual(0, result["skipped_existing"])
        self.assertEqual(2, result["coverage"]["practice_rows"])


class _FakeF1Client:
    season = 2026
    backtest_disk_cache = False

    def __init__(self):
        self.historical_race_loads = 0
        self.historical_qualifying_loads = 0
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

    async def get_historical_race_results(self, season):
        self.historical_race_loads += 1
        return [_historical_race(1, "max", "charles"), _historical_race(2, "charles", "max")]

    async def get_historical_qualifying_results(self, season):
        self.historical_qualifying_loads += 1
        return [
            {
                "round": "2",
                "QualifyingResults": [
                    {
                        "position": "1",
                        "Driver": {
                            "driverId": "max",
                            "permanentNumber": "1",
                            "code": "VER",
                            "givenName": "Max",
                            "familyName": "Verstappen",
                            "nationality": "Dutch",
                        },
                        "Constructor": {"constructorId": "red_bull", "name": "Red Bull Racing", "nationality": "Austrian"},
                        "Q1": "1:30.000",
                        "Q2": "1:29.500",
                        "Q3": "1:29.100",
                    },
                    {
                        "position": "2",
                        "Driver": {
                            "driverId": "charles",
                            "permanentNumber": "16",
                            "code": "LEC",
                            "givenName": "Charles",
                            "familyName": "Leclerc",
                            "nationality": "Monegasque",
                        },
                        "Constructor": {"constructorId": "ferrari", "name": "Ferrari", "nationality": "Italian"},
                        "Q1": "1:30.100",
                        "Q2": "1:29.700",
                        "Q3": "1:29.300",
                    },
                ],
            }
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

    async def get_prediction_features(self):
        return self.features

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


class _ExplodingOpenF1:
    async def get_session_features(self, *args, **kwargs):
        raise AssertionError("OpenF1 should not be called in profile-only mode")


class _FailingOutcomePublisher:
    def publish_batch(self, probs):
        raise RuntimeError("redis offline")


def _historical_race(round_num, winner_id, runner_up_id):
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
        "raceName": f"Test GP {round_num}",
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


if __name__ == "__main__":
    unittest.main()
