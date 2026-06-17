import unittest
from datetime import datetime, timezone

from sports.f1.predictor.features.builder import F1FeatureBuilder
from sports.f1.predictor.features.performance import build_performance_table
from sports.f1.predictor.backtesting.replay import _constructor_feature, _driver_feature
from sports.f1.predictor.service import F1PredictionService
from sports.f1.predictor.models.baseline import BaselineRaceModel
from sports.f1.predictor.models.ml_simulator import MLSimulatorRaceModel
from sports.f1.predictor.models.registry import F1ModelRegistry
from sports.f1.data.openf1_client import _summarize_laps, _summarize_stints
from sports.f1.models.f1 import Constructor, Driver, Race


class F1PredictorModuleTests(unittest.TestCase):
    def setUp(self):
        self.drivers = [
            Driver(
                id="antonelli",
                number=12,
                code="ANT",
                first_name="Andrea Kimi",
                last_name="Antonelli",
                nationality="Italian",
                team="Mercedes",
                points=75,
                wins=1,
                position=1,
            ),
            Driver(
                id="russell",
                number=63,
                code="RUS",
                first_name="George",
                last_name="Russell",
                nationality="British",
                team="Mercedes",
                points=62,
                wins=0,
                position=2,
            ),
        ]
        self.constructors = [
            Constructor(id="mercedes", name="Mercedes", nationality="German", points=137, wins=1, position=1),
        ]
        self.features = {
            "completed_races": 3,
            "total_races": 22,
            "drivers": {
                "antonelli": {"form_score": 0.82, "reliability_score": 0.90, "recent_wins": 1, "recent_podiums": 2, "recent_starts": 3, "starts": 3},
                "russell": {"form_score": 0.70, "reliability_score": 0.86, "recent_wins": 0, "recent_podiums": 2, "recent_starts": 3, "starts": 120},
            },
            "constructors": {
                "mercedes": {"team_score": 0.84, "recent_points": 137},
            },
        }
        self.race = Race(
            round=4,
            name="Miami Grand Prix",
            circuit="Miami International Autodrome",
            country="USA",
            date=datetime.now(timezone.utc),
        )

    def test_performance_table_returns_driver_and_car_scores(self):
        rows = build_performance_table(self.drivers, self.constructors, self.features["drivers"], self.features["constructors"])

        self.assertIn("antonelli", rows)
        self.assertGreater(rows["antonelli"]["driver_skill_score"], 0)
        self.assertGreater(rows["antonelli"]["car_performance_score"], 0)
        self.assertGreater(rows["antonelli"]["form_score"], rows["russell"]["form_score"])
        self.assertGreater(rows["russell"]["performance_score"], 0)

    def test_feature_builder_keeps_fallback_missing_data_flags(self):
        snapshot = F1FeatureBuilder(self.drivers, self.constructors, self.features, sentiment={}).build(self.race)

        self.assertIn("antonelli", snapshot.drivers)
        self.assertIn("weather", snapshot.missing_data)
        self.assertIn("sentiment", snapshot.missing_data)
        self.assertEqual("circuit_trait_registry", snapshot.track["source"])
        self.assertEqual("track_tire_trait_registry", snapshot.tires["source"])

    def test_feature_builder_blends_openf1_practice_pace_when_laps_exist(self):
        features = {
            **self.features,
            "openf1_session": {
                "ok": True,
                "source": "openf1_practice_laps",
                "laps": {
                    "drivers": {
                        "12": {
                            "best_lap": 72.1,
                            "representative_lap": 73.0,
                            "long_run_lap": 73.4,
                            "laps": 18,
                            "representative_sector_1": 23.1,
                            "representative_sector_2": 25.0,
                            "representative_sector_3": 24.9,
                            "track_evolution_delta": 0.4,
                            "lap_distribution": {"sample_size": 18, "p10": 72.4, "median": 73.0, "p90": 73.6, "spread_p90_p10": 1.2},
                            "compounds": ["MEDIUM"],
                        },
                        "63": {
                            "best_lap": 73.8,
                            "representative_lap": 75.0,
                            "long_run_lap": 75.2,
                            "laps": 16,
                            "representative_sector_1": 24.0,
                            "representative_sector_2": 25.8,
                            "representative_sector_3": 25.2,
                            "track_evolution_delta": 0.4,
                            "lap_distribution": {"sample_size": 16, "p10": 74.4, "median": 75.0, "p90": 76.2, "spread_p90_p10": 1.8},
                            "compounds": ["HARD"],
                        },
                    }
                },
                "stints": {
                    "drivers": {
                        "12": {"avg_stint_laps": 18, "compounds": ["MEDIUM"]},
                        "63": {"avg_stint_laps": 16, "compounds": ["HARD"]},
                    }
                },
            },
        }

        snapshot = F1FeatureBuilder(self.drivers, self.constructors, features, sentiment={}).build(
            self.race,
            session_stage="practice",
        )
        antonelli = snapshot.drivers["antonelli"]
        russell = snapshot.drivers["russell"]

        self.assertIn("practice_pace_score", antonelli)
        self.assertGreater(antonelli["practice_pace_score"], russell["practice_pace_score"])
        self.assertIn("practice_sector_scores", antonelli)
        self.assertGreater(antonelli["practice_sector_score"], russell["practice_sector_score"])
        self.assertGreater(antonelli["practice_long_run_score"], russell["practice_long_run_score"])
        self.assertLess(antonelli["practice_fuel_uncertainty"], 0.35)
        self.assertGreater(antonelli["practice_distribution_stability"], 0.0)
        self.assertEqual(18, antonelli["practice_lap_distribution"]["sample_size"])
        self.assertEqual(["MEDIUM"], antonelli["practice_compounds"])
        self.assertGreater(antonelli["qualifying_pace_score"], russell["qualifying_pace_score"])

    def test_openf1_lap_summary_keeps_sector_long_run_and_evolution(self):
        rows = [
            {"driver_number": 12, "lap_duration": 75.0, "duration_sector_1": 24.0, "duration_sector_2": 26.0, "duration_sector_3": 25.0, "compound": "MEDIUM", "date": "2026-06-05T10:00:00Z"},
            {"driver_number": 12, "lap_duration": 73.0, "duration_sector_1": 23.0, "duration_sector_2": 25.0, "duration_sector_3": 25.0, "compound": "MEDIUM", "date": "2026-06-05T10:01:00Z"},
            {"driver_number": 12, "lap_duration": 72.8, "duration_sector_1": 22.9, "duration_sector_2": 24.9, "duration_sector_3": 25.0, "compound": "SOFT", "date": "2026-06-05T10:02:00Z"},
            {"driver_number": 12, "lap_duration": 74.2, "duration_sector_1": 23.4, "duration_sector_2": 25.5, "duration_sector_3": 25.3, "compound": "SOFT", "date": "2026-06-05T10:03:00Z"},
            {"driver_number": 12, "lap_duration": 72.4, "duration_sector_1": 22.8, "duration_sector_2": 24.7, "duration_sector_3": 24.9, "compound": "SOFT", "date": "2026-06-05T10:04:00Z"},
            {"driver_number": 12, "lap_duration": 73.8, "duration_sector_1": 23.2, "duration_sector_2": 25.3, "duration_sector_3": 25.3, "compound": "SOFT", "date": "2026-06-05T10:05:00Z"},
            {"driver_number": 12, "lap_duration": 74.0, "duration_sector_1": 23.3, "duration_sector_2": 25.4, "duration_sector_3": 25.3, "compound": "SOFT", "date": "2026-06-05T10:06:00Z"},
            {"driver_number": 12, "lap_duration": 74.1, "duration_sector_1": 23.5, "duration_sector_2": 25.2, "duration_sector_3": 25.4, "compound": "SOFT", "date": "2026-06-05T10:07:00Z"},
        ]

        summary = _summarize_laps(rows, self.drivers)
        antonelli = summary["drivers"]["12"]

        self.assertEqual(8, antonelli["laps"])
        self.assertEqual(["MEDIUM", "SOFT"], antonelli["compounds"])
        self.assertIn("representative_sector_1", antonelli)
        self.assertIsNotNone(antonelli["long_run_lap"])
        self.assertGreater(antonelli["track_evolution_delta"], 0)
        self.assertIn("lap_distribution", antonelli)
        self.assertEqual(8, antonelli["lap_distribution"]["sample_size"])
        self.assertGreater(antonelli["pace_stability"], 0)

    def test_openf1_stint_summary_keeps_compound_sequence_and_distribution(self):
        rows = [
            {"driver_number": 12, "stint_number": 1, "lap_start": 1, "lap_end": 15, "compound": "SOFT"},
            {"driver_number": 12, "stint_number": 2, "lap_start": 16, "lap_end": 42, "compound": "HARD"},
            {"driver_number": 63, "stint_number": 1, "lap_start": 1, "lap_end": 20, "compound": "MEDIUM"},
        ]

        summary = _summarize_stints(rows, self.drivers)
        antonelli = summary["drivers"]["12"]

        self.assertEqual(["SOFT", "HARD"], antonelli["compound_sequence"])
        self.assertEqual(27, antonelli["max_stint_laps"])
        self.assertEqual(27, antonelli["final_stint_laps"])
        self.assertEqual(2, antonelli["stint_lap_distribution"]["sample_size"])

    def test_prediction_service_works_without_sentiment(self):
        service = F1PredictionService()
        service.load(self.drivers, self.constructors, self.features, sentiment={})

        prediction = service.predict_race(self.race)
        antonelli = prediction.driver_predictions["antonelli"]

        self.assertEqual("f1-live-historical-sentiment-v4", prediction.model_version)
        self.assertGreater(antonelli.win_prob, 0)
        self.assertIsNotNone(antonelli.performance_score)
        self.assertIsNotNone(antonelli.track_fit_score)
        self.assertIsNotNone(antonelli.dnf_prob)
        self.assertEqual("Neutral", antonelli.sentiment_label)

    def test_model_registry_dispatches_only_ml_simulator_to_ml_adapter(self):
        self.assertIsInstance(F1ModelRegistry(model_id="production_v1")._model, BaselineRaceModel)
        self.assertIsInstance(F1ModelRegistry(model_id="calibrated_candidate_v1")._model, BaselineRaceModel)
        self.assertIsInstance(F1ModelRegistry(model_id="conservative_v1")._model, BaselineRaceModel)
        self.assertIsInstance(F1ModelRegistry(model_id="ml_simulator_v1")._model, MLSimulatorRaceModel)

    def test_ml_simulator_model_uses_existing_prediction_shape(self):
        features = {
            **self.features,
            "ml_simulator_iterations": 300,
            "weekend_evidence": {
                "ok": True,
                "session": "race",
                "source_mode": "historical",
                "confidence": 0.72,
                "coverage_counts": {"practice_drivers": 2, "grid_drivers": 2, "race_input_drivers": 0},
                "drivers": {
                    "antonelli": {
                        "practice": {
                            "representative_lap": 72.8,
                            "long_run_lap": 73.2,
                            "best_lap": 72.2,
                            "race_evidence_score": 0.88,
                            "qualifying_evidence_score": 0.90,
                            "pace_stability": 0.82,
                            "fuel_uncertainty": 0.18,
                            "confidence": 0.82,
                            "compounds": ["MEDIUM"],
                        },
                        "grid": {"grid_position": 1, "qualifying_position": 1, "confidence": 0.86},
                    },
                    "russell": {
                        "practice": {
                            "representative_lap": 74.2,
                            "long_run_lap": 74.6,
                            "best_lap": 73.8,
                            "race_evidence_score": 0.50,
                            "qualifying_evidence_score": 0.52,
                            "pace_stability": 0.62,
                            "fuel_uncertainty": 0.28,
                            "confidence": 0.72,
                            "compounds": ["HARD"],
                        },
                        "grid": {"grid_position": 2, "qualifying_position": 2, "confidence": 0.86},
                    },
                },
            },
        }
        service = F1PredictionService(model_id="ml_simulator_v1")
        service.load(self.drivers, self.constructors, features, sentiment={})

        prediction = service.predict_race(self.race)
        antonelli = prediction.driver_predictions["antonelli"]
        russell = prediction.driver_predictions["russell"]
        total_win = sum(item.win_prob for item in prediction.driver_predictions.values())

        self.assertIn("ml-simulator-v1", prediction.model_version)
        self.assertIn("sports.f1.ml Monte Carlo simulator", prediction.data_sources)
        self.assertEqual("ml_simulator_v1", prediction.model_id)
        self.assertEqual("evidence_fallback", prediction.ml_input_source)
        self.assertEqual("trained_ml_artifacts_unavailable", prediction.ml_fallback_reason)
        self.assertFalse(prediction.trained_artifacts_used)
        self.assertIn("practice", prediction.evidence_groups_used)
        self.assertEqual(300, prediction.simulator_iterations)
        self.assertAlmostEqual(1.0, total_win, places=2)
        self.assertGreater(antonelli.win_prob, russell.win_prob)
        self.assertGreaterEqual(antonelli.podium_prob, antonelli.win_prob)
        self.assertIsNotNone(antonelli.expected_finish)
        self.assertIsNotNone(antonelli.dnf_prob)

    def test_ml_simulator_uses_trained_artifact_overlay_when_supplied(self):
        features = {
            **self.features,
            "ml_simulator_iterations": 300,
            "ml_trained_inputs": {
                "drivers": {
                    "antonelli": {
                        "pace_mean_seconds": 74.5,
                        "pace_sigma_seconds": 0.35,
                        "dnf_hazard_per_lap": 0.001,
                        "sources": ["trained_gbm_pace", "trained_gbm_dnf"],
                        "confidence": 0.82,
                    },
                    "russell": {
                        "pace_mean_seconds": 71.5,
                        "pace_sigma_seconds": 0.28,
                        "dnf_hazard_per_lap": 0.0004,
                        "sources": ["trained_gbm_pace", "trained_gbm_dnf"],
                        "confidence": 0.84,
                    },
                }
            },
        }
        service = F1PredictionService(model_id="ml_simulator_v1")
        service.load(self.drivers, self.constructors, features, sentiment={})

        prediction = service.predict_race(self.race)

        self.assertEqual("trained_artifacts", prediction.ml_input_source)
        self.assertTrue(prediction.trained_artifacts_used)
        self.assertIsNone(prediction.ml_fallback_reason)
        self.assertIn("trained_gbm_pace", prediction.ml_provider_sources)
        self.assertGreater(
            prediction.driver_predictions["russell"].win_prob,
            prediction.driver_predictions["antonelli"].win_prob,
        )

    def test_ml_simulator_uses_fitted_model_artifacts_when_supplied(self):
        class FakePaceModel:
            def predict(self, frame):
                return [74.5, 71.5][:len(frame)]

        class FakeDNFModel:
            def hazard_per_lap(self, frame):
                return [0.0002, 0.004][:len(frame)]

        features = {
            **self.features,
            "ml_simulator_iterations": 300,
            "ml_trained_inputs": {
                "pace_model": FakePaceModel(),
                "dnf_model": FakeDNFModel(),
                "pace_model_confidence": 0.83,
                "dnf_model_confidence": 0.79,
            },
        }
        service = F1PredictionService(model_id="ml_simulator_v1")
        service.load(self.drivers, self.constructors, features, sentiment={})

        prediction = service.predict_race(self.race)

        self.assertEqual("trained_artifacts", prediction.ml_input_source)
        self.assertTrue(prediction.trained_artifacts_used)
        self.assertIn("trained_gbm_pace", prediction.ml_provider_sources)
        self.assertIn("trained_gbm_dnf", prediction.ml_provider_sources)
        self.assertGreater(
            prediction.driver_predictions["russell"].win_prob,
            prediction.driver_predictions["antonelli"].win_prob,
        )

    def test_ml_simulator_trained_dnf_overlay_changes_dnf_probability(self):
        features = {
            **self.features,
            "ml_simulator_iterations": 500,
            "ml_trained_inputs": {
                "drivers": {
                    "antonelli": {"dnf_hazard_per_lap": 0.0002, "sources": ["trained_gbm_dnf"], "confidence": 0.80},
                    "russell": {"dnf_hazard_per_lap": 0.04, "sources": ["trained_gbm_dnf"], "confidence": 0.80},
                }
            },
        }
        service = F1PredictionService(model_id="ml_simulator_v1")
        service.load(self.drivers, self.constructors, features, sentiment={})

        prediction = service.predict_race(self.race)

        self.assertGreater(
            prediction.driver_predictions["russell"].dnf_prob,
            prediction.driver_predictions["antonelli"].dnf_prob,
        )

    def test_ml_simulator_seed_is_stable_for_same_inputs(self):
        features = {**self.features, "ml_simulator_iterations": 300}
        service_a = F1PredictionService(model_id="ml_simulator_v1")
        service_b = F1PredictionService(model_id="ml_simulator_v1")
        service_a.load(self.drivers, self.constructors, features, sentiment={})
        service_b.load(self.drivers, self.constructors, features, sentiment={})

        first = service_a.predict_race(self.race)
        second = service_b.predict_race(self.race)

        self.assertEqual(
            {k: v.win_prob for k, v in first.driver_predictions.items()},
            {k: v.win_prob for k, v in second.driver_predictions.items()},
        )

    def test_championship_leader_does_not_override_race_evidence(self):
        drivers = [
            Driver(
                id="leader",
                number=1,
                code="LED",
                first_name="Points",
                last_name="Leader",
                nationality="Test",
                team="Legacy",
                points=160,
                wins=4,
                position=1,
            ),
            Driver(
                id="pace",
                number=2,
                code="PAC",
                first_name="Race",
                last_name="Pace",
                nationality="Test",
                team="Momentum",
                points=55,
                wins=0,
                position=6,
            ),
        ]
        constructors = [
            Constructor(id="legacy", name="Legacy", nationality="Test", points=190, wins=4, position=1),
            Constructor(id="momentum", name="Momentum", nationality="Test", points=80, wins=0, position=4),
        ]
        features = {
            "completed_races": 3,
            "total_races": 22,
            "drivers": {
                "leader": {
                    "form_score": 0.38,
                    "reliability_score": 0.68,
                    "qualifying_pace_score": 0.35,
                    "race_pace_score": 0.36,
                    "teammate_score": 0.42,
                    "trend_score": 0.35,
                    "recent_wins": 0,
                    "recent_podiums": 0,
                    "recent_starts": 3,
                    "starts": 120,
                },
                "pace": {
                    "form_score": 0.91,
                    "reliability_score": 0.93,
                    "qualifying_pace_score": 0.88,
                    "race_pace_score": 0.94,
                    "teammate_score": 0.72,
                    "trend_score": 0.80,
                    "recent_wins": 1,
                    "recent_podiums": 3,
                    "recent_starts": 3,
                    "starts": 70,
                },
            },
            "constructors": {
                "legacy": {"team_score": 0.48, "recent_points": 8, "reliability_score": 0.72},
                "momentum": {"team_score": 0.88, "recent_points": 58, "reliability_score": 0.90},
            },
        }
        service = F1PredictionService()
        service.load(drivers, constructors, features, sentiment={})

        prediction = service.predict_race(self.race)
        leader = prediction.driver_predictions["leader"]
        pace = prediction.driver_predictions["pace"]

        self.assertGreater(pace.win_prob, leader.win_prob)
        self.assertLess(leader.win_prob, 0.45)

    def test_backtest_replay_does_not_turn_points_into_perfect_form(self):
        driver = Driver(
            id="leader",
            code="LED",
            first_name="Points",
            last_name="Leader",
            nationality="Test",
            team="Legacy",
            points=300,
            wins=8,
            position=1,
        )
        constructor = Constructor(id="legacy", name="Legacy", nationality="Test", points=300, wins=8, position=1)

        driver_feature = _driver_feature(driver, recent=[], all_results=[], teammate_delta=0.0, lookback=8)
        constructor_feature = _constructor_feature(constructor, recent=[], lookback=8)

        self.assertLess(driver_feature["form_score"], 0.45)
        self.assertLess(constructor_feature["team_score"], 0.50)

    def test_feature_builder_uses_weather_and_reliability_context(self):
        features = {
            **self.features,
            "weather_by_round": {
                "4": {
                    "rain_probability": 0.65,
                    "rain_intensity": 0.4,
                    "air_temperature": 27.0,
                    "track_temperature": None,
                    "humidity": 78.0,
                    "wind_speed": 18.0,
                    "weather_change_probability": 0.55,
                    "chaos_score": 0.6,
                    "source": "test_weather",
                    "confidence": 0.7,
                    "missing_data": False,
                }
            },
        }
        snapshot = F1FeatureBuilder(self.drivers, self.constructors, features, sentiment={}).build(self.race)

        self.assertEqual("test_weather", snapshot.weather["source"])
        self.assertNotIn("weather", snapshot.missing_data)
        self.assertGreater(snapshot.reliability["antonelli"]["incident_dnf_probability"], 0.01)

    def test_monte_carlo_simulation_returns_dnf_and_expected_finish(self):
        service = F1PredictionService()
        service.load(self.drivers, self.constructors, self.features, sentiment={})
        prediction = service.predict_race(self.race).model_dump(mode="json")

        simulation = service.build_session_simulation(
            race=self.race,
            prediction=prediction,
            qualifying=[],
            sprint=[],
            results=[],
            session="race",
            live=False,
        )

        first = simulation["simulations"][0]
        self.assertIn("monte_carlo", simulation)
        self.assertIn("dnf_probability", first)
        self.assertIn("expected_finish", first)


if __name__ == "__main__":
    unittest.main()
