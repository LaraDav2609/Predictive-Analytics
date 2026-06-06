import unittest
from datetime import datetime, timezone

from f1_predictor.features.builder import F1FeatureBuilder
from f1_predictor.features.performance import build_performance_table
from f1_predictor.backtesting.replay import _constructor_feature, _driver_feature
from f1_predictor.service import F1PredictionService
from models.f1 import Constructor, Driver, Race


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
