import unittest
from datetime import datetime, timezone

from sports.f1.predictor.simulation.session_projection import build_session_projection
from sports.f1.models.f1 import Constructor, Driver, Race


class F1SessionProjectionTests(unittest.TestCase):
    def _two_driver_context(self, race_name="Monaco Grand Prix", circuit="Circuit de Monaco", country="Monaco"):
        drivers = [
            Driver(id="low", number=16, code="LOW", first_name="Low", last_name="Speed", nationality="Test", team="Low Team", points=100, position=1),
            Driver(id="high", number=1, code="HIG", first_name="High", last_name="Speed", nationality="Test", team="High Team", points=100, position=2),
        ]
        constructors = [
            Constructor(id="low_team", name="Low Team", nationality="Test", points=100, position=1),
            Constructor(id="high_team", name="High Team", nationality="Test", points=100, position=2),
        ]
        race = Race(
            round=6,
            name=race_name,
            circuit=circuit,
            country=country,
            date=datetime.now(timezone.utc),
        )
        features = {
            "drivers": {
                "low": {"form_score": 0.80, "race_pace_score": 0.80, "qualifying_pace_score": 0.80, "reliability_score": 0.95},
                "high": {"form_score": 0.80, "race_pace_score": 0.80, "qualifying_pace_score": 0.80, "reliability_score": 0.95},
            },
            "constructors": {
                "low team": {"team_score": 0.80, "reliability_score": 0.95},
                "high team": {"team_score": 0.80, "reliability_score": 0.95},
            },
            "car_model": {
                "source": "test_car_model",
                "confidence": 0.90,
                "drivers": {
                    "low": {
                        "overall_car_score": 0.80,
                        "car_modifier": 1.0,
                        "confidence": 0.90,
                        "scores": {
                            "qualifying_pace": 0.80,
                            "race_pace": 0.80,
                            "long_run_pace": 0.80,
                            "low_speed_track_fit": 0.93,
                            "high_speed_track_fit": 0.54,
                            "braking_traction_strength": 0.92,
                            "drs_straight_line_strength": 0.50,
                            "tire_behavior": 0.72,
                            "strategy_operations": 0.76,
                            "reliability": 0.88,
                        },
                    },
                    "high": {
                        "overall_car_score": 0.80,
                        "car_modifier": 1.0,
                        "confidence": 0.90,
                        "scores": {
                            "qualifying_pace": 0.80,
                            "race_pace": 0.80,
                            "long_run_pace": 0.80,
                            "low_speed_track_fit": 0.54,
                            "high_speed_track_fit": 0.94,
                            "braking_traction_strength": 0.52,
                            "drs_straight_line_strength": 0.95,
                            "tire_behavior": 0.72,
                            "strategy_operations": 0.76,
                            "reliability": 0.88,
                        },
                    },
                },
            },
        }
        prediction = {
            "driver_predictions": {
                "low": {"win_prob": 0.50, "form_score": 0.80, "team_score": 0.80, "race_pace_score": 0.80, "qualifying_pace_score": 0.80, "reliability_score": 0.95},
                "high": {"win_prob": 0.50, "form_score": 0.80, "team_score": 0.80, "race_pace_score": 0.80, "qualifying_pace_score": 0.80, "reliability_score": 0.95},
            }
        }
        return race, drivers, constructors, features, prediction

    def test_grid_penalty_lowers_race_projection_without_changing_quali_session(self):
        drivers = [
            Driver(id="max", number=1, code="VER", first_name="Max", last_name="Verstappen", nationality="Dutch", team="Red Bull", points=100, position=1),
            Driver(id="charles", number=16, code="LEC", first_name="Charles", last_name="Leclerc", nationality="Monegasque", team="Ferrari", points=100, position=2),
        ]
        constructors = [
            Constructor(id="red_bull", name="Red Bull", nationality="Austrian", points=100, position=1),
            Constructor(id="ferrari", name="Ferrari", nationality="Italian", points=100, position=2),
        ]
        race = Race(
            round=6,
            name="Monaco Grand Prix",
            circuit="Circuit de Monaco",
            country="Monaco",
            date=datetime.now(timezone.utc),
        )
        features = {
            "drivers": {
                "max": {"form_score": 0.80, "race_pace_score": 0.80, "qualifying_pace_score": 0.80, "reliability_score": 0.95},
                "charles": {"form_score": 0.80, "race_pace_score": 0.80, "qualifying_pace_score": 0.80, "reliability_score": 0.95},
            },
            "constructors": {
                "red bull": {"team_score": 0.80, "reliability_score": 0.95},
                "ferrari": {"team_score": 0.80, "reliability_score": 0.95},
            },
        }
        prediction = {
            "driver_predictions": {
                "max": {"win_prob": 0.50, "form_score": 0.80, "team_score": 0.80, "race_pace_score": 0.80, "qualifying_pace_score": 0.80, "reliability_score": 0.95},
                "charles": {"win_prob": 0.50, "form_score": 0.80, "team_score": 0.80, "race_pace_score": 0.80, "qualifying_pace_score": 0.80, "reliability_score": 0.95},
            }
        }
        clean_qualifying = [
            {"driver_id": "max", "position": 1},
            {"driver_id": "charles", "position": 2},
        ]
        penalized_qualifying = [
            {"driver_id": "max", "position": 1, "grid": 11, "grid_penalty": 10},
            {"driver_id": "charles", "position": 2, "grid": 2},
        ]

        clean = build_session_projection(race, drivers, constructors, prediction, features, clean_qualifying, [], [], session="race")
        penalized = build_session_projection(race, drivers, constructors, prediction, features, penalized_qualifying, [], [], session="race")
        clean_max = next(row for row in clean["simulations"] if row["driver_id"] == "max")
        penalized_max = next(row for row in penalized["simulations"] if row["driver_id"] == "max")

        self.assertEqual(11, penalized_max["components"]["grid_position"])
        self.assertEqual(10, penalized_max["components"]["grid_penalty"])
        self.assertLess(penalized_max["components"]["grid_modifier"], 1.0)
        self.assertLess(penalized_max["win_probability"], clean_max["win_probability"])

    def test_car_trait_fit_favors_low_speed_car_at_monaco(self):
        race, drivers, constructors, features, prediction = self._two_driver_context()

        projection = build_session_projection(race, drivers, constructors, prediction, features, [], [], [], session="qualifying")
        low = next(row for row in projection["simulations"] if row["driver_id"] == "low")
        high = next(row for row in projection["simulations"] if row["driver_id"] == "high")

        self.assertGreater(low["components"]["car_trait_fit"], high["components"]["car_trait_fit"])
        self.assertGreater(low["components"]["car_trait_modifier"], high["components"]["car_trait_modifier"])
        self.assertIn("low-speed", " ".join(low["components"]["car_trait_explanations"]))

    def test_car_trait_fit_favors_straight_line_car_at_monza(self):
        race, drivers, constructors, features, prediction = self._two_driver_context(
            race_name="Italian Grand Prix",
            circuit="Autodromo Nazionale Monza",
            country="Italy",
        )

        projection = build_session_projection(race, drivers, constructors, prediction, features, [], [], [], session="race")
        low = next(row for row in projection["simulations"] if row["driver_id"] == "low")
        high = next(row for row in projection["simulations"] if row["driver_id"] == "high")

        self.assertGreater(high["components"]["car_trait_fit"], low["components"]["car_trait_fit"])
        self.assertGreater(high["components"]["car_trait_modifier"], low["components"]["car_trait_modifier"])
        self.assertIn("high-speed", " ".join(high["components"]["car_trait_explanations"]))

    def test_projection_exposes_practice_evidence_components_for_dashboard(self):
        race, drivers, constructors, features, prediction = self._two_driver_context()
        features["drivers"]["low"].update({
            "practice_pace_score": 0.91,
            "practice_qualifying_evidence_score": 0.88,
            "practice_race_evidence_score": 0.84,
            "practice_long_run_score": 0.80,
            "practice_sector_score": 0.86,
            "practice_fuel_uncertainty": 0.18,
            "practice_compounds": ["SOFT"],
            "practice_source": "openf1_practice_laps",
            "practice_confidence": 0.72,
        })

        projection = build_session_projection(race, drivers, constructors, prediction, features, [], [], [], session="qualifying")
        low = next(row for row in projection["simulations"] if row["driver_id"] == "low")

        self.assertEqual(0.91, low["components"]["practice_pace_score"])
        self.assertEqual(0.86, low["components"]["practice_sector_score"])
        self.assertEqual(["SOFT"], low["components"]["practice_compounds"])
        self.assertEqual("openf1_practice_laps", low["components"]["practice_source"])


if __name__ == "__main__":
    unittest.main()
