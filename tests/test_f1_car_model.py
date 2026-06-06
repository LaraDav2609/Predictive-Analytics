import unittest
from datetime import datetime, timezone

from f1_predictor.features.car_model import build_car_model_analysis, summarize_car_data
from models.f1 import Constructor, Driver, Race


class F1CarModelTests(unittest.TestCase):
    def setUp(self):
        self.race = Race(
            round=6,
            name="Monaco Grand Prix",
            circuit="Circuit de Monaco",
            country="Monaco",
            date=datetime(2026, 6, 7, tzinfo=timezone.utc),
        )
        self.drivers = [
            Driver(id="russell", number=63, code="RUS", first_name="George", last_name="Russell", nationality="British", team="Mercedes", points=88, position=2),
            Driver(id="antonelli", number=12, code="ANT", first_name="Kimi", last_name="Antonelli", nationality="Italian", team="Mercedes", points=131, position=1),
            Driver(id="leclerc", number=16, code="LEC", first_name="Charles", last_name="Leclerc", nationality="Monegasque", team="Ferrari", points=75, position=3),
            Driver(id="hamilton", number=44, code="HAM", first_name="Lewis", last_name="Hamilton", nationality="British", team="Ferrari", points=72, position=4),
        ]
        self.constructors = [
            Constructor(id="mercedes", name="Mercedes", nationality="German", points=219, position=1),
            Constructor(id="ferrari", name="Ferrari", nationality="Italian", points=147, position=2),
        ]
        self.features = {
            "drivers": {
                "russell": {"qualifying_pace_score": 0.78, "race_pace_score": 0.76, "reliability_score": 0.90},
                "antonelli": {"qualifying_pace_score": 0.82, "race_pace_score": 0.80, "reliability_score": 0.86},
                "leclerc": {"qualifying_pace_score": 0.80, "race_pace_score": 0.72, "reliability_score": 0.82},
                "hamilton": {"qualifying_pace_score": 0.77, "race_pace_score": 0.74, "reliability_score": 0.84},
            },
            "constructors": {
                "mercedes": {"team_score": 0.88, "recent_points": 168, "reliability_score": 0.89},
                "ferrari": {"team_score": 0.78, "recent_points": 120, "reliability_score": 0.83},
            },
        }

    def test_builds_fallback_car_model_with_bounded_modifiers(self):
        payload = build_car_model_analysis(
            race=self.race,
            drivers=self.drivers,
            constructors=self.constructors,
            features=self.features,
            session="qualifying",
            track={"street_circuit": True, "qualifying_importance": 0.96, "tire_stress": 0.28, "source": "test"},
            weather={"chaos_score": 0.05, "source": "test_weather"},
            tires={"degradation_rate": 0.28},
        )

        self.assertTrue(payload["ok"])
        self.assertIn("mercedes", payload["constructors"])
        self.assertIn("antonelli", payload["drivers"])
        self.assertGreater(payload["constructors"]["mercedes"]["scores"]["overall_car_score"], 0.0)
        self.assertGreaterEqual(payload["drivers"]["antonelli"]["car_modifier"], 0.94)
        self.assertLessEqual(payload["drivers"]["antonelli"]["car_modifier"], 1.06)
        self.assertIn("car_data", payload["constructors"]["mercedes"]["missing_data"])

    def test_openf1_laps_pits_stints_and_car_data_raise_coverage(self):
        openf1_session = {
            "laps": {
                "drivers": {
                    "63": {"representative_lap": 72.1, "laps": 18},
                    "12": {"representative_lap": 72.0, "laps": 20},
                    "16": {"representative_lap": 72.9, "laps": 17},
                    "44": {"representative_lap": 72.8, "laps": 18},
                }
            },
            "pits": {
                "drivers": {
                    "63": {"avg_pit_duration": 2.42},
                    "12": {"avg_pit_duration": 2.38},
                    "16": {"avg_pit_duration": 2.70},
                    "44": {"avg_pit_duration": 2.65},
                }
            },
            "stints": {
                "drivers": {
                    "63": {"avg_stint_laps": 24, "compounds": ["MEDIUM"]},
                    "12": {"avg_stint_laps": 25, "compounds": ["MEDIUM"]},
                    "16": {"avg_stint_laps": 19, "compounds": ["SOFT"]},
                    "44": {"avg_stint_laps": 20, "compounds": ["SOFT"]},
                }
            },
            "car_data": {
                "drivers": {
                    "63": {"max_speed": 296, "drs_usage": 0.20},
                    "12": {"max_speed": 298, "drs_usage": 0.22},
                    "16": {"max_speed": 290, "drs_usage": 0.18},
                    "44": {"max_speed": 291, "drs_usage": 0.18},
                }
            },
            "positions": {
                "drivers": {
                    "63": {"position": 2},
                    "12": {"position": 1},
                    "16": {"position": 4},
                    "44": {"position": 3},
                }
            },
            "intervals": {
                "drivers": {
                    "63": {"gap_to_leader": "+2.0", "interval": "+2.0"},
                    "12": {"gap_to_leader": "Leader", "interval": None},
                    "16": {"gap_to_leader": "+8.0", "interval": "+3.0"},
                    "44": {"gap_to_leader": "+6.5", "interval": "+4.5"},
                }
            },
            "race_control": {"chaos_score": 0.08},
            "track": {"racing_line": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]},
        }

        payload = build_car_model_analysis(
            race=self.race,
            drivers=self.drivers,
            constructors=self.constructors,
            features={**self.features, "openf1_session": openf1_session},
            session="race",
            track={"street_circuit": True, "high_speed": False, "qualifying_importance": 0.96, "tire_stress": 0.28, "source": "test"},
            weather={"chaos_score": 0.04, "source": "test_weather"},
            tires={"degradation_rate": 0.28},
            openf1_session=openf1_session,
        )

        mercedes = payload["constructors"]["mercedes"]
        ferrari = payload["constructors"]["ferrari"]
        self.assertGreater(mercedes["confidence"], ferrari["confidence"] - 0.01)
        self.assertGreater(mercedes["scores"]["car_pace"], ferrari["scores"]["car_pace"])
        self.assertTrue(mercedes["source_coverage"]["has_car_data"])
        self.assertTrue(mercedes["source_coverage"]["has_positions"])
        self.assertTrue(mercedes["source_coverage"]["has_intervals"])
        self.assertTrue(mercedes["source_coverage"]["has_location_trace"])

    def test_summarize_car_data(self):
        rows = [
            {"driver_number": 63, "speed": 290, "throttle": 98, "brake": 0, "drs": 12},
            {"driver_number": 63, "speed": 300, "throttle": 100, "brake": 1, "drs": 0},
        ]
        summary = summarize_car_data(rows, self.drivers)

        self.assertFalse(summary["missing_data"])
        self.assertEqual(300, summary["drivers"]["63"]["max_speed"])
        self.assertEqual("RUS", summary["drivers"]["63"]["driver_code"])


if __name__ == "__main__":
    unittest.main()
