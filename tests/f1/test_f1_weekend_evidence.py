import unittest
from datetime import datetime, timezone

from sports.f1.models.f1 import Driver, Race
from sports.f1.predictor.features.weekend import (
    apply_weekend_evidence_adjustments,
    build_weekend_evidence,
    has_real_grid_evidence,
    has_real_practice_evidence,
)
from sports.f1.predictor.simulation.session_projection import build_session_projection
from sports.f1.models.f1 import Constructor


class F1WeekendEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.drivers = [
            Driver(id="max", number=1, code="VER", first_name="Max", last_name="Verstappen", nationality="Dutch", team="Red Bull", points=100, position=1),
            Driver(id="charles", number=16, code="LEC", first_name="Charles", last_name="Leclerc", nationality="Monegasque", team="Ferrari", points=80, position=2),
        ]
        self.race = Race(
            round=6,
            name="Monaco Grand Prix",
            circuit="Circuit de Monaco",
            country="Monaco",
            date=datetime.now(timezone.utc),
        )

    def test_practice_rows_produce_pace_long_run_sector_compound_and_confidence(self):
        evidence = build_weekend_evidence(
            race=self.race,
            drivers=self.drivers,
            profile={},
            openf1_sessions={
                "fp1": {
                    "ok": True,
                    "source": "openf1",
                    "laps": {
                        "drivers": {
                            "1": {
                                "best_lap": 72.1,
                                "representative_lap": 72.4,
                                "long_run_lap": 74.0,
                                "laps": 18,
                                "compounds": ["SOFT", "MEDIUM"],
                                "representative_sector_1": 20.0,
                                "representative_sector_2": 31.0,
                                "representative_sector_3": 21.1,
                            },
                            "16": {
                                "best_lap": 72.8,
                                "representative_lap": 73.1,
                                "long_run_lap": 74.8,
                                "laps": 16,
                                "compounds": ["SOFT"],
                                "representative_sector_1": 20.2,
                                "representative_sector_2": 31.4,
                                "representative_sector_3": 21.2,
                            },
                        }
                    },
                    "stints": {"drivers": {}},
                    "raw_counts": {"laps": 34},
                }
            },
            session="qualifying",
        )

        self.assertTrue(has_real_practice_evidence(evidence))
        max_practice = evidence["drivers"]["max"]["practice"]
        self.assertEqual(["MEDIUM", "SOFT"], max_practice["compounds"])
        self.assertGreater(max_practice["practice_pace_score"], evidence["drivers"]["charles"]["practice"]["practice_pace_score"])
        self.assertGreater(max_practice["long_run_score"], evidence["drivers"]["charles"]["practice"]["long_run_score"])
        self.assertGreater(max_practice["sector_score"], 0.0)
        self.assertGreater(max_practice["confidence"], 0.2)

    def test_grid_evidence_overrides_historical_grid_assumptions(self):
        evidence = build_weekend_evidence(
            race=self.race,
            drivers=self.drivers,
            profile={
                "qualifying": [
                    {"driver_id": "max", "driver_code": "VER", "position": 1, "grid": 11},
                    {"driver_id": "charles", "driver_code": "LEC", "position": 2, "grid": 2},
                ]
            },
            openf1_sessions={},
            session="race",
        )
        self.assertTrue(has_real_grid_evidence(evidence))
        features = apply_weekend_evidence_adjustments(
            self.drivers,
            {"max": {"qualifying_pace_score": 0.8}, "charles": {"qualifying_pace_score": 0.8}},
            evidence,
            session_stage="race",
        )

        self.assertEqual(11, features["max"]["grid_position"])
        self.assertEqual(10, features["max"]["grid_penalty"])
        self.assertEqual("jolpica_qualifying_grid", features["max"]["grid_source"])

    def test_simulation_uses_weekend_evidence_grid_penalty_without_quali_rows(self):
        constructors = [
            Constructor(id="red_bull", name="Red Bull", nationality="Austrian", points=100, position=1),
            Constructor(id="ferrari", name="Ferrari", nationality="Italian", points=80, position=2),
        ]
        evidence = build_weekend_evidence(
            race=self.race,
            drivers=self.drivers,
            profile={
                "qualifying": [
                    {"driver_id": "max", "driver_code": "VER", "position": 1, "grid": 11},
                    {"driver_id": "charles", "driver_code": "LEC", "position": 2, "grid": 2},
                ]
            },
            openf1_sessions={},
            session="race",
        )
        features = {
            "weekend_evidence": evidence,
            "drivers": {
                "max": {"form_score": 0.8, "race_pace_score": 0.8, "qualifying_pace_score": 0.8, "reliability_score": 0.95},
                "charles": {"form_score": 0.8, "race_pace_score": 0.8, "qualifying_pace_score": 0.8, "reliability_score": 0.95},
            },
            "constructors": {
                "red bull": {"team_score": 0.8},
                "ferrari": {"team_score": 0.8},
            },
        }
        prediction = {
            "driver_predictions": {
                "max": {"win_prob": 0.5, "form_score": 0.8, "team_score": 0.8, "race_pace_score": 0.8, "qualifying_pace_score": 0.8, "reliability_score": 0.95},
                "charles": {"win_prob": 0.5, "form_score": 0.8, "team_score": 0.8, "race_pace_score": 0.8, "qualifying_pace_score": 0.8, "reliability_score": 0.95},
            }
        }

        projection = build_session_projection(self.race, self.drivers, constructors, prediction, features, [], [], [], session="race")
        max_row = next(row for row in projection["simulations"] if row["driver_id"] == "max")

        self.assertEqual(11, max_row["components"]["grid_position"])
        self.assertLess(max_row["components"]["grid_modifier"], 1.0)
        self.assertIn("weekend_evidence", projection)


if __name__ == "__main__":
    unittest.main()
