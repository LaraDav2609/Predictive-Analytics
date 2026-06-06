import unittest
from datetime import datetime, timezone

from f1_predictor.truth import build_race_truth_snapshot
from models.f1 import Driver, Race


class F1RaceTruthTests(unittest.TestCase):
    def setUp(self):
        self.drivers = [
            Driver(id="max", number=1, code="VER", first_name="Max", last_name="Verstappen", nationality="Dutch", team="Red Bull Racing", points=25, wins=1, position=1),
            Driver(id="charles", number=16, code="LEC", first_name="Charles", last_name="Leclerc", nationality="Monegasque", team="Ferrari", points=18, wins=0, position=2),
        ]
        self.race = Race(
            round=1,
            name="Bahrain Grand Prix",
            circuit="Bahrain International Circuit",
            country="Bahrain",
            date=datetime.now(timezone.utc),
        )

    def test_official_results_become_historical_truth(self):
        profile = {
            "ok": True,
            "results": [
                {"driver_id": "charles", "position": 1, "driver_code": "LEC", "team": "Ferrari", "points": 25, "status": "Finished"},
                {"driver_id": "max", "position": 2, "driver_code": "VER", "team": "Red Bull Racing", "points": 18, "status": "Finished"},
            ],
        }

        truth = build_race_truth_snapshot(self.race, self.drivers, profile=profile)

        self.assertTrue(truth["ok"])
        self.assertEqual("historical", truth["source_mode"])
        self.assertEqual(1, truth["by_driver_id"]["charles"]["position"])
        self.assertEqual("Finished", truth["by_driver_id"]["max"]["result_status"])
        self.assertNotIn("live_timing", truth["missing_groups"])

    def test_live_state_overrides_recent_rows_when_live(self):
        openf1_session = {
            "ok": True,
            "source": "openf1",
            "positions": {"drivers": {"1": {"position": 2}, "16": {"position": 1}}},
            "raw_counts": {"positions": 2, "intervals": 2, "laps": 2},
        }
        live_state = {
            "ok": True,
            "source_mode": "live",
            "confidence": 0.91,
            "drivers": [
                {"driver_id": "max", "position": 1, "gap_to_leader": "0.000", "interval": "0.000", "compound": "MEDIUM", "tyre_age": 8},
                {"driver_id": "charles", "position": 2, "gap_to_leader": "+1.200", "interval": "+1.200", "compound": "HARD", "tyre_age": 8},
            ],
            "signals": {"chaos_score": 0.2},
        }

        truth = build_race_truth_snapshot(
            self.race,
            self.drivers,
            openf1_session=openf1_session,
            live_state=live_state,
            live=True,
        )

        self.assertEqual("live", truth["source_mode"])
        self.assertEqual(0.91, truth["confidence"])
        self.assertEqual(1, truth["by_driver_id"]["max"]["position"])
        self.assertEqual("MEDIUM", truth["by_driver_id"]["max"]["compound"])

    def test_missing_sources_are_explicitly_estimated(self):
        truth = build_race_truth_snapshot(self.race, self.drivers, profile={"ok": True})

        self.assertEqual("estimated", truth["source_mode"])
        self.assertTrue(truth["is_estimated"])
        self.assertIn("live_timing", truth["missing_groups"])
        self.assertIn("official_race_classification", truth["missing_groups"])


if __name__ == "__main__":
    unittest.main()
