import unittest
from datetime import datetime, timezone

from sports.f1.predictor.data_quality import build_data_quality_report
from sports.f1.predictor.data_quality.leakage_guard import leakage_guard_status
from sports.f1.predictor.data_quality.source_registry import get_source_policy
from sports.f1.predictor.truth import build_race_truth_snapshot
from sports.f1.models.f1 import Driver, Race


class F1DataQualityTests(unittest.TestCase):
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

    def test_source_policy_caps_sentiment_and_estimated_sources(self):
        self.assertLess(get_source_policy("f1_sentiment").max_influence, 0.05)
        self.assertLess(get_source_policy("estimated").reliability, get_source_policy("live").reliability)

    def test_estimated_truth_gets_low_quality_and_bias_warning(self):
        truth = build_race_truth_snapshot(self.race, self.drivers, profile={"ok": True})
        report = build_data_quality_report(truth=truth, profile={"ok": True}, probabilities=[])

        self.assertLess(report["data_quality_score"], 0.55)
        self.assertTrue(report["estimated_data_discounted"])
        self.assertTrue(any(w["type"] == "estimated_source_bias" for w in report["bias_warnings"]))

    def test_disagreement_lowers_confidence_when_model_leader_differs_from_live_leader(self):
        truth = build_race_truth_snapshot(
            self.race,
            self.drivers,
            live=True,
            live_state={
                "ok": True,
                "source_mode": "live",
                "confidence": 0.9,
                "drivers": [
                    {"driver_id": "charles", "position": 1},
                    {"driver_id": "max", "position": 2},
                ],
            },
        )
        report = build_data_quality_report(
            truth=truth,
            profile={"ok": True},
            probabilities=[
                {"driver_id": "max", "win_probability": 0.65},
                {"driver_id": "charles", "win_probability": 0.35},
            ],
            live=True,
        )

        self.assertGreater(report["source_disagreement"]["score"], 0)
        self.assertTrue(any(w["type"] == "source_disagreement" for w in report["bias_warnings"]))

    def test_leakage_guard_blocks_future_features_for_pre_weekend(self):
        guard = leakage_guard_status(
            stage="pre_weekend",
            truth={},
            profile={},
            feature_groups=["historical", "qualifying", "official_results"],
        )

        self.assertEqual("blocked", guard["status"])
        self.assertIn("qualifying", guard["blocked_feature_groups"])
        self.assertIn("official_results", guard["blocked_feature_groups"])


if __name__ == "__main__":
    unittest.main()
