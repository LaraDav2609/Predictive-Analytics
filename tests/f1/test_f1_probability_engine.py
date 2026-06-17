import unittest

from sports.f1.predictor.probability import build_probability_audit, enrich_probability_payload
from sports.f1.predictor.probability.calibration import apply_temperature, build_calibration_profile
from sports.f1.predictor.probability.stage import detect_stage
from sports.f1.predictor.scoring.normalization import prior_score
from sports.f1.predictor.simulation.monte_carlo import MonteCarloSimulator


class F1ProbabilityEngineTests(unittest.TestCase):
    def test_stage_detection(self):
        self.assertEqual("pre_weekend", detect_stage({"ok": True}, {}))
        self.assertEqual(
            "pre_weekend",
            detect_stage({"sessions": [{"name": "Practice 1", "status": "completed"}]}, {}),
        )
        self.assertEqual("post_qualifying", detect_stage({"qualifying": [{"driver_id": "max", "position": 1}]}, {}))
        self.assertEqual(
            "pre_weekend",
            detect_stage({"context": {"has_qualifying": True, "completed_sessions": 3}}, {}),
        )
        self.assertEqual(
            "practice_available",
            detect_stage({}, {"weekend_evidence": {"practice": {"available": True, "coverage_count": 18}}}),
        )
        self.assertEqual(
            "post_qualifying",
            detect_stage({}, {"weekend_evidence": {"grid": {"available": True, "coverage_count": 20}}}),
        )
        self.assertEqual(
            "pre_weekend",
            detect_stage({"qualifying": []}, {"source_mode": "live", "drivers": [{"driver_id": "max"}]}, live=True),
        )
        self.assertEqual(
            "live",
            detect_stage(
                {"qualifying": []},
                {
                    "source_mode": "live",
                    "drivers": [{"driver_id": f"d{i}", "position": i, "lap": 12} for i in range(1, 11)],
                    "weekend_evidence": {"race_inputs": {"available": True, "coverage_count": 10}},
                },
                live=True,
            ),
        )
        self.assertEqual("completed", detect_stage({"results": [{"driver_id": "max", "position": 1}]}, {}))

    def test_temperature_flattens_overconfident_probability(self):
        raw = {"max": 0.80, "charles": 0.20}
        calibrated = apply_temperature(raw, 1.25)

        self.assertLess(calibrated["max"], 0.80)
        self.assertGreater(calibrated["charles"], 0.20)
        self.assertAlmostEqual(sum(calibrated.values()), 1.0, places=3)

    def test_prior_score_does_not_saturate_front_runners(self):
        field = [object() for _ in range(22)]

        self.assertAlmostEqual(prior_score(1 / 22, field), 0.50, places=2)
        self.assertLess(prior_score(0.08, field), 0.75)
        self.assertLess(prior_score(0.12, field), 0.82)

    def test_probability_audit_adds_calibrated_fields(self):
        simulation = {
            "ok": True,
            "simulations": [
                {"driver_id": "max", "driver_code": "VER", "driver_name": "Max Verstappen", "win_probability": 0.70, "podium_probability": 0.90, "top5_probability": 0.96, "points_probability": 0.98, "expected_finish": 1.7},
                {"driver_id": "charles", "driver_code": "LEC", "driver_name": "Charles Leclerc", "win_probability": 0.30, "podium_probability": 0.60, "top5_probability": 0.88, "points_probability": 0.94, "expected_finish": 3.2},
            ],
        }

        audit = build_probability_audit(simulation, profile={}, truth={"source_mode": "estimated", "confidence": 0.2})

        self.assertEqual("pre_weekend", audit["stage"])
        self.assertIn("calibration_profile", audit)
        self.assertEqual(2, len(audit["probabilities"]))
        self.assertIn("finish_distribution", audit["probabilities"][0])

    def test_tyre_stress_calibration_depends_on_stage(self):
        track = {"tire_stress": 0.82, "qualifying_importance": 0.62, "overtaking_difficulty": 0.52}

        pre = build_calibration_profile("pre_weekend", track=track, truth={"confidence": 0.6})
        post = build_calibration_profile("post_qualifying", track=track, truth={"confidence": 0.6})

        self.assertGreater(pre.temperature, 1.18)
        self.assertLess(post.temperature, 0.95)
        self.assertTrue(any(item["target"] == "tires" and item["magnitude"] == "-0.03" for item in post.adjustments))

    def test_low_evidence_governance_caps_overconfident_top_pick(self):
        simulation = {
            "ok": True,
            "simulations": [
                {"driver_id": "leader", "driver_code": "LED", "win_probability": 0.86},
                {"driver_id": "pace", "driver_code": "PAC", "win_probability": 0.10},
                {"driver_id": "outsider", "driver_code": "OUT", "win_probability": 0.04},
            ],
        }

        audit = build_probability_audit(
            simulation,
            profile={},
            truth={"source_mode": "estimated", "confidence": 0.18, "missing_groups": ["live_positions", "gaps"]},
        )
        leader = next(row for row in audit["probabilities"] if row["driver_id"] == "leader")

        self.assertTrue(audit["probability_governance"]["applied"])
        self.assertLessEqual(leader["governed_probability"], audit["probability_governance"]["top_cap"])
        self.assertLess(leader["calibrated_probability"], leader["raw_probability"])

    def test_enrich_payload_keeps_old_fields_and_adds_new_fields(self):
        payload = {
            "simulations": [
                {"driver_id": "max", "win_probability": 0.65, "podium_probability": 0.90, "top5_probability": 0.95, "points_probability": 0.98, "expected_finish": 1.8},
                {"driver_id": "charles", "win_probability": 0.35, "podium_probability": 0.65, "top5_probability": 0.90, "points_probability": 0.94, "expected_finish": 2.7},
            ],
        }

        enriched = enrich_probability_payload(payload, profile={}, truth={"source_mode": "estimated", "confidence": 0.3})
        first = enriched["simulations"][0]

        self.assertIn("win_probability", first)
        self.assertIn("raw_probability", first)
        self.assertIn("calibrated_probability", first)
        self.assertIn("probability_audit", enriched)

    def test_enrich_payload_uses_finish_distribution_for_points_buckets(self):
        payload = {
            "simulations": [
                {
                    "driver_id": "max",
                    "win_probability": 0.65,
                    "podium_probability": 0.99,
                    "top5_probability": 0.99,
                    "points_probability": 0.99,
                    "expected_finish": 4.0,
                    "finish_distribution": {"1": 0.10, "2": 0.10, "6": 0.80},
                },
                {
                    "driver_id": "charles",
                    "win_probability": 0.35,
                    "podium_probability": 0.99,
                    "top5_probability": 0.99,
                    "points_probability": 0.99,
                    "expected_finish": 5.0,
                    "finish_distribution": {"1": 0.05, "3": 0.10, "11": 0.85},
                },
            ],
        }

        enriched = enrich_probability_payload(payload, profile={}, truth={"source_mode": "recent", "confidence": 0.7})
        by_driver = {row["driver_id"]: row for row in enriched["simulations"]}

        self.assertAlmostEqual(0.20, by_driver["max"]["podium_probability"], places=3)
        self.assertAlmostEqual(1.00, by_driver["max"]["points_probability"], places=3)
        self.assertAlmostEqual(0.15, by_driver["charles"]["podium_probability"], places=3)
        self.assertAlmostEqual(0.15, by_driver["charles"]["points_probability"], places=3)

    def test_monte_carlo_emits_top5_and_finish_distribution(self):
        result = MonteCarloSimulator(iterations=25, seed=7).run({"max": 0.9, "charles": 0.7}, {"laps": 2})
        first = result["drivers"][0]

        self.assertIn("top5_probability", first)
        self.assertIn("finish_distribution", first)
        self.assertTrue(first["finish_distribution"])

    def test_monte_carlo_smoothing_keeps_longshots_nonzero(self):
        result = MonteCarloSimulator(iterations=40, seed=11).run(
            {"front": 0.82, "mid": 0.62, "longshot": 0.28},
            {"laps": 4},
        )
        by_driver = {row["driver_id"]: row for row in result["drivers"]}

        self.assertGreater(by_driver["longshot"]["win_probability"], 0.0)
        self.assertLess(by_driver["front"]["win_probability"], 0.95)

    def test_monte_carlo_uses_live_dynamics_time_anchors(self):
        result = MonteCarloSimulator(iterations=80, seed=9).run(
            {"leader": 0.75, "chaser": 0.75},
            {
                "laps": 3,
                "live_confidence": 0.9,
                "live_dynamics": {
                    "drivers": {
                        "leader": {"time_anchor_seconds": 0.0, "confidence": 0.9},
                        "chaser": {"time_anchor_seconds": 8.0, "confidence": 0.9},
                    }
                },
            },
        )

        by_driver = {row["driver_id"]: row for row in result["drivers"]}
        self.assertLess(by_driver["leader"]["expected_finish"], by_driver["chaser"]["expected_finish"])


if __name__ == "__main__":
    unittest.main()
