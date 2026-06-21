"""Tests for the empirical (isotonic / Platt) probability calibrator and its
serve-time application in the probability engine."""
import os
import random
import tempfile
import unittest

from sports.f1.predictor.probability import build_probability_audit
from sports.f1.predictor.probability.empirical_calibration import (
    CALIBRATION_ARTIFACT_ENV,
    EmpiricalCalibrator,
    brier_score,
    fit_isotonic,
    fit_platt,
    fit_winner_calibrator,
    load_default,
    winner_pairs_from_runs,
)


def _miscalibrated_dataset(n: int = 600, seed: int = 11):
    """Predicted probabilities that are systematically overconfident: a predicted
    p actually wins at rate 0.15 + 0.45*p (so 0.95 -> ~0.58)."""
    rng = random.Random(seed)
    probs, outcomes = [], []
    for _ in range(n):
        p = rng.choice([0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95])
        true_p = 0.15 + 0.45 * p
        probs.append(p)
        outcomes.append(1.0 if rng.random() < true_p else 0.0)
    return probs, outcomes


class EmpiricalCalibratorTests(unittest.TestCase):
    def test_identity_is_noop(self):
        cal = EmpiricalCalibrator.identity()
        self.assertTrue(cal.is_identity())
        self.assertAlmostEqual(cal.apply(0.42), 0.42)
        out = cal.calibrate_field({"a": 0.6, "b": 0.4}, normalize_to=None)
        self.assertAlmostEqual(out["a"], 0.6)
        self.assertAlmostEqual(out["b"], 0.4)

    def test_isotonic_reduces_brier(self):
        probs, outcomes = _miscalibrated_dataset()
        cal = fit_isotonic(probs, outcomes, source="test")
        self.assertFalse(cal.is_identity())
        self.assertEqual(cal.method, "isotonic")
        raw = brier_score(probs, outcomes)
        calibrated = brier_score([cal.apply(p) for p in probs], outcomes)
        self.assertLess(calibrated, raw)

    def test_isotonic_is_monotonic(self):
        probs, outcomes = _miscalibrated_dataset()
        cal = fit_isotonic(probs, outcomes)
        mapped = [cal.apply(i / 20) for i in range(21)]
        for earlier, later in zip(mapped, mapped[1:]):
            self.assertLessEqual(earlier, later + 1e-9)

    def test_isotonic_pulls_overconfident_down(self):
        probs, outcomes = _miscalibrated_dataset()
        cal = fit_isotonic(probs, outcomes)
        self.assertLess(cal.apply(0.95), 0.95)

    def test_platt_pulls_overconfident_down(self):
        probs, outcomes = _miscalibrated_dataset()
        cal = fit_platt(probs, outcomes)
        self.assertEqual(cal.method, "platt")
        self.assertFalse(cal.is_identity())
        self.assertLess(cal.apply(0.95), 0.95)

    def test_too_few_samples_returns_identity(self):
        self.assertTrue(fit_isotonic([0.5, 0.6, 0.7], [1, 0, 1]).is_identity())

    def test_calibrate_field_normalizes_to_one(self):
        probs, outcomes = _miscalibrated_dataset()
        cal = fit_isotonic(probs, outcomes)
        out = cal.calibrate_field({"a": 0.5, "b": 0.3, "c": 0.2}, normalize_to=1.0)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=4)

    def test_save_load_round_trip(self):
        probs, outcomes = _miscalibrated_dataset()
        cal = fit_isotonic(probs, outcomes, source="orig")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cal.json")
            cal.save(path)
            loaded = EmpiricalCalibrator.load(path)
        self.assertEqual(loaded.method, "isotonic")
        self.assertEqual(len(loaded.breakpoints), len(cal.breakpoints))
        for p in (0.1, 0.5, 0.9):
            self.assertAlmostEqual(loaded.apply(p), cal.apply(p), places=5)

    def test_winner_pairs_from_runs(self):
        runs = [
            {"actual_winner": "max", "probability_distribution": [
                {"driver_id": "max", "win_probability": 0.6},
                {"driver_id": "lec", "win_probability": 0.4},
            ]},
            {"actual_winner": {"driver_id": "lec"}, "probability_distribution": [
                {"driver_id": "max", "win_probability": 0.55},
                {"driver_id": "lec", "win_probability": 0.45},
            ]},
        ]
        pairs = winner_pairs_from_runs(runs)
        self.assertEqual(len(pairs), 4)
        self.assertIn((0.6, 1.0), pairs)
        self.assertIn((0.55, 0.0), pairs)
        self.assertIn((0.45, 1.0), pairs)

    def test_fit_winner_calibrator_from_runs(self):
        rng = random.Random(3)
        runs = []
        for _ in range(120):
            fav_wins = rng.random() < 0.5  # the 0.9 favourite only wins ~half the time
            runs.append({
                "actual_winner": "fav" if fav_wins else "dog",
                "probability_distribution": [
                    {"driver_id": "fav", "win_probability": 0.9},
                    {"driver_id": "dog", "win_probability": 0.1},
                ],
            })
        cal = fit_winner_calibrator(runs, method="isotonic", source="backtest")
        self.assertFalse(cal.is_identity())
        self.assertLess(cal.apply(0.9), 0.9)

    def test_fit_winner_calibrator_empty_is_identity(self):
        self.assertTrue(fit_winner_calibrator([]).is_identity())

    def test_load_default_identity_without_env(self):
        prior = os.environ.pop(CALIBRATION_ARTIFACT_ENV, None)
        try:
            self.assertTrue(load_default().is_identity())
        finally:
            if prior is not None:
                os.environ[CALIBRATION_ARTIFACT_ENV] = prior


class EngineCalibrationIntegrationTests(unittest.TestCase):
    def _simulation(self):
        return {
            "ok": True,
            "simulations": [
                {"driver_id": "max", "driver_code": "VER", "driver_name": "Max", "win_probability": 0.70,
                 "podium_probability": 0.9, "top5_probability": 0.96, "points_probability": 0.98, "expected_finish": 1.7},
                {"driver_id": "lec", "driver_code": "LEC", "driver_name": "Charles", "win_probability": 0.30,
                 "podium_probability": 0.6, "top5_probability": 0.88, "points_probability": 0.94, "expected_finish": 3.2},
            ],
        }

    @staticmethod
    def _reset_cache():
        import sports.f1.predictor.probability.engine as engine
        engine._EMPIRICAL_STATE = None

    def test_not_applied_without_artifact(self):
        prior = os.environ.pop(CALIBRATION_ARTIFACT_ENV, None)
        self._reset_cache()
        try:
            audit = build_probability_audit(self._simulation(), profile={}, truth={"confidence": 0.6})
            self.assertIn("empirical_calibration", audit)
            self.assertFalse(audit["empirical_calibration"]["applied"])
        finally:
            if prior is not None:
                os.environ[CALIBRATION_ARTIFACT_ENV] = prior
            self._reset_cache()

    def test_applied_with_artifact(self):
        probs, outcomes = _miscalibrated_dataset()
        cal = fit_isotonic(probs, outcomes)
        prior = os.environ.get(CALIBRATION_ARTIFACT_ENV)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cal.json")
            cal.save(path)
            os.environ[CALIBRATION_ARTIFACT_ENV] = path
            self._reset_cache()
            try:
                audit = build_probability_audit(self._simulation(), profile={}, truth={"confidence": 0.6})
                self.assertTrue(audit["empirical_calibration"]["applied"])
                self.assertEqual(audit["empirical_calibration"]["method"], "isotonic")
                self.assertAlmostEqual(sum(audit["calibrated_probabilities"].values()), 1.0, places=3)
            finally:
                if prior is None:
                    os.environ.pop(CALIBRATION_ARTIFACT_ENV, None)
                else:
                    os.environ[CALIBRATION_ARTIFACT_ENV] = prior
                self._reset_cache()


if __name__ == "__main__":
    unittest.main()
