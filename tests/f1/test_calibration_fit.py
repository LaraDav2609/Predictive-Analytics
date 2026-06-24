import os
import tempfile
import unittest

from sports.f1.api import f1_routes
from sports.f1.predictor.probability.empirical_calibration import CALIBRATION_ARTIFACT_ENV


class _CalibBacktester:
    """Returns enough backtest rows to clear MIN_SAMPLES (60) / MIN_POSITIVES (8)."""

    async def backtest_season(self, season, include_races=False, allow_partial=False, model_id=None, stage="pre_weekend"):
        runs = []
        for r in range(16):
            winner = "d0" if r % 2 == 0 else f"d{1 + (r % 4)}"
            dist = [{"driver_id": f"d{i}", "win_probability": 0.3 if i == 0 else 0.0778} for i in range(10)]
            runs.append({"round": r + 1, "actual_winner": winner, "probability_distribution": dist})
        return {"ok": True, "season": season, "races": runs}


class CalibrationFitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._bt = f1_routes._backtester
        self._env = os.environ.get(CALIBRATION_ARTIFACT_ENV)
        f1_routes._backtester = lambda: _CalibBacktester()
        self._tmp = tempfile.mkdtemp()
        self._path = os.path.join(self._tmp, "cal.json")

    def tearDown(self):
        f1_routes._backtester = self._bt
        if self._env is None:
            os.environ.pop(CALIBRATION_ARTIFACT_ENV, None)
        else:
            os.environ[CALIBRATION_ARTIFACT_ENV] = self._env
        for fn in (lambda: os.remove(self._path), lambda: os.rmdir(self._tmp)):
            try:
                fn()
            except OSError:
                pass

    async def test_fit_single_season_in_sample_enables_calibrator(self):
        res = await f1_routes.post_f1_fit_calibration(start_season=2024, end_season=2024, method="isotonic", path=self._path)
        self.assertTrue(res["ok"])
        self.assertEqual(res["evaluation"], "in_sample")
        self.assertIn("brier", res)
        self.assertTrue(res["enabled"])
        self.assertTrue(os.path.exists(self._path))
        self.assertGreaterEqual(res["fit_positive_count"], 8)
        # The engine now resolves a non-identity calibrator from the saved artifact.
        from sports.f1.predictor.probability.engine import _get_empirical_calibrator
        self.assertFalse(_get_empirical_calibrator().is_identity())

    async def test_fit_multi_season_is_held_out(self):
        res = await f1_routes.post_f1_fit_calibration(start_season=2024, end_season=2025, method="isotonic", path=self._path)
        self.assertTrue(res["ok"])
        self.assertEqual(res["evaluation"], "held_out")
        self.assertGreaterEqual(res["brier"]["raw"], 0.0)
        self.assertEqual(res["seasons_used"], [2024, 2025])

    async def test_state_endpoint_reports_enabled_after_fit(self):
        await f1_routes.post_f1_fit_calibration(start_season=2024, end_season=2024, path=self._path)
        state = await f1_routes.get_f1_calibration_state()
        self.assertTrue(state["enabled"])
        self.assertEqual(state["method"], "isotonic")


if __name__ == "__main__":
    unittest.main()
