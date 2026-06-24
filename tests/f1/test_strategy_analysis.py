import unittest

from sports.f1.api import f1_routes
from sports.f1.predictor.features.strategy_analysis import analyze_race_strategy, driver_strategy_delta


def test_high_deg_favors_two_stops():
    s = analyze_race_strategy(laps=57, pit_loss_s=22, tire_stress=0.9, degradation_rate=0.9, safety_car_probability=0.4)
    assert s["recommended_stops"] == 2
    assert s["two_stop_delta_s"] < 0


def test_low_deg_favors_one_stop():
    s = analyze_race_strategy(laps=57, pit_loss_s=22, tire_stress=0.2, degradation_rate=0.2)
    assert s["recommended_stops"] == 1


def test_pit_window_within_race():
    s = analyze_race_strategy(laps=57, pit_loss_s=22)
    w = s["optimal_pit_window"]
    assert 0 < w["early"] <= w["ideal"] <= w["late"] < 57


def test_safety_car_value_scales_with_probability():
    low = analyze_race_strategy(laps=57, pit_loss_s=22, safety_car_probability=0.2)
    high = analyze_race_strategy(laps=57, pit_loss_s=22, safety_car_probability=0.8)
    assert high["safety_car_expected_value_s"] > low["safety_car_expected_value_s"]


def test_driver_delta_returns_tactic():
    base = analyze_race_strategy(laps=57, pit_loss_s=22, tire_stress=0.7, degradation_rate=0.7, undercut_strength=0.7)
    d = driver_strategy_delta(base, grid_position=8, tire_management=0.5)
    assert d["recommended_tactic"] in ("undercut", "overcut", "hold")
    assert "estimated_position_delta" in d


class _FakeRace:
    round = 1
    name = "Bahrain Grand Prix"
    circuit = "Bahrain"
    circuit_id = "bahrain"
    country = "Bahrain"


class _FakeDriver:
    def __init__(self, did):
        self.id = did
        self.code = did


class _FakeClient:
    season = 2026

    def get_race_by_round(self, r):
        return _FakeRace()

    def get_drivers(self):
        return [_FakeDriver("VER"), _FakeDriver("HAM")]


class _FakePredictor:
    _features: dict = {}


class StrategyRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._c = f1_routes.client
        self._p = f1_routes.predictor
        self._pr = f1_routes._race_profile_for_round
        f1_routes.client = _FakeClient()
        f1_routes.predictor = _FakePredictor()

        async def fake_profile(round_num):
            return {
                "qualifying": [
                    {"driver_id": "VER", "driver_code": "VER", "position": 1, "grid": 1},
                    {"driver_id": "HAM", "driver_code": "HAM", "position": 8, "grid": 8},
                ],
                "results": [],
            }

        f1_routes._race_profile_for_round = fake_profile

    def tearDown(self):
        f1_routes.client = self._c
        f1_routes.predictor = self._p
        f1_routes._race_profile_for_round = self._pr

    async def test_strategy_route(self):
        res = await f1_routes.get_f1_race_strategy(1)
        self.assertTrue(res["ok"])
        self.assertIn(res["strategy"]["recommended_stops"], (1, 2))
        self.assertEqual(res["track_key"], "bahrain")
        self.assertEqual(len(res["drivers"]), 2)


if __name__ == "__main__":
    unittest.main()
