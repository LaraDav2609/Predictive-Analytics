import unittest

from sports.f1.api import f1_routes
from sports.f1.predictor.features.penalties import (
    build_penalty_report, classify_result_status, is_disqualified,
)


def test_is_disqualified():
    assert is_disqualified("Disqualified")
    assert is_disqualified("Excluded")
    assert not is_disqualified("Finished")
    assert not is_disqualified("Engine")


def test_classify_result_status():
    assert classify_result_status("Disqualified") == "disqualified"
    assert classify_result_status("Finished") == "finished"
    assert classify_result_status("+1 Lap") == "finished"
    assert classify_result_status("Engine") == "mechanical"
    assert classify_result_status("Accident") == "incident"
    assert classify_result_status("Collision damage") == "incident"
    assert classify_result_status("") == "unknown"


def test_build_penalty_report():
    grid_evidence = {
        "drivers": {
            "VER": {"driver_id": "VER", "driver_code": "VER", "qualifying_position": 1,
                    "grid_position": 6, "grid_penalty": 5, "pit_lane_start": False, "confidence": 0.86},
            "HAM": {"driver_id": "HAM", "driver_code": "HAM", "qualifying_position": 3,
                    "grid_position": 0, "grid_penalty": 0, "pit_lane_start": True, "confidence": 0.86},
            "NOR": {"driver_id": "NOR", "driver_code": "NOR", "qualifying_position": 2,
                    "grid_position": 2, "grid_penalty": 0, "pit_lane_start": False, "confidence": 0.86},
        },
        "source": "jolpica_qualifying_grid", "confidence": 0.86,
    }
    results = [
        {"driver_id": "PER", "driver_code": "PER", "position": None, "status": "Disqualified"},
        {"driver_id": "VER", "driver_code": "VER", "position": 4, "status": "Finished"},
    ]
    report = build_penalty_report(grid_evidence, results)
    assert report["summary"] == {"grid_penalties": 1, "pit_lane_starts": 1, "disqualifications": 1}
    assert report["grid_penalties"][0]["driver_code"] == "VER"
    assert report["grid_penalties"][0]["grid_penalty"] == 5
    assert report["pit_lane_starts"][0]["driver_code"] == "HAM"
    assert report["disqualifications"][0]["driver_code"] == "PER"
    assert report["available"] is True


def test_build_penalty_report_empty():
    report = build_penalty_report({}, [])
    assert report["available"] is False
    assert report["summary"]["grid_penalties"] == 0


class _FakeDriver:
    def __init__(self, did):
        self.id = did
        self.code = did


class _FakeClient:
    def get_race_by_round(self, r):
        return object()

    def get_drivers(self):
        return [_FakeDriver("VER"), _FakeDriver("HAM")]


class PenaltyRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._client = f1_routes.client
        self._profile = f1_routes._race_profile_for_round
        f1_routes.client = _FakeClient()

        async def fake_profile(round_num):
            return {
                "qualifying": [
                    {"driver_id": "VER", "driver_code": "VER", "position": 1, "grid": 6},
                    {"driver_id": "HAM", "driver_code": "HAM", "position": 2, "grid": 2},
                ],
                "results": [
                    {"driver_id": "HAM", "driver_code": "HAM", "position": None, "status": "Disqualified"},
                    {"driver_id": "VER", "driver_code": "VER", "position": 3, "status": "Finished"},
                ],
            }

        f1_routes._race_profile_for_round = fake_profile

    def tearDown(self):
        f1_routes.client = self._client
        f1_routes._race_profile_for_round = self._profile

    async def test_penalties_route(self):
        res = await f1_routes.get_f1_race_penalties(1)
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(res["summary"]["grid_penalties"], 1)   # VER P1→grid P6 = 5 places
        self.assertTrue(any(g["driver_code"] == "VER" for g in res["grid_penalties"]))
        self.assertTrue(any(d["driver_code"] == "HAM" for d in res["disqualifications"]))


if __name__ == "__main__":
    unittest.main()
