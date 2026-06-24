import unittest

from sports.f1.api import f1_routes
from sports.f1.predictor.features.track_archetype import (
    archetype_fit_score, archetype_ratings, archetypes_for_track, classify_archetypes,
)


def test_classify_monaco():
    monaco = {"street_circuit": True, "high_speed": False, "tire_stress": 0.28,
              "overtaking_difficulty": 0.94, "safety_car_probability": 0.64, "qualifying_importance": 0.96}
    tags = classify_archetypes(monaco)
    assert "street" in tags and "low_speed" in tags
    assert "low_overtake" in tags and "high_safety_car" in tags and "qualifying_critical" in tags
    assert "high_speed" not in tags


def test_classify_monza_is_high_speed_only():
    monza = {"street_circuit": False, "high_speed": True, "tire_stress": 0.48,
             "overtaking_difficulty": 0.35, "safety_car_probability": 0.32, "qualifying_importance": 0.54}
    assert classify_archetypes(monza) == ["high_speed"]


def test_archetypes_for_track_resolves_name_and_sprint():
    assert "street" in archetypes_for_track("Monaco Grand Prix")
    assert "sprint" in archetypes_for_track("Monaco Grand Prix", is_sprint=True)


def test_ratings_and_fit_score():
    history = {
        "monza": {"drivers": {"VER": {"track_score": 0.9, "starts": 3}, "HAM": {"track_score": 0.4, "starts": 3}}},
        "suzuka": {"drivers": {"VER": {"track_score": 0.8, "starts": 2}, "HAM": {"track_score": 0.5, "starts": 2}}},
    }
    ratings = archetype_ratings(history)
    assert "high_speed" in ratings["VER"]
    ver_hs = ratings["VER"]["high_speed"]["rating"]
    ham_hs = ratings["HAM"]["high_speed"]["rating"]
    assert ver_hs > ham_hs                      # VER stronger at high-speed circuits
    assert ratings["VER"]["high_speed"]["starts"] == 5
    # Single-archetype race fit == that archetype's rating.
    assert archetype_fit_score(ratings["VER"], ["high_speed"]) == ver_hs
    # No overlapping history → None.
    assert archetype_fit_score(ratings["VER"], ["wet"]) is None


class _FakeRace:
    round = 1
    name = "Italian Grand Prix"
    circuit = "Monza"
    circuit_id = "monza"
    country = "Italy"
    has_sprint = False


class _FakeDriver:
    def __init__(self, did, code, team):
        self.id = did
        self.code = code
        self.team = team


class _FakeClient:
    season = 2026

    def get_race_by_round(self, r):
        return _FakeRace()

    def get_drivers(self):
        return [_FakeDriver("VER", "VER", "Red Bull"), _FakeDriver("HAM", "HAM", "Mercedes")]


class _FakePredictor:
    _features = {
        "track_history": {
            "monza": {"drivers": {"VER": {"track_score": 0.9, "starts": 3},
                                  "HAM": {"track_score": 0.4, "starts": 3}}},
        }
    }


class ArchetypeRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._client = f1_routes.client
        self._predictor = f1_routes.predictor
        f1_routes.client = _FakeClient()
        f1_routes.predictor = _FakePredictor()

    def tearDown(self):
        f1_routes.client = self._client
        f1_routes.predictor = self._predictor

    async def test_archetype_fit_route(self):
        res = await f1_routes.get_f1_race_archetype_fit(1)
        self.assertTrue(res["ok"])
        self.assertIn("high_speed", res["race_archetypes"])      # Monza
        ver = next(d for d in res["drivers"] if d["driver_id"] == "VER")
        self.assertIsNotNone(ver["archetype_fit_score"])
        # VER (0.9) ranks above HAM (0.4) at this high-speed circuit.
        self.assertEqual(res["drivers"][0]["driver_id"], "VER")


if __name__ == "__main__":
    unittest.main()
