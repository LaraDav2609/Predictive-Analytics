import unittest

from sports.f1.api import f1_routes
from sports.f1.predictor.features.market_consensus import (
    MARKET_MAX_INFLUENCE, build_market_consensus, market_strength_modifier,
)


def test_build_consensus_basic():
    c = build_market_consensus(
        {"VER": 0.5, "HAM": 0.2, "NOR": 0.3},
        driver_team={"VER": "Red Bull", "HAM": "Mercedes", "NOR": "McLaren"},
    )
    assert c["top_driver_id"] == "VER"
    assert set(c["drivers"]) == {"VER", "HAM", "NOR"}
    assert "Red Bull" in c["constructors"]
    assert c["source"] == "market_consensus"
    assert c["max_influence"] == MARKET_MAX_INFLUENCE


def test_build_consensus_empty():
    c = build_market_consensus({})
    assert c["drivers"] == {}
    assert c["top_driver_id"] is None


def test_modifier_absent_is_neutral():
    assert market_strength_modifier("VER", None) == 1.0
    assert market_strength_modifier("VER", {"drivers": {}}) == 1.0
    assert market_strength_modifier("ZZZ", {"drivers": {"VER": 0.5}, "field_mean": 0.5, "field_std": 0.1}) == 1.0


def test_modifier_bounded_and_directional():
    sig = build_market_consensus({"VER": 0.6, "HAM": 0.1, "NOR": 0.3})
    fav = market_strength_modifier("VER", sig)
    fade = market_strength_modifier("HAM", sig)
    assert fav > 1.0
    assert fade < 1.0
    assert 1.0 - MARKET_MAX_INFLUENCE <= fade <= fav <= 1.0 + MARKET_MAX_INFLUENCE


class _FakeDriver:
    def __init__(self, did, team):
        self.id = did
        self.team = team


class _FakeClient:
    season = 2026

    def get_race_by_round(self, r):
        return object()  # truthy, non-None

    def get_drivers(self):
        return [_FakeDriver("VER", "Red Bull"), _FakeDriver("HAM", "Mercedes")]


class MarketConsensusRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._client = f1_routes.client
        f1_routes.client = _FakeClient()
        f1_routes._market_consensus_by_round.clear()

    def tearDown(self):
        f1_routes.client = self._client
        f1_routes._market_consensus_by_round.clear()

    async def test_post_caches_consensus_and_get_returns_it(self):
        res = await f1_routes.post_f1_market_consensus(5, {"driver_implied": {"VER": 0.55, "HAM": 0.2}})
        self.assertTrue(res["ok"])
        self.assertEqual(res["consensus"]["top_driver_id"], "VER")
        self.assertIn(5, f1_routes._market_consensus_by_round)
        got = await f1_routes.get_f1_market_consensus(5)
        self.assertEqual(got["consensus"]["top_driver_id"], "VER")

    async def test_post_requires_driver_implied(self):
        res = await f1_routes.post_f1_market_consensus(5, {})
        self.assertFalse(res["ok"])
        self.assertEqual(res["code"], "missing_driver_implied")


if __name__ == "__main__":
    unittest.main()
