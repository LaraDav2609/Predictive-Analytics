import unittest
from datetime import datetime, timezone

from common.ml.types import OutcomeProbability
from sports.f1.api import f1_routes
from sports.f1.ml.markets.edge_service import (
    F1EdgeQuoteIn, F1EdgeRequest, RaceQuote, compute_race_edges, probability_map_from_records,
)


def _q(market="winner", code="VER", yes_bid=0.40, yes_ask=0.44, fee_bps=0.0, venue="kalshi", market_id="M"):
    return RaceQuote(market_id, venue, market, code, yes_bid, yes_ask, None, None, fee_bps)


def test_positive_yes_edge_is_tradeable_and_sized():
    rows = compute_race_edges({("VER", "winner"): 0.60}, [_q(yes_ask=0.44)], bankroll_usd=1000.0, min_edge_bps=200.0)
    r = rows[0]
    assert r["tradeable"] is True
    assert r["direction"] == "YES"
    assert r["edge_bps"] == 1600.0          # (0.60 - 0.44) * 10000
    assert r["stake_usd"] > 0.0
    assert 0.0 < r["stake_fraction"] <= 0.05  # capped at max_per_market_pct


def test_below_min_edge_not_tradeable():
    rows = compute_race_edges({("VER", "winner"): 0.45}, [_q(yes_ask=0.44)], min_edge_bps=200.0)
    r = rows[0]
    assert r["tradeable"] is False          # (0.45 - 0.44) * 10000 = 100 < 200
    assert r["stake_usd"] == 0.0
    assert r["best_edge_bps"] == 100.0


def test_no_side_edge_detected():
    rows = compute_race_edges({("VER", "winner"): 0.20}, [_q(yes_bid=0.40, yes_ask=0.44)], min_edge_bps=200.0)
    r = rows[0]
    assert r["tradeable"] is True
    assert r["direction"] == "NO"           # (0.40 - 0.20) * 10000 = 2000
    assert r["edge_bps"] == 2000.0
    assert r["stake_usd"] > 0.0


def test_fee_reduces_edge_below_threshold():
    # Raw YES edge = (0.46 - 0.44) * 10000 = 200 bps; minus a 300 bps fee → negative.
    rows = compute_race_edges({("VER", "winner"): 0.46}, [_q(yes_ask=0.44, fee_bps=300.0)], min_edge_bps=200.0)
    assert rows[0]["tradeable"] is False


def test_missing_probability_is_flagged():
    rows = compute_race_edges({}, [_q(code="HAM")], min_edge_bps=200.0)
    assert rows[0]["ok"] is False
    assert rows[0]["reason"] == "no_model_probability"
    assert rows[0]["tradeable"] is False


def test_results_sorted_by_edge_desc():
    prob_map = {("VER", "winner"): 0.60, ("HAM", "winner"): 0.30}
    quotes = [_q(code="HAM", yes_ask=0.44), _q(code="VER", yes_ask=0.44)]
    rows = compute_race_edges(prob_map, quotes)
    assert rows[0]["entity_code"] == "VER"   # bigger edge sorts first


def test_probability_map_from_records():
    recs = [OutcomeProbability(domain="f1", entity_id="E", entity_code="VER", market="winner",
                               probability=0.6, knowable_as_of=datetime.now(timezone.utc), model_version="t")]
    assert probability_map_from_records(recs)[("VER", "winner")] == 0.6


class _FakeRace:
    round = 1
    circuit_id = "bahrain"
    country = "Bahrain"
    name = "Bahrain Grand Prix"


class _FakeClient:
    season = 2026

    def get_race_by_round(self, r):
        return _FakeRace()


class F1EdgeRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig_client = f1_routes.client
        self._orig_sim = f1_routes.get_race_simulation
        f1_routes.client = _FakeClient()

        async def fake_sim(round_num, session="race", live=False, model_id=None):
            return {
                "simulations": [
                    {"driver_code": "VER", "win_probability": 0.60, "podium_probability": 0.85},
                    {"driver_code": "HAM", "win_probability": 0.10, "podium_probability": 0.40},
                ],
                "generated_at": None,
                "model_id": "production_v1",
            }

        f1_routes.get_race_simulation = fake_sim

    def tearDown(self):
        f1_routes.client = self._orig_client
        f1_routes.get_race_simulation = self._orig_sim

    async def test_edges_route_joins_probs_to_quotes(self):
        body = F1EdgeRequest(quotes=[
            F1EdgeQuoteIn(market_id="KXF1WIN-VER", venue="kalshi", market="winner",
                          entity_code="VER", yes_bid=0.40, yes_ask=0.44),
        ])
        result = await f1_routes.post_f1_race_edges(1, body)
        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 1)
        self.assertGreaterEqual(result["tradeable_count"], 1)
        self.assertIn("winner", result["model_markets"])
        self.assertEqual(result["entity_id"], "2026-01-BAHRAIN")
        edge = result["edges"][0]
        self.assertEqual(edge["entity_code"], "VER")
        self.assertEqual(edge["direction"], "YES")

    async def test_edges_route_race_not_found(self):
        class _NoRace:
            season = 2026

            def get_race_by_round(self, r):
                return None

        f1_routes.client = _NoRace()
        result = await f1_routes.post_f1_race_edges(99, F1EdgeRequest(quotes=[]))
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "race_not_found")


if __name__ == "__main__":
    unittest.main()
