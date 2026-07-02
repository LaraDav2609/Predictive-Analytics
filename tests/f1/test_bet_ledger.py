import unittest

try:
    from sports.f1.api import f1_routes
except ModuleNotFoundError:
    f1_routes = None
from sports.f1.ml.markets.bet_ledger import run_bet_ledger
from sports.f1.ml.markets.synthetic_market import build_synthetic_decisions, synthetic_quote


def test_winning_yes_edge_grows_bankroll():
    # model 0.7 vs market ask 0.5 → large YES edge; YES wins → profit.
    led = run_bet_ledger([{"model_prob": 0.7, "yes_bid": 0.48, "yes_ask": 0.5, "outcome": 1}],
                         bankroll_usd=1000, min_edge_bps=200)
    assert led["bets_placed"] == 1
    assert led["wins"] == 1
    assert led["realized_pnl"] > 0
    assert led["end_bankroll"] > 1000


def test_losing_bet_drops_bankroll():
    led = run_bet_ledger([{"model_prob": 0.7, "yes_bid": 0.48, "yes_ask": 0.5, "outcome": 0}],
                         min_edge_bps=200)
    assert led["realized_pnl"] < 0
    assert led["end_bankroll"] < 1000


def test_below_min_edge_is_skipped():
    led = run_bet_ledger([{"model_prob": 0.51, "yes_bid": 0.49, "yes_ask": 0.5, "outcome": 1}],
                         min_edge_bps=200)
    assert led["bets_placed"] == 0
    assert led["skipped"] == 1


def test_metrics_and_clv_present():
    led = run_bet_ledger([
        {"model_prob": 0.7, "yes_bid": 0.48, "yes_ask": 0.5, "outcome": 1, "close_yes": 0.65},
        {"model_prob": 0.3, "yes_bid": 0.5, "yes_ask": 0.52, "outcome": 0},  # NO-side edge
    ], min_edge_bps=200)
    assert "roi" in led and "max_drawdown" in led
    assert "calibration_by_edge_bucket" in led
    assert led["avg_clv"] is not None


def test_no_side_edge_bets():
    # model 0.2 vs market bid 0.5 → BUY NO edge; YES does not happen → NO wins.
    led = run_bet_ledger([{"model_prob": 0.2, "yes_bid": 0.5, "yes_ask": 0.52, "outcome": 0}],
                         min_edge_bps=200)
    assert led["bets_placed"] == 1
    assert led["realized_pnl"] > 0


def test_synthetic_quote_and_decisions():
    q = synthetic_quote(0.5, overround=0.04, spread=0.02)
    assert q["yes_bid"] < q["yes_ask"]
    decisions = build_synthetic_decisions([{"model_prob": 0.7, "outcome": 1, "market_ref": 0.5}])
    assert decisions[0]["model_prob"] == 0.7
    assert "yes_ask" in decisions[0]


class LedgerRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        if f1_routes is None:
            self.skipTest("fastapi route stack is not installed in this test runtime")

    async def test_ledger_route_with_samples(self):
        body = {"samples": [
            {"model_prob": 0.7, "outcome": 1, "market_ref": 0.5},
            {"model_prob": 0.65, "outcome": 1, "market_ref": 0.5},
            {"model_prob": 0.3, "outcome": 0, "market_ref": 0.5},
        ], "min_edge_bps": 200}
        res = await f1_routes.post_f1_ledger_backtest(body)
        self.assertTrue(res["ok"])
        self.assertTrue(res["synthetic"])
        self.assertGreaterEqual(res["bets_placed"], 1)

    async def test_ledger_route_requires_input(self):
        res = await f1_routes.post_f1_ledger_backtest({})
        self.assertFalse(res["ok"])
        self.assertEqual(res["code"], "missing_decisions")


class _FakeBacktester:
    async def backtest_season(self, season, include_races=False, allow_partial=False, model_id=None, stage="pre_weekend"):
        return {"ok": True, "season": season, "model_id": "production_v1", "races": [
            {"round": 1, "actual_winner": "VER", "probability_distribution": [
                {"driver_id": "VER", "win_probability": 0.6},
                {"driver_id": "HAM", "win_probability": 0.2},
                {"driver_id": "NOR", "win_probability": 0.1},
            ]},
            {"round": 2, "actual_winner": "NOR", "probability_distribution": [
                {"driver_id": "VER", "win_probability": 0.5},
                {"driver_id": "NOR", "win_probability": 0.3},
            ]},
        ]}


class SeasonLedgerRouteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        if f1_routes is None:
            self.skipTest("fastapi route stack is not installed in this test runtime")
        self._bt = f1_routes._backtester
        f1_routes._backtester = lambda: _FakeBacktester()

    def tearDown(self):
        f1_routes._backtester = self._bt

    async def test_season_ledger_runs_over_backtest_rows(self):
        res = await f1_routes.get_f1_season_ledger(2025, market_shrink=0.3, min_edge_bps=100)
        self.assertTrue(res["ok"])
        self.assertEqual(res["races_evaluated"], 2)
        self.assertEqual(res["samples"], 5)        # 3 + 2 drivers
        self.assertTrue(res["synthetic"])
        self.assertIn("roi", res)
        self.assertIn("calibration_by_edge_bucket", res)

    async def test_season_ledger_handles_backtest_failure(self):
        class _FailBt:
            async def backtest_season(self, season, include_races=False, allow_partial=False, model_id=None, stage="pre_weekend"):
                return {"ok": False, "reason": "no_data"}
        f1_routes._backtester = lambda: _FailBt()
        res = await f1_routes.get_f1_season_ledger(2099)
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "no_data")


if __name__ == "__main__":
    unittest.main()
