import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from common.ml.types import OutcomeProbability
try:
    from sports.f1.api import f1_routes
except ModuleNotFoundError:
    f1_routes = None
try:
    from typer.testing import CliRunner
    from sports.f1.ml.cli import app as f1_ml_cli_app
except ModuleNotFoundError:
    CliRunner = None
    f1_ml_cli_app = None
from sports.f1.ml.markets.edge_service import (
    F1EdgeQuoteIn,
    F1EdgeRequest,
    F1ExposureState,
    F1RiskPolicy,
    F1TradeReadinessGate,
    RaceQuote,
    append_order_submissions,
    build_exposure_state,
    build_order_intents,
    compute_race_edges,
    evaluate_order_risk,
    evaluate_trade_readiness,
    load_order_submissions,
    probability_map_from_records,
    settle_order_submissions,
    submit_order_intents,
)
from sports.f1.ml.markets.venues import (
    match_f1_market,
    normalize_kalshi_quote,
    normalize_matched_venue_quote,
    normalize_matched_venue_quotes,
    normalize_polymarket_quote,
    normalize_venue_quote,
    normalize_venue_quotes,
)


def _q(market="winner", code="VER", yes_bid=0.40, yes_ask=0.44, fee_bps=0.0, venue="kalshi", market_id="M"):
    return RaceQuote(market_id, venue, market, code, yes_bid, yes_ask, None, None, fee_bps)


def test_positive_yes_edge_is_tradeable_and_sized():
    rows = compute_race_edges({("VER", "winner"): 0.60}, [_q(yes_ask=0.44)], bankroll_usd=1000.0, min_edge_bps=200.0)
    r = rows[0]
    assert r["would_trade"] is True
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


def test_edge_request_defaults_to_analysis_mode():
    request = F1EdgeRequest(quotes=[
        F1EdgeQuoteIn(market_id="M", market="winner", entity_code="VER", yes_bid=0.40, yes_ask=0.44),
    ])

    assert request.trade_readiness is None


def test_readiness_gate_blocks_good_edge_when_live_controls_missing():
    rows = compute_race_edges(
        {("VER", "winner"): 0.60},
        [_q(yes_ask=0.44)],
        bankroll_usd=1000.0,
        min_edge_bps=200.0,
        trade_readiness=F1TradeReadinessGate(),
    )
    r = rows[0]
    assert r["would_trade"] is True
    assert r["tradeable"] is False
    assert r["trade_blocked"] is True
    assert r["stake_usd"] == 0.0
    assert "live_trading_disabled" in r["trade_block_reasons"]
    assert "risk_manager_inactive" in r["trade_block_reasons"]


def test_readiness_gate_allows_edge_when_all_live_controls_pass():
    readiness = F1TradeReadinessGate(
        live_trading_enabled=True,
        risk_manager_active=True,
        validation_gate={"passed": True, "status": "passed"},
        artifact_readiness={"live_trading_ready": True},
        calibration_status={"status": "passed"},
    )

    rows = compute_race_edges(
        {("VER", "winner"): 0.60},
        [_q(yes_ask=0.44)],
        bankroll_usd=1000.0,
        min_edge_bps=200.0,
        trade_readiness=readiness,
    )

    assert rows[0]["tradeable"] is True
    assert rows[0]["stake_usd"] > 0.0
    assert rows[0]["trade_readiness"]["live_trading_ready"] is True


def test_evaluate_trade_readiness_requires_calibration_and_artifacts():
    readiness = evaluate_trade_readiness({
        "live_trading_enabled": True,
        "risk_manager_active": True,
        "validation_gate": {"passed": True, "status": "passed"},
    })

    assert readiness["live_trading_ready"] is False
    assert "artifact_readiness_missing" in readiness["reasons"]
    assert "calibration_status_missing" in readiness["reasons"]


class F1EdgeReadinessTests(unittest.TestCase):
    def _live_ready(self, **overrides):
        payload = {
            "live_trading_enabled": True,
            "risk_manager_active": True,
            "validation_gate": {"passed": True, "status": "passed"},
            "artifact_readiness": {"live_trading_ready": True},
            "calibration_status": {"status": "passed"},
        }
        payload.update(overrides)
        return F1TradeReadinessGate(**payload)

    def test_readiness_gate_blocks_good_edge_when_live_controls_missing(self):
        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=F1TradeReadinessGate(),
        )
        row = rows[0]

        self.assertTrue(row["would_trade"])
        self.assertFalse(row["tradeable"])
        self.assertTrue(row["trade_blocked"])
        self.assertEqual(0.0, row["stake_usd"])
        self.assertIn("live_trading_disabled", row["trade_block_reasons"])
        self.assertIn("risk_manager_inactive", row["trade_block_reasons"])

    def test_readiness_gate_allows_edge_when_all_live_controls_pass(self):
        readiness = F1TradeReadinessGate(
            live_trading_enabled=True,
            risk_manager_active=True,
            validation_gate={"passed": True, "status": "passed"},
            artifact_readiness={"live_trading_ready": True},
            calibration_status={"status": "passed"},
        )

        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )

        self.assertTrue(rows[0]["tradeable"])
        self.assertGreater(rows[0]["stake_usd"], 0.0)
        self.assertTrue(rows[0]["trade_readiness"]["live_trading_ready"])
        self.assertTrue(rows[0]["risk_check"]["ok"])

    def test_evaluate_trade_readiness_requires_calibration_and_artifacts(self):
        readiness = evaluate_trade_readiness({
            "live_trading_enabled": True,
            "risk_manager_active": True,
            "validation_gate": {"passed": True, "status": "passed"},
        })

        self.assertFalse(readiness["live_trading_ready"])
        self.assertIn("artifact_readiness_missing", readiness["reasons"])
        self.assertIn("calibration_status_missing", readiness["reasons"])

    def test_risk_policy_blocks_market_exposure_limit(self):
        readiness = self._live_ready(
            exposure_state=F1ExposureState(market_exposure_usd={"winner:VER": 95.0}),
            risk_policy=F1RiskPolicy(max_market_exposure_pct_bankroll=0.10),
        )

        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )

        self.assertTrue(rows[0]["would_trade"])
        self.assertFalse(rows[0]["tradeable"])
        self.assertEqual(0.0, rows[0]["stake_usd"])
        self.assertEqual(50.0, rows[0]["suggested_stake_usd"])
        self.assertIn("market_exposure_limit_exceeded", rows[0]["trade_block_reasons"])

    def test_risk_policy_blocks_daily_loss_limit(self):
        readiness = self._live_ready(
            exposure_state=F1ExposureState(daily_pnl_usd=-50.0),
            risk_policy=F1RiskPolicy(max_daily_loss_pct_bankroll=0.05),
        )

        risk = evaluate_order_risk(
            venue="kalshi",
            market_key="winner:VER",
            stake_usd=25.0,
            bankroll_usd=1000.0,
            trade_readiness=readiness,
        )

        self.assertFalse(risk["ok"])
        self.assertIn("daily_loss_limit_exceeded", risk["reasons"])

    def test_risk_policy_allows_order_under_limits(self):
        readiness = self._live_ready(
            exposure_state=F1ExposureState(market_exposure_usd={"winner:VER": 10.0}, daily_exposure_usd=20.0),
            risk_policy=F1RiskPolicy(),
        )

        risk = evaluate_order_risk(
            venue="kalshi",
            market_key="winner:VER",
            stake_usd=25.0,
            bankroll_usd=1000.0,
            trade_readiness=readiness,
        )

        self.assertTrue(risk["ok"])
        self.assertEqual([], risk["reasons"])
        self.assertEqual(35.0, risk["market_exposure_after_usd"])

    def test_build_exposure_state_aggregates_open_positions(self):
        exposure = build_exposure_state([
            {"status": "open", "venue": "kalshi", "market": "winner", "entity_code": "VER", "stake_usd": 40.0},
            {"status": "pending", "venue": "kalshi", "market": "winner", "entity_code": "VER", "notional_usd": 15.0},
            {"status": "filled", "venue": "polymarket", "market": "dnf", "entity_code": "LEC", "open_stake_usd": 10.0},
            {"status": "settled", "venue": "kalshi", "market": "winner", "entity_code": "VER", "stake_usd": 100.0, "pnl_usd": -12.5},
        ])

        self.assertEqual(55.0, exposure.market_exposure_usd["winner:VER"])
        self.assertEqual(10.0, exposure.market_exposure_usd["dnf:LEC"])
        self.assertEqual(55.0, exposure.venue_exposure_usd["kalshi"])
        self.assertEqual(10.0, exposure.venue_exposure_usd["polymarket"])
        self.assertEqual(65.0, exposure.daily_exposure_usd)
        self.assertEqual(-12.5, exposure.daily_pnl_usd)

    def test_exposure_state_from_positions_blocks_next_order(self):
        exposure = build_exposure_state([
            {"status": "open", "venue": "kalshi", "market": "winner", "entity_code": "VER", "stake_usd": 90.0},
        ])
        readiness = self._live_ready(
            exposure_state=exposure,
            risk_policy=F1RiskPolicy(max_market_exposure_pct_bankroll=0.10),
        )

        risk = evaluate_order_risk(
            venue="kalshi",
            market_key="winner:VER",
            stake_usd=15.0,
            bankroll_usd=1000.0,
            trade_readiness=readiness,
        )

        self.assertFalse(risk["ok"])
        self.assertIn("market_exposure_limit_exceeded", risk["reasons"])
        self.assertEqual(105.0, risk["market_exposure_after_usd"])

    def test_order_intents_created_only_from_tradeable_edges(self):
        readiness = self._live_ready()
        rows = compute_race_edges(
            {("VER", "winner"): 0.60, ("HAM", "winner"): 0.45},
            [_q(code="VER", yes_ask=0.44), _q(code="HAM", yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )

        intents = build_order_intents(rows, race_id="2026-01-BAHRAIN", mode="paper")

        self.assertEqual(1, len(intents))
        intent = intents[0]
        self.assertEqual("paper", intent.mode)
        self.assertEqual("ready_for_paper_submit", intent.status)
        self.assertEqual("YES", intent.direction)
        self.assertEqual("VER", intent.entity_code)
        self.assertEqual(0.44, intent.limit_price)
        self.assertEqual(50.0, intent.stake_usd)
        self.assertEqual("2026-01-BAHRAIN", intent.audit["race_id"])
        self.assertTrue(intent.idempotency_key)

    def test_order_intent_idempotency_key_is_stable(self):
        readiness = self._live_ready()
        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(code="VER", yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )

        first = build_order_intents(rows, race_id="2026-01-BAHRAIN", mode="paper")
        second = build_order_intents(rows, race_id="2026-01-BAHRAIN", mode="paper")
        live = build_order_intents(rows, race_id="2026-01-BAHRAIN", mode="live")

        self.assertEqual(first[0].idempotency_key, second[0].idempotency_key)
        self.assertNotEqual(first[0].idempotency_key, live[0].idempotency_key)
        self.assertEqual("ready_for_live_submit", live[0].status)

    def test_order_intents_skip_risk_blocked_edges(self):
        readiness = self._live_ready(
            exposure_state=F1ExposureState(market_exposure_usd={"winner:VER": 95.0}),
            risk_policy=F1RiskPolicy(max_market_exposure_pct_bankroll=0.10),
        )
        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(code="VER", yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )

        self.assertFalse(rows[0]["tradeable"])
        self.assertEqual([], build_order_intents(rows, race_id="2026-01-BAHRAIN"))

    def test_submit_order_intents_records_paper_submission(self):
        readiness = self._live_ready()
        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(code="VER", yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )
        intents = build_order_intents(rows, race_id="2026-01-BAHRAIN")

        submissions = submit_order_intents(intents)

        self.assertEqual(1, len(submissions))
        self.assertEqual("paper_submitted", submissions[0].status)
        self.assertEqual(intents[0].idempotency_key, submissions[0].idempotency_key)
        self.assertEqual("paper", submissions[0].mode)
        self.assertEqual(50.0, submissions[0].stake_usd)
        self.assertEqual(intents[0].model_dump(), submissions[0].intent)

    def test_settle_order_submissions_scores_yes_win_and_updates_pnl(self):
        readiness = self._live_ready()
        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(code="VER", yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )
        submissions = submit_order_intents(build_order_intents(rows, race_id="2026-01-BAHRAIN"))

        result = settle_order_submissions(submissions, {"winner:VER": True})
        settlement = result["settlements"][0]
        exposure = build_exposure_state(result["settlements"])

        self.assertEqual(1, result["settlement_count"])
        self.assertEqual(0, result["skipped_count"])
        self.assertEqual("settled_win", settlement["status"])
        self.assertTrue(settlement["won"])
        self.assertEqual(63.64, settlement["pnl_usd"])
        self.assertEqual(63.64, result["realized_pnl"])
        self.assertEqual(0.0, exposure.daily_exposure_usd)
        self.assertEqual(63.64, exposure.daily_pnl_usd)

    def test_settle_order_submissions_scores_no_side_win(self):
        readiness = self._live_ready()
        rows = compute_race_edges(
            {("VER", "winner"): 0.20},
            [_q(code="VER", yes_bid=0.40, yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )
        submissions = submit_order_intents(build_order_intents(rows, race_id="2026-01-BAHRAIN"))

        result = settle_order_submissions(submissions, {"winner": {"VER": False}})
        settlement = result["settlements"][0]

        self.assertEqual("NO", settlement["direction"])
        self.assertEqual(0.6, settlement["limit_price"])
        self.assertTrue(settlement["won"])
        self.assertEqual(33.33, settlement["pnl_usd"])
        self.assertEqual(1.0, result["hit_rate"])

    def test_settle_order_submissions_skips_missing_outcome(self):
        readiness = self._live_ready()
        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(code="VER", yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )
        submissions = submit_order_intents(build_order_intents(rows, race_id="2026-01-BAHRAIN"))

        result = settle_order_submissions(submissions, {"winner:HAM": True})

        self.assertEqual(0, result["settlement_count"])
        self.assertEqual(1, result["skipped_count"])
        self.assertEqual("missing_outcome", result["skipped"][0]["reason"])

    def test_submit_order_intents_deduplicates_existing_submission(self):
        readiness = self._live_ready()
        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(code="VER", yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )
        intents = build_order_intents(rows, race_id="2026-01-BAHRAIN")
        existing = submit_order_intents(intents)

        duplicate = submit_order_intents(intents, existing_submissions=existing)

        self.assertEqual("duplicate_ignored", duplicate[0].status)
        self.assertEqual("idempotency_key_seen", duplicate[0].reason)

    def test_order_submission_jsonl_round_trip_and_dedupes(self):
        readiness = self._live_ready()
        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(code="VER", yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )
        intents = build_order_intents(rows, race_id="2026-01-BAHRAIN")
        submissions = submit_order_intents(intents)

        with TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "orders.jsonl"

            first = append_order_submissions(ledger, submissions)
            second = append_order_submissions(ledger, submissions)
            loaded = load_order_submissions(ledger)

        self.assertTrue(first["ok"])
        self.assertEqual(1, first["appended_count"])
        self.assertEqual(0, first["duplicate_count"])
        self.assertEqual(0, second["appended_count"])
        self.assertEqual(1, second["duplicate_count"])
        self.assertEqual(1, len(loaded))
        self.assertEqual(submissions[0].idempotency_key, loaded[0].idempotency_key)
        self.assertEqual("paper_submitted", loaded[0].status)

    def test_order_submission_jsonl_dedupes_same_batch_records(self):
        readiness = self._live_ready()
        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(code="VER", yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )
        submissions = submit_order_intents(build_order_intents(rows, race_id="2026-01-BAHRAIN"))

        with TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "orders.jsonl"
            status = append_order_submissions(ledger, [submissions[0], submissions[0].model_dump()])
            loaded = load_order_submissions(ledger)

        self.assertEqual(1, status["appended_count"])
        self.assertEqual(1, status["duplicate_count"])
        self.assertEqual(1, len(loaded))

    def test_loaded_order_submissions_feed_exposure_state(self):
        readiness = self._live_ready()
        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(code="VER", yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )
        submissions = submit_order_intents(build_order_intents(rows, race_id="2026-01-BAHRAIN"))

        with TemporaryDirectory() as temp_dir:
            ledger = Path(temp_dir) / "orders.jsonl"
            append_order_submissions(ledger, submissions)
            loaded = load_order_submissions(ledger)

        exposure = build_exposure_state([item.model_dump() for item in loaded])

        self.assertEqual(50.0, exposure.market_exposure_usd["winner:VER"])
        self.assertEqual(50.0, exposure.venue_exposure_usd["kalshi"])
        self.assertEqual(50.0, exposure.daily_exposure_usd)

    def test_cli_appends_lists_and_settles_order_submission_ledger(self):
        if CliRunner is None or f1_ml_cli_app is None:
            self.skipTest("typer is not installed in this test runtime")
        readiness = self._live_ready()
        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(code="VER", yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )
        submissions = submit_order_intents(build_order_intents(rows, race_id="2026-01-BAHRAIN"))

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "submission.json"
            outcomes_path = Path(temp_dir) / "outcomes.json"
            ledger_path = Path(temp_dir) / "orders.jsonl"
            input_path.write_text(json.dumps({"submissions": [submissions[0].model_dump()]}), encoding="utf-8")
            outcomes_path.write_text(json.dumps({"winner:VER": True}), encoding="utf-8")

            runner = CliRunner()
            appended = runner.invoke(f1_ml_cli_app, [
                "append-order-submissions",
                "--input-path",
                str(input_path),
                "--ledger-path",
                str(ledger_path),
            ])
            listed = runner.invoke(f1_ml_cli_app, [
                "order-submissions",
                "--ledger-path",
                str(ledger_path),
                "--json",
            ])
            settled = runner.invoke(f1_ml_cli_app, [
                "settle-order-submissions",
                "--ledger-path",
                str(ledger_path),
                "--outcomes-path",
                str(outcomes_path),
                "--json",
            ])

        self.assertEqual(0, appended.exit_code, appended.output)
        self.assertIn("appended=1", appended.output)
        self.assertEqual(0, listed.exit_code, listed.output)
        payload = json.loads(listed.output)
        self.assertEqual(1, payload["count"])
        self.assertEqual(50.0, payload["exposure_state"]["daily_exposure_usd"])
        self.assertEqual("paper_submitted", payload["submissions"][0]["status"])
        self.assertEqual(0, settled.exit_code, settled.output)
        settlement_payload = json.loads(settled.output)
        self.assertEqual(1, settlement_payload["settlement_count"])
        self.assertEqual(63.64, settlement_payload["realized_pnl"])

    def test_submit_order_intents_blocks_live_by_default(self):
        readiness = self._live_ready()
        rows = compute_race_edges(
            {("VER", "winner"): 0.60},
            [_q(code="VER", yes_ask=0.44)],
            bankroll_usd=1000.0,
            min_edge_bps=200.0,
            trade_readiness=readiness,
        )
        intents = build_order_intents(rows, race_id="2026-01-BAHRAIN", mode="live")

        blocked = submit_order_intents(intents)
        allowed = submit_order_intents(intents, allow_live=True)

        self.assertEqual("blocked", blocked[0].status)
        self.assertEqual("live_submission_disabled", blocked[0].reason)
        self.assertEqual("live_handoff_ready", allowed[0].status)
        self.assertIsNone(allowed[0].reason)


class F1VenueQuoteNormalizationTests(unittest.TestCase):
    def test_normalize_kalshi_cents_quote(self):
        quote = normalize_kalshi_quote(
            {
                "ticker": "KXF1WIN-VER",
                "yes_bid": 42,
                "yes_ask": 47,
                "no_bid": 51,
                "no_ask": 56,
                "taker_fee_bps": 75,
            },
            market="winner",
            entity_code="ver",
        )

        self.assertEqual("KXF1WIN-VER", quote.market_id)
        self.assertEqual("kalshi", quote.venue)
        self.assertEqual("VER", quote.entity_code)
        self.assertEqual(0.42, quote.yes_bid)
        self.assertEqual(0.47, quote.yes_ask)
        self.assertEqual(75.0, quote.fee_bps)

    def test_normalize_polymarket_decimal_quote(self):
        quote = normalize_polymarket_quote(
            {
                "condition_id": "0xabc",
                "bestBid": "0.38",
                "bestAsk": "0.41",
                "fee_bps": 25,
            },
            market="podium",
            entity_code="nor",
        )

        self.assertEqual("0xabc", quote.market_id)
        self.assertEqual("polymarket", quote.venue)
        self.assertEqual("NOR", quote.entity_code)
        self.assertEqual(0.38, quote.yes_bid)
        self.assertEqual(0.41, quote.yes_ask)
        self.assertEqual(25.0, quote.fee_bps)

    def test_normalize_venue_quotes_batches_driver_rows(self):
        quotes = normalize_venue_quotes(
            [
                {"ticker": "KX-VER", "driver_code": "VER", "yes_bid": 40, "yes_ask": 44},
                {"ticker": "KX-HAM", "driver_code": "HAM", "yes_bid": 12, "yes_ask": 15},
                {"ticker": "KX-MISSING", "yes_bid": 5, "yes_ask": 8},
            ],
            venue="kalshi",
            market="winner",
        )

        self.assertEqual(["VER", "HAM"], [quote.entity_code for quote in quotes])
        self.assertEqual([0.44, 0.15], [quote.yes_ask for quote in quotes])

    def test_normalize_venue_quote_rejects_unknown_venue(self):
        with self.assertRaises(ValueError):
            normalize_venue_quote(
                {"id": "X", "yes_bid": 0.1, "yes_ask": 0.2},
                venue="unknown",
                market="winner",
                entity_code="VER",
            )

    def test_match_f1_market_from_title_and_aliases(self):
        match = match_f1_market(
            {"title": "Will Max Verstappen win the Bahrain Grand Prix?"},
            driver_aliases={"VER": ["Max Verstappen", "Verstappen"]},
        )

        self.assertEqual("winner", match["market"])
        self.assertEqual("VER", match["entity_code"])

    def test_match_f1_market_detects_dnf_and_explicit_driver_code(self):
        match = match_f1_market(
            {"question": "Will driver DNF in the race?", "driver_code": "lec"},
        )

        self.assertEqual("dnf", match["market"])
        self.assertEqual("LEC", match["entity_code"])

    def test_normalize_matched_venue_quote_uses_title_match(self):
        quote = normalize_matched_venue_quote(
            {
                "condition_id": "0xverwin",
                "question": "Max Verstappen to win the race",
                "bestBid": "0.52",
                "bestAsk": "0.57",
            },
            venue="polymarket",
            driver_aliases={"VER": ["Max Verstappen"]},
        )

        self.assertEqual("winner", quote.market)
        self.assertEqual("VER", quote.entity_code)
        self.assertEqual(0.57, quote.yes_ask)

    def test_normalize_matched_venue_quotes_skips_unmatched_rows(self):
        quotes = normalize_matched_venue_quotes(
            [
                {"ticker": "KX-VER-POD", "title": "Max Verstappen podium finish", "yes_bid": 62, "yes_ask": 68},
                {"ticker": "KX-UNKNOWN", "title": "Formula 1 miscellaneous", "yes_bid": 1, "yes_ask": 2},
            ],
            venue="kalshi",
            driver_aliases={"VER": ["Max Verstappen"]},
        )

        self.assertEqual(1, len(quotes))
        self.assertEqual("podium", quotes[0].market)
        self.assertEqual("VER", quotes[0].entity_code)


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
        if f1_routes is None or not hasattr(f1_routes, "post_f1_race_edges"):
            self.skipTest("fastapi edge route stack is not installed in this test runtime")
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

    async def test_order_submit_and_settlement_route_flow(self):
        with TemporaryDirectory() as temp_dir:
            original_root = f1_routes._ORDER_SUBMISSION_LEDGER_ROOT
            f1_routes._ORDER_SUBMISSION_LEDGER_ROOT = Path(temp_dir)
            try:
                edges = await f1_routes.post_f1_race_edges(
                    1,
                    F1EdgeRequest(quotes=[
                        F1EdgeQuoteIn(market_id="KXF1WIN-VER", venue="kalshi", market="winner",
                                      entity_code="VER", yes_bid=0.40, yes_ask=0.44),
                    ]),
                )
                submitted = await f1_routes.post_f1_race_order_submissions(
                    1,
                    {"edges": edges["edges"], "mode": "paper"},
                )
                settled = await f1_routes.post_f1_race_order_settlement(
                    1,
                    {"outcomes": {"winner:VER": True}, "persist_report": True},
                )
                report_exists = Path(settled["report_path"]).exists()
            finally:
                f1_routes._ORDER_SUBMISSION_LEDGER_ROOT = original_root

        self.assertTrue(submitted["ok"])
        self.assertEqual(1, submitted["persisted_count"])
        self.assertTrue(settled["ok"])
        self.assertEqual(1, settled["settlement_count"])
        self.assertEqual(63.64, settled["realized_pnl"])
        self.assertTrue(report_exists)

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
