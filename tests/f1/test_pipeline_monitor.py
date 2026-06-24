import unittest
from datetime import datetime, timezone

from common.ml.bridge.ops_publisher import InMemoryOpsPublisher
from common.ml.bridge.outcome_publisher import InMemoryOutcomePublisher
from common.ml.types import OutcomeProbability
from sports.f1.api import f1_routes


class F1PipelineMonitorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # The pipeline-monitor surface needs no F1 client / predictor / storage —
        # it reports pipeline *state*, so None globals exercise the null-safe path.
        f1_routes.client = None
        f1_routes.predictor = None
        f1_routes.storage = None
        f1_routes.live_engine = None
        f1_routes.openf1 = None
        f1_routes._ops_publisher_override = InMemoryOpsPublisher()
        f1_routes._outcome_publisher_override = InMemoryOutcomePublisher()
        f1_routes._recent_ops_events.clear()
        f1_routes._last_pipeline_run = {"source_mode": None, "reason": "no_run_yet"}
        f1_routes._last_bridge_publish_status = {"ok": None, "reason": "not_requested", "record_count": 0}
        f1_routes._active_freshness_alert_keys = set()

    def test_emit_ops_event_buffers_and_publishes(self):
        record = f1_routes._emit_ops_event("refresh_started", message="hi", foo="bar")
        self.assertEqual(record["event_type"], "refresh_started")
        self.assertEqual(record["detail"]["foo"], "bar")
        self.assertEqual(f1_routes._recent_ops_events[-1]["event_type"], "refresh_started")
        pub = f1_routes._ops_publisher_override
        self.assertEqual(pub.channel_messages[0][0], "f1:ops:refresh_started")

    def test_emit_ops_event_caps_ring_buffer(self):
        for i in range(f1_routes._RECENT_OPS_EVENTS_MAX + 20):
            f1_routes._emit_ops_event("tick", message=str(i))
        self.assertEqual(len(f1_routes._recent_ops_events), f1_routes._RECENT_OPS_EVENTS_MAX)

    def test_publish_outcome_probabilities_emits_prediction_ready(self):
        records = [OutcomeProbability(
            domain="f1", entity_id="2026-01-BAHRAIN", entity_code="VER", market="winner",
            probability=0.6, knowable_as_of=datetime.now(timezone.utc), model_version="t")]
        status = f1_routes._publish_outcome_probabilities(records)
        self.assertTrue(status["ok"])
        types = [e["event_type"] for e in f1_routes._recent_ops_events]
        self.assertIn("prediction_ready", types)

    def test_record_pipeline_run_emits_degraded_once_per_transition(self):
        f1_routes._record_pipeline_run(
            payload={},
            simulation={"trained_artifacts_used": False, "ml_input_source": "evidence_fallback"},
            truth={"source_mode": "estimated", "confidence": 0.22, "confidence_ceiling": 0.25},
            bridge_records=[])
        self.assertEqual(f1_routes._last_pipeline_run["source_mode"], "estimated")
        degraded = [e for e in f1_routes._recent_ops_events if e["event_type"] == "degraded"]
        self.assertEqual(len(degraded), 1)
        # Staying estimated must not re-emit; recovering emits 'recovered'.
        f1_routes._record_pipeline_run(payload={}, simulation={}, truth={"source_mode": "estimated"}, bridge_records=[])
        self.assertEqual(len([e for e in f1_routes._recent_ops_events if e["event_type"] == "degraded"]), 1)
        f1_routes._record_pipeline_run(payload={}, simulation={}, truth={"source_mode": "live"}, bridge_records=[])
        self.assertIn("recovered", [e["event_type"] for e in f1_routes._recent_ops_events])

    async def test_pipeline_health_snapshot_shape(self):
        result = await f1_routes.get_f1_pipeline_health()
        self.assertTrue(result["ok"])
        self.assertIn(result["provenance"]["mode"], ("trained", "heuristic_fallback"))
        self.assertIn("empirical_active", result["calibration"])
        self.assertIn("winner", result["market_coverage"]["computed"])
        self.assertIn("safety_car", result["market_coverage"]["computed_not_published"])
        self.assertIn("confidence_ceilings", result)
        self.assertEqual(result["bridge"]["domain"], "f1")
        self.assertIn("last_publish", result["bridge"])
        # Freshness alerts: with predictor/client/storage all None, expect the
        # missing-results error to be present and the summary to be populated.
        self.assertIsInstance(result["freshness_alerts"], list)
        self.assertIn("official_results", {a["source"] for a in result["freshness_alerts"]})
        self.assertEqual(result["freshness_summary"]["total"], len(result["freshness_alerts"]))

    async def test_pipeline_health_includes_recent_events(self):
        f1_routes._emit_ops_event("refresh_started", message="hi")
        result = await f1_routes.get_f1_pipeline_health()
        self.assertTrue(any(e["event_type"] == "refresh_started" for e in result["recent_events"]))


if __name__ == "__main__":
    unittest.main()
