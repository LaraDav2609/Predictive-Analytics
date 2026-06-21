import asyncio
import unittest
from types import SimpleNamespace
from time import monotonic
from unittest.mock import AsyncMock

from sports.f1.predictor.storage.clickhouse_store import (
    F1ClickHouseStore,
    backtest_rows,
    live_state_rows,
    probability_rows,
    sentiment_item_rows,
    track_geometry_row,
)
from sports.f1.predictor.storage.manager import F1Storage
from sports.f1.predictor.storage.redis_store import F1RedisKeys


class F1StorageMappingTests(unittest.TestCase):
    def test_redis_key_policy_uses_f1_namespace_and_normalized_sessions(self):
        keys = F1RedisKeys()

        self.assertEqual(keys.live_state(2026, 4, "Race"), "f1:live:2026:4:race:state")
        self.assertEqual(keys.probability_latest(2026, 4, "qualies", "production_v1"), "f1:prob:2026:4:qualifying:production_v1:latest")
        self.assertEqual(keys.probability_channel(2026, 4, "sprint", "m1"), "f1:pub:prob:2026:4:sprint:m1")
        self.assertTrue(keys.racehub(4, "simulation", "abc").startswith("f1:racehub:4:simulation:"))
        self.assertEqual(keys.sentiment_latest(), "f1:sentiment:latest")

    def test_live_state_rows_preserve_order_tyre_gap_and_confidence(self):
        race = SimpleNamespace(round=4, name="Miami GP")
        state = {
            "refreshed_at": "2026-05-03T19:00:00+00:00",
            "mode": "live",
            "confidence": 0.82,
            "drivers": [
                {
                    "driver_id": "norris",
                    "driver_code": "NOR",
                    "team": "McLaren",
                    "position": 1,
                    "laps": 12,
                    "estimated_progress": 0.42,
                    "gap_to_leader": "0.000",
                    "interval": "0.000",
                    "compounds": ["MEDIUM"],
                    "pit_stops": 0,
                }
            ],
        }

        rows = live_state_rows(2026, race, "race", state)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["position"], 1)
        self.assertEqual(rows[0]["compound"], "MEDIUM")
        self.assertEqual(rows[0]["gap_to_leader"], "0.000")
        self.assertEqual(rows[0]["confidence"], 0.82)

    def test_probability_rows_preserve_prediction_fields(self):
        race = SimpleNamespace(round=4)
        payload = {
            "generated_at": "2026-05-03T19:00:00+00:00",
            "probabilities": [
                {
                    "driver_id": "leclerc",
                    "driver_code": "LEC",
                    "win_probability": 0.2,
                    "podium_probability": 0.55,
                    "top5_probability": 0.8,
                    "dnf_probability": 0.05,
                    "wdc_probability": 0.12,
                    "expected_finish": 3.2,
                }
            ],
        }

        rows = probability_rows(2026, race, "race", "production_v1", payload)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["driver_id"], "leclerc")
        self.assertAlmostEqual(rows[0]["win_probability"], 0.2)
        self.assertAlmostEqual(rows[0]["expected_finish"], 3.2)

    def test_track_sentiment_and_backtest_rows_are_compact_audit_rows(self):
        race = SimpleNamespace(round=20, season=2026, circuit="Las Vegas")
        track = {
            "track_key": "lasvegas",
            "source": "estimated",
            "mapping_source": "curated_registry",
            "geometry_confidence": 0.72,
            "display_points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
            "racing_line": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        }

        track_row = track_geometry_row(race, track)
        sentiment_rows = sentiment_item_rows(2026, [{"Title": "Ferrari upgrade", "Url": "https://example.test/f1", "Score": 0.4}])
        backtest = backtest_rows({
            "ok": True,
            "season": 2025,
            "round": 1,
            "model_id": "production_v1",
            "race_name": "Australian GP",
            "actual_winner": "verstappen",
            "predicted_winner": "norris",
            "metrics": {"winner_hit": False},
        })

        self.assertEqual(track_row["track_key"], "lasvegas")
        self.assertIn("geometry_hash", track_row)
        self.assertEqual(len(sentiment_rows), 1)
        self.assertEqual(sentiment_rows[0]["season"], 2026)
        self.assertEqual(len(backtest), 1)
        self.assertEqual(backtest[0]["actual_winner"], "verstappen")


class F1ClickHouseSoftFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_storage_health_is_short_ttl_cached(self):
        redis = _FakeRedisHealth()
        clickhouse = _FakeClickHouseHealth()
        storage = F1Storage(redis_store=redis, clickhouse_store=clickhouse)

        first = await storage.health()
        second = await storage.health()

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(1, redis.calls)
        self.assertEqual(1, clickhouse.calls)

    async def test_storage_health_clickhouse_timeout_is_best_effort_and_cached(self):
        redis = _FakeRedisHealth()
        clickhouse = _SlowFakeClickHouseHealth()
        storage = F1Storage(redis_store=redis, clickhouse_store=clickhouse)
        storage._clickhouse_health_timeout_seconds = 0.01

        first = await storage.health()
        second = await storage.health()

        self.assertFalse(first["cached"])
        self.assertFalse(first["clickhouse"]["available"])
        self.assertTrue(first["clickhouse"]["best_effort"])
        self.assertEqual("clickhouse_health_timeout", first["clickhouse"]["last_error"])
        self.assertTrue(second["cached"])
        self.assertEqual(1, redis.calls)
        self.assertEqual(1, clickhouse.calls)

    async def test_clickhouse_write_backoff_skips_operation_factory(self):
        redis = _FakeRedisHealth()
        clickhouse = _FakeClickHouseHealth()
        clickhouse._disabled_until = monotonic() + 30.0
        clickhouse._last_error = "clickhouse_down"
        storage = F1Storage(redis_store=redis, clickhouse_store=clickhouse)
        called = False

        async def _operation():
            nonlocal called
            called = True
            return {"ok": True}

        status = await storage._clickhouse_best_effort(lambda: _operation(), "f1_probability_snapshots")

        self.assertFalse(status["ok"])
        self.assertTrue(status["skipped"])
        self.assertEqual("clickhouse_down", status["reason"])
        self.assertFalse(called)

    async def test_clickhouse_health_failure_enters_backoff(self):
        store = F1ClickHouseStore(enabled=True)
        store._query = AsyncMock(side_effect=RuntimeError("clickhouse down"))

        first = await store.health()
        second = await store.health()

        self.assertFalse(first["available"])
        self.assertIn("clickhouse down", first["last_error"])
        self.assertFalse(second["available"])
        self.assertIn("backoff_seconds", second)
        self.assertEqual(1, store._query.await_count)
        await store.close()

    async def test_clickhouse_insert_failure_returns_status_not_exception(self):
        store = F1ClickHouseStore(enabled=True)
        store._query = AsyncMock(side_effect=RuntimeError("clickhouse down"))

        status = await store.append_probability_snapshot(
            2026,
            SimpleNamespace(round=1),
            "race",
            "production_v1",
            {"probabilities": [{"driver_id": "hamilton", "win_probability": 0.1}]},
        )

        self.assertFalse(status["ok"])
        self.assertIn("clickhouse down", status["reason"])
        await store.close()


class _FakeRedisHealth:
    def __init__(self):
        self.calls = 0

    def health(self):
        self.calls += 1
        return {"available": True, "kind": "redis"}


class _FakeClickHouseHealth:
    def __init__(self):
        self.calls = 0

    async def health(self):
        self.calls += 1
        return {"available": True, "kind": "clickhouse"}

    async def close(self):
        return None


class _SlowFakeClickHouseHealth:
    database = "f1_analytics"

    def __init__(self):
        self.calls = 0
        self._disabled_until = 0.0
        self._last_error = None

    async def health(self):
        self.calls += 1
        await asyncio.sleep(1)
        return {"available": True, "kind": "clickhouse"}

    async def close(self):
        return None


if __name__ == "__main__":
    unittest.main()
