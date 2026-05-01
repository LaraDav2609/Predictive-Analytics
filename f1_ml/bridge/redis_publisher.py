"""Redis publisher — push RaceOutcomeProbability snapshots to the .NET trader
and dashboard.

Two patterns, used together:
  - **Pub/sub**: `f1:prob:{race_id}:{driver_code}:{market}` channels for
    trader hot-path subscribers (low-latency).
  - **Key snapshot**: `f1:snapshot:{race_id}` HASH of `{driver}:{market}` →
    JSON for the dashboard / cold readers (point-in-time fetch).

The .NET side (Common/Redis/RedisManager.cs via IRedisManager) subscribes for
live pipelines and reads snapshot hashes for the predictions page.
"""

from __future__ import annotations

import json
from typing import Protocol

from f1_ml.common.types import RaceOutcomeProbability


class _RedisLike(Protocol):
    def publish(self, channel: str, message: str) -> int: ...
    def hset(self, name: str, mapping: dict[str, str]) -> int: ...
    def expire(self, name: str, time: int) -> bool: ...


class RedisPublisher:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        client: _RedisLike | None = None,
        snapshot_ttl_seconds: int = 6 * 3600,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.snapshot_ttl_seconds = snapshot_ttl_seconds
        self._client = client  # if None, lazily constructed in _ensure_client

    def _ensure_client(self) -> _RedisLike:
        if self._client is None:
            import redis  # imported here so the module loads without redis-server running
            self._client = redis.Redis(host=self.host, port=self.port, db=self.db, decode_responses=True)
        return self._client

    @staticmethod
    def channel_for(prob: RaceOutcomeProbability) -> str:
        return f"f1:prob:{prob.race_id}:{prob.driver_code}:{prob.market}"

    @staticmethod
    def snapshot_key(race_id: str) -> str:
        return f"f1:snapshot:{race_id}"

    @staticmethod
    def snapshot_field(prob: RaceOutcomeProbability) -> str:
        return f"{prob.driver_code}:{prob.market}"

    def publish(self, prob: RaceOutcomeProbability) -> None:
        client = self._ensure_client()
        payload = prob.model_dump_json()
        client.publish(self.channel_for(prob), payload)
        client.hset(self.snapshot_key(prob.race_id), mapping={self.snapshot_field(prob): payload})
        client.expire(self.snapshot_key(prob.race_id), self.snapshot_ttl_seconds)

    def publish_batch(self, probs: list[RaceOutcomeProbability]) -> None:
        if not probs:
            return
        client = self._ensure_client()
        # Group snapshot writes by race for fewer round-trips.
        snapshot_payloads: dict[str, dict[str, str]] = {}
        for prob in probs:
            payload = prob.model_dump_json()
            client.publish(self.channel_for(prob), payload)
            snapshot_payloads.setdefault(self.snapshot_key(prob.race_id), {})[
                self.snapshot_field(prob)
            ] = payload
        for key, mapping in snapshot_payloads.items():
            client.hset(key, mapping=mapping)
            client.expire(key, self.snapshot_ttl_seconds)


class InMemoryPublisher:
    """Drop-in replacement for tests / dry runs — captures published payloads
    in lists instead of calling Redis."""

    def __init__(self) -> None:
        self.channel_messages: list[tuple[str, str]] = []
        self.snapshots: dict[str, dict[str, str]] = {}

    def publish(self, prob: RaceOutcomeProbability) -> None:
        payload = prob.model_dump_json()
        self.channel_messages.append((RedisPublisher.channel_for(prob), payload))
        self.snapshots.setdefault(RedisPublisher.snapshot_key(prob.race_id), {})[
            RedisPublisher.snapshot_field(prob)
        ] = payload

    def publish_batch(self, probs: list[RaceOutcomeProbability]) -> None:
        for prob in probs:
            self.publish(prob)

    def parsed_snapshot(self, race_id: str) -> dict[str, RaceOutcomeProbability]:
        snap = self.snapshots.get(RedisPublisher.snapshot_key(race_id), {})
        return {
            field: RaceOutcomeProbability.model_validate_json(json_payload)
            for field, json_payload in snap.items()
        }
