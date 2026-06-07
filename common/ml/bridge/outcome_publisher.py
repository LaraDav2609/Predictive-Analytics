"""Redis publisher — push OutcomeProbability snapshots to the .NET trader and dashboard.

Domain-agnostic generalization of the per-sport publishers: channels and keys are
namespaced by each prediction's own `domain`, so f1, baseball, and csgo all share
this one bridge. Two patterns, used together:
  - **Pub/sub**: `{domain}:prob:{entity_id}:{entity_code}:{market}` channels for
    trader hot-path subscribers (low-latency).
  - **Key snapshot**: `{domain}:snapshot:{entity_id}` HASH of `{entity_code}:{market}`
    -> JSON for the dashboard / cold readers (point-in-time fetch).

The .NET side (Services/Markets/ProbabilityRedisSubscriber) subscribes to
`{domain}:prob:*` and reads the snapshot hashes.
"""
from __future__ import annotations

from typing import Protocol

from common.ml.types import OutcomeProbability


class _RedisLike(Protocol):
    def publish(self, channel: str, message: str) -> int: ...
    def hset(self, name: str, mapping: dict[str, str]) -> int: ...
    def expire(self, name: str, time: int) -> bool: ...


class OutcomePublisher:
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
    def channel_for(prob: OutcomeProbability) -> str:
        return f"{prob.domain}:prob:{prob.entity_id}:{prob.entity_code}:{prob.market}"

    @staticmethod
    def snapshot_key(domain: str, entity_id: str) -> str:
        return f"{domain}:snapshot:{entity_id}"

    @staticmethod
    def snapshot_field(prob: OutcomeProbability) -> str:
        return f"{prob.entity_code}:{prob.market}"

    def publish(self, prob: OutcomeProbability) -> None:
        client = self._ensure_client()
        payload = prob.model_dump_json()
        client.publish(self.channel_for(prob), payload)
        key = self.snapshot_key(prob.domain, prob.entity_id)
        client.hset(key, mapping={self.snapshot_field(prob): payload})
        client.expire(key, self.snapshot_ttl_seconds)

    def publish_batch(self, probs: list[OutcomeProbability]) -> None:
        if not probs:
            return
        client = self._ensure_client()
        # Group snapshot writes by event for fewer round-trips.
        snapshot_payloads: dict[str, dict[str, str]] = {}
        for prob in probs:
            payload = prob.model_dump_json()
            client.publish(self.channel_for(prob), payload)
            key = self.snapshot_key(prob.domain, prob.entity_id)
            snapshot_payloads.setdefault(key, {})[self.snapshot_field(prob)] = payload
        for key, mapping in snapshot_payloads.items():
            client.hset(key, mapping=mapping)
            client.expire(key, self.snapshot_ttl_seconds)


class InMemoryOutcomePublisher:
    """Drop-in replacement for tests / dry runs — captures published payloads
    in lists instead of calling Redis."""

    def __init__(self) -> None:
        self.channel_messages: list[tuple[str, str]] = []
        self.snapshots: dict[str, dict[str, str]] = {}

    def publish(self, prob: OutcomeProbability) -> None:
        payload = prob.model_dump_json()
        self.channel_messages.append((OutcomePublisher.channel_for(prob), payload))
        key = OutcomePublisher.snapshot_key(prob.domain, prob.entity_id)
        self.snapshots.setdefault(key, {})[OutcomePublisher.snapshot_field(prob)] = payload

    def publish_batch(self, probs: list[OutcomeProbability]) -> None:
        for prob in probs:
            self.publish(prob)

    def parsed_snapshot(self, domain: str, entity_id: str) -> dict[str, OutcomeProbability]:
        snap = self.snapshots.get(OutcomePublisher.snapshot_key(domain, entity_id), {})
        return {
            field: OutcomeProbability.model_validate_json(json_payload)
            for field, json_payload in snap.items()
        }
