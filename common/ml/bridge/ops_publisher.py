"""Redis publisher for pipeline ops/health events — the monitoring sibling of
`outcome_publisher`.

Where `OutcomeProbability` carries *predictions*, `OpsEvent` carries *pipeline
state*: a refresh starting/finishing, a prediction being published, a
bridge-publish failure, the live source degrading to `estimated`, etc.
Domain-namespaced exactly like the outcome bridge so f1 / baseball / csgo can
all reuse it:
  - **Pub/sub**: `{domain}:ops:{event_type}` channels for the live monitor feed.
  - **Recent list**: `{domain}:ops:recent` — a capped LIST of the latest events
    for cold readers (a monitor page that loads after an event fired).

The .NET side subscribes to `{domain}:ops:*` and forwards each event to the
SignalR pipeline-monitor group.
"""
from __future__ import annotations

from typing import Protocol

from common.ml.types import OpsEvent


class _RedisLike(Protocol):
    def publish(self, channel: str, message: str) -> int: ...
    def lpush(self, name: str, *values: str) -> int: ...
    def ltrim(self, name: str, start: int, end: int) -> bool: ...
    def expire(self, name: str, time: int) -> bool: ...


class OpsEventPublisher:
    def __init__(
        self,
        domain: str = "f1",
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        client: _RedisLike | None = None,
        recent_max: int = 200,
        recent_ttl_seconds: int = 24 * 3600,
    ) -> None:
        self.domain = domain
        self.host = host
        self.port = port
        self.db = db
        self.recent_max = recent_max
        self.recent_ttl_seconds = recent_ttl_seconds
        self._client = client  # if None, lazily constructed in _ensure_client

    def _ensure_client(self) -> _RedisLike:
        if self._client is None:
            import redis  # imported here so the module loads without redis-server running
            self._client = redis.Redis(host=self.host, port=self.port, db=self.db, decode_responses=True)
        return self._client

    @staticmethod
    def channel_for(event: OpsEvent) -> str:
        return f"{event.domain}:ops:{event.event_type}"

    @staticmethod
    def recent_key(domain: str) -> str:
        return f"{domain}:ops:recent"

    def publish(self, event: OpsEvent) -> None:
        client = self._ensure_client()
        payload = event.model_dump_json()
        client.publish(self.channel_for(event), payload)
        key = self.recent_key(event.domain)
        client.lpush(key, payload)
        client.ltrim(key, 0, self.recent_max - 1)
        client.expire(key, self.recent_ttl_seconds)


class InMemoryOpsPublisher:
    """Drop-in replacement for tests / dry runs — captures published events in a
    list instead of calling Redis."""

    def __init__(self) -> None:
        self.channel_messages: list[tuple[str, str]] = []
        self.events: list[OpsEvent] = []

    def publish(self, event: OpsEvent) -> None:
        self.channel_messages.append((OpsEventPublisher.channel_for(event), event.model_dump_json()))
        self.events.append(event)
