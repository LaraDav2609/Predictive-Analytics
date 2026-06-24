import json
from datetime import datetime, timezone

from common.ml.bridge.ops_publisher import InMemoryOpsPublisher, OpsEventPublisher
from common.ml.types import OpsEvent


def make_event(event_type: str = "refresh_started", **detail) -> OpsEvent:
    return OpsEvent(
        domain="f1",
        event_type=event_type,
        severity="info",
        message="hello",
        entity_id="2026-01-BAHRAIN",
        detail=detail,
        emitted_at=datetime(2026, 4, 29, 17, 38, tzinfo=timezone.utc),
    )


def test_channel_naming():
    assert OpsEventPublisher.channel_for(make_event("prediction_ready")) == "f1:ops:prediction_ready"
    assert OpsEventPublisher.recent_key("f1") == "f1:ops:recent"


def test_inmemory_publish_captures_channel_and_payload():
    pub = InMemoryOpsPublisher()
    pub.publish(make_event("degraded", source_mode="estimated"))
    channel, payload = pub.channel_messages[0]
    assert channel == "f1:ops:degraded"
    parsed = json.loads(payload)
    assert parsed["event_type"] == "degraded"
    assert parsed["domain"] == "f1"
    assert parsed["detail"]["source_mode"] == "estimated"
    assert pub.events[0].event_type == "degraded"


def test_publish_uses_injected_client_and_caps_recent_list():
    # A fake redis-like client records calls; OpsEventPublisher must publish to the
    # event channel and maintain the capped recent list without a real server.
    calls = []

    class FakeRedis:
        def publish(self, channel, message):
            calls.append(("publish", channel))
            return 1

        def lpush(self, name, *values):
            calls.append(("lpush", name))
            return 1

        def ltrim(self, name, start, end):
            calls.append(("ltrim", name, start, end))
            return True

        def expire(self, name, time):
            calls.append(("expire", name))
            return True

    pub = OpsEventPublisher(domain="f1", client=FakeRedis(), recent_max=10)
    pub.publish(make_event("refresh_completed"))
    assert [c[0] for c in calls] == ["publish", "lpush", "ltrim", "expire"]
    assert calls[0][1] == "f1:ops:refresh_completed"
    assert calls[2] == ("ltrim", "f1:ops:recent", 0, 9)
