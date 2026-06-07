"""Tests for sports.f1.ml.bridge — channel naming, snapshot keying, payload format."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from sports.f1.ml.bridge.redis_publisher import InMemoryPublisher, RedisPublisher
from sports.f1.ml.common.types import RaceOutcomeProbability


def make_prob(race: str = "2026-01-BAHRAIN", driver: str = "VER",
              market: str = "winner", probability: float = 0.62) -> RaceOutcomeProbability:
    return RaceOutcomeProbability(
        race_id=race,
        driver_code=driver,
        market=market,
        probability=probability,
        knowable_as_of=datetime(2026, 4, 26, 13, 0, tzinfo=timezone.utc),
        model_version="test",
    )


def test_channel_for_format():
    """Channel name is f1:prob:{race_id}:{driver}:{market}."""
    prob = make_prob()
    assert RedisPublisher.channel_for(prob) == "f1:prob:2026-01-BAHRAIN:VER:winner"


def test_snapshot_key_format():
    prob = make_prob()
    assert RedisPublisher.snapshot_key("2026-01-BAHRAIN") == "f1:snapshot:2026-01-BAHRAIN"


def test_snapshot_field_format():
    prob = make_prob(driver="HAM", market="podium")
    assert RedisPublisher.snapshot_field(prob) == "HAM:podium"


def test_inmemory_publish_records_channel_and_snapshot():
    pub = InMemoryPublisher()
    pub.publish(make_prob())
    assert len(pub.channel_messages) == 1
    channel, payload = pub.channel_messages[0]
    assert channel == "f1:prob:2026-01-BAHRAIN:VER:winner"
    parsed = json.loads(payload)
    assert parsed["driver_code"] == "VER"
    assert parsed["probability"] == 0.62
    assert parsed["model_version"] == "test"
    snap = pub.snapshots["f1:snapshot:2026-01-BAHRAIN"]
    assert "VER:winner" in snap


def test_inmemory_publish_batch_groups_by_race():
    pub = InMemoryPublisher()
    pub.publish_batch([
        make_prob(race="R1", driver="A", market="winner"),
        make_prob(race="R1", driver="B", market="winner"),
        make_prob(race="R2", driver="X", market="winner"),
    ])
    assert len(pub.channel_messages) == 3
    assert "f1:snapshot:R1" in pub.snapshots
    assert "f1:snapshot:R2" in pub.snapshots
    assert len(pub.snapshots["f1:snapshot:R1"]) == 2
    assert len(pub.snapshots["f1:snapshot:R2"]) == 1


def test_parsed_snapshot_round_trips():
    pub = InMemoryPublisher()
    pub.publish_batch([
        make_prob(driver="VER", market="winner", probability=0.62),
        make_prob(driver="VER", market="podium", probability=0.91),
        make_prob(driver="HAM", market="winner", probability=0.10),
    ])
    snap = pub.parsed_snapshot("2026-01-BAHRAIN")
    assert "VER:winner" in snap
    assert "HAM:winner" in snap
    assert snap["VER:winner"].probability == pytest.approx(0.62)
    assert snap["VER:podium"].probability == pytest.approx(0.91)


def test_inmemory_publish_overwrites_same_field():
    pub = InMemoryPublisher()
    pub.publish(make_prob(probability=0.50))
    pub.publish(make_prob(probability=0.62))
    snap = pub.parsed_snapshot("2026-01-BAHRAIN")
    # Latest write wins.
    assert snap["VER:winner"].probability == pytest.approx(0.62)


def test_publish_empty_batch_is_safe():
    pub = InMemoryPublisher()
    pub.publish_batch([])
    assert pub.channel_messages == []
    assert pub.snapshots == {}
