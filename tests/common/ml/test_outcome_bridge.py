"""Tests for the shared OutcomeProbability contract + domain-namespaced Redis bridge."""

from datetime import datetime, timezone

from common.ml.bridge.outcome_publisher import InMemoryOutcomePublisher, OutcomePublisher
from common.ml.types import OutcomeProbability


def _prob(domain="csgo", entity_id="M1", entity_code="NAVI", market="winner", p=0.6):
    return OutcomeProbability(
        domain=domain, entity_id=entity_id, entity_code=entity_code, market=market,
        probability=p, knowable_as_of=datetime(2026, 6, 7, tzinfo=timezone.utc),
        model_version="v0",
    )


def test_channel_is_domain_namespaced():
    assert OutcomePublisher.channel_for(_prob()) == "csgo:prob:M1:NAVI:winner"
    f1 = _prob(domain="f1", entity_id="2026-BAHRAIN", entity_code="VER")
    assert OutcomePublisher.channel_for(f1) == "f1:prob:2026-BAHRAIN:VER:winner"


def test_snapshot_key_and_field():
    assert OutcomePublisher.snapshot_key("csgo", "M1") == "csgo:snapshot:M1"
    assert OutcomePublisher.snapshot_field(_prob()) == "NAVI:winner"


def test_inmemory_publish_and_roundtrip():
    pub = InMemoryOutcomePublisher()
    pub.publish_batch([_prob(entity_code="NAVI", p=0.6), _prob(entity_code="FAZE", p=0.4)])
    assert len(pub.channel_messages) == 2
    snap = pub.parsed_snapshot("csgo", "M1")
    assert set(snap.keys()) == {"NAVI:winner", "FAZE:winner"}
    assert snap["NAVI:winner"].probability == 0.6


def test_different_domains_are_isolated():
    pub = InMemoryOutcomePublisher()
    pub.publish(_prob(domain="csgo", entity_id="M1"))
    pub.publish(_prob(domain="f1", entity_id="R1"))
    assert "csgo:snapshot:M1" in pub.snapshots
    assert "f1:snapshot:R1" in pub.snapshots
