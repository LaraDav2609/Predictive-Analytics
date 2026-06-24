from sports.f1.predictor.data_quality.freshness import (
    alert_key, build_freshness_alerts, summarize_alerts,
)

_HEALTHY = dict(
    features={"drivers": [{"id": "VER"}], "weather_by_round": {1: {}}, "sentiment": {"drivers": {"VER": 1}}},
    last_run={"source_mode": "live"},
    storage_health={"redis": {"available": True}, "clickhouse": {"available": True}},
    service={"predictor_loaded": True, "openf1": True},
)


def test_healthy_pipeline_has_no_alerts():
    alerts = build_freshness_alerts(**_HEALTHY)
    assert alerts == []
    assert summarize_alerts(alerts) == {"total": 0, "error": 0, "warn": 0, "info": 0}


def test_missing_results_is_error():
    args = {**_HEALTHY, "features": {}}
    sources = {a["source"]: a for a in build_freshness_alerts(**args)}
    assert sources["official_results"]["severity"] == "error"
    assert sources["official_results"]["status"] == "missing"


def test_degraded_live_source_and_redis_down():
    alerts = build_freshness_alerts(
        features={"drivers": [1], "weather_by_round": {1: {}}, "sentiment": {"drivers": {"x": 1}}},
        last_run={"source_mode": "estimated", "confidence_ceiling": 0.25, "fallback_reason": "openf1_unavailable"},
        storage_health={"redis": {"available": False}, "clickhouse": {"available": True}},
        service={"predictor_loaded": True, "openf1": True},
    )
    keys = {alert_key(a) for a in alerts}
    assert "live_session_engine:degraded" in keys
    assert "live_session_engine:fallback" in keys
    assert "redis:unavailable" in keys
    summary = summarize_alerts(alerts)
    assert summary["error"] >= 1  # redis down
    assert summary["warn"] >= 1   # estimated live source


def test_weather_fallback_is_warn():
    alerts = build_freshness_alerts(
        features={"drivers": [1], "weather_by_round": {1: {"missing_data": True}, 2: {}}, "sentiment": {"drivers": {"x": 1}}},
        last_run={"source_mode": "live"},
        storage_health={"redis": {"available": True}, "clickhouse": {"available": True}},
        service={"predictor_loaded": True, "openf1": True},
    )
    wf = [a for a in alerts if a["source"] == "weather_fallback"]
    assert wf and wf[0]["severity"] == "warn"
    assert wf[0]["freshness_seconds"] is None  # weather_fallback policy has no freshness limit


def test_unavailable_source_carries_freshness_limit_for_known_sources():
    # openf1 down → its policy freshness limit (120s) is attached for context.
    args = {**_HEALTHY, "service": {"predictor_loaded": True, "openf1": False}}
    openf1 = [a for a in build_freshness_alerts(**args) if a["source"] == "openf1_session_facts"]
    assert openf1 and openf1[0]["freshness_seconds"] == 120


def test_alert_key_is_source_and_status():
    assert alert_key({"source": "redis", "status": "unavailable"}) == "redis:unavailable"
