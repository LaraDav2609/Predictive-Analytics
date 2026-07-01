"""Formula 1 API routes."""

from __future__ import annotations

import asyncio
import json
from asyncio import sleep
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
from time import monotonic

from fastapi import APIRouter
import redis

from common.ml.bridge.outcome_publisher import OutcomePublisher
from common.ml.bridge.ops_publisher import OpsEventPublisher
from common.ml.types import OutcomeProbability, OpsEvent
from sports.f1.data.f1_client import F1Client
from sports.f1.data.openf1_client import OpenF1Client
from sports.f1.data.f1_sentiment import DEFAULT_RSS_FEEDS, read_f1_sentiment, refresh_f1_sentiment
from sports.f1.analytics.f1_predictor import F1Predictor
from sports.f1.analytics.f1_simulator import build_session_simulation
from sports.f1.ml.live_runner import LiveRunnerConfig, run_live_once
from sports.f1.ml.markets.edge_service import F1EdgeRequest, RaceQuote, compute_race_edges, probability_map_from_records
from sports.f1.ml.markets.bet_ledger import run_bet_ledger
from sports.f1.ml.markets.synthetic_market import build_synthetic_decisions
from sports.f1.predictor.probability.empirical_calibration import (
    CALIBRATION_ARTIFACT_ENV, MIN_POSITIVES, MIN_SAMPLES,
    brier_score, fit_winner_calibrator, winner_pairs_from_runs,
)
from sports.f1.predictor.features.market_consensus import build_market_consensus
from sports.f1.predictor.features.track_archetype import archetype_fit_score, archetype_ratings, archetypes_for_track
from sports.f1.predictor.features.penalties import build_penalty_report
from sports.f1.predictor.features.strategy_analysis import analyze_race_strategy, driver_strategy_delta
from sports.f1.predictor.features.upgrade_impact import build_upgrade_impacts, load_upgrade_calendar
from sports.f1.predictor.backtesting import F1BacktestService
from sports.f1.predictor.backtesting.evidence_cache import (
    WeekendEvidenceCacheWriter,
    drivers_from_historical_raw,
    profile_from_historical_raw,
    race_from_historical_raw,
)
from sports.f1.predictor.live import F1LiveSessionEngine, attach_confidence_report, build_live_confidence_report, build_live_dynamics
from sports.f1.predictor.live.recorder import FastF1LiveRecorderManager
from sports.f1.predictor.models.configs import PRODUCTION_MODEL_ID
from sports.f1.predictor.models.registry import F1ModelRegistry
from sports.f1.predictor.service import F1PredictionService
from sports.f1.predictor.storage import F1Storage
from sports.f1.predictor.truth import build_race_truth_snapshot
from sports.f1.predictor.agents import build_car_performance_agent
from sports.f1.predictor.features.car_model import build_car_model_analysis, summarize_car_data
from sports.f1.predictor.features.sentiment import build_race_sentiment_impact
from sports.f1.predictor.features.tires import TireFeatureProvider
from sports.f1.predictor.features.track import TrackFeatureProvider
from sports.f1.predictor.features.weather import WeatherFeatureProvider
from sports.f1.predictor.features.weekend import _grid_evidence, build_weekend_evidence, normalize_weekend_session
from sports.f1.predictor.probability import detect_stage, enrich_probability_payload
from sports.f1.predictor.probability.calibration import build_calibration_profile
from sports.f1.predictor.data_quality import attach_data_quality, build_data_quality_report
from sports.f1.predictor.data_quality.freshness import alert_key, build_freshness_alerts, summarize_alerts
from sports.f1.assistant import AssistantActionRequest, AssistantChatRequest, F1AssistantService

router = APIRouter(prefix="/f1", tags=["f1"])

# Shared instances (initialized in server.py lifespan)
client: F1Client | None = None
predictor: F1Predictor | None = None
openf1: OpenF1Client | None = None
live_engine: F1LiveSessionEngine | None = None
live_recorder: FastF1LiveRecorderManager | None = None
storage: F1Storage | None = None
backtester: F1BacktestService | None = None

_STATIC_RESPONSE_CACHE_TTL_SECONDS = 60.0
_LIVE_RESPONSE_CACHE_TTL_SECONDS = 8.0
_simulation_response_cache: dict[tuple[int, str, str, bool], tuple[float, dict]] = {}
_track_response_cache: dict[tuple[int, str], tuple[float, dict]] = {}
_weekend_evidence_response_cache: dict[tuple[int, str], tuple[float, dict]] = {}
_live_state_response_cache: dict[tuple[int, str], tuple[float, dict]] = {}
_live_truth_response_cache: dict[tuple[int, str, bool], tuple[float, dict]] = {}
_live_probability_response_cache: dict[tuple[int, str, str, bool], tuple[float, dict]] = {}
_race_profile_response_cache: dict[tuple[int], tuple[float, dict]] = {}
_race_profile_tasks: dict[tuple[int], asyncio.Task] = {}
_live_state_tasks: dict[tuple[int, str], asyncio.Task] = {}
_outcome_publisher_override = None
_last_bridge_publish_status: dict = {"ok": None, "reason": "not_requested", "record_count": 0}
# Pipeline-monitor state: ops-event publisher (wired in server.py), an in-process
# ring buffer of recent events for the /pipeline/health snapshot, and the trust
# state of the most recent live prediction run.
_ops_publisher_override = None
_recent_ops_events: list[dict] = []
_RECENT_OPS_EVENTS_MAX = 50
_last_pipeline_run: dict = {"source_mode": None, "reason": "no_run_yet"}
_active_freshness_alert_keys: set = set()
# Per-round market-implied consensus supplied by the dashboard; consumed as a
# bounded feature on the next prediction for that round.
_market_consensus_by_round: dict = {}

_WEEKEND_SESSION_LABELS = {
    "fp1": "Practice 1",
    "fp2": "Practice 2",
    "fp3": "Practice 3",
    "sprint_qualifying": "Sprint Qualifying",
    "sprint": "Sprint",
    "qualifying": "Qualifying",
    "race": "Race",
}


def _get_static_cache(cache: dict, key: tuple) -> dict | None:
    entry = cache.get(key)
    if not entry:
        return None
    cached_at, payload = entry
    age = monotonic() - cached_at
    if age > _STATIC_RESPONSE_CACHE_TTL_SECONDS:
        cache.pop(key, None)
        return None
    cached = deepcopy(payload)
    existing_cache = cached.get("response_cache") or {}
    cached["response_cache"] = {
        **existing_cache,
        "hit": True,
        "age_seconds": round(age, 2),
        "ttl_seconds": _STATIC_RESPONSE_CACHE_TTL_SECONDS,
    }
    return cached


def _get_ttl_cache(cache: dict, key: tuple, ttl_seconds: float, label: str) -> dict | None:
    entry = cache.get(key)
    if not entry:
        return None
    cached_at, payload = entry
    age = monotonic() - cached_at
    if age > ttl_seconds:
        cache.pop(key, None)
        return None
    cached = deepcopy(payload)
    existing_cache = cached.get("response_cache") or {}
    cached["response_cache"] = {
        **existing_cache,
        "scope": label,
        "hit": True,
        "age_seconds": round(age, 2),
        "ttl_seconds": ttl_seconds,
    }
    return cached


def _set_ttl_cache(cache: dict, key: tuple, payload: dict, ttl_seconds: float, label: str) -> None:
    stored = deepcopy(payload)
    existing_cache = stored.get("response_cache") or {}
    stored["response_cache"] = {
        **existing_cache,
        "scope": label,
        "hit": False,
        "age_seconds": 0,
        "ttl_seconds": ttl_seconds,
    }
    cache[key] = (monotonic(), stored)


def _set_static_cache(cache: dict, key: tuple, payload: dict) -> None:
    stored = deepcopy(payload)
    existing_cache = stored.get("response_cache") or {}
    stored["response_cache"] = {
        **existing_cache,
        "hit": False,
        "age_seconds": 0,
        "ttl_seconds": _STATIC_RESPONSE_CACHE_TTL_SECONDS,
    }
    cache[key] = (monotonic(), stored)


def _clear_static_response_caches() -> None:
    _simulation_response_cache.clear()
    _track_response_cache.clear()
    _live_state_response_cache.clear()
    _live_truth_response_cache.clear()
    _live_probability_response_cache.clear()
    _race_profile_response_cache.clear()


async def _race_profile_for_round(round_num: int) -> dict:
    cache_key = (round_num,)
    cached = _get_ttl_cache(_race_profile_response_cache, cache_key, _STATIC_RESPONSE_CACHE_TTL_SECONDS, "race_profile")
    if cached:
        return cached

    task = _race_profile_tasks.get(cache_key)
    if task is None or task.done():
        task = asyncio.create_task(client.get_race_profile(round_num, predictor))
        _race_profile_tasks[cache_key] = task
    try:
        profile = await task
    finally:
        if _race_profile_tasks.get(cache_key) is task:
            _race_profile_tasks.pop(cache_key, None)

    if isinstance(profile, dict) and profile.get("ok", True):
        _set_ttl_cache(_race_profile_response_cache, cache_key, profile, _STATIC_RESPONSE_CACHE_TTL_SECONDS, "race_profile")
        cached_profile = _get_ttl_cache(_race_profile_response_cache, cache_key, _STATIC_RESPONSE_CACHE_TTL_SECONDS, "race_profile")
        return cached_profile or profile
    return profile


def _available_model_ids() -> set[str]:
    return {str(model.get("model_id")) for model in F1ModelRegistry.list_models() if model.get("model_id")}


def _resolve_model_id(model_id: str | None) -> tuple[str | None, dict | None]:
    selected = (model_id or PRODUCTION_MODEL_ID).strip()
    available = sorted(_available_model_ids())
    if selected not in available:
        return None, {
            "ok": False,
            "reason": f"Unknown F1 model id '{selected}'",
            "code": "invalid_model_id",
            "model_id": selected,
            "available_models": available,
            "default_model_id": PRODUCTION_MODEL_ID,
        }
    return selected, None


def _predict_race_with_model(
    *,
    selected_model_id: str,
    race,
    drivers: list,
    constructors: list,
    features: dict,
    sentiment: dict,
):
    service = F1PredictionService(model_id=selected_model_id)
    service.load(drivers, constructors, features, sentiment)
    return service.predict_race(race)


def _prediction_model_fields(prediction: dict | None, selected_model_id: str) -> dict:
    prediction = prediction or {}
    return {
        "model_id": prediction.get("model_id") or selected_model_id,
        "model_version": prediction.get("model_version"),
        "ml_input_source": prediction.get("ml_input_source"),
        "ml_provider_sources": prediction.get("ml_provider_sources") or [],
        "ml_fallback_reason": prediction.get("ml_fallback_reason"),
        "ml_confidence": prediction.get("ml_confidence"),
        "simulator_iterations": prediction.get("simulator_iterations"),
        "trained_artifacts_used": prediction.get("trained_artifacts_used"),
        "evidence_groups_used": prediction.get("evidence_groups_used") or [],
        "ml_artifact_id": prediction.get("ml_artifact_id"),
        "ml_artifact_version": prediction.get("ml_artifact_version"),
        "ml_model_contract_used": prediction.get("ml_model_contract_used"),
        "ml_model_adapters_used": prediction.get("ml_model_adapters_used") or [],
        "ml_model_fallback_reason": prediction.get("ml_model_fallback_reason"),
        "pace_adapter_source": prediction.get("pace_adapter_source"),
        "dnf_adapter_source": prediction.get("dnf_adapter_source"),
        "rating_adapter_source": prediction.get("rating_adapter_source"),
    }


def _assistant_service() -> F1AssistantService:
    return F1AssistantService(
        client=client,
        predictor=predictor,
        openf1=openf1,
        live_engine=live_engine,
        live_recorder=live_recorder,
        storage=storage,
    )


def _ml_live_runner_payload(
    *,
    round_num: int,
    session: str,
    race,
    drivers: list,
    constructors: list,
    features: dict,
    truth: dict,
    weekend_evidence: dict,
    live_state: dict | None,
    sentiment: dict,
    publish: bool = False,
) -> dict:
    try:
        result = run_live_once(
            LiveRunnerConfig(
                race=_race_entity_id(client.season, race),
                season=client.season,
                round_num=round_num,
                session=session,
                model_id="ml_simulator_v1",
                source="auto",
                once=True,
                dry_run=True,
                publish=False,
                n_iterations=int((features or {}).get("ml_simulator_iterations") or 900),
                physical=True,
                race_obj=race,
                drivers=list(drivers or []),
                constructors=list(constructors or []),
                features=features or {},
                sentiment=sentiment or {},
                race_truth=truth or {},
                weekend_evidence=weekend_evidence or {},
                live_state=live_state or {},
            )
        )
        payload = result.payload
        payload["api_publish_requested"] = bool(publish)
        payload["runner_bridge_record_count"] = len(result.published_records)
        return payload
    except Exception as exc:
        return {
            "ok": False,
            "model_id": "ml_simulator_v1",
            "source_mode": (truth or {}).get("source_mode") or "estimated",
            "confidence": (truth or {}).get("confidence"),
            "fallback_reason": f"ml_live_runner_failed:{exc.__class__.__name__}",
            "probabilities": [],
            "production_context": True,
            "race_truth_used": bool(truth),
            "weekend_evidence_used": bool(weekend_evidence),
            "live_state_used": bool(live_state),
        }


def _apply_ml_live_runner_payload(simulation: dict, ml_payload: dict, drivers: list) -> dict:
    if not ml_payload:
        return simulation
    rows = simulation.get("simulations") or simulation.get("probabilities") or []
    probabilities = ml_payload.get("probabilities") or []
    by_code = {str(row.get("driver_code") or "").upper(): row for row in probabilities}
    driver_by_code = {str(getattr(driver, "code", "") or "").upper(): driver for driver in drivers or []}
    if not rows and probabilities:
        rows = []
        for item in probabilities:
            code = str(item.get("driver_code") or "").upper()
            driver = driver_by_code.get(code)
            rows.append({
                "driver_id": getattr(driver, "id", code.lower()),
                "driver_code": code,
                "driver_name": f"{getattr(driver, 'first_name', '')} {getattr(driver, 'last_name', '')}".strip() or code,
                "team": getattr(driver, "team", None),
                "components": {},
            })
    for row in rows:
        code = str(row.get("driver_code") or "").upper()
        item = by_code.get(code)
        if not item:
            continue
        raw = row.get("win_probability") or row.get("calibrated_probability")
        row["ml_live_runner_probability"] = True
        row["win_probability"] = item.get("win_probability")
        row["calibrated_probability"] = item.get("win_probability")
        row["podium_probability"] = item.get("podium_probability")
        row["top5_probability"] = item.get("top5_probability")
        row["dnf_probability"] = item.get("dnf_probability")
        row["fastest_lap_probability"] = item.get("fastest_lap_probability")
        row["expected_finish"] = item.get("expected_finish")
        if raw is not None:
            try:
                row["ml_live_probability_delta"] = round(float(item.get("win_probability") or 0.0) - float(raw or 0.0), 6)
            except (TypeError, ValueError):
                row["ml_live_probability_delta"] = None
        components = row.setdefault("components", {})
        components["ml_live_runner"] = {
            "win_probability": item.get("win_probability"),
            "expected_finish": item.get("expected_finish"),
            "source_mode": ml_payload.get("source_mode"),
            "confidence": ml_payload.get("confidence"),
        }
    simulation["simulations"] = sorted(rows, key=lambda row: float(row.get("win_probability") or 0.0), reverse=True)
    simulation["ml_live_runner"] = ml_payload
    simulation["ml_live_runner_used"] = bool(ml_payload.get("ok", True) and probabilities)
    simulation["ml_live_runner_fallback_reason"] = ml_payload.get("fallback_reason")
    for key in (
        "source_mode",
        "confidence",
        "confidence_ceiling",
        "confidence_reason",
        "ml_input_source",
        "ml_provider_sources",
        "ml_fallback_reason",
        "ml_confidence",
        "simulator_iterations",
        "trained_artifacts_used",
        "evidence_groups_used",
        "ml_artifact_id",
        "ml_artifact_version",
        "ml_model_contract_used",
        "ml_model_adapters_used",
        "ml_model_fallback_reason",
        "pace_adapter_source",
        "dnf_adapter_source",
        "rating_adapter_source",
        "race_truth_used",
        "weekend_evidence_used",
        "live_state_used",
    ):
        value = ml_payload.get(key)
        if value is not None:
            simulation[key] = value
    if ml_payload.get("top_probability_movers"):
        simulation["top_probability_movers"] = ml_payload.get("top_probability_movers")
    return simulation


def _slug(value: str | None) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value or "").strip("-").upper()
    return value or "RACE"


def _race_entity_id(season: int, race) -> str:
    key = race.circuit_id or race.country or race.name or f"round-{race.round}"
    return f"{season}-{int(race.round):02d}-{_slug(key)}"


def _parse_datetime(value: str | None):
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def _simulation_rows_to_outcome_probabilities(
    *,
    season: int,
    race,
    rows: list[dict],
    model_version: str,
    generated_at: str | None,
) -> list[OutcomeProbability]:
    entity_id = _race_entity_id(season, race)
    knowable_as_of = _parse_datetime(generated_at)
    market_keys = {
        "winner": ("win_probability", "calibrated_probability"),
        "podium": ("podium_probability",),
        "top5": ("top5_probability",),
        "points": ("points_probability",),
        "dnf": ("dnf_probability",),
    }
    probabilities: list[OutcomeProbability] = []
    for row in rows or []:
        entity_code = str(row.get("driver_code") or row.get("driver_id") or "").upper()
        if not entity_code:
            continue
        for market, keys in market_keys.items():
            value = None
            for key in keys:
                if row.get(key) is not None:
                    value = row.get(key)
                    break
            if value is None:
                continue
            try:
                probability = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
            probabilities.append(
                OutcomeProbability(
                    domain="f1",
                    entity_id=entity_id,
                    entity_code=entity_code,
                    market=market,
                    probability=probability,
                    knowable_as_of=knowable_as_of,
                    model_version=model_version,
                )
            )
    return probabilities


def set_ops_publisher(publisher) -> None:
    """Install the process-wide ops-event publisher (wired in server.py lifespan).
    Tests can inject an InMemoryOpsPublisher here."""
    global _ops_publisher_override
    _ops_publisher_override = publisher


def _emit_ops_event(event_type: str, *, severity: str = "info", message: str = "",
                    entity_id: str | None = None, **detail) -> dict:
    """Record a pipeline ops/health event: append to the in-process ring buffer
    (read by /pipeline/health) and best-effort publish to the Redis ops bridge
    (f1:ops:{event_type}) for the live monitor feed. Never raises."""
    event = OpsEvent(
        domain="f1",
        event_type=event_type,
        severity=severity,
        message=message,
        entity_id=entity_id,
        detail=detail,
        emitted_at=datetime.now(timezone.utc),
    )
    record = event.model_dump(mode="json")
    _recent_ops_events.append(record)
    if len(_recent_ops_events) > _RECENT_OPS_EVENTS_MAX:
        del _recent_ops_events[:-_RECENT_OPS_EVENTS_MAX]
    try:
        publisher = _ops_publisher_override or OpsEventPublisher(domain="f1")
        publisher.publish(event)
    except Exception:
        # Monitoring must never break the pipeline; the in-process buffer still
        # holds the event for the next /pipeline/health poll.
        pass
    return record


def _record_pipeline_run(payload: dict, simulation: dict, truth: dict, bridge_records: list) -> None:
    """Capture the trust state of the most recent live prediction run (source mode,
    confidence, provenance) for /pipeline/health, and emit a degraded/recovered
    ops event when the live source quality transitions."""
    global _last_pipeline_run
    truth = truth or {}
    simulation = simulation or {}
    source_mode = truth.get("source_mode") or payload.get("source_mode") or "unknown"
    previous_mode = _last_pipeline_run.get("source_mode")
    entity_id = bridge_records[0].entity_id if bridge_records else None
    _last_pipeline_run = {
        "source_mode": source_mode,
        "confidence": truth.get("confidence"),
        "confidence_ceiling": truth.get("confidence_ceiling"),
        "confidence_reason": truth.get("confidence_reason"),
        "trained_artifacts_used": bool(simulation.get("trained_artifacts_used")),
        "ml_input_source": simulation.get("ml_input_source"),
        "fallback_reason": simulation.get("ml_fallback_reason") or truth.get("fallback_reason"),
        "simulator_iterations": simulation.get("simulator_iterations"),
        "model_version": simulation.get("prediction_model_version") or simulation.get("model_id"),
        "generated_at": simulation.get("generated_at"),
        "entity_id": entity_id,
        "bridge_record_count": len(bridge_records),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    degraded_modes = {"estimated", "synthetic", "unavailable", "recording_pending"}
    if source_mode in degraded_modes and source_mode != previous_mode:
        _emit_ops_event(
            "degraded", severity="warn",
            message=f"Live source degraded to '{source_mode}' (confidence ceiling {truth.get('confidence_ceiling')})",
            entity_id=entity_id, source_mode=source_mode, confidence=truth.get("confidence"),
        )
    elif previous_mode in degraded_modes and source_mode not in degraded_modes:
        _emit_ops_event(
            "recovered", severity="info",
            message=f"Live source recovered to '{source_mode}'",
            entity_id=entity_id, source_mode=source_mode,
        )


def _emit_freshness_alert_transitions(alerts: list[dict]) -> None:
    """Emit an ops event when a warn/error freshness alert newly appears or clears,
    so the f1:ops feed (and the monitor) shows transitions without spamming on every
    poll of /pipeline/health."""
    global _active_freshness_alert_keys
    current = {alert_key(a) for a in alerts if a.get("severity") in {"warn", "error"}}
    by_key = {alert_key(a): a for a in alerts}
    for key in current - _active_freshness_alert_keys:
        a = by_key[key]
        _emit_ops_event("freshness_alert", severity=a["severity"], message=a["message"],
                        source=a["source"], status=a["status"])
    for key in _active_freshness_alert_keys - current:
        _emit_ops_event("freshness_cleared", severity="info",
                        message=f"Freshness alert cleared: {key}", alert_key=key)
    _active_freshness_alert_keys = current


def _publish_outcome_probabilities(records: list[OutcomeProbability]) -> dict:
    global _last_bridge_publish_status
    if not records:
        _last_bridge_publish_status = {"ok": None, "reason": "no_records", "record_count": 0}
        return _last_bridge_publish_status
    try:
        publisher = _outcome_publisher_override or OutcomePublisher()
        publisher.publish_batch(records)
        _last_bridge_publish_status = {"ok": True, "reason": "published", "record_count": len(records)}
        _emit_ops_event(
            "prediction_ready",
            message=f"Published {len(records)} probability records to the bridge",
            entity_id=records[0].entity_id,
            record_count=len(records),
            markets=sorted({r.market for r in records}),
        )
    except Exception as exc:
        _last_bridge_publish_status = {
            "ok": False,
            "reason": f"{exc.__class__.__name__}: {exc}",
            "record_count": len(records),
        }
        _emit_ops_event(
            "publish_failed",
            severity="error",
            message=f"Bridge publish failed: {exc.__class__.__name__}: {exc}",
            entity_id=records[0].entity_id if records else None,
            record_count=len(records),
        )
    return _last_bridge_publish_status


def init(fc: F1Client, fp: F1Predictor):
    global client, predictor, openf1, live_engine, live_recorder, storage, backtester
    client, predictor = fc, fp
    openf1 = OpenF1Client()
    live_engine = F1LiveSessionEngine(openf1)
    live_recorder = FastF1LiveRecorderManager()
    storage = F1Storage()
    backtester = F1BacktestService(client)


def _backtester() -> F1BacktestService:
    global backtester
    if backtester is None:
        backtester = F1BacktestService(client)
    return backtester


async def close():
    if openf1:
        await openf1.close()
    if storage:
        await storage.close()


def _sentiment_snapshot() -> dict:
    if predictor:
        return predictor.get_sentiment()
    return {}


async def _ensure_sentiment() -> dict:
    sentiment = _sentiment_snapshot()
    if sentiment.get("drivers") and sentiment.get("teams"):
        return sentiment

    sentiment = read_f1_sentiment(client.get_drivers(), client.get_constructors(), limit=80)
    if sentiment.get("drivers") and sentiment.get("teams") and predictor:
        features = await client.get_prediction_features()
        predictor.load_drivers(client.get_drivers(), client.get_constructors(), features, sentiment)
        await _persist_sentiment(sentiment)
        return sentiment

    refreshed = await refresh_f1_sentiment(client.get_drivers(), client.get_constructors(), client.season)
    if predictor:
        features = await client.get_prediction_features()
        predictor.load_drivers(client.get_drivers(), client.get_constructors(), features, refreshed)
    await _persist_sentiment(refreshed)
    return refreshed


async def _persist_sentiment(sentiment: dict) -> None:
    if storage and sentiment:
        await storage.persist_sentiment(client.season, sentiment.get("items") or [], sentiment)


async def _race_sentiment_impact_for_round(round_num: int, session: str = "race") -> dict:
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    sentiment = await _ensure_sentiment()
    impact = build_race_sentiment_impact(
        race=race,
        drivers=client.get_drivers(),
        constructors=client.get_constructors(),
        sentiment=sentiment,
        session=session,
    )
    if storage and impact.get("ok"):
        try:
            impact["storage"] = await storage.persist_race_sentiment_impact(client.season, race, session, impact)
        except AttributeError:
            impact["storage"] = {"ok": False, "reason": "race_sentiment_storage_unavailable"}
    return impact


async def _sentiment_with_race_impact(round_num: int, session: str = "race") -> tuple[dict, dict]:
    sentiment = await _ensure_sentiment()
    impact = await _race_sentiment_impact_for_round(round_num, session=session)
    enriched = {**(sentiment or {}), "race_sentiment_impact": impact}
    return enriched, impact


def _sentiment_impact_fields(impact: dict | None) -> dict:
    impact = impact or {}
    return {
        "sentiment_impact": impact,
        "sentiment_source_count": impact.get("source_count") or 0,
        "sentiment_article_count": impact.get("article_count") or 0,
        "sentiment_confidence": impact.get("confidence") or 0.0,
        "sentiment_missing_groups": impact.get("missing_groups") or [],
        "sentiment_explanations": impact.get("explanations") or [],
    }


def _apply_sentiment_impact_to_rows(rows: list[dict], impact: dict | None) -> list[dict]:
    drivers = ((impact or {}).get("drivers") or {})
    for row in rows or []:
        driver_id = row.get("driver_id")
        driver_impact = drivers.get(driver_id) or {}
        if not driver_impact:
            continue
        row.setdefault("race_sentiment_impact_score", driver_impact.get("driver_impact_score") or 0.0)
        row.setdefault("race_sentiment_delta", driver_impact.get("prediction_delta") or 0.0)
        row.setdefault("race_sentiment_confidence", driver_impact.get("confidence") or 0.0)
        row.setdefault("race_sentiment_articles", driver_impact.get("article_count") or 0)
        row.setdefault("race_sentiment_explanations", driver_impact.get("explanations") or [])
        components = row.setdefault("components", {})
        components.setdefault("race_sentiment_impact", row.get("race_sentiment_impact_score"))
        components.setdefault("race_sentiment_delta", row.get("race_sentiment_delta"))
        components.setdefault("race_sentiment_confidence", row.get("race_sentiment_confidence"))
    return rows or []


def _apply_probability_audit(
    payload: dict,
    profile: dict | None,
    truth: dict | None,
    *,
    stage: str | None = "auto",
    live: bool = False,
) -> dict:
    enriched = enrich_probability_payload(
        payload,
        profile=profile,
        truth=truth or payload.get("truth") or {},
        stage=stage,
        live=live,
        model_id=payload.get("model_id") or payload.get("model_version") or "production_v1",
    )
    data_quality = _data_quality_report(
        truth=truth or payload.get("truth") or {},
        profile=profile,
        probabilities=enriched.get("simulations") or enriched.get("probabilities") or [],
        stage=enriched.get("stage") or stage,
        live=live,
        sentiment_impact=enriched.get("sentiment_impact") or (truth or {}).get("sentiment_impact"),
    )
    attach_data_quality(enriched, data_quality)
    multiplier = float((data_quality.get("influence_caps_applied") or {}).get("quality_confidence_multiplier") or 1.0)
    enriched["confidence"] = round(max(0.05, min(0.98, float(enriched.get("confidence") or 0.0) * multiplier)), 4)
    existing = enriched.get("probability_explanations") or []
    audit_explanations = (enriched.get("probability_audit") or {}).get("explanations") or []
    quality_explanations = _data_quality_explanations(data_quality)
    audit = enriched.get("probability_audit") or {}
    audit["data_quality"] = data_quality
    audit["bias_warnings"] = data_quality.get("bias_warnings") or []
    audit["source_disagreement"] = data_quality.get("source_disagreement") or {}
    audit["influence_caps_applied"] = data_quality.get("influence_caps_applied") or {}
    audit["leakage_guard_status"] = data_quality.get("leakage_guard_status") or {}
    enriched["probability_audit"] = audit
    enriched["probability_explanations"] = [*quality_explanations, *audit_explanations, *existing]
    return enriched


def _data_quality_report(
    *,
    truth: dict | None,
    profile: dict | None = None,
    probabilities: list[dict] | None = None,
    stage: str | None = "auto",
    live: bool = False,
    sentiment_impact: dict | None = None,
) -> dict:
    try:
        return build_data_quality_report(
            truth=truth or {},
            profile=profile or {},
            probabilities=probabilities or [],
            stage=stage,
            live=live,
            sentiment=sentiment_impact,
        )
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"data_quality_unavailable: {exc}",
            "data_quality_score": 0.0,
            "data_quality_label": "unavailable",
            "source_coverage": {},
            "source_disagreement": {"score": 0.0, "label": "unavailable", "items": []},
            "bias_warnings": [],
            "influence_caps_applied": {"quality_confidence_multiplier": 0.65},
            "leakage_guard_status": {"status": "unavailable", "warnings": []},
        }


def _data_quality_explanations(report: dict | None) -> list[dict]:
    report = report or {}
    explanations = []
    score = report.get("data_quality_score")
    if score is not None:
        explanations.append({
            "type": "data_quality",
            "message": f"Data quality is {report.get('data_quality_label', 'unknown')} ({float(score) * 100:.0f}%).",
        })
    for warning in (report.get("bias_warnings") or [])[:3]:
        explanations.append({
            "type": warning.get("type") or "bias_warning",
            "message": warning.get("message") or "Bias guard active.",
            "severity": warning.get("severity"),
        })
    if (report.get("source_disagreement") or {}).get("score", 0) >= 0.12:
        explanations.append({
            "type": "source_disagreement",
            "message": "Source disagreement lowers model confidence and flattens probabilities.",
        })
    return explanations


def _driver_sentiment(driver_id: str) -> dict:
    sentiment = _sentiment_snapshot()
    driver = (sentiment.get("drivers") or {}).get(driver_id) or {}
    composite = sentiment.get("composite") or {}
    personal_score = driver.get("personal_score", driver.get("score", 0.0))
    team_score = driver.get("team_score", 0.0)
    overall_score = driver.get("overall_score", composite.get("Composite", 0.0))
    return {
        "score": driver.get("score", 0.0),
        "label": driver.get("label", "Neutral"),
        "confidence": driver.get("confidence", 0.0),
        "mentions": driver.get("mentions", 0),
        "personal_score": personal_score,
        "personal_label": driver.get("personal_label", "Neutral"),
        "personal_mentions": driver.get("personal_mentions", 0),
        "team_score": team_score,
        "team_label": driver.get("team_label", "Neutral"),
        "team_mentions": driver.get("team_mentions", 0),
        "overall_score": overall_score,
        "overall_label": driver.get("overall_label", composite.get("Label", "Neutral")),
        "win_probability_modifier": driver.get("win_probability_modifier", 1.0),
        "wdc_probability_modifier": driver.get("wdc_probability_modifier", 1.0),
        "source_breakdown": driver.get("source_breakdown", {}),
        "topic_scores": driver.get("topic_scores", {}),
        "dimensions": [
            {"key": "personal", "label": "Personal", "score": personal_score, "mentions": driver.get("personal_mentions", 0), "sentiment": driver.get("personal_label", "Neutral")},
            {"key": "team", "label": "Team", "score": team_score, "mentions": driver.get("team_mentions", 0), "sentiment": driver.get("team_label", "Neutral")},
            {"key": "overall", "label": "Overall", "score": overall_score, "mentions": (composite.get("TotalItems") or 0), "sentiment": driver.get("overall_label", composite.get("Label", "Neutral"))},
        ],
        "latest_items": driver.get("latest_items", []),
    }


def _constructor_sentiment(constructor_name: str) -> dict:
    sentiment = _sentiment_snapshot()
    team = (sentiment.get("teams") or {}).get((constructor_name or "").lower()) or {}
    return {
        "score": team.get("score", 0.0),
        "label": team.get("label", "Neutral"),
        "confidence": team.get("confidence", 0.0),
        "mentions": team.get("mentions", 0),
        "source_breakdown": team.get("source_breakdown", {}),
        "topic_scores": team.get("topic_scores", {}),
        "dimensions": [
            {"key": key, "label": key.replace("_", " ").title(), "score": value}
            for key, value in sorted((team.get("topic_scores") or {}).items())
        ],
        "latest_items": team.get("latest_items", []),
    }


def _probability_explanations(rows: list[dict], truth: dict) -> list[dict]:
    source_mode = truth.get("source_mode") or "model"
    confidence = float(truth.get("confidence") or 0.0)
    truth_by_driver = truth.get("by_driver_id") or {}
    weather = (truth.get("signals") or {}).get("weather") or {}
    weather_explanations = weather.get("weather_explanations") or []
    explanations = []
    for row in rows[:10]:
        driver_id = row.get("driver_id")
        driver_truth = truth_by_driver.get(driver_id) or {}
        components = row.get("components") or {}
        reasons = []
        live_position = components.get("live_position") or driver_truth.get("position")
        if live_position:
            reasons.append(f"track position P{live_position}")
        if driver_truth.get("gap_to_leader"):
            reasons.append(f"gap {driver_truth.get('gap_to_leader')}")
        if driver_truth.get("compound"):
            tyre = driver_truth.get("compound")
            age = driver_truth.get("tyre_age")
            reasons.append(f"{tyre} tyre{f' age {age}' if age is not None else ''}")
        if components.get("race_pace") is not None:
            reasons.append(f"race pace {float(components.get('race_pace') or 0.0) * 100:.0f}")
        for live_reason in components.get("live_dynamics_explanations") or []:
            if live_reason not in reasons:
                reasons.append(str(live_reason))
        if components.get("reliability") is not None:
            reasons.append(f"reliability {float(components.get('reliability') or 0.0) * 100:.0f}")
        sentiment_delta = row.get("race_sentiment_delta") or components.get("race_sentiment_delta")
        if sentiment_delta:
            reasons.append(f"news impact {float(sentiment_delta) * 100:+.1f}%")
        if weather_explanations:
            reasons.append(str(weather_explanations[0]))
        elif float(weather.get("chaos_score") or 0.0) > 0.12:
            reasons.append("weather volatility increased model variance")
        if source_mode == "estimated":
            reasons.append("estimated-source discount applied")
        explanations.append({
            "driver_id": driver_id,
            "driver_code": row.get("driver_code"),
            "source_mode": driver_truth.get("source_mode") or source_mode,
            "confidence": round(float(driver_truth.get("confidence") if driver_truth.get("confidence") is not None else confidence), 4),
            "reasons": reasons[:5],
        })
    return explanations


def _apply_live_probability_fields(payload: dict, truth: dict | None = None) -> dict:
    live_dynamics = payload.get("live_dynamics") or {}
    dynamics_by_driver = live_dynamics.get("drivers") or {}
    rows = payload.get("simulations") or payload.get("probabilities") or []
    for row in rows:
        driver_id = str(row.get("driver_id") or "")
        dynamic = dynamics_by_driver.get(driver_id) or {}
        components = row.setdefault("components", {})
        win = float(row.get("win_probability") or row.get("calibrated_probability") or 0.0)
        raw = float(row.get("raw_probability") or win)
        live_delta = float(dynamic.get("live_delta") or components.get("live_delta") or 0.0)
        row["live_adjusted_probability"] = round(win, 4)
        row["live_probability_delta"] = round(win - raw + live_delta * 0.18, 4)
        row["live_dynamics_explanations"] = dynamic.get("explanations") or components.get("live_dynamics_explanations") or []
        row["live_evidence_score"] = dynamic.get("evidence_score") or components.get("live_evidence_score") or {}
        row["strategy_state"] = payload.get("strategy_state") or {}
        row["tyre_phase"] = _tyre_phase(dynamic, components)
        row["pit_window_status"] = (payload.get("strategy_state") or {}).get("pit_window_status")
        for key, source in {
            "live_position_delta": "position_delta",
            "gap_delta": "gap_delta",
            "pace_trend_delta": "pace_delta",
            "tyre_risk_delta": "tyre_delta",
            "pit_strategy_delta": "pit_delta",
            "live_strength_multiplier": "strength_multiplier",
        }.items():
            if dynamic.get(source) is not None:
                components[key] = dynamic.get(source)
                row[key] = dynamic.get(source)
    live_movers = sorted(
        [
            {
                "driver_id": row.get("driver_id"),
                "driver_code": row.get("driver_code"),
                "driver_name": row.get("driver_name"),
                "team": row.get("team"),
                "delta": row.get("live_probability_delta"),
                "live_adjusted_probability": row.get("live_adjusted_probability"),
                "top_reason": (row.get("live_dynamics_explanations") or ["live state"])[0],
            }
            for row in rows
            if row.get("live_probability_delta") is not None
        ],
        key=lambda item: abs(float(item.get("delta") or 0.0)),
        reverse=True,
    )[:5]
    if live_movers:
        payload["top_probability_movers"] = live_movers
    payload["probability_timeline"] = {
        "source_mode": (truth or {}).get("source_mode") or payload.get("source_mode"),
        "confidence": (truth or {}).get("confidence") or payload.get("confidence"),
        "top5": [
            {
                "driver_id": row.get("driver_id"),
                "driver_code": row.get("driver_code"),
                "driver_name": row.get("driver_name"),
                "win_probability": row.get("win_probability"),
                "live_probability_delta": row.get("live_probability_delta"),
            }
            for row in rows[:5]
        ],
        "movers": live_movers,
    }
    return payload


_LIVE_PROBABILITY_ROW_KEYS = {
    "driver_id",
    "driver_code",
    "driver_name",
    "name",
    "code",
    "team",
    "constructor_id",
    "position",
    "rank",
    "win_probability",
    "calibrated_probability",
    "raw_probability",
    "live_adjusted_probability",
    "live_probability_delta",
    "podium_probability",
    "top5_probability",
    "points_probability",
    "dnf_probability",
    "wdc_probability",
    "expected_finish",
    "confidence",
    "source_mode",
    "top_reason",
    "live_dynamics_explanations",
    "live_evidence_score",
    "race_sentiment_delta",
    "race_sentiment_confidence",
    "strategy_state",
    "tyre_phase",
    "pit_window_status",
    "live_position_delta",
    "gap_delta",
    "pace_trend_delta",
    "tyre_risk_delta",
    "pit_strategy_delta",
    "live_strength_multiplier",
}


_LIVE_COMPONENT_KEYS = {
    "live_position_delta",
    "gap_delta",
    "pace_trend_delta",
    "tyre_risk_delta",
    "pit_strategy_delta",
    "live_strength_multiplier",
    "live_dynamics_explanations",
    "live_evidence_score",
    "race_sentiment_delta",
    "race_sentiment_confidence",
}


_LIVE_STATE_DRIVER_KEYS = {
    "driver_id",
    "driver_code",
    "driver_name",
    "team",
    "position",
    "gap_to_leader",
    "interval",
    "lap",
    "compound",
    "tyre_age",
    "pit_stops",
    "progress",
    "estimated_progress",
    "x",
    "y",
    "source_mode",
    "confidence",
}


def _compact_dict(source: dict | None, keys: set[str]) -> dict:
    source = source or {}
    return {key: source.get(key) for key in keys if key in source and source.get(key) is not None}


def _compact_data_quality(report: dict | None) -> dict:
    report = report or {}
    return {
        "ok": report.get("ok", True),
        "data_quality_score": report.get("data_quality_score"),
        "data_quality_label": report.get("data_quality_label"),
        "source_coverage": report.get("source_coverage") or {},
        "source_disagreement": {
            key: value
            for key, value in (report.get("source_disagreement") or {}).items()
            if key in {"score", "label", "items"}
        },
        "bias_warnings": (report.get("bias_warnings") or [])[:5],
        "influence_caps_applied": report.get("influence_caps_applied") or {},
        "leakage_guard_status": report.get("leakage_guard_status") or {},
    }


def _compact_truth_snapshot(truth: dict | None) -> dict:
    truth = truth or {}
    signals = truth.get("signals") or {}
    compact = {
        "ok": truth.get("ok", True),
        "round": truth.get("round"),
        "session": truth.get("session"),
        "source_mode": truth.get("source_mode"),
        "confidence": truth.get("confidence"),
        "confidence_ceiling": truth.get("confidence_ceiling"),
        "confidence_reason": truth.get("confidence_reason"),
        "data_age_seconds": truth.get("data_age_seconds"),
        "fallback_reason": truth.get("fallback_reason"),
        "missing_groups": truth.get("missing_groups") or [],
        "source_coverage": truth.get("source_coverage") or {},
        "data_quality": _compact_data_quality(truth.get("data_quality") or {}),
    }
    if signals.get("weather"):
        compact["signals"] = {"weather": signals.get("weather")}
    return compact


def _compact_live_state_payload(live_state: dict | None) -> dict:
    live_state = live_state or {}
    drivers = live_state.get("drivers") or live_state.get("live_positions") or []
    return {
        "ok": live_state.get("ok", True),
        "round": live_state.get("round"),
        "session": live_state.get("session"),
        "mode": live_state.get("mode"),
        "source_mode": live_state.get("source_mode") or live_state.get("mode"),
        "confidence": live_state.get("confidence"),
        "data_age_seconds": live_state.get("data_age_seconds"),
        "fallback_reason": live_state.get("fallback_reason"),
        "last_successful_source": live_state.get("last_successful_source"),
        "driver_count": live_state.get("driver_count") or len(drivers),
        "drivers": [_compact_dict(driver, _LIVE_STATE_DRIVER_KEYS) for driver in drivers[:24] if isinstance(driver, dict)],
    }


def _compact_probability_row(row: dict) -> dict:
    compact = _compact_dict(row, _LIVE_PROBABILITY_ROW_KEYS)
    components = _compact_dict(row.get("components") or {}, _LIVE_COMPONENT_KEYS)
    if components:
        compact["components"] = components
    return compact


def _compact_live_probability_payload(payload: dict) -> dict:
    rows = payload.get("probabilities") or []
    compact_rows = [_compact_probability_row(row) for row in rows if isinstance(row, dict)]
    compact = {
        "ok": payload.get("ok", True),
        "round": payload.get("round"),
        "session": payload.get("session"),
        "model_id": payload.get("model_id"),
        "selected_model_id": payload.get("selected_model_id"),
        "prediction_model_version": payload.get("prediction_model_version"),
        "source_mode": payload.get("source_mode"),
        "confidence": payload.get("confidence"),
        "confidence_ceiling": payload.get("confidence_ceiling"),
        "confidence_reason": payload.get("confidence_reason"),
        "data_age_seconds": payload.get("data_age_seconds"),
        "fallback_reason": payload.get("fallback_reason"),
        "missing_groups": payload.get("missing_groups") or [],
        "probability_explanations": (payload.get("probability_explanations") or [])[:12],
        "probabilities": compact_rows,
        "context": payload.get("context") or {},
        "stage": payload.get("stage"),
        "calibration_profile": payload.get("calibration_profile") or {},
        "top_probability_movers": payload.get("top_probability_movers") or [],
        "strategy_state": payload.get("strategy_state") or {},
        "probability_timeline": payload.get("probability_timeline") or {},
        "weather": payload.get("weather") or {},
        "weather_source": payload.get("weather_source"),
        "weather_confidence": payload.get("weather_confidence"),
        "weather_missing": payload.get("weather_missing"),
        "weather_model_impact": payload.get("weather_model_impact") or {},
        "weather_explanations": payload.get("weather_explanations") or [],
        "confidence_report": payload.get("confidence_report") or {},
        "data_quality": _compact_data_quality(payload.get("data_quality") or {}),
        "truth": _compact_truth_snapshot(payload.get("truth") or {}),
        "live_state": _compact_live_state_payload(payload.get("live_state") or {}),
        "weekend_evidence_confidence": payload.get("weekend_evidence_confidence"),
        "weekend_evidence_missing_groups": payload.get("weekend_evidence_missing_groups") or [],
        "sentiment_source_count": payload.get("sentiment_source_count") or 0,
        "sentiment_article_count": payload.get("sentiment_article_count") or 0,
        "sentiment_confidence": payload.get("sentiment_confidence") or 0.0,
        "sentiment_missing_groups": payload.get("sentiment_missing_groups") or [],
        "sentiment_explanations": (payload.get("sentiment_explanations") or [])[:5],
        "timeline_record": payload.get("timeline_record") or {},
        "storage": payload.get("storage") or {},
    }
    for key in (
        "ml_input_source",
        "ml_provider_sources",
        "ml_fallback_reason",
        "ml_confidence",
        "trained_artifacts_used",
        "ml_artifact_id",
        "ml_artifact_version",
        "ml_live_runner",
        "ml_live_runner_used",
    ):
        if payload.get(key) is not None:
            compact[key] = payload.get(key)
    return compact


def _compact_timeline_event(event: dict) -> dict:
    event = event or {}
    compact = {
        key: event.get(key)
        for key in (
            "at",
            "type",
            "lap",
            "time",
            "date",
            "message",
            "event",
            "category",
            "source",
            "driver",
            "mode",
            "source_mode",
            "confidence",
            "stage",
            "chaos_score",
            "fallback_reason",
        )
        if event.get(key) is not None
    }
    top_rows = event.get("top5") or event.get("top") or []
    if isinstance(top_rows, list) and top_rows:
        compact["top5"] = [
            _compact_dict(row, {"driver_id", "driver_code", "driver_name", "team", "win_probability", "live_probability_delta"})
            for row in top_rows[:5]
            if isinstance(row, dict)
        ]
    movers = event.get("movers") or []
    if isinstance(movers, list) and movers:
        compact["movers"] = [
            _compact_dict(row, {"driver_id", "driver_code", "driver_name", "team", "delta", "live_adjusted_probability", "top_reason"})
            for row in movers[:5]
            if isinstance(row, dict)
        ]
    summary = event.get("live_dynamics_summary") or []
    if isinstance(summary, list) and summary:
        compact["live_dynamics_summary"] = [
            _compact_dict(row, {"driver_id", "driver_code", "delta", "reason"})
            for row in summary[:5]
            if isinstance(row, dict)
        ]
    return compact


def _compact_live_timeline_payload(payload: dict) -> dict:
    events = payload.get("events") or []
    return {
        "ok": payload.get("ok", True),
        "round": payload.get("round"),
        "session": payload.get("session"),
        "current": _compact_live_state_payload(payload.get("current") or {}),
        "events": [_compact_timeline_event(event) for event in events[-40:] if isinstance(event, dict)],
        "event_count": len(events),
    }


def _confidence_report(
    truth: dict | None,
    *,
    live_state: dict | None = None,
    diagnostics: dict | None = None,
    probabilities: list[dict] | None = None,
) -> dict:
    round_num = (live_state or {}).get("round") or (truth or {}).get("round")
    session = (live_state or {}).get("session") or (truth or {}).get("session") or "race"
    return build_live_confidence_report(
        truth or {},
        live_state=live_state or {},
        diagnostics=diagnostics or {},
        recorder_status=(live_recorder.status(round_num=round_num, session=session) if live_recorder else {"ok": False, "reason": "live_recorder_unavailable"}),
        probabilities=probabilities or [],
    )


def _tyre_phase(dynamic: dict, components: dict) -> dict:
    compound = dynamic.get("compound") or components.get("compound")
    tyre_age = dynamic.get("tyre_age") if dynamic.get("tyre_age") is not None else components.get("tyre_age")
    try:
        age = int(tyre_age) if tyre_age is not None else None
    except (TypeError, ValueError):
        age = None
    if age is None:
        phase = "unknown"
    elif age >= 28:
        phase = "cliff_risk"
    elif age >= 18:
        phase = "worn"
    elif age <= 4:
        phase = "fresh"
    else:
        phase = "stable"
    return {
        "compound": compound,
        "tyre_age": age,
        "phase": phase,
    }


async def _sprint_unavailable(round_num: int, session: str) -> dict | None:
    if (session or "").lower() != "sprint":
        return None
    profile = await _race_profile_for_round(round_num)
    context = profile.get("context") or {}
    if context.get("has_sprint"):
        return None
    return {
        "ok": False,
        "reason": "Sprint session is not scheduled for this Grand Prix",
        "code": "sprint_unavailable",
        "context": context,
    }


async def _live_state_for_round(round_num: int, session: str = "race", force: bool = False) -> dict:
    session_key = (session or "race").lower()
    cache_key = (round_num, session_key)
    if not force:
        cached = _get_ttl_cache(_live_state_response_cache, cache_key, _LIVE_RESPONSE_CACHE_TTL_SECONDS, "live_state")
        if cached:
            return cached
        task = _live_state_tasks.get(cache_key)
        if task is not None and not task.done():
            state = await task
            return deepcopy(state)
        task = asyncio.create_task(_live_state_for_round_uncached(round_num, session=session, force=False))
        _live_state_tasks[cache_key] = task
        try:
            state = await task
        finally:
            if _live_state_tasks.get(cache_key) is task:
                _live_state_tasks.pop(cache_key, None)
        return deepcopy(state)
    else:
        _live_state_response_cache.pop(cache_key, None)
        _live_truth_response_cache.pop((round_num, session_key, True), None)
        for key in list(_live_probability_response_cache):
            if key[0] == round_num and key[1] == session_key:
                _live_probability_response_cache.pop(key, None)
    return await _live_state_for_round_uncached(round_num, session=session, force=force)


async def _live_state_for_round_uncached(round_num: int, session: str = "race", force: bool = False) -> dict:
    session_key = (session or "race").lower()
    cache_key = (round_num, session_key)
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    sprint_error = await _sprint_unavailable(round_num, session)
    if sprint_error:
        return {**sprint_error, "race": race.model_dump(mode="json")}
    if not live_engine:
        return {
            "ok": False,
            "reason": "Live session engine unavailable",
            "code": "live_engine_unavailable",
            "race": race.model_dump(mode="json"),
        }
    state = await live_engine.get_state(race, client.get_drivers(), session=session, force=force)
    if live_recorder:
        recorder_status = live_recorder.status(round_num=round_num, session=session)
        state["recorder_status"] = recorder_status
        recording_mode = recorder_status.get("recording_source_mode")
        if recording_mode == "recording_pending" and (state.get("source_mode") in {"estimated", "unavailable"} or state.get("mode") in {"estimated", "unavailable"}):
            state["mode"] = "recording_pending"
            state["status"] = "recording_pending"
            state["source_mode"] = "recording_pending"
            state["is_estimated"] = True
            state["confidence"] = min(float(state.get("confidence") or 0.16), 0.18)
            state["fallback_reason"] = recorder_status.get("next_setup_action") or "fastf1_recorder_running_waiting_for_timing_rows"
        elif recording_mode in {"recorded", "recorded_confident"} and state.get("source_mode") == "recorded":
            state["mode"] = recording_mode
            state["status"] = recording_mode
            state["source_mode"] = recording_mode
    attach_confidence_report(state, _confidence_report({}, live_state=state))
    if storage and state:
        state["storage"] = await storage.persist_live_state(client.season, race, session, state)
    _set_ttl_cache(_live_state_response_cache, cache_key, state, _LIVE_RESPONSE_CACHE_TTL_SECONDS, "live_state")
    return state


async def _weather_snapshot_for_round(
    round_num: int,
    session: str = "race",
    live: bool = False,
    features: dict | None = None,
    openf1_session: dict | None = None,
) -> dict:
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}

    sprint_error = await _sprint_unavailable(round_num, session)
    if sprint_error:
        return {**sprint_error, "race": race.model_dump(mode="json")}

    features = features or await client.get_prediction_features()
    if openf1_session is None and openf1:
        openf1_session = await openf1.get_session_features(
            race=race,
            session=session,
            drivers=client.get_drivers(),
            live=live,
        )
        if openf1_session.get("ok"):
            features = {**features, "openf1_session": openf1_session}
    weather = WeatherFeatureProvider(features).get_features(
        race,
        openf1_session or features.get("openf1_session") or {},
        session=session,
    )
    return {
        "ok": True,
        "race": race.model_dump(mode="json"),
        "round": round_num,
        "session": (session or "race").lower(),
        "live": live,
        "weather": weather,
        "weather_source": weather.get("source"),
        "weather_confidence": weather.get("confidence"),
        "weather_missing": bool(weather.get("missing_data")),
        "weather_model_impact": weather.get("weather_model_impact") or {},
        "weather_explanations": weather.get("weather_explanations") or [],
        "fallback_reason": weather.get("reason"),
    }


async def _car_model_for_round(
    round_num: int,
    session: str = "race",
    live: bool = False,
    *,
    include_car_data: bool = False,
) -> dict:
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}

    sprint_error = await _sprint_unavailable(round_num, session)
    if sprint_error:
        return {**sprint_error, "race": race.model_dump(mode="json")}

    features = await client.get_prediction_features()
    openf1_session = {}
    if openf1:
        openf1_session = await openf1.get_session_features(
            race=race,
            session=session,
            drivers=client.get_drivers(),
            live=live,
        )
        if include_car_data and openf1_session.get("ok") and openf1_session.get("session_key"):
            car_rows = await openf1.get_car_data(int(openf1_session["session_key"]))
            openf1_session = {
                **openf1_session,
                "car_data": summarize_car_data(car_rows, client.get_drivers()),
                "raw_counts": {
                    **(openf1_session.get("raw_counts") or {}),
                    "car_data": len(car_rows),
                },
            }
        if openf1_session.get("ok"):
            features = {**features, "openf1_session": openf1_session}

    weather_payload = await _weather_snapshot_for_round(
        round_num,
        session=session,
        live=live,
        features=features,
        openf1_session=openf1_session,
    )
    weather = weather_payload.get("weather") if weather_payload.get("ok") else {}
    track = TrackFeatureProvider(features).get_features(race)
    tires = TireFeatureProvider().get_features(track, openf1_session or {}, weather=weather)
    sentiment_impact = await _race_sentiment_impact_for_round(round_num, session=session)
    car_model = build_car_model_analysis(
        race=race,
        drivers=client.get_drivers(),
        constructors=client.get_constructors(),
        features=features,
        session=session,
        track=track,
        weather=weather,
        tires=tires,
        openf1_session=openf1_session,
        sentiment_impact=sentiment_impact,
    )
    return {
        "ok": True,
        "race": race.model_dump(mode="json"),
        "round": round_num,
        "session": (session or "race").lower(),
        "live": live,
        "track": track,
        "weather": weather,
        "tires": tires,
        "openf1_raw_counts": (openf1_session or {}).get("raw_counts") or {},
        "sentiment_impact": sentiment_impact,
        "car_model": car_model,
        "constructors": car_model.get("constructors") or {},
        "drivers": car_model.get("drivers") or {},
        "rankings": car_model.get("rankings") or [],
        "top_edges": car_model.get("top_edges") or [],
        "confidence": car_model.get("confidence"),
        "missing_data": car_model.get("missing_data") or [],
    }


async def _car_performance_agent_for_round(
    round_num: int,
    session: str = "race",
    live: bool = False,
    *,
    constructor_id: str | None = None,
    include_car_data: bool = False,
) -> dict:
    payload = await _car_model_for_round(
        round_num,
        session=session,
        live=live,
        include_car_data=include_car_data,
    )
    if not payload.get("ok"):
        return payload
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}

    weekend_evidence = await _weekend_evidence_for_round(
        round_num,
        session=session,
        live=live,
        weather=payload.get("weather") or {},
    )
    analysis = build_car_performance_agent(
        race=race,
        drivers=client.get_drivers(),
        constructors=client.get_constructors(),
        features={},
        session=session,
        constructor_id=constructor_id,
        track=payload.get("track") or {},
        weather=payload.get("weather") or {},
        tires=payload.get("tires") or {},
        weekend_evidence=weekend_evidence if weekend_evidence.get("ok", True) else {},
        openf1_session={"raw_counts": payload.get("openf1_raw_counts") or {}},
        car_model=payload.get("car_model") or {},
        sentiment_impact=payload.get("sentiment_impact") or {},
    )
    if analysis.get("ok"):
        analysis["race"] = payload.get("race")
        analysis["live"] = live
        analysis["include_car_data"] = include_car_data
    return analysis


async def _weekend_evidence_for_round(
    round_num: int,
    session: str = "race",
    live: bool = False,
    *,
    profile: dict | None = None,
    features: dict | None = None,
    openf1_session: dict | None = None,
    live_state: dict | None = None,
    weather: dict | None = None,
) -> dict:
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}

    session_code = normalize_weekend_session(session)
    sprint_error = await _sprint_unavailable(round_num, session_code)
    if sprint_error:
        return {**sprint_error, "race": race.model_dump(mode="json")}

    if not live:
        cached = _get_static_cache(_weekend_evidence_response_cache, (round_num, session_code))
        if cached:
            return cached

    profile = profile if profile is not None else await _race_profile_for_round(round_num)
    if not profile.get("ok"):
        return profile

    openf1_sessions: dict[str, dict] = {}
    if openf1:
        sessions_to_fetch = {"fp1", "fp2", "fp3", session_code}
        if session_code == "race":
            sessions_to_fetch.add("qualifying")
        if (profile.get("context") or {}).get("has_sprint"):
            sessions_to_fetch.update({"sprint", "sprint_qualifying"})
        future_weekend = race.date > datetime.now(timezone.utc)
        missing_timing_sessions = 0
        for code in sorted(sessions_to_fetch):
            if openf1_session is not None and code == session_code:
                openf1_sessions[code] = openf1_session
                continue
            if future_weekend and missing_timing_sessions >= 2:
                openf1_sessions[code] = _openf1_probe_short_circuited(code)
                continue
            try:
                openf1_sessions[code] = await openf1.get_session_features(
                    race=race,
                    session=_openf1_session_arg(code),
                    drivers=client.get_drivers(),
                    live=live and code == session_code,
                )
                if _openf1_detail_unavailable(openf1_sessions[code]):
                    missing_timing_sessions += 1
            except Exception as exc:
                openf1_sessions[code] = {
                    "ok": False,
                    "source": "openf1",
                    "session": code,
                    "reason": f"openf1_error:{exc.__class__.__name__}",
                }
                missing_timing_sessions += 1
    elif openf1_session is not None:
        openf1_sessions[session_code] = openf1_session

    if weather is None:
        weather_payload = await _weather_snapshot_for_round(
            round_num,
            session=session_code,
            live=live,
            features=features,
            openf1_session=openf1_sessions.get(session_code) or openf1_session,
        )
        weather = weather_payload.get("weather") if weather_payload.get("ok") else None

    evidence = build_weekend_evidence(
        race=race,
        drivers=client.get_drivers(),
        profile=profile,
        openf1_sessions=openf1_sessions,
        live_state=live_state or {},
        session=session_code,
        live=live,
        weather=weather,
    )
    evidence["race"] = race.model_dump(mode="json")
    evidence["openf1_raw_counts"] = {
        code: (payload.get("raw_counts") or {})
        for code, payload in openf1_sessions.items()
    }
    if not live and evidence.get("ok", True):
        _set_static_cache(_weekend_evidence_response_cache, (round_num, session_code), evidence)
        evidence["response_cache"] = {
            "hit": False,
            "age_seconds": 0,
            "ttl_seconds": _STATIC_RESPONSE_CACHE_TTL_SECONDS,
        }
    return evidence


async def _truth_snapshot_for_round(
    round_num: int,
    session: str = "race",
    live: bool = False,
    profile: dict | None = None,
    features: dict | None = None,
    openf1_session: dict | None = None,
    live_state: dict | None = None,
) -> dict:
    session_key = (session or "race").lower()
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}

    sprint_error = await _sprint_unavailable(round_num, session)
    if sprint_error:
        return {**sprint_error, "race": race.model_dump(mode="json")}

    cache_key = (round_num, session_key, bool(live))
    if live:
        cached = _get_ttl_cache(_live_truth_response_cache, cache_key, _LIVE_RESPONSE_CACHE_TTL_SECONDS, "race_truth")
        if cached:
            return cached

    profile = profile if profile is not None else await _race_profile_for_round(round_num)
    if not profile.get("ok"):
        return profile

    features = features or await client.get_prediction_features()
    if openf1_session is None and openf1:
        openf1_session = await openf1.get_session_features(
            race=race,
            session=session,
            drivers=client.get_drivers(),
            live=live,
        )
    if live_state is None and live:
        live_state = await _live_state_for_round(round_num, session=session, force=False)
    weather_payload = await _weather_snapshot_for_round(
        round_num,
        session=session,
        live=live,
        features=features,
        openf1_session=openf1_session or features.get("openf1_session"),
    )
    weather = weather_payload.get("weather") if weather_payload.get("ok") else None
    weekend_evidence = features.get("weekend_evidence") or await _weekend_evidence_for_round(
        round_num,
        session=session,
        live=live,
        profile=profile,
        features=features,
        openf1_session=openf1_session or features.get("openf1_session"),
        live_state=live_state or features.get("live_state") or {},
        weather=weather,
    )

    truth = build_race_truth_snapshot(
        race=race,
        drivers=client.get_drivers(),
        session=session,
        profile=profile,
        openf1_session=openf1_session or features.get("openf1_session") or {},
        live_state=live_state or features.get("live_state") or {},
        weather=weather,
        live=live,
    )
    truth["weekend_evidence"] = weekend_evidence
    truth["weekend_evidence_confidence"] = weekend_evidence.get("confidence")
    truth["weekend_evidence_missing_groups"] = weekend_evidence.get("missing_groups") or []
    truth["missing_groups"] = sorted(set((truth.get("missing_groups") or []) + [f"weekend_{item}" for item in (weekend_evidence.get("missing_groups") or [])]))
    sentiment_impact = await _race_sentiment_impact_for_round(round_num, session=session)
    truth.update(_sentiment_impact_fields(sentiment_impact))
    track = TrackFeatureProvider(features).get_features(race)
    tires = TireFeatureProvider().get_features(track, openf1_session or features.get("openf1_session") or {}, weather=weather or {})
    car_model = build_car_model_analysis(
        race=race,
        drivers=client.get_drivers(),
        constructors=client.get_constructors(),
        features={**features, "openf1_session": openf1_session or features.get("openf1_session") or {}},
        session=session,
        track=track,
        weather=weather or {},
        tires=tires,
        openf1_session=openf1_session or features.get("openf1_session") or {},
        sentiment_impact=sentiment_impact,
    )
    truth["car_model"] = car_model
    truth["car_model_confidence"] = car_model.get("confidence")
    truth["car_model_missing_data"] = car_model.get("missing_data") or []
    attach_data_quality(
        truth,
        _data_quality_report(
            truth=truth,
            profile=profile,
            probabilities=[],
            stage="auto",
            live=live,
            sentiment_impact=sentiment_impact,
        ),
    )
    attach_confidence_report(truth, _confidence_report(truth, live_state=live_state or features.get("live_state") or {}))
    if storage and truth.get("ok"):
        truth["storage"] = await storage.persist_truth_snapshot(client.season, race, session, truth)
    if live:
        _set_ttl_cache(_live_truth_response_cache, cache_key, truth, _LIVE_RESPONSE_CACHE_TTL_SECONDS, "race_truth")
    return truth


def _canonical_weekend_session(session: str | None) -> str:
    value = (session or "race").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "practice1": "fp1",
        "practice_1": "fp1",
        "free_practice_1": "fp1",
        "p1": "fp1",
        "practice2": "fp2",
        "practice_2": "fp2",
        "free_practice_2": "fp2",
        "p2": "fp2",
        "practice3": "fp3",
        "practice_3": "fp3",
        "free_practice_3": "fp3",
        "p3": "fp3",
        "quali": "qualifying",
        "qualification": "qualifying",
        "sq": "sprint_qualifying",
        "sprint_quali": "sprint_qualifying",
        "sprint_shootout": "sprint_qualifying",
        "grand_prix": "race",
    }
    return aliases.get(value, value if value in _WEEKEND_SESSION_LABELS else "race")


def _openf1_session_arg(session_code: str) -> str:
    return {
        "fp1": "fp1",
        "fp2": "fp2",
        "fp3": "fp3",
        "qualifying": "qualifying",
        "sprint_qualifying": "sprint_qualifying",
        "sprint": "sprint",
        "race": "race",
    }.get(session_code, "race")


def _openf1_detail_unavailable(payload: dict | None) -> bool:
    if not payload or not payload.get("ok"):
        return True
    raw_counts = payload.get("raw_counts") or {}
    return not any(int(raw_counts.get(key) or 0) > 0 for key in ("laps", "positions", "intervals"))


def _openf1_probe_short_circuited(session_code: str) -> dict:
    return {
        "ok": False,
        "source": "openf1",
        "session": session_code,
        "reason": "openf1_detail_probe_short_circuited",
        "raw_counts": {},
        "detail_probe_stopped": True,
    }


async def _weekend_session_result(round_num: int, session: str) -> dict:
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}

    session_code = _canonical_weekend_session(session)
    profile = await _race_profile_for_round(round_num)
    if not profile.get("ok"):
        return profile

    meta = next((item for item in profile.get("sessions") or [] if item.get("code") == session_code), None)
    if not meta and session_code in {"fp1", "fp2", "fp3", "qualifying", "race"}:
        meta = {"code": session_code, "name": _WEEKEND_SESSION_LABELS.get(session_code, session_code), "status": "unknown"}
    if not meta:
        return {
            "ok": True,
            "available": False,
            "code": "session_not_scheduled",
            "reason": f"{_WEEKEND_SESSION_LABELS.get(session_code, session_code)} is not scheduled for this Grand Prix",
            "race": race.model_dump(mode="json"),
            "session": {"code": session_code, "name": _WEEKEND_SESSION_LABELS.get(session_code, session_code)},
            "results": [],
        }

    official_rows = _official_session_rows(profile, session_code)
    openf1_payload = {}
    if openf1:
        try:
            openf1_payload = await openf1.get_session_features(
                race=race,
                session=_openf1_session_arg(session_code),
                drivers=client.get_drivers(),
                live=False,
            )
        except Exception as exc:
            openf1_payload = {"ok": False, "source": "openf1", "reason": f"openf1_error:{exc.__class__.__name__}"}

    timing_rows = _openf1_timing_rows(openf1_payload, client.get_drivers())
    if official_rows:
        rows = _merge_official_and_timing(official_rows, timing_rows, session_code)
        source = "jolpica_official"
        if timing_rows:
            source += "+openf1_timing"
    else:
        rows = timing_rows
        source = (openf1_payload or {}).get("source") or "openf1"

    rows = _rank_session_rows(rows, session_code)
    available = bool(rows)
    reason = None
    if not available:
        reason = "Session has not started yet" if meta.get("status") != "completed" else "No classified timing/result rows are available yet"
        if (openf1_payload or {}).get("reason"):
            reason = f"{reason}; {(openf1_payload or {}).get('reason')}"

    return {
        "ok": True,
        "available": available,
        "race": race.model_dump(mode="json"),
        "session": {
            "code": session_code,
            "name": meta.get("name") or _WEEKEND_SESSION_LABELS.get(session_code, session_code),
            "date": meta.get("date"),
            "status": "completed" if available else meta.get("status") or "unavailable",
            "note": meta.get("note"),
        },
        "source": source if available else "unavailable",
        "source_detail": {
            "official_rows": len(official_rows),
            "openf1_rows": len(timing_rows),
            "openf1_ok": bool((openf1_payload or {}).get("ok")),
            "openf1_reason": (openf1_payload or {}).get("reason"),
            "raw_counts": (openf1_payload or {}).get("raw_counts") or {},
        },
        "reason": reason,
        "results": rows,
        "result_count": len(rows),
    }


def _official_session_rows(profile: dict, session_code: str) -> list[dict]:
    if session_code == "race":
        return [dict(row) for row in profile.get("results") or []]
    if session_code == "qualifying":
        rows = []
        for row in profile.get("qualifying") or []:
            item = dict(row)
            best_seconds = _best_qualifying_seconds(item)
            item["best_time"] = _format_lap_time(best_seconds) if best_seconds else None
            item["best_time_seconds"] = best_seconds
            rows.append(item)
        return rows
    if session_code == "sprint":
        return [dict(row) for row in profile.get("sprint") or []]
    return []


def _openf1_timing_rows(openf1_payload: dict | None, drivers: list) -> list[dict]:
    if not openf1_payload or not openf1_payload.get("ok"):
        return []
    by_number = _driver_lookup_by_number(drivers)
    rows = []
    lap_drivers = ((openf1_payload.get("laps") or {}).get("drivers") or {})
    positions = ((openf1_payload.get("positions") or {}).get("drivers") or {})
    intervals = ((openf1_payload.get("intervals") or {}).get("drivers") or {})
    stints = ((openf1_payload.get("stints") or {}).get("drivers") or {})
    pits = ((openf1_payload.get("pits") or {}).get("drivers") or {})
    numbers = set(lap_drivers.keys()) | set(positions.keys()) | set(intervals.keys()) | set(stints.keys()) | set(pits.keys())
    for number_key in numbers:
        lap = lap_drivers.get(str(number_key)) or {}
        position = positions.get(str(number_key)) or {}
        interval = intervals.get(str(number_key)) or {}
        stint = stints.get(str(number_key)) or {}
        pit = pits.get(str(number_key)) or {}
        try:
            number = int(number_key)
        except (TypeError, ValueError):
            number = int(lap.get("driver_number") or position.get("driver_number") or 0)
        driver = by_number.get(number)
        best_seconds = _safe_float(lap.get("best_lap"))
        rows.append({
            "position": _safe_int(position.get("position")),
            "driver_id": getattr(driver, "id", None),
            "driver_number": number or lap.get("driver_number"),
            "driver_code": lap.get("driver_code") or position.get("driver_code") or getattr(driver, "code", None),
            "driver_name": f"{getattr(driver, 'first_name', '')} {getattr(driver, 'last_name', '')}".strip() or None,
            "team": getattr(driver, "team", None),
            "best_time": _format_lap_time(best_seconds) if best_seconds else None,
            "best_time_seconds": best_seconds,
            "representative_time": _format_lap_time(_safe_float(lap.get("representative_lap"))) if lap.get("representative_lap") else None,
            "median_time": _format_lap_time(_safe_float(lap.get("median_lap"))) if lap.get("median_lap") else None,
            "laps": _safe_int(lap.get("laps")),
            "compound": "/".join(lap.get("compounds") or []) or stint.get("compound"),
            "stint_laps": stint.get("avg_stint_laps"),
            "pit_stops": pit.get("pit_stops"),
            "gap_to_leader": interval.get("gap_to_leader"),
            "interval": interval.get("interval"),
            "source": "openf1_timing",
        })
    return rows


def _merge_official_and_timing(official_rows: list[dict], timing_rows: list[dict], session_code: str) -> list[dict]:
    by_id = {str(row.get("driver_id") or "").lower(): row for row in timing_rows if row.get("driver_id")}
    by_code = {str(row.get("driver_code") or "").upper(): row for row in timing_rows if row.get("driver_code")}
    merged = []
    for row in official_rows:
        timing = by_id.get(str(row.get("driver_id") or "").lower()) or by_code.get(str(row.get("driver_code") or "").upper()) or {}
        best_seconds = _safe_float(timing.get("best_time_seconds")) or _safe_float(row.get("best_time_seconds"))
        item = {
            **row,
            "driver_number": timing.get("driver_number") or row.get("driver_number"),
            "best_time": timing.get("best_time") or row.get("best_time") or row.get("time"),
            "best_time_seconds": best_seconds,
            "representative_time": timing.get("representative_time"),
            "median_time": timing.get("median_time"),
            "laps": timing.get("laps"),
            "compound": timing.get("compound"),
            "stint_laps": timing.get("stint_laps"),
            "pit_stops": timing.get("pit_stops"),
            "gap_to_leader": timing.get("gap_to_leader"),
            "interval": timing.get("interval"),
            "source": "official+openf1_timing" if timing else "official",
        }
        if session_code == "race":
            item["finish_time"] = row.get("time")
        merged.append(item)
    return merged


def _rank_session_rows(rows: list[dict], session_code: str) -> list[dict]:
    if not rows:
        return []
    if session_code in {"fp1", "fp2", "fp3"}:
        ranked = sorted(rows, key=lambda row: (_safe_float(row.get("best_time_seconds")) or 99999, _safe_int(row.get("position")) or 999))
        for index, row in enumerate(ranked, start=1):
            row["position"] = row.get("position") or index
        return ranked
    return sorted(rows, key=lambda row: _safe_int(row.get("position")) or 999)


def _driver_lookup_by_number(drivers: list) -> dict[int, object]:
    lookup = {}
    for driver in drivers:
        try:
            if driver.number is not None:
                lookup[int(driver.number)] = driver
        except (TypeError, ValueError):
            continue
    return lookup


def _best_qualifying_seconds(row: dict) -> float | None:
    values = [_time_to_seconds(row.get(key)) for key in ("q1", "q2", "q3")]
    values = [value for value in values if value is not None]
    return min(values) if values else None


def _time_to_seconds(value) -> float | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return int(minutes) * 60 + float(seconds)
        return float(text)
    except (TypeError, ValueError):
        return None


def _format_lap_time(value) -> str | None:
    seconds = _safe_float(value)
    if seconds is None:
        return None
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}:{remainder:06.3f}" if minutes else f"{remainder:.3f}"


def _safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _openf1_mode(value: str | None) -> str:
    mode = str(value or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    if mode in {"profile", "profile_only", "jolpica", "jolpica_only", "no_openf1", "skip_openf1"}:
        return "profile_only"
    return "auto"


def _fastf1_mode(value: str | None) -> str:
    mode = str(value or "off").strip().lower().replace("-", "_").replace(" ", "_")
    return mode if mode in {"off", "fallback", "force"} else "off"


@router.get("/assistant/context")
async def get_f1_assistant_context(route: str = "/F1"):
    return (await _assistant_service().context(route)).model_dump()


@router.get("/assistant/tools")
async def get_f1_assistant_tools():
    return _assistant_service().tools()


@router.get("/assistant/diagnostics")
async def get_f1_assistant_diagnostics(route: str = "/F1"):
    return await _assistant_service().diagnostics(route)


@router.post("/assistant/chat")
async def post_f1_assistant_chat(request: AssistantChatRequest):
    return (await _assistant_service().chat(request)).model_dump()


@router.post("/assistant/action")
async def post_f1_assistant_action(request: AssistantActionRequest):
    return (await _assistant_service().action(request)).model_dump()


@router.get("/drivers")
async def get_drivers():
    drivers = client.get_drivers()
    return {"ok": True, "drivers": [d.model_dump() for d in drivers]}


@router.get("/drivers/{driver_id}")
async def get_driver(driver_id: str, season: int | None = None):
    profile = await client.get_driver_profile(driver_id, season)
    if profile.get("ok"):
        await _ensure_sentiment()
        driver = profile.get("driver") or {}
        profile["sentiment"] = _driver_sentiment(driver.get("id") or driver_id)
    return profile


@router.get("/constructors")
async def get_constructors():
    constructors = client.get_constructors()
    return {"ok": True, "constructors": [c.model_dump() for c in constructors]}


@router.get("/constructors/{constructor_id}")
async def get_constructor(constructor_id: str, season: int | None = None):
    profile = await client.get_constructor_profile(constructor_id, season)
    if profile.get("ok"):
        await _ensure_sentiment()
        constructor = profile.get("constructor") or {}
        profile["sentiment"] = _constructor_sentiment(constructor.get("name") or constructor_id)
    return profile


@router.get("/constructors/{constructor_id}/car-model")
async def get_constructor_car_model(
    constructor_id: str,
    round: int | None = None,
    session: str = "race",
    include_car_data: bool = False,
):
    round_num = round or next((race.round for race in client.get_races() if race.status != "COMPLETED"), None) or 1
    payload = await _car_model_for_round(round_num, session=session, include_car_data=include_car_data)
    if not payload.get("ok"):
        return payload
    constructors = payload.get("constructors") or {}
    constructor = next((item for item in client.get_constructors() if item.id == constructor_id), None)
    key = (constructor.name if constructor else constructor_id or "").lower()
    model = constructors.get(constructor_id) or constructors.get(key)
    if not model:
        return {
            "ok": False,
            "reason": "Constructor car model not found",
            "code": "constructor_car_model_not_found",
            "constructor_id": constructor_id,
            "round": round_num,
            "session": session,
        }
    return {
        "ok": True,
        "constructor": (constructor.model_dump(mode="json") if constructor else {"id": constructor_id}),
        "round": round_num,
        "session": payload.get("session"),
        "race": payload.get("race"),
        "car_model": model,
        "field_car_model": payload.get("car_model"),
        "confidence": model.get("confidence"),
        "missing_data": model.get("missing_data") or [],
    }


@router.get("/constructors/{constructor_id}/car-performance-agent")
async def get_constructor_car_performance_agent(
    constructor_id: str,
    round: int | None = None,
    session: str = "race",
    include_car_data: bool = False,
):
    round_num = round or next((race.round for race in client.get_races() if race.status != "COMPLETED"), None) or 1
    return await _car_performance_agent_for_round(
        round_num,
        session=session,
        constructor_id=constructor_id,
        include_car_data=include_car_data,
    )


@router.get("/calendar")
async def get_calendar():
    races = client.get_races()
    if predictor:
        races = predictor.predict_races(races)
    return {"ok": True, "races": [r.model_dump(mode="json") for r in races]}


@router.get("/models")
async def get_f1_models():
    return {"ok": True, "models": F1ModelRegistry.list_models(), "default_model_id": PRODUCTION_MODEL_ID}


@router.get("/models/compare")
async def compare_f1_models(season: int | None = None, include_races: bool = False, allow_partial: bool = False, stage: str = "pre_weekend"):
    return await _backtester().compare_season(season=season, include_races=include_races, allow_partial=allow_partial, stage=stage)


@router.get("/models/compare/summary")
async def compare_f1_models_summary(
    start_season: int = 2023,
    end_season: int | None = None,
    include_races: bool = False,
    allow_partial: bool = False,
    stage: str = "pre_weekend",
):
    end = end_season if end_season is not None else max(start_season, client.season - 1)
    return await _backtester().compare_summary(
        start_season=start_season,
        end_season=end,
        include_races=include_races,
        allow_partial=allow_partial,
        stage=stage,
    )


@router.get("/health")
async def get_f1_health():
    features = (predictor._features if predictor else {}) or {}
    sentiment = _sentiment_snapshot()
    weather_by_round = features.get("weather_by_round") or {}
    weather_by_round_session = features.get("weather_by_round_session") or {}
    storage_health = await storage.health() if storage else {
        "redis": {"available": False, "last_error": "storage_unavailable"},
        "clickhouse": {"available": False, "last_error": "storage_unavailable"},
    }
    return {
        "ok": True,
        "season": client.season,
        "drivers": len(client.get_drivers()),
        "constructors": len(client.get_constructors()),
        "races": len(client.get_races()),
        "model": {
            "default_model_id": PRODUCTION_MODEL_ID,
            "available_models": [model["model_id"] for model in F1ModelRegistry.list_models()],
            "active_version": predictor._version if predictor else None,
            "bridge": {
                "publisher": "common.ml.bridge.outcome_publisher.OutcomePublisher",
                "domain": "f1",
                "channel_pattern": "f1:prob:{entity_id}:{entity_code}:{market}",
                "snapshot_pattern": "f1:snapshot:{entity_id}",
                "last_publish": _last_bridge_publish_status,
            },
        },
        "sources": {
            "jolpica": {
                "available": bool(client.get_drivers() and client.get_races()),
                "prediction_features_cached": bool(features.get("drivers")),
                "completed_races": features.get("completed_races", 0),
                "total_races": features.get("total_races", 0),
            },
            "openf1": {
                "available": openf1 is not None,
                "session_cache_entries": len(getattr(openf1, "_session_cache", {}) or {}) if openf1 else 0,
                "trace_cache_entries": len(getattr(openf1, "_trace_cache", {}) or {}) if openf1 else 0,
            },
            "weather": {
                "covered_races": len(weather_by_round),
                "covered_sessions": sum(len(items) for items in weather_by_round_session.values()),
                "fallback_races": sum(1 for item in weather_by_round.values() if (item or {}).get("missing_data")),
                "fallback_sessions": sum(
                    1
                    for session_items in weather_by_round_session.values()
                    for item in (session_items or {}).values()
                    if (item or {}).get("missing_data")
                ),
            },
            "sentiment": {
                "available": bool((sentiment.get("drivers") or {}) or (sentiment.get("teams") or {})),
                "driver_entities": len(sentiment.get("drivers") or {}),
                "team_entities": len(sentiment.get("teams") or {}),
                "source_items": sentiment.get("source_items") or sentiment.get("published_items") or 0,
            },
            "live": live_engine.health() if live_engine else {
                "cache_entries": 0,
                "latest_stale": True,
                "latest_mode": "unavailable",
                "latest_reason": "live_engine_unavailable",
            },
            "data_quality": {
                "available": True,
                "default_policy": "quality-governed bounded influence",
                "guards": [
                    "source_coverage",
                    "freshness",
                    "source_disagreement",
                    "bias_warnings",
                    "leakage_guard",
                    "influence_caps",
                ],
            },
            "storage": storage_health,
        },
        "fallback_heavy": [
            key for key, enabled in {
                "prediction_features": not bool(features.get("drivers")),
                "weather": not bool(weather_by_round),
                "sentiment": not bool((sentiment.get("drivers") or {}) or (sentiment.get("teams") or {})),
                "live": not bool(live_engine),
                "redis": not bool(storage_health.get("redis", {}).get("available")),
                "clickhouse": not bool(storage_health.get("clickhouse", {}).get("available")),
            }.items()
            if enabled
        ],
        "freshness_alerts": build_freshness_alerts(
            features=features, last_run=_last_pipeline_run, storage_health=storage_health,
            service={"predictor_loaded": predictor is not None, "openf1": openf1 is not None},
        ),
    }


@router.get("/pipeline/health")
async def get_f1_pipeline_health():
    """Snapshot of the F1 ML pipeline's *trust state* for the monitor page:
    model provenance (trained vs heuristic fallback), calibration status, live
    source mode + confidence ceilings, market coverage, bridge/Redis health, the
    last run's trust state, and recent ops events. Pairs with the f1:ops:* live
    feed pushed over SignalR."""
    import os
    from sports.f1.predictor.probability.empirical_calibration import CALIBRATION_ARTIFACT_ENV
    from sports.f1.predictor.probability.engine import _get_empirical_calibrator

    artifact_path = os.environ.get("F1_ML_ARTIFACT_PATH")
    artifact_configured = bool(artifact_path)
    artifact_present = bool(artifact_path and os.path.exists(artifact_path))

    calibrator = _get_empirical_calibrator()
    is_identity = calibrator.is_identity()
    calibration = {
        "empirical_active": not is_identity,
        "method": None if is_identity else calibrator.method,
        "source": None if is_identity else calibrator.source,
        "sample_count": getattr(calibrator, "sample_count", 0),
        "artifact_env": CALIBRATION_ARTIFACT_ENV,
        "artifact_configured": bool(os.environ.get(CALIBRATION_ARTIFACT_ENV)),
        "temperature_scaling": "active (heuristic stage table)",
    }

    storage_health = await storage.health() if storage else {
        "redis": {"available": False, "last_error": "storage_unavailable"},
        "clickhouse": {"available": False, "last_error": "storage_unavailable"},
    }
    redis_available = bool(storage_health.get("redis", {}).get("available"))

    features = (predictor._features if predictor else {}) or {}
    service_block = {
        "predictor_loaded": predictor is not None,
        "client_loaded": client is not None,
        "season": getattr(client, "season", None),
        "active_model": predictor._version if predictor else None,
        "live_engine": live_engine is not None,
        "openf1": openf1 is not None,
    }
    freshness_alerts = build_freshness_alerts(
        features=features, last_run=_last_pipeline_run,
        storage_health=storage_health, service=service_block,
    )
    _emit_freshness_alert_transitions(freshness_alerts)

    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "service": service_block,
        "freshness_alerts": freshness_alerts,
        "freshness_summary": summarize_alerts(freshness_alerts),
        "provenance": {
            "mode": "trained" if artifact_present else "heuristic_fallback",
            "trained_artifacts_configured": artifact_configured,
            "trained_artifacts_present": artifact_present,
            "artifact_env": "F1_ML_ARTIFACT_PATH",
            "fallback_reason": None if artifact_present else "trained_ml_artifacts_unavailable",
            "note": "Monte-Carlo simulation over heuristic inputs until a trained artifact is deployed.",
        },
        "calibration": calibration,
        "confidence_ceilings": {
            "unavailable": 0.05, "estimated": 0.25, "recording_pending": 0.25,
            "recent": 0.55, "historical": 0.65, "recorded": 0.75,
            "recorded_confident": 0.85, "live": 0.90,
        },
        "market_coverage": {
            "computed": ["winner", "podium", "top_k", "h2h", "fastest_lap", "safety_car", "dnf"],
            "published_live": ["winner", "podium", "top5", "fastest_lap", "dnf"],
            "published_route": ["winner", "podium", "top5", "points", "dnf"],
            "computed_not_published": ["h2h", "safety_car"],
        },
        "bridge": {
            "publisher": "common.ml.bridge.outcome_publisher.OutcomePublisher",
            "domain": "f1",
            "prob_channel_pattern": "f1:prob:{entity_id}:{entity_code}:{market}",
            "ops_channel_pattern": "f1:ops:{event_type}",
            "redis_available": redis_available,
            "last_publish": _last_bridge_publish_status,
        },
        "last_run": _last_pipeline_run,
        "recent_events": list(reversed(_recent_ops_events))[:25],
    }


@router.get("/storage/health")
async def get_f1_storage_health():
    if not storage:
        return {"ok": False, "reason": "storage_unavailable"}
    health = await storage.health()
    return {"ok": True, **health}


@router.get("/backtest")
async def get_f1_backtest(season: int | None = None, include_races: bool = False, allow_partial: bool = False, model_id: str | None = None, stage: str = "pre_weekend"):
    result = await _backtester().backtest_season(season=season, include_races=include_races, allow_partial=allow_partial, model_id=model_id, stage=stage)
    if storage and result.get("ok"):
        result["storage"] = await storage.persist_backtest_result(result)
    return result


@router.get("/backtest/deep")
async def get_f1_deep_backtest(
    start_season: int = 2023,
    end_season: int | None = None,
    stages: str = "pre_weekend,practice_available,post_qualifying",
    model_ids: str | None = None,
    include_races: bool = False,
    allow_partial: bool = False,
    include_ablations: bool = True,
    export_artifact: bool = False,
    cache_policy: str = "read_through",
):
    end = end_season if end_season is not None else max(start_season, client.season - 1)
    result = await _backtester().deep_backtest(
        start_season=start_season,
        end_season=end,
        stages=_csv_values(stages),
        model_ids=_csv_values(model_ids),
        include_races=include_races,
        allow_partial=allow_partial,
        include_ablations=include_ablations,
        export_artifact=export_artifact,
        cache_policy=cache_policy,
    )
    if storage and result.get("ok"):
        result["storage"] = await storage.persist_backtest_result(result)
    return result


async def _build_backtest_evidence_cache_round(
    round_num: int,
    season: int | None = None,
    sessions: str = "fp1,fp2,fp3,qualifying,race",
    write: bool = True,
    openf1_mode: str = "auto",
    fastf1_mode: str = "off",
    *,
    writer: WeekendEvidenceCacheWriter | None = None,
    raw_races: list[dict] | None = None,
    qualifying_by_round: dict[int, list[dict]] | None = None,
):
    target_season = int(season or client.season)
    historical_mode = target_season != int(client.season)
    writer = writer or WeekendEvidenceCacheWriter()
    openf1_mode = _openf1_mode(openf1_mode)
    fastf1_mode = _fastf1_mode(fastf1_mode)
    if historical_mode:
        raw_races = raw_races if raw_races is not None else await client.get_historical_race_results(target_season)
        if qualifying_by_round is None:
            qualifying_races = await client.get_historical_qualifying_results(target_season)
            qualifying_by_round = {
                int(item.get("round") or 0): item.get("QualifyingResults") or []
                for item in qualifying_races
                if str(item.get("round") or "").isdigit()
            }
        raw = next((item for item in raw_races or [] if int(item.get("round") or 0) == int(round_num)), None)
        if not raw:
            return {
                "ok": False,
                "reason": "Historical race not found",
                "code": "historical_race_not_found",
                "season": target_season,
                "round": round_num,
            }
        raw = {**raw, "QualifyingResults": raw.get("QualifyingResults") or qualifying_by_round.get(int(round_num), [])}
        race = race_from_historical_raw(raw)
        drivers = drivers_from_historical_raw(raw)
        profile = profile_from_historical_raw(raw)
    else:
        race = client.get_race_by_round(round_num)
        if not race:
            return {"ok": False, "reason": "Race not found", "code": "race_not_found", "round": round_num}
        drivers = client.get_drivers()
        profile = await _race_profile_for_round(round_num)
        if not profile.get("ok"):
            return profile
    result = await writer.build_round(
        season=target_season,
        race=race,
        drivers=drivers,
        profile=profile,
        openf1=None if openf1_mode == "profile_only" else openf1,
        fastf1_mode=fastf1_mode,
        sessions=_csv_values(sessions),
        write=write,
    )
    result["profile_context"] = profile.get("context") or {}
    result["openf1_available"] = bool(openf1) and openf1_mode != "profile_only"
    result["openf1_mode"] = openf1_mode
    result["fastf1_mode"] = fastf1_mode
    result["historical_mode"] = historical_mode
    result["note"] = (
        "Evidence cache populated. Deep backtests can merge this on refresh/read-through runs."
        if result.get("available")
        else "No real weekend evidence rows were found; cache file is diagnostic only."
    )
    return result


@router.post("/backtest/evidence-cache/batch")
async def build_f1_backtest_evidence_cache_batch(
    season: int | None = None,
    sessions: str = "fp1",
    write: bool = True,
    start_round: int | None = None,
    end_round: int | None = None,
    max_rounds: int = 5,
    skip_existing: bool = True,
    refresh_missing_practice: bool = False,
    refresh_missing_practice_detail: bool = False,
    refresh_missing_practice_distribution: bool = False,
    refresh_missing_race_inputs: bool = False,
    refresh_missing_race_stints: bool = False,
    delay_seconds: float = 1.5,
    openf1_mode: str = "auto",
    fastf1_mode: str = "off",
):
    target_season = int(season or client.season)
    writer = WeekendEvidenceCacheWriter()
    historical_mode = target_season != int(client.season)
    openf1_mode = _openf1_mode(openf1_mode)
    fastf1_mode = _fastf1_mode(fastf1_mode)
    max_rounds = max(1, min(int(max_rounds or 1), 25))
    start = int(start_round or 1)
    end = int(end_round or 99)
    raw_races = None
    qualifying_by_round = None
    candidates: list[int] = []

    if historical_mode:
        raw_races = await client.get_historical_race_results(target_season)
        qualifying_races = await client.get_historical_qualifying_results(target_season)
        qualifying_by_round = {
            int(item.get("round") or 0): item.get("QualifyingResults") or []
            for item in qualifying_races
            if str(item.get("round") or "").isdigit()
        }
        candidates = [
            int(item.get("round") or 0)
            for item in raw_races
            if str(item.get("round") or "").isdigit()
            and item.get("Results")
            and start <= int(item.get("round") or 0) <= end
        ]
    else:
        candidates = [
            int(race.round)
            for race in client.get_races()
            if start <= int(race.round) <= end and str(race.status or "").upper() == "COMPLETED"
        ]

    candidates = sorted(dict.fromkeys(candidates))
    results = []
    processed = 0
    skipped_existing = 0
    for round_value in candidates:
        if processed >= max_rounds:
            break
        path = writer.path_for(target_season, round_value)
        should_skip_existing = skip_existing and write and path.exists()
        if should_skip_existing and refresh_missing_practice and _evidence_practice_rows(path) <= 0:
            should_skip_existing = False
        if should_skip_existing and refresh_missing_practice_detail and _evidence_practice_detail_rows(path) <= 0:
            should_skip_existing = False
        if should_skip_existing and refresh_missing_practice_distribution and _evidence_practice_distribution_rows(path) <= 0:
            should_skip_existing = False
        if should_skip_existing and refresh_missing_race_inputs and _evidence_race_input_rows(path) <= 0:
            should_skip_existing = False
        if should_skip_existing and refresh_missing_race_stints and _evidence_race_stint_rows(path) <= 0:
            should_skip_existing = False
        if should_skip_existing:
            skipped_existing += 1
            results.append({
                "ok": True,
                "season": target_season,
                "round": round_value,
                "path": str(path),
                "skipped": True,
                "reason": "evidence_cache_exists",
            })
            continue
        result = await _build_backtest_evidence_cache_round(
            round_value,
            season=target_season,
            sessions=sessions,
            write=write,
            openf1_mode=openf1_mode,
            fastf1_mode=fastf1_mode,
            writer=writer,
            raw_races=raw_races,
            qualifying_by_round=qualifying_by_round,
        )
        results.append(result)
        processed += 1
        if delay_seconds > 0 and processed < max_rounds and processed < len(candidates):
            await sleep(min(float(delay_seconds), 10.0))

    coverage_totals = {
        "practice_rows": sum(int((item.get("coverage") or {}).get("practice_rows") or 0) for item in results),
        "qualifying_rows": sum(int((item.get("coverage") or {}).get("qualifying_rows") or 0) for item in results),
        "grid_rows": sum(int((item.get("coverage") or {}).get("grid_rows") or 0) for item in results),
        "race_input_drivers": sum(int((item.get("coverage") or {}).get("race_input_drivers") or 0) for item in results),
        "openf1_sessions_ok": sum(int((item.get("coverage") or {}).get("openf1_sessions_ok") or 0) for item in results),
        "practice_distribution_rows": sum(_evidence_practice_distribution_rows(Path(item.get("path") or "")) for item in results if item.get("path")),
        "race_input_rows": sum(_evidence_race_input_rows(Path(item.get("path") or "")) for item in results if item.get("path")),
        "race_stint_rows": sum(_evidence_race_stint_rows(Path(item.get("path") or "")) for item in results if item.get("path")),
    }
    limitations = [
        item for item in [
            "batch_size_limited_to_reduce_provider_pressure" if len(candidates) > max_rounds else None,
            "existing_files_skipped" if skipped_existing else None,
            "openf1_practice_lap_rows_empty" if any(
                (warning.get("code") == "openf1_practice_lap_rows_empty")
                for result in results
                for warning in (result.get("warnings") or [])
            ) else None,
            "practice_distribution_missing" if any(
                not result.get("skipped") and _evidence_practice_distribution_rows(Path(result.get("path") or "")) <= 0
                for result in results
                if result.get("path")
            ) else None,
            "race_inputs_missing" if any(
                not result.get("skipped") and _evidence_race_input_rows(Path(result.get("path") or "")) <= 0
                for result in results
                if result.get("path")
            ) else None,
            "race_stints_missing" if any(
                not result.get("skipped") and _evidence_race_stint_rows(Path(result.get("path") or "")) <= 0
                for result in results
                if result.get("path")
            ) else None,
        ]
        if item
    ]
    return {
        "ok": True,
        "season": target_season,
        "historical_mode": historical_mode,
        "sessions": _csv_values(sessions),
        "openf1_mode": openf1_mode,
        "fastf1_mode": fastf1_mode,
        "write": write,
        "skip_existing": skip_existing,
        "refresh_missing_practice": refresh_missing_practice,
        "refresh_missing_practice_detail": refresh_missing_practice_detail,
        "refresh_missing_practice_distribution": refresh_missing_practice_distribution,
        "refresh_missing_race_inputs": refresh_missing_race_inputs,
        "refresh_missing_race_stints": refresh_missing_race_stints,
        "candidate_rounds": candidates,
        "processed": processed,
        "skipped_existing": skipped_existing,
        "result_count": len(results),
        "coverage": coverage_totals,
        "results": results,
        "limitations": limitations,
    }


def _evidence_practice_rows(path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        coverage = payload.get("coverage") or {}
        if coverage.get("practice_rows") is not None:
            return int(coverage.get("practice_rows") or 0)
        rows = payload.get("PracticeResults") or payload.get("practice_results") or []
        return len(rows) if isinstance(rows, list) else 0
    except Exception:
        return 0


def _evidence_practice_detail_rows(path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("PracticeResults") or payload.get("practice_results") or []
        if not isinstance(rows, list):
            return 0
        count = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            has_sector = any(row.get(f"representative_sector_{idx}") or row.get(f"best_sector_{idx}") for idx in (1, 2, 3))
            has_stability = row.get("pace_stability") is not None or row.get("lap_time_stddev") is not None
            if has_sector or has_stability:
                count += 1
        return count
    except Exception:
        return 0


def _evidence_practice_distribution_rows(path) -> int:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload.get("PracticeResults") or payload.get("practice_results") or []
        if not isinstance(rows, list):
            return 0
        count = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            distribution = row.get("lap_distribution")
            if isinstance(distribution, dict) and int(distribution.get("sample_size") or 0) > 0:
                count += 1
        return count
    except Exception:
        return 0


def _evidence_race_input_rows(path) -> int:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        drivers = ((payload.get("RaceInputs") or {}).get("drivers") or {})
        if isinstance(drivers, dict):
            return len([row for row in drivers.values() if isinstance(row, dict)])
        if isinstance(drivers, list):
            return len([row for row in drivers if isinstance(row, dict)])
        return 0
    except Exception:
        return 0


def _evidence_race_stint_rows(path) -> int:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        drivers = ((payload.get("RaceInputs") or {}).get("drivers") or {})
        rows = drivers.values() if isinstance(drivers, dict) else drivers if isinstance(drivers, list) else []
        count = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            distribution = row.get("stint_lap_distribution")
            if (
                row.get("compound_sequence")
                or row.get("avg_stint_laps") is not None
                or row.get("max_stint_laps") is not None
                or row.get("final_stint_laps") is not None
                or row.get("stints") is not None
                or (isinstance(distribution, dict) and int(distribution.get("sample_size") or 0) > 0)
            ):
                count += 1
        return count
    except Exception:
        return 0


@router.post("/backtest/evidence-cache/{round_num}")
async def build_f1_backtest_evidence_cache(
    round_num: int,
    season: int | None = None,
    sessions: str = "fp1,fp2,fp3,qualifying,race",
    write: bool = True,
    openf1_mode: str = "auto",
    fastf1_mode: str = "off",
):
    return await _build_backtest_evidence_cache_round(
        round_num,
        season=season,
        sessions=sessions,
        write=write,
        openf1_mode=openf1_mode,
        fastf1_mode=fastf1_mode,
    )


@router.get("/live/{round_num}")
async def get_f1_live_state(round_num: int, session: str = "race"):
    return await _live_state_for_round(round_num, session=session, force=False)


@router.post("/live/{round_num}/refresh")
async def refresh_f1_live_state(round_num: int, session: str = "race"):
    return await _live_state_for_round(round_num, session=session, force=True)


@router.get("/live/{round_num}/timeline")
async def get_f1_live_timeline(round_num: int, session: str = "race"):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    sprint_error = await _sprint_unavailable(round_num, session)
    if sprint_error:
        return {**sprint_error, "race": race.model_dump(mode="json")}
    if not live_engine:
        return {"ok": False, "reason": "Live session engine unavailable", "events": []}
    return _compact_live_timeline_payload(live_engine.get_timeline(race, session=session))


@router.get("/live/{round_num}/sources")
async def get_f1_live_sources(round_num: int, session: str = "race"):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    sprint_error = await _sprint_unavailable(round_num, session)
    if sprint_error:
        return {**sprint_error, "race": race.model_dump(mode="json")}
    if not live_engine:
        return {
            "ok": False,
            "round": round_num,
            "session": (session or "race").lower(),
            "source_mode": "unavailable",
            "source_chain": [],
            "confidence": 0.0,
            "fallback_reason": "live_engine_unavailable",
        }
    state = await _live_state_for_round(round_num, session=session, force=False)
    return live_engine.get_sources(race, session=session) or {
        "ok": False,
        "round": round_num,
        "session": (session or "race").lower(),
        "source_mode": state.get("source_mode") or state.get("mode") or "unavailable",
    }


@router.get("/live/{round_num}/diagnostics")
async def get_f1_live_diagnostics(round_num: int, session: str = "race"):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    sprint_error = await _sprint_unavailable(round_num, session)
    if sprint_error:
        return {**sprint_error, "race": race.model_dump(mode="json")}

    started_at = monotonic()
    state = await _live_state_for_round(round_num, session=session, force=False)
    truth = await _truth_snapshot_for_round(round_num, session=session, live=True, live_state=state)
    dynamics = build_live_dynamics(truth, session=session)
    sources = live_engine.get_sources(race, session=session) if live_engine else {
        "ok": False,
        "source_mode": "unavailable",
        "fallback_reason": "live_engine_unavailable",
        "source_chain": [],
    }
    timeline = live_engine.get_timeline(race, session=session) if live_engine else {"events": []}
    recorder_status = live_recorder.status(round_num=round_num, session=session) if live_recorder else {"ok": False, "reason": "live_recorder_unavailable"}
    storage_health = await storage.health() if storage else {
        "redis": {"available": False, "last_error": "storage_unavailable"},
        "clickhouse": {"available": False, "last_error": "storage_unavailable"},
    }
    redis_status = storage_health.get("redis") or {}
    probability_events = [
        event for event in (timeline.get("events") or [])
        if event.get("type") == "probability_snapshot"
    ]
    latest_probability = probability_events[-1] if probability_events else None
    report = _confidence_report(truth, live_state=state, diagnostics={"openf1": {"raw_counts": ((state.get("signals") or {}).get("raw_counts") or {})}})
    response = {
        "ok": True,
        "round": round_num,
        "session": (session or "race").lower(),
        "race": race.model_dump(mode="json"),
        "latency_ms": round((monotonic() - started_at) * 1000.0, 2),
        "source_mode": truth.get("source_mode") or state.get("source_mode") or state.get("mode"),
        "confidence": truth.get("confidence") or state.get("confidence"),
        "data_age_seconds": truth.get("data_age_seconds") or state.get("data_age_seconds"),
        "fallback_reason": truth.get("fallback_reason") or state.get("fallback_reason"),
        "missing_live_groups": truth.get("missing_groups") or [],
        "source_chain_status": sources.get("source_chain") or [],
        "openf1": {
            "available": openf1 is not None,
            "session_key": state.get("session_key"),
            "meeting_key": state.get("meeting_key"),
            "raw_counts": ((state.get("signals") or {}).get("raw_counts") or {}),
            "row_counts": report.get("openf1_row_counts") or {},
            "latest_reason": state.get("reason"),
        },
        "fastf1_recorder": recorder_status,
        "latest_refresh": live_engine.health() if live_engine else {"latest_reason": "live_engine_unavailable"},
        "redis_publish": {
            "available": bool(redis_status.get("available")),
            "last_error": redis_status.get("last_error"),
            "latest_probability_published": bool(latest_probability),
            "latest_probability_at": (latest_probability or {}).get("at"),
        },
        "clickhouse": storage_health.get("clickhouse") or {},
        "live_dynamics": {
            "ok": dynamics.get("ok"),
            "confidence": dynamics.get("confidence"),
            "driver_count": dynamics.get("driver_count"),
            "summary": dynamics.get("summary") or [],
            "reason": dynamics.get("reason"),
        },
        "live_test_checklist": {
            "source_mode": report.get("source_mode"),
            "usable_driver_count": (report.get("coverage_counts") or {}).get("usable_driver_count"),
            "real_positions": (report.get("coverage_counts") or {}).get("real_positions"),
            "real_gaps": (report.get("coverage_counts") or {}).get("real_gaps"),
            "real_intervals": (report.get("coverage_counts") or {}).get("real_intervals"),
            "tyre_count": (report.get("coverage_counts") or {}).get("tyre_count"),
            "race_control_available": (report.get("coverage_counts") or {}).get("race_control_available"),
            "confidence_blocker": ((report.get("confidence_blockers") or [{}])[0] or {}).get("message"),
            "fastf1_available": recorder_status.get("fastf1_available"),
            "signalrcore_available": recorder_status.get("signalrcore_available"),
            "recorder_running": recorder_status.get("running"),
            "file_growth": recorder_status.get("recording_file_growth"),
            "parsed_driver_count": recorder_status.get("parsed_driver_count"),
            "recording_source_mode": recorder_status.get("recording_source_mode"),
        },
        "probability_timeline": {
            "event_count": len(probability_events),
            "latest": latest_probability,
        },
    }
    attach_confidence_report(response, report)
    return response


@router.get("/live/{round_num}/confidence")
async def get_f1_live_confidence(round_num: int, session: str = "race"):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    sprint_error = await _sprint_unavailable(round_num, session)
    if sprint_error:
        return {**sprint_error, "race": race.model_dump(mode="json")}
    state = await _live_state_for_round(round_num, session=session, force=False)
    truth = await _truth_snapshot_for_round(round_num, session=session, live=True, live_state=state)
    report = _confidence_report(truth, live_state=state)
    return {
        "ok": True,
        "round": round_num,
        "session": (session or "race").lower(),
        "race": race.model_dump(mode="json"),
        **report,
    }


@router.get("/live/{round_num}/recorder")
async def get_f1_live_recorder(round_num: int, session: str = "race"):
    if not live_recorder:
        return {"ok": False, "reason": "live_recorder_unavailable"}
    return live_recorder.status(round_num=round_num, session=session)


@router.post("/live/{round_num}/recorder/start")
async def start_f1_live_recorder(round_num: int, session: str = "race"):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    sprint_error = await _sprint_unavailable(round_num, session)
    if sprint_error:
        return {**sprint_error, "race": race.model_dump(mode="json")}
    if not live_recorder:
        return {"ok": False, "reason": "live_recorder_unavailable"}
    return live_recorder.start(round_num, session=session)


@router.post("/live/{round_num}/recorder/stop")
async def stop_f1_live_recorder(round_num: int, session: str = "race"):
    if not live_recorder:
        return {"ok": False, "reason": "live_recorder_unavailable"}
    return live_recorder.stop()


@router.get("/live/{round_num}/probabilities")
async def get_f1_live_probabilities(round_num: int, session: str = "race", model_id: str | None = None, compact: bool = False):
    selected_model_id, model_error = _resolve_model_id(model_id)
    if model_error:
        return model_error
    session_key = (session or "race").lower()
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    profile = await _race_profile_for_round(round_num)
    if not profile.get("ok"):
        return profile
    context = profile.get("context") or {}
    if (session or "").lower() == "sprint" and not context.get("has_sprint"):
        return {
            "ok": False,
            "reason": "Sprint session is not scheduled for this Grand Prix",
            "code": "sprint_unavailable",
            "race": race.model_dump(mode="json"),
            "context": context,
        }
    cache_key = (round_num, session_key, selected_model_id, bool(compact))
    cached = _get_ttl_cache(_live_probability_response_cache, cache_key, _LIVE_RESPONSE_CACHE_TTL_SECONDS, "live_probabilities")
    if cached:
        return cached
    features = await client.get_prediction_features()
    live_state = await _live_state_for_round(round_num, session=session, force=False)
    if live_state.get("ok") or live_state.get("mode"):
        features = {**features, "live_state": live_state}
    truth = await _truth_snapshot_for_round(
        round_num,
        session=session,
        live=True,
        profile=profile,
        features=features,
        live_state=live_state,
    )
    weekend_evidence = truth.get("weekend_evidence") or {}
    features = {**features, "race_truth": truth, "weekend_evidence": weekend_evidence, "car_model": truth.get("car_model") or {}, "market_signals": _market_consensus_by_round.get(round_num) or {}}
    sentiment, sentiment_impact = await _sentiment_with_race_impact(round_num, session=session)
    prediction_payload = {}
    if race.status == "SCHEDULED":
        race.prediction = _predict_race_with_model(
            selected_model_id=selected_model_id,
            race=race,
            drivers=client.get_drivers(),
            constructors=client.get_constructors(),
            features=features,
            sentiment=sentiment,
        )
        prediction_payload = race.prediction.model_dump(mode="json") if race.prediction else {}
    simulation = build_session_simulation(
        race=race,
        drivers=client.get_drivers(),
        constructors=client.get_constructors(),
        prediction=prediction_payload,
        features=features,
        qualifying=profile.get("qualifying") or [],
        sprint=profile.get("sprint") or [],
        results=profile.get("results") or [],
        session=session,
        live=True,
    )
    model_fields = _prediction_model_fields(prediction_payload, selected_model_id)
    prediction_model_version = model_fields.pop("model_version", None)
    simulation["model_id"] = model_fields.pop("model_id", selected_model_id)
    simulation["selected_model_id"] = selected_model_id
    simulation["prediction_model_version"] = prediction_model_version
    simulation.update(model_fields)
    probability_rows = _apply_sentiment_impact_to_rows(simulation.get("simulations") or [], sentiment_impact)
    simulation["simulations"] = probability_rows
    simulation["truth"] = truth
    simulation["context"] = context
    simulation = _apply_probability_audit(simulation, profile, truth, live=True)
    if selected_model_id == "ml_simulator_v1":
        ml_payload = _ml_live_runner_payload(
            round_num=round_num,
            session=session,
            race=race,
            drivers=client.get_drivers(),
            constructors=client.get_constructors(),
            features=features,
            truth=truth,
            weekend_evidence=weekend_evidence,
            live_state=live_state,
            sentiment=sentiment,
        )
        simulation = _apply_ml_live_runner_payload(simulation, ml_payload, client.get_drivers())
    simulation = _apply_live_probability_fields(simulation, truth)
    probability_rows = simulation.get("simulations") or []
    confidence_report = _confidence_report(truth, live_state=live_state, probabilities=probability_rows)
    payload = {
        "ok": True,
        "race_id": _race_entity_id(client.season, race),
        "round": round_num,
        "session": (session or "race").lower(),
        "model_id": simulation.get("model_id") or selected_model_id,
        "selected_model_id": selected_model_id,
        "prediction_model_version": simulation.get("prediction_model_version"),
        "ml_input_source": simulation.get("ml_input_source"),
        "ml_provider_sources": simulation.get("ml_provider_sources") or [],
        "ml_fallback_reason": simulation.get("ml_fallback_reason"),
        "ml_confidence": simulation.get("ml_confidence"),
        "trained_artifacts_used": simulation.get("trained_artifacts_used"),
        "ml_artifact_id": simulation.get("ml_artifact_id"),
        "ml_artifact_version": simulation.get("ml_artifact_version"),
        "ml_live_runner": simulation.get("ml_live_runner"),
        "ml_live_runner_used": simulation.get("ml_live_runner_used"),
        "live_state": live_state,
        "truth": truth,
        "weekend_evidence": weekend_evidence,
        "weekend_evidence_confidence": weekend_evidence.get("confidence"),
        "weekend_evidence_missing_groups": weekend_evidence.get("missing_groups") or [],
        "source_mode": simulation.get("source_mode") or truth.get("source_mode"),
        "confidence": simulation.get("confidence") if simulation.get("confidence") is not None else truth.get("confidence"),
        "confidence_ceiling": simulation.get("confidence_ceiling"),
        "confidence_reason": simulation.get("confidence_reason"),
        "data_age_seconds": simulation.get("data_age_seconds") or truth.get("data_age_seconds"),
        "fallback_reason": simulation.get("fallback_reason") or truth.get("fallback_reason"),
        "missing_groups": simulation.get("missing_groups") or truth.get("missing_groups") or [],
        "probability_explanations": _probability_explanations(probability_rows, truth),
        "probabilities": probability_rows,
        "monte_carlo": simulation.get("monte_carlo") or {},
        "context": context,
        "stage": simulation.get("stage"),
        "calibration_profile": simulation.get("calibration_profile"),
        "raw_probabilities": simulation.get("raw_probabilities") or {},
        "calibrated_probabilities": simulation.get("calibrated_probabilities") or {},
        "finish_distribution": simulation.get("finish_distribution") or {},
        "probability_audit": simulation.get("probability_audit") or {},
        "top_probability_movers": simulation.get("top_probability_movers") or [],
        "live_dynamics": simulation.get("live_dynamics") or {},
        "strategy_state": simulation.get("strategy_state") or {},
        "probability_timeline": simulation.get("probability_timeline") or {},
        "weather": simulation.get("weather") or ((truth.get("signals") or {}).get("weather") if truth else {}),
        "weather_source": simulation.get("weather_source") or (((truth.get("signals") or {}).get("weather") or {}).get("source") if truth else None),
        "weather_confidence": simulation.get("weather_confidence") or (((truth.get("signals") or {}).get("weather") or {}).get("confidence") if truth else None),
        "weather_missing": bool((simulation.get("weather") or {}).get("missing_data")),
        "weather_model_impact": simulation.get("weather_model_impact") or {},
        "weather_explanations": simulation.get("weather_explanations") or [],
        **_sentiment_impact_fields(sentiment_impact),
    }
    attach_confidence_report(payload, confidence_report)
    attach_data_quality(payload, simulation.get("data_quality") or truth.get("data_quality") or {})
    payload["probability_explanations"] = [
        *((simulation.get("probability_audit") or {}).get("explanations") or []),
        *_probability_explanations(probability_rows, truth),
    ]
    if live_engine:
        payload["timeline_record"] = live_engine.record_probability_snapshot(race, session, payload)
    if storage:
        payload["storage"] = await storage.persist_probability_snapshot(
            client.season,
            race,
            session,
            payload,
            model_id=simulation.get("model_id") or simulation.get("model_version") or "production_v1",
        )
    # Live path: publish per-driver market probabilities to the Redis bridge so the
    # dashboard's F1ProbabilityRedisSubscriber -> SignalR (group f1:race:{entity_id})
    # pushes them to connected browsers. Publishing only happens on cache-miss (every
    # _LIVE_RESPONSE_CACHE_TTL_SECONDS), which is the right cadence for poll-driven fanout.
    bridge_records = _simulation_rows_to_outcome_probabilities(
        season=client.season,
        race=race,
        rows=probability_rows,
        model_version=str(
            simulation.get("prediction_model_version")
            or simulation.get("model_id")
            or PRODUCTION_MODEL_ID
        ),
        generated_at=simulation.get("generated_at"),
    )
    payload["bridge_record_count"] = len(bridge_records)
    payload["bridge_publish"] = _publish_outcome_probabilities(bridge_records)
    _record_pipeline_run(payload, simulation, truth, bridge_records)
    response_payload = _compact_live_probability_payload(payload) if compact else payload
    _set_ttl_cache(_live_probability_response_cache, cache_key, response_payload, _LIVE_RESPONSE_CACHE_TTL_SECONDS, "live_probabilities")
    return response_payload


@router.get("/backtest/summary")
async def get_f1_backtest_summary(
    start_season: int = 2023,
    end_season: int | None = None,
    include_races: bool = False,
    allow_partial: bool = False,
    model_id: str | None = None,
    stage: str = "pre_weekend",
):
    end = end_season if end_season is not None else max(start_season, client.season - 1)
    result = await _backtester().backtest_summary(
        start_season=start_season,
        end_season=end,
        include_races=include_races,
        allow_partial=allow_partial,
        model_id=model_id,
        stage=stage,
    )
    if storage and result.get("ok"):
        result["storage"] = await storage.persist_backtest_result(result)
    return result


@router.get("/backtest/races/{season}/{round_num}")
async def get_f1_backtest_race(season: int, round_num: int, allow_partial: bool = True, model_id: str | None = None, stage: str = "pre_weekend"):
    result = await _backtester().backtest_race(season=season, round_num=round_num, allow_partial=allow_partial, model_id=model_id, stage=stage)
    if storage and result.get("ok"):
        result["storage"] = await storage.persist_backtest_result(result)
    return result


@router.get("/sentiment")
async def get_sentiment(limit: int = 12):
    try:
        r = redis.from_url("redis://localhost:6379/0", decode_responses=True)
        composite_raw = r.get("sentiment:composite:latest:f1")
        composite = json.loads(composite_raw) if composite_raw else None
        keys = r.zrevrange("feeditems:category:f1", 0, max(limit - 1, 0))
        items = []
        for key in keys:
            raw = r.get(key)
            if not raw:
                continue
            item = json.loads(raw)
            items.append({
                "title": item.get("Title"),
                "content": item.get("Content"),
                "url": item.get("Url"),
                "source": item.get("Source"),
                "feed": item.get("Feed"),
                "published_utc": item.get("PublishedUtc"),
                "score": item.get("SentimentScore"),
                "label": item.get("SentimentLabel"),
                "confidence": item.get("SentimentConfidence"),
                "source_kind": item.get("SourceKind") or item.get("source_kind"),
                "bias_flags": item.get("BiasFlags") or item.get("bias_flags") or [],
                "prediction_impact_score": item.get("PredictionImpactScore") or item.get("prediction_impact_score"),
                "prediction_impact_cap": item.get("PredictionImpactCap") or item.get("prediction_impact_cap"),
                "confidence_ceiling": item.get("SentimentConfidenceCeiling") or item.get("confidence_ceiling"),
                "analyzer": item.get("SentimentAnalyzerUsed") or item.get("SentimentAnalyzer"),
            })
        entities = read_f1_sentiment(client.get_drivers(), client.get_constructors(), limit=60)
        enriched_items = [
            {
                "title": item.get("Title"),
                "content": item.get("Content"),
                "url": item.get("Url"),
                "source": item.get("Source"),
                "feed": item.get("Feed"),
                "published_utc": item.get("PublishedUtc"),
                "score": item.get("SentimentScore"),
                "label": item.get("SentimentLabel"),
                "confidence": item.get("SentimentConfidence"),
                "source_kind": item.get("SourceKind") or item.get("source_kind"),
                "bias_flags": item.get("BiasFlags") or item.get("bias_flags") or [],
                "prediction_impact_score": item.get("PredictionImpactScore") or item.get("prediction_impact_score"),
                "prediction_impact_cap": item.get("PredictionImpactCap") or item.get("prediction_impact_cap"),
                "confidence_ceiling": item.get("SentimentConfidenceCeiling") or item.get("confidence_ceiling"),
                "analyzer": item.get("SentimentAnalyzerUsed") or item.get("SentimentAnalyzer"),
            }
            for item in (entities.get("items") or [])[:limit]
        ]
        result = {
            "ok": True,
            "composite": composite or entities.get("composite"),
            "items": enriched_items or items,
            "drivers": entities.get("drivers", {}),
            "teams": entities.get("teams", {}),
            "source_items": entities.get("source_items") or entities.get("published_items") or len(enriched_items or items),
            "configured_sources": entities.get("configured_sources") or len(DEFAULT_RSS_FEEDS),
        }
        if storage:
            result["storage"] = await storage.persist_sentiment(client.season, entities.get("items") or [], entities)
        return result
    except Exception as ex:
        try:
            features = await client.get_prediction_features()
            entities = await refresh_f1_sentiment(client.get_drivers(), client.get_constructors(), client.season)
            if predictor:
                predictor.load_drivers(client.get_drivers(), client.get_constructors(), features, entities)
            result = {
                "ok": True,
                "reason": f"Redis unavailable, refreshed live news directly: {ex}",
                "composite": entities.get("composite"),
                "items": [
                    {
                        "title": item.get("Title"),
                        "content": item.get("Content"),
                        "url": item.get("Url"),
                        "source": item.get("Source"),
                        "feed": item.get("Feed"),
                        "published_utc": item.get("PublishedUtc"),
                        "score": item.get("SentimentScore"),
                        "label": item.get("SentimentLabel"),
                        "confidence": item.get("SentimentConfidence"),
                        "source_kind": item.get("SourceKind") or item.get("source_kind"),
                        "bias_flags": item.get("BiasFlags") or item.get("bias_flags") or [],
                        "prediction_impact_score": item.get("PredictionImpactScore") or item.get("prediction_impact_score"),
                        "prediction_impact_cap": item.get("PredictionImpactCap") or item.get("prediction_impact_cap"),
                        "confidence_ceiling": item.get("SentimentConfidenceCeiling") or item.get("confidence_ceiling"),
                        "topics": item.get("Topics") or [],
                        "analyzer": item.get("SentimentAnalyzerUsed") or item.get("SentimentAnalyzer"),
                    }
                    for item in (entities.get("items") or [])[:limit]
                ],
                "drivers": entities.get("drivers", {}),
                "teams": entities.get("teams", {}),
                "source_items": entities.get("source_items") or entities.get("published_items") or 0,
                "configured_sources": entities.get("configured_sources") or 0,
            }
            if storage:
                result["storage"] = await storage.persist_sentiment(client.season, entities.get("items") or [], entities)
            return result
        except Exception as refresh_ex:
            return {"ok": False, "reason": f"{ex}; live refresh failed: {refresh_ex}", "composite": None, "items": []}


@router.get("/performance")
async def get_performance():
    if predictor:
        if not (predictor._features or {}).get("drivers"):
            features = await client.get_prediction_features()
            predictor.load_drivers(client.get_drivers(), client.get_constructors(), features, await _ensure_sentiment())
        return predictor.get_performance_intelligence()
    features = await client.get_prediction_features()
    fallback = F1Predictor()
    fallback.load_drivers(client.get_drivers(), client.get_constructors(), features, await _ensure_sentiment())
    return fallback.get_performance_intelligence()


@router.get("/races/{round_num}")
async def get_race(round_num: int):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found"}
    race.race_id = _race_entity_id(client.season, race)
    if race.status == "SCHEDULED" and predictor:
        race.prediction = predictor.predict_race(race)
    return {"ok": True, "race": race.model_dump(mode="json")}


@router.get("/races/{round_num}/profile")
async def get_race_profile(round_num: int):
    return await _race_profile_for_round(round_num)


@router.get("/races/{round_num}/sessions/{session}")
async def get_race_session_result(round_num: int, session: str):
    return await _weekend_session_result(round_num, session)


@router.get("/races/{round_num}/weekend-evidence")
async def get_race_weekend_evidence(round_num: int, session: str = "race", live: bool = False):
    return await _weekend_evidence_for_round(round_num, session=session, live=live)


@router.get("/races/{round_num}/truth")
async def get_race_truth(round_num: int, session: str = "race", live: bool = False):
    truth = await _truth_snapshot_for_round(round_num, session=session, live=live)
    race = client.get_race_by_round(round_num)
    if race and truth.get("ok"):
        return {"ok": True, "race": race.model_dump(mode="json"), "truth": truth}
    return truth


@router.get("/races/{round_num}/data-quality")
async def get_race_data_quality(round_num: int, session: str = "race", live: bool = False):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    profile = await _race_profile_for_round(round_num)
    if not profile.get("ok"):
        return profile
    truth = await _truth_snapshot_for_round(round_num, session=session, live=live, profile=profile)
    probabilities: list[dict] = []
    simulation = None
    try:
        simulation = await get_race_simulation(round_num, session=session, live=live)
        probabilities = simulation.get("simulations") or simulation.get("probabilities") or []
    except Exception:
        probabilities = []
    report = _data_quality_report(
        truth=truth,
        profile=profile,
        probabilities=probabilities,
        stage=(simulation or {}).get("stage") or "auto",
        live=live,
        sentiment_impact=truth.get("sentiment_impact"),
    )
    return {
        "ok": True,
        "race": race.model_dump(mode="json"),
        "round": round_num,
        "session": (session or "race").lower(),
        "live": live,
        "truth": truth,
        "probability_count": len(probabilities),
        **report,
    }


@router.get("/races/{round_num}/weather")
async def get_race_weather(round_num: int, session: str = "race", live: bool = False):
    return await _weather_snapshot_for_round(round_num, session=session, live=live)


@router.get("/races/{round_num}/car-model")
async def get_race_car_model(
    round_num: int,
    session: str = "race",
    live: bool = False,
    include_car_data: bool = False,
):
    return await _car_model_for_round(round_num, session=session, live=live, include_car_data=include_car_data)


@router.get("/races/{round_num}/car-performance-agent")
async def get_race_car_performance_agent(
    round_num: int,
    session: str = "race",
    constructor_id: str | None = None,
    live: bool = False,
    include_car_data: bool = False,
):
    return await _car_performance_agent_for_round(
        round_num,
        session=session,
        live=live,
        constructor_id=constructor_id,
        include_car_data=include_car_data,
    )


@router.get("/races/{round_num}/sentiment-impact")
async def get_race_sentiment_impact(round_num: int, session: str = "race"):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    impact = await _race_sentiment_impact_for_round(round_num, session=session)
    return {"ok": True, "race": race.model_dump(mode="json"), **impact}


@router.get("/races/{round_num}/probability-audit")
async def get_race_probability_audit(
    round_num: int,
    session: str = "race",
    stage: str = "auto",
    live: bool = False,
    model_id: str | None = None,
):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    profile = await _race_profile_for_round(round_num)
    if not profile.get("ok"):
        return profile
    simulation = await get_race_simulation(round_num, session=session, live=live, model_id=model_id)
    if not simulation.get("ok", True):
        return simulation
    truth = simulation.get("truth") or await _truth_snapshot_for_round(round_num, session=session, live=live, profile=profile)
    audited = _apply_probability_audit(deepcopy(simulation), profile, truth, stage=stage, live=live)
    audit = audited.get("probability_audit") or {}
    return {
        "ok": True,
        "race": race.model_dump(mode="json"),
        "round": round_num,
        "session": (session or "race").lower(),
        "model_id": simulation.get("model_id"),
        "prediction_model_version": simulation.get("prediction_model_version"),
        "ml_input_source": simulation.get("ml_input_source"),
        "ml_provider_sources": simulation.get("ml_provider_sources") or [],
        "ml_fallback_reason": simulation.get("ml_fallback_reason"),
        "ml_confidence": simulation.get("ml_confidence"),
        "trained_artifacts_used": simulation.get("trained_artifacts_used"),
        "evidence_groups_used": simulation.get("evidence_groups_used") or [],
        "ml_artifact_id": simulation.get("ml_artifact_id"),
        "ml_artifact_version": simulation.get("ml_artifact_version"),
        "stage": audited.get("stage"),
        "truth": truth,
        "weekend_evidence": audited.get("weekend_evidence") or simulation.get("weekend_evidence") or truth.get("weekend_evidence") or {},
        "weekend_evidence_confidence": audited.get("weekend_evidence_confidence") or simulation.get("weekend_evidence_confidence") or truth.get("weekend_evidence_confidence"),
        "weekend_evidence_missing_groups": audited.get("weekend_evidence_missing_groups") or simulation.get("weekend_evidence_missing_groups") or truth.get("weekend_evidence_missing_groups") or [],
        "source_mode": truth.get("source_mode"),
        "confidence": audited.get("confidence"),
        "calibration_profile": audited.get("calibration_profile"),
        "probability_governance": audited.get("probability_governance") or audit.get("probability_governance") or {},
        "raw_probabilities": audited.get("raw_probabilities") or {},
        "governed_probabilities": audited.get("governed_probabilities") or {},
        "calibrated_probabilities": audited.get("calibrated_probabilities") or {},
        "finish_distribution": audited.get("finish_distribution") or {},
        "probabilities": audit.get("probabilities") or [],
        "top_probability_movers": audited.get("top_probability_movers") or [],
        "explanations": audit.get("explanations") or [],
        "probability_audit": audit,
        "data_quality": audited.get("data_quality") or audit.get("data_quality") or truth.get("data_quality") or {},
        "data_quality_score": audited.get("data_quality_score") or truth.get("data_quality_score"),
        "source_coverage": audited.get("source_coverage") or truth.get("source_coverage") or {},
        "source_disagreement": audited.get("source_disagreement") or truth.get("source_disagreement") or {},
        "bias_warnings": audited.get("bias_warnings") or truth.get("bias_warnings") or [],
        "influence_caps_applied": audited.get("influence_caps_applied") or truth.get("influence_caps_applied") or {},
        "leakage_guard_status": audited.get("leakage_guard_status") or truth.get("leakage_guard_status") or {},
    }


@router.get("/races/{round_num}/features")
async def get_race_features(round_num: int, session: str = "race", live: bool = False):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found"}
    if (session or "").lower() == "sprint":
        profile = await _race_profile_for_round(round_num)
        context = profile.get("context") or {}
        if not context.get("has_sprint"):
            return {
                "ok": False,
                "reason": "Sprint session is not scheduled for this Grand Prix",
                "code": "sprint_unavailable",
                "context": context,
            }
    features = await client.get_prediction_features()
    openf1_session = None
    openf1_session = None
    if openf1:
        openf1_session = await openf1.get_session_features(
            race=race,
            session=session,
            drivers=client.get_drivers(),
            live=live,
        )
        if openf1_session.get("ok"):
            features = {**features, "openf1_session": openf1_session}
    if live and live_engine:
        live_state = await live_engine.get_state(race, client.get_drivers(), session=session, force=False)
        features = {**features, "live_state": live_state}
    profile = await _race_profile_for_round(round_num)
    truth = await _truth_snapshot_for_round(
        round_num,
        session=session,
        live=live,
        profile=profile,
        features=features,
        openf1_session=features.get("openf1_session"),
        live_state=features.get("live_state"),
    )
    weekend_evidence = truth.get("weekend_evidence") or {}
    features = {**features, "race_truth": truth, "weekend_evidence": weekend_evidence, "car_model": truth.get("car_model") or {}, "market_signals": _market_consensus_by_round.get(round_num) or {}}
    sentiment, sentiment_impact = await _sentiment_with_race_impact(round_num, session=session)
    if predictor:
        predictor.load_drivers(client.get_drivers(), client.get_constructors(), features, sentiment)
        snapshot = predictor.build_feature_snapshot(race, session)
    else:
        fallback = F1Predictor()
        fallback.load_drivers(client.get_drivers(), client.get_constructors(), features, sentiment)
        snapshot = fallback.build_feature_snapshot(race, session)
    stage_name = detect_stage(profile, truth, live=live)
    calibration_profile = build_calibration_profile(stage_name, truth=truth).as_dict()
    snapshot_payload = snapshot.model_dump(mode="json")
    weather = snapshot_payload.get("weather") or {}
    car_model = snapshot_payload.get("car_model") or truth.get("car_model") or {}
    return {
        "ok": True,
        "race": race.model_dump(mode="json"),
        "features": snapshot_payload,
        "truth": truth,
        "weekend_evidence": weekend_evidence,
        "weekend_evidence_confidence": weekend_evidence.get("confidence"),
        "weekend_evidence_missing_groups": weekend_evidence.get("missing_groups") or [],
        "car_model": car_model,
        "car_model_confidence": car_model.get("confidence"),
        "car_model_missing_data": car_model.get("missing_data") or [],
        "stage": stage_name,
        "calibration_profile": calibration_profile,
        "source_mode": truth.get("source_mode"),
        "confidence": truth.get("confidence"),
        "data_age_seconds": truth.get("data_age_seconds"),
        "fallback_reason": truth.get("fallback_reason"),
        "missing_groups": truth.get("missing_groups") or [],
        "weather": weather,
        "weather_source": weather.get("source"),
        "weather_confidence": weather.get("confidence"),
        "weather_missing": bool(weather.get("missing_data")),
        "weather_model_impact": weather.get("weather_model_impact") or {},
        "weather_explanations": weather.get("weather_explanations") or [],
        "data_quality": truth.get("data_quality") or {},
        "data_quality_score": truth.get("data_quality_score"),
        "source_coverage": truth.get("source_coverage") or {},
        "source_disagreement": truth.get("source_disagreement") or {},
        "bias_warnings": truth.get("bias_warnings") or [],
        "influence_caps_applied": truth.get("influence_caps_applied") or {},
        "leakage_guard_status": truth.get("leakage_guard_status") or {},
        **_sentiment_impact_fields(sentiment_impact),
    }


@router.get("/races/{round_num}/simulation")
async def get_race_simulation(
    round_num: int,
    session: str = "race",
    live: bool = False,
    model_id: str | None = None,
    publish: bool = False,
):
    normalized_session = (session or "race").lower()
    selected_model_id, model_error = _resolve_model_id(model_id)
    if model_error:
        return model_error
    cache_key = (round_num, normalized_session, selected_model_id, bool(live))
    if not live and not publish:
        cached = _get_static_cache(_simulation_response_cache, cache_key)
        if cached:
            return cached

    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found"}

    features = await client.get_prediction_features()
    profile = await _race_profile_for_round(round_num)
    if not profile.get("ok"):
        return profile
    context = profile.get("context") or {}
    if normalized_session == "sprint" and not context.get("has_sprint"):
        return {
            "ok": False,
            "reason": "Sprint session is not scheduled for this Grand Prix",
            "code": "sprint_unavailable",
            "race": race.model_dump(mode="json"),
            "context": context,
        }

    openf1_session = None
    if openf1:
        openf1_session = await openf1.get_session_features(
            race=race,
            session=session,
            drivers=client.get_drivers(),
            live=live,
        )
        if openf1_session.get("ok"):
            features = {**features, "openf1_session": openf1_session}
    live_state = None
    if live and live_engine:
        live_state = await live_engine.get_state(race, client.get_drivers(), session=session, force=False)
        features = {**features, "live_state": live_state}
    truth = await _truth_snapshot_for_round(
        round_num,
        session=session,
        live=live,
        profile=profile,
        features=features,
        openf1_session=openf1_session,
        live_state=live_state,
    )
    weekend_evidence = truth.get("weekend_evidence") or {}
    features = {**features, "race_truth": truth, "weekend_evidence": weekend_evidence, "car_model": truth.get("car_model") or {}, "market_signals": _market_consensus_by_round.get(round_num) or {}}
    sentiment, sentiment_impact = await _sentiment_with_race_impact(round_num, session=session)
    prediction_payload = {}
    if race.status == "SCHEDULED":
        race.prediction = _predict_race_with_model(
            selected_model_id=selected_model_id,
            race=race,
            drivers=client.get_drivers(),
            constructors=client.get_constructors(),
            features=features,
            sentiment=sentiment,
        )
        prediction_payload = race.prediction.model_dump(mode="json") if race.prediction else {}

    simulation = build_session_simulation(
        race=race,
        drivers=client.get_drivers(),
        constructors=client.get_constructors(),
        prediction=prediction_payload,
        features=features,
        qualifying=profile.get("qualifying") or [],
        sprint=profile.get("sprint") or [],
        results=profile.get("results") or [],
        session=session,
        live=live,
    )
    model_fields = _prediction_model_fields(prediction_payload, selected_model_id)
    prediction_model_version = model_fields.pop("model_version", None)
    simulation["model_id"] = model_fields.pop("model_id", selected_model_id)
    simulation["selected_model_id"] = selected_model_id
    simulation["prediction_model_version"] = prediction_model_version
    simulation["session_simulation_version"] = simulation.get("model_version")
    simulation.update(model_fields)
    simulation["context"] = context
    simulation["truth"] = truth
    simulation["weekend_evidence"] = weekend_evidence
    simulation["weekend_evidence_confidence"] = weekend_evidence.get("confidence")
    simulation["weekend_evidence_missing_groups"] = weekend_evidence.get("missing_groups") or []
    simulation["source_mode"] = truth.get("source_mode")
    simulation["confidence"] = truth.get("confidence")
    simulation["data_age_seconds"] = truth.get("data_age_seconds")
    simulation["fallback_reason"] = truth.get("fallback_reason")
    simulation["missing_groups"] = truth.get("missing_groups") or []
    simulation["simulations"] = _apply_sentiment_impact_to_rows(simulation.get("simulations") or [], sentiment_impact)
    simulation["probability_explanations"] = _probability_explanations(simulation.get("simulations") or [], truth)
    simulation.update(_sentiment_impact_fields(sentiment_impact))
    if live and selected_model_id == "ml_simulator_v1":
        ml_payload = _ml_live_runner_payload(
            round_num=round_num,
            session=session,
            race=race,
            drivers=client.get_drivers(),
            constructors=client.get_constructors(),
            features=features,
            truth=truth,
            weekend_evidence=weekend_evidence,
            live_state=live_state,
            sentiment=sentiment,
            publish=publish,
        )
        simulation = _apply_ml_live_runner_payload(simulation, ml_payload, client.get_drivers())
    simulation = _apply_probability_audit(simulation, profile, truth, live=live)
    simulation = _apply_live_probability_fields(simulation, truth) if live else simulation
    if live:
        attach_confidence_report(
            simulation,
            _confidence_report(truth, live_state=live_state or {}, probabilities=simulation.get("simulations") or []),
        )
    simulation["probability_explanations"] = _probability_explanations(simulation.get("simulations") or [], truth)
    if live_state is not None:
        simulation["live_state"] = live_state
    if openf1:
        simulation["track"] = await openf1.get_track_data(
            race=race,
            drivers=client.get_drivers(),
            session=session,
            live=live,
        )
        if storage:
            simulation["track_storage"] = await storage.persist_track_geometry(race, simulation["track"])
    if storage:
        simulation["storage"] = await storage.persist_simulation_run(client.season, race, session, simulation, live=live)
        await storage.persist_probability_snapshot(
            client.season,
            race,
            session,
            {
                "ok": True,
                "round": round_num,
                "session": normalized_session,
                "probabilities": simulation.get("simulations") or [],
                "generated_at": simulation.get("generated_at"),
                "source_mode": simulation.get("source_mode"),
                "confidence": simulation.get("confidence"),
                "live_dynamics": simulation.get("live_dynamics") or {},
                "probability_timeline": simulation.get("probability_timeline") or {},
            },
            model_id=simulation.get("model_id") or PRODUCTION_MODEL_ID,
        )
    bridge_records = _simulation_rows_to_outcome_probabilities(
        season=client.season,
        race=race,
        rows=simulation.get("simulations") or [],
        model_version=prediction_model_version or simulation.get("model_version") or selected_model_id,
        generated_at=simulation.get("generated_at"),
    )
    simulation["bridge_record_count"] = len(bridge_records)
    simulation["bridge_publish"] = (
        _publish_outcome_probabilities(bridge_records)
        if publish
        else {"ok": None, "reason": "not_requested", "record_count": len(bridge_records)}
    )
    if not live and not publish and simulation.get("ok", True):
        simulation["response_cache"] = {
            "hit": False,
            "age_seconds": 0,
            "ttl_seconds": _STATIC_RESPONSE_CACHE_TTL_SECONDS,
            "model_id": selected_model_id,
            "live": bool(live),
        }
        _set_static_cache(_simulation_response_cache, cache_key, simulation)
    return simulation


@router.get("/races/{round_num}/track")
async def get_race_track(round_num: int, session: str = "race", live: bool = False):
    normalized_session = (session or "race").lower()
    if not live:
        cached = _get_static_cache(_track_response_cache, (round_num, normalized_session))
        if cached:
            return cached

    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found"}
    if not openf1:
        return {"ok": False, "reason": "OpenF1 client unavailable"}
    if normalized_session == "sprint":
        profile = await _race_profile_for_round(round_num)
        context = profile.get("context") or {}
        if not context.get("has_sprint"):
            return {
                "ok": False,
                "reason": "Sprint session is not scheduled for this Grand Prix",
                "code": "sprint_unavailable",
                "context": context,
            }
    track = await openf1.get_track_data(
        race=race,
        drivers=client.get_drivers(),
        session=session,
        live=live,
    )
    if not live and track.get("ok", True):
        _set_static_cache(_track_response_cache, (round_num, normalized_session), track)
        track["response_cache"] = {
            "hit": False,
            "age_seconds": 0,
            "ttl_seconds": _STATIC_RESPONSE_CACHE_TTL_SECONDS,
        }
    if storage:
        track["storage"] = await storage.persist_track_geometry(race, track)
    return track


@router.post("/races/{round_num}/edges")
async def post_f1_race_edges(
    round_num: int,
    body: F1EdgeRequest,
    session: str = "race",
    live: bool = False,
    model_id: str | None = None,
):
    """ANALYSIS-ONLY: join the round's model probabilities to posted venue quotes,
    returning de-vigged edges + fractional-Kelly stakes. Does NOT place orders —
    the F1 model is still uncalibrated heuristic Monte-Carlo (see /pipeline/health),
    so paper-trade until calibration is proven."""
    race = client.get_race_by_round(round_num) if client else None
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}

    simulation = await get_race_simulation(round_num, session=session, live=live, model_id=model_id)
    rows = (simulation or {}).get("simulations") or []
    records = _simulation_rows_to_outcome_probabilities(
        season=client.season,
        race=race,
        rows=rows,
        model_version=str(
            (simulation or {}).get("prediction_model_version")
            or (simulation or {}).get("model_id")
            or PRODUCTION_MODEL_ID
        ),
        generated_at=(simulation or {}).get("generated_at"),
    )
    prob_map = probability_map_from_records(records)
    quotes = [RaceQuote.from_input(q) for q in body.quotes]
    edges = compute_race_edges(
        prob_map, quotes,
        bankroll_usd=body.bankroll_usd,
        min_edge_bps=body.min_edge_bps,
        shrinkage=body.shrinkage,
        max_per_market_pct=body.max_per_market_pct,
    )
    tradeable = [e for e in edges if e.get("tradeable")]
    return {
        "ok": True,
        "round": round_num,
        "entity_id": _race_entity_id(client.season, race),
        "session": (session or "race").lower(),
        "bankroll_usd": body.bankroll_usd,
        "min_edge_bps": body.min_edge_bps,
        "model_markets": sorted({market for (_code, market) in prob_map.keys()}),
        "edges": edges,
        "tradeable_count": len(tradeable),
        "count": len(edges),
        "disclaimer": "Analysis-only. Model is uncalibrated heuristic Monte-Carlo (see /pipeline/health); paper-trade until calibration is proven.",
    }


@router.post("/races/{round_num}/market-consensus")
async def post_f1_market_consensus(round_num: int, body: dict):
    """Supply venue-implied per-driver win probabilities for a round. Builds the
    governed (capped) market-consensus signal and caches it so the next prediction
    for this round incorporates it as a bounded feature (≤ 0.045 influence). The
    market informs but cannot dominate, and this does not place orders."""
    race = client.get_race_by_round(round_num) if client else None
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    driver_implied = (body or {}).get("driver_implied") or (body or {}).get("drivers") or {}
    if not isinstance(driver_implied, dict) or not driver_implied:
        return {"ok": False, "reason": "driver_implied map required", "code": "missing_driver_implied"}
    driver_team = {str(d.id): d.team for d in (client.get_drivers() or [])}
    consensus = build_market_consensus(
        {str(k): v for k, v in driver_implied.items()},
        driver_team=driver_team,
        confidence=float((body or {}).get("confidence") or 0.58),
    )
    _market_consensus_by_round[round_num] = consensus
    _clear_static_response_caches()
    return {"ok": True, "round": round_num, "consensus": consensus}


@router.get("/races/{round_num}/market-consensus")
async def get_f1_market_consensus(round_num: int):
    return {"ok": True, "round": round_num, "consensus": _market_consensus_by_round.get(round_num)}


@router.get("/races/{round_num}/archetype-fit")
async def get_f1_race_archetype_fit(round_num: int):
    """Per-driver circuit-archetype ratings + fit for this race, from historical
    teammate-context track performance (the existing per-track scores grouped by
    archetype). Read-only analysis."""
    race = client.get_race_by_round(round_num) if client else None
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    features = (predictor._features if predictor else {}) or {}
    track_history = features.get("track_history") or {}
    race_name = " ".join(str(x) for x in [
        getattr(race, "name", ""),
        getattr(race, "circuit", "") or getattr(race, "circuit_id", ""),
        getattr(race, "country", ""),
    ] if x)
    is_sprint = bool(getattr(race, "has_sprint", False) or getattr(race, "sprint", False))
    race_archetypes = archetypes_for_track(race_name, is_sprint=is_sprint)
    ratings = archetype_ratings(track_history)
    drivers = []
    for d in (client.get_drivers() or []):
        did = str(d.id)
        drivers.append({
            "driver_id": did,
            "driver_code": getattr(d, "code", None),
            "team": getattr(d, "team", None),
            "archetype_fit_score": archetype_fit_score(ratings.get(did), race_archetypes),
            "archetype_ratings": ratings.get(did) or {},
        })
    drivers.sort(key=lambda x: (x["archetype_fit_score"] if x["archetype_fit_score"] is not None else -1.0), reverse=True)
    return {
        "ok": True,
        "round": round_num,
        "race_archetypes": race_archetypes,
        "drivers": drivers,
        "source": "archetype_fit (historical teammate-context track scores)",
        "missing_data": not bool(track_history),
    }


@router.get("/backtest/ledger/{season}")
async def get_f1_season_ledger(
    season: int,
    model_id: str | None = None,
    stage: str = "pre_weekend",
    bankroll_usd: float = 1000.0,
    min_edge_bps: float = 200.0,
    market_shrink: float = 0.25,
    overround: float = 0.04,
    spread: float = 0.02,
    allow_partial: bool = True,
):
    """Run the bet-ledger over a season's historical backtest (winner market) against a
    SYNTHETIC market — each driver's market price is the model probability shrunk toward
    the race's field mean by ``market_shrink`` (so the model has edge on its sharp picks).
    Paper-only and partly circular: validates the ledger pipeline, not real edge (no real
    F1 market history yet — enable F1OddsSnapshotService for that)."""
    bt = _backtester()
    result = await bt.backtest_season(season, include_races=True, allow_partial=allow_partial,
                                      model_id=model_id, stage=stage)
    if not result.get("ok", False):
        return {"ok": False, "season": season, "reason": result.get("reason", "backtest_failed")}

    samples = []
    for row in result.get("races") or []:
        winner = row.get("actual_winner")
        dist = [it for it in (row.get("probability_distribution") or []) if float(it.get("win_probability") or 0.0) > 0]
        if not dist:
            continue
        field_mean = sum(float(it["win_probability"]) for it in dist) / len(dist)
        for item in dist:
            prob = float(item.get("win_probability") or 0.0)
            samples.append({
                "model_prob": prob,
                "outcome": 1 if item.get("driver_id") == winner else 0,
                "market_ref": (1.0 - market_shrink) * prob + market_shrink * field_mean,
                "label": f"R{row.get('round')}:{item.get('driver_id')}",
            })

    decisions = build_synthetic_decisions(samples, overround=overround, spread=spread)
    ledger = run_bet_ledger(decisions, bankroll_usd=bankroll_usd, min_edge_bps=min_edge_bps)
    return {
        "ok": True,
        "season": season,
        "stage": stage,
        "model_id": result.get("model_id"),
        "races_evaluated": len(result.get("races") or []),
        "samples": len(samples),
        "synthetic": True,
        "market_shrink": market_shrink,
        **ledger,
    }


@router.post("/backtest/ledger")
async def post_f1_ledger_backtest(body: dict):
    """Run the model-vs-market bet-ledger over supplied decisions (or samples →
    synthetic market). Returns P&L / ROI / drawdown / hit-rate / CLV / edge-bucket
    calibration. Paper-only analysis — no real F1 market history exists yet, so a
    synthetic market is partly circular; use it to validate the ledger, not the edge."""
    body = body or {}
    decisions = body.get("decisions")
    synthetic = bool(not decisions and body.get("samples"))
    if synthetic:
        decisions = build_synthetic_decisions(
            body.get("samples") or [],
            market_bias=float(body.get("market_bias", 0.0)),
            overround=float(body.get("overround", 0.04)),
            spread=float(body.get("spread", 0.02)),
        )
    if not decisions:
        return {"ok": False, "reason": "decisions or samples required", "code": "missing_decisions"}
    ledger = run_bet_ledger(
        decisions,
        bankroll_usd=float(body.get("bankroll_usd", 1000.0)),
        min_edge_bps=float(body.get("min_edge_bps", 200.0)),
        shrinkage=float(body.get("shrinkage", 0.25)),
        max_per_market_pct=float(body.get("max_per_market_pct", 0.05)),
        default_fee_bps=float(body.get("fee_bps", 0.0)),
    )
    return {"ok": True, "synthetic": synthetic, **ledger}


@router.get("/races/{round_num}/strategy")
async def get_f1_race_strategy(round_num: int):
    """Compound-aware stint-strategy analysis: optimal pit window, undercut/overcut
    value, safety-car pit value, 1-vs-2-stop, and a per-driver tactic. Analytic over
    the real per-circuit pit-loss/tire-stress + tire-degradation features. Read-only."""
    race = client.get_race_by_round(round_num) if client else None
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    features = (predictor._features if predictor else {}) or {}
    track = TrackFeatureProvider(features).get_features(race)
    tires = TireFeatureProvider().get_features(track=track)
    base = analyze_race_strategy(
        laps=track.get("laps") or 57,
        pit_loss_s=track.get("pit_loss") or 22.0,
        tire_stress=track.get("tire_stress") or 0.5,
        degradation_rate=tires.get("degradation_rate") or 0.5,
        undercut_strength=tires.get("undercut_strength") or 0.5,
        overcut_strength=tires.get("overcut_strength") or 0.4,
        safety_car_probability=track.get("safety_car_probability") or 0.3,
        overtaking_difficulty=track.get("overtaking_difficulty") or 0.5,
        compound_set=tires.get("compound_set"),
    )
    profile = await _race_profile_for_round(round_num)
    drivers_by_id = {str(d.id): d for d in (client.get_drivers() or [])}
    grid_drivers = _grid_evidence(profile, drivers_by_id).get("drivers") or {}
    driver_strategies = []
    for did, gitem in grid_drivers.items():
        delta = driver_strategy_delta(base, grid_position=gitem.get("grid_position"))
        driver_strategies.append({"driver_id": did, "driver_code": gitem.get("driver_code"), **delta})
    driver_strategies.sort(key=lambda x: -(x.get("estimated_gain_s") or 0.0))
    return {
        "ok": True,
        "round": round_num,
        "track_key": track.get("track_key"),
        "tire_source": tires.get("source"),
        "strategy": base,
        "drivers": driver_strategies,
    }


@router.get("/races/{round_num}/penalties")
async def get_f1_race_penalties(round_num: int):
    """Structured grid penalties, pit-lane starts, and disqualifications for a race,
    from official Jolpica qualifying-vs-grid deltas + result status. Penalty reasons
    and component/PU changes are not in any structured feed and are omitted (not
    guessed). Read-only."""
    race = client.get_race_by_round(round_num) if client else None
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    profile = await _race_profile_for_round(round_num)
    drivers_by_id = {str(d.id): d for d in (client.get_drivers() or [])}
    grid_evidence = _grid_evidence(profile, drivers_by_id)
    report = build_penalty_report(grid_evidence, profile.get("results") or [])
    return {"ok": True, "round": round_num, **report}


@router.post("/races/{round_num}/upgrade-impact")
async def post_f1_upgrade_impact(round_num: int, body: dict):
    """Estimate car-upgrade impact via a manual upgrade calendar + diff-in-diff on
    field-relative pace. Body: {calendar?, pace_history, cap?, min_post?}. With no
    calendar configured (and none posted) it returns no impacts — honest by
    construction; it does NOT fabricate an 'upgrade impact' from the sentiment signal."""
    body = body or {}
    calendar = body.get("calendar") or load_upgrade_calendar()
    pace_history = body.get("pace_history") or {}
    impacts = build_upgrade_impacts(
        calendar, pace_history, current_round=round_num,
        cap=float(body.get("cap", 0.06)), min_post=int(body.get("min_post", 2)),
    )
    applied = [i for i in impacts if i.get("applied")]
    return {
        "ok": True,
        "round": round_num,
        "calendar_entries": len(calendar),
        "impacts": impacts,
        "applied_count": len(applied),
        "note": "No structured upgrade feed exists; treatment is a manual calendar (F1_UPGRADE_CALENDAR). Empty calendar → no impacts.",
    }


@router.get("/backtest/calibration")
async def get_f1_calibration_state():
    """Current empirical-calibrator state (enabled / method / source / fit counts)."""
    import os
    from sports.f1.predictor.probability.engine import _get_empirical_calibrator
    cal = _get_empirical_calibrator()
    return {
        "ok": True,
        "enabled": not cal.is_identity(),
        "method": cal.method,
        "source": cal.source,
        "sample_count": getattr(cal, "sample_count", 0),
        "positive_count": getattr(cal, "positive_count", 0),
        "artifact_env": CALIBRATION_ARTIFACT_ENV,
        "artifact_configured": bool(os.environ.get(CALIBRATION_ARTIFACT_ENV)),
        "breakpoints": [[round(x, 4), round(y, 4)] for x, y in getattr(cal, "breakpoints", [])[:14]],
    }


@router.post("/backtest/calibration/fit")
async def post_f1_fit_calibration(
    start_season: int = 2024,
    end_season: int = 2026,
    method: str = "isotonic",
    stage: str = "pre_weekend",
    path: str | None = None,
):
    """Fit the empirical winner-market calibrator on historical backtest outcomes and
    enable it (sets F1_CALIBRATION_ARTIFACT for this process; applied on the next
    prediction). When the range spans >1 season, fits on all but the last season and
    reports HELD-OUT Brier on the last; otherwise reports in-sample Brier."""
    import os
    bt = _backtester()
    fit_runs, eval_runs, seasons_used = [], [], []
    multi = int(end_season) > int(start_season)
    for season in range(int(start_season), int(end_season) + 1):
        try:
            result = await bt.backtest_season(season, include_races=True, allow_partial=True, model_id=None, stage=stage)
        except Exception:
            continue
        races = result.get("races") if result.get("ok") else None
        if not races:
            continue
        seasons_used.append(season)
        if multi and season == int(end_season):
            eval_runs.extend(races)
        else:
            fit_runs.extend(races)

    fit_pairs = winner_pairs_from_runs(fit_runs)
    positives = sum(1 for _, y in fit_pairs if y >= 0.5)
    if len(fit_pairs) < MIN_SAMPLES or positives < MIN_POSITIVES:
        return {"ok": False, "reason": "insufficient_data", "fit_samples": len(fit_pairs),
                "fit_positives": positives, "min_samples": MIN_SAMPLES, "min_positives": MIN_POSITIVES,
                "seasons_used": seasons_used}

    calibrator = fit_winner_calibrator(fit_runs, method=method, source=f"backtest:{start_season}-{end_season}:{method}")
    if calibrator.is_identity():
        return {"ok": False, "reason": "fit_returned_identity", "fit_samples": len(fit_pairs), "seasons_used": seasons_used}

    eval_pairs = winner_pairs_from_runs(eval_runs) if eval_runs else fit_pairs
    eval_kind = "held_out" if eval_runs else "in_sample"
    ep = [p for p, _ in eval_pairs]
    eo = [y for _, y in eval_pairs]
    brier_raw = brier_score(ep, eo)
    brier_cal = brier_score([calibrator.apply(p) for p in ep], eo)

    target = path or os.environ.get(CALIBRATION_ARTIFACT_ENV) or os.path.join(os.getcwd(), "artifacts", "f1_winner_calibrator.json")
    calibrator.save(target)
    os.environ[CALIBRATION_ARTIFACT_ENV] = target
    _clear_static_response_caches()

    return {
        "ok": True,
        "method": calibrator.method,
        "seasons_used": seasons_used,
        "fit_sample_count": calibrator.sample_count,
        "fit_positive_count": calibrator.positive_count,
        "evaluation": eval_kind,
        "brier": {"raw": round(brier_raw, 6), "calibrated": round(brier_cal, 6), "improvement": round(brier_raw - brier_cal, 6)},
        "breakpoints": [[round(x, 4), round(y, 4)] for x, y in calibrator.breakpoints[:14]],
        "artifact_path": target,
        "enabled": True,
        "note": "Applied on the next prediction (caches cleared). "
                + ("Held-out Brier on the last season." if eval_runs else "In-sample Brier (single season)."),
    }


@router.get("/backtest/validation")
async def get_f1_walk_forward_validation(
    start_season: int = 2023,
    end_season: int = 2026,
    stage: str = "pre_weekend",
    method: str = "isotonic",
    model_id: str | None = None,
    min_train_races: int = 20,
    val_block: int = 5,
):
    """Walk-forward, out-of-sample validation of the served win-probability model.

    Gathers backtest race rows across [start_season, end_season], builds expanding-
    window folds (each trains only on prior races), and reports pooled HELD-OUT
    Brier / log loss / winner accuracy against a uniform baseline, plus a pass/fail
    gate. This is the A7 go/no-go check: the model must beat the baseline
    out-of-sample before its probabilities (and the edges derived from them) can be
    trusted for trading. Also reports whether empirical calibration helps OOS."""
    from sports.f1.predictor.backtesting.validation import run_walk_forward_validation

    bt = _backtester()
    rows: list[dict] = []
    seasons_used: list[int] = []
    for season in range(int(start_season), int(end_season) + 1):
        try:
            result = await bt.backtest_season(
                season, include_races=True, allow_partial=True, model_id=model_id, stage=stage
            )
        except Exception:
            continue
        races = result.get("races") if result.get("ok") else None
        if not races:
            continue
        seasons_used.append(season)
        rows.extend(races)

    report = run_walk_forward_validation(
        rows, method=method, min_train_races=int(min_train_races), val_block=int(val_block)
    )
    report["seasons_used"] = seasons_used
    report["stage"] = stage
    report["model_id"] = model_id or "production_v1"
    return report


@router.get("/backtest/train-eval")
async def get_f1_trained_model_eval(
    start_season: int = 2023,
    end_season: int = 2025,
    stage: str = "pre_weekend",
    model_kind: str = "logistic",
    min_train_races: int = 20,
    val_block: int = 5,
):
    """Walk-forward evaluation of a TRAINED win model (A1) vs the heuristic.

    Gathers full backtest rows (with leakage-safe per-driver ``component_scores``)
    across the season range, and for each walk-forward fold trains a model on prior
    races and predicts the held-out block. Returns pooled out-of-sample Brier /
    log-loss / winner-accuracy for the trained model, the heuristic, and a uniform
    baseline, plus a gate: does the trained model beat the heuristic out-of-sample?
    ``model_kind`` is ``logistic`` (default; robust on the small F1 sample) or
    ``gbm``. Training/eval only — a winning model is wired into serving separately."""
    from sports.f1.ml.training.win_model import walk_forward_train_eval

    bt = _backtester()
    rows: list[dict] = []
    seasons_used: list[int] = []
    for season in range(int(start_season), int(end_season) + 1):
        try:
            result = await bt.backtest_season(
                season, include_races=True, allow_partial=True, model_id=None, stage=stage, _compact=False
            )
        except Exception:
            continue
        races = result.get("races") if result.get("ok") else None
        if not races:
            continue
        seasons_used.append(season)
        rows.extend(races)

    report = walk_forward_train_eval(
        rows, model_kind=model_kind, min_train_races=int(min_train_races), val_block=int(val_block)
    )
    report["seasons_used"] = seasons_used
    report["stage"] = stage
    return report


@router.post("/refresh")
async def refresh():
    _emit_ops_event("refresh_started", message="Catalog + feature refresh started")
    _clear_static_response_caches()
    await client.refresh()
    features = await client.get_prediction_features()
    sentiment = await refresh_f1_sentiment(
        client.get_drivers(),
        client.get_constructors(),
        client.season,
    )
    await _persist_sentiment(sentiment)
    if predictor:
        predictor.load_drivers(
            client.get_drivers(),
            client.get_constructors(),
            features,
            sentiment,
        )
    result = {
        "ok": True,
        "drivers": len(client.get_drivers()),
        "constructors": len(client.get_constructors()),
        "races": len(client.get_races()),
        "completed_races": features.get("completed_races", 0),
        "sentiment_items": sentiment.get("source_items", 0),
        "sentiment_published_items": sentiment.get("published_items", 0),
        "sentiment_configured_sources": sentiment.get("configured_sources", 0),
        "prediction_model": "f1-live-historical-sentiment-v4",
    }
    _emit_ops_event(
        "refresh_completed",
        message=f"Refresh complete — {result['drivers']} drivers, {result['completed_races']} completed races",
        drivers=result["drivers"],
        completed_races=result["completed_races"],
    )
    return result
