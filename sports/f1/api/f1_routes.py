"""Formula 1 API routes."""

from __future__ import annotations

import json
from copy import deepcopy
from time import monotonic

from fastapi import APIRouter
import redis

from sports.f1.data.f1_client import F1Client
from sports.f1.data.openf1_client import OpenF1Client
from sports.f1.data.f1_sentiment import DEFAULT_RSS_FEEDS, read_f1_sentiment, refresh_f1_sentiment
from sports.f1.analytics.f1_predictor import F1Predictor
from sports.f1.analytics.f1_simulator import build_session_simulation
from sports.f1.predictor.backtesting import F1BacktestService
from sports.f1.predictor.live import F1LiveSessionEngine, attach_confidence_report, build_live_confidence_report, build_live_dynamics
from sports.f1.predictor.live.recorder import FastF1LiveRecorderManager
from sports.f1.predictor.models.registry import F1ModelRegistry
from sports.f1.predictor.storage import F1Storage
from sports.f1.predictor.truth import build_race_truth_snapshot
from sports.f1.predictor.features.car_model import build_car_model_analysis, summarize_car_data
from sports.f1.predictor.features.sentiment import build_race_sentiment_impact
from sports.f1.predictor.features.tires import TireFeatureProvider
from sports.f1.predictor.features.track import TrackFeatureProvider
from sports.f1.predictor.features.weather import WeatherFeatureProvider
from sports.f1.predictor.probability import detect_stage, enrich_probability_payload
from sports.f1.predictor.probability.calibration import build_calibration_profile
from sports.f1.predictor.data_quality import attach_data_quality, build_data_quality_report

router = APIRouter(prefix="/f1", tags=["f1"])

# Shared instances (initialized in server.py lifespan)
client: F1Client | None = None
predictor: F1Predictor | None = None
openf1: OpenF1Client | None = None
live_engine: F1LiveSessionEngine | None = None
live_recorder: FastF1LiveRecorderManager | None = None
storage: F1Storage | None = None

_STATIC_RESPONSE_CACHE_TTL_SECONDS = 60.0
_simulation_response_cache: dict[tuple[int, str], tuple[float, dict]] = {}
_track_response_cache: dict[tuple[int, str], tuple[float, dict]] = {}

_WEEKEND_SESSION_LABELS = {
    "fp1": "Practice 1",
    "fp2": "Practice 2",
    "fp3": "Practice 3",
    "sprint_qualifying": "Sprint Qualifying",
    "sprint": "Sprint",
    "qualifying": "Qualifying",
    "race": "Race",
}


def _get_static_cache(cache: dict, key: tuple[int, str]) -> dict | None:
    entry = cache.get(key)
    if not entry:
        return None
    cached_at, payload = entry
    age = monotonic() - cached_at
    if age > _STATIC_RESPONSE_CACHE_TTL_SECONDS:
        cache.pop(key, None)
        return None
    cached = deepcopy(payload)
    cached["response_cache"] = {
        "hit": True,
        "age_seconds": round(age, 2),
        "ttl_seconds": _STATIC_RESPONSE_CACHE_TTL_SECONDS,
    }
    return cached


def _set_static_cache(cache: dict, key: tuple[int, str], payload: dict) -> None:
    stored = deepcopy(payload)
    stored["response_cache"] = {
        "hit": False,
        "age_seconds": 0,
        "ttl_seconds": _STATIC_RESPONSE_CACHE_TTL_SECONDS,
    }
    cache[key] = (monotonic(), stored)


def _clear_static_response_caches() -> None:
    _simulation_response_cache.clear()
    _track_response_cache.clear()


def init(fc: F1Client, fp: F1Predictor):
    global client, predictor, openf1, live_engine, live_recorder, storage
    client, predictor = fc, fp
    openf1 = OpenF1Client()
    live_engine = F1LiveSessionEngine(openf1)
    live_recorder = FastF1LiveRecorderManager()
    storage = F1Storage()


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
    profile = await client.get_race_profile(round_num, predictor)
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


async def _truth_snapshot_for_round(
    round_num: int,
    session: str = "race",
    live: bool = False,
    profile: dict | None = None,
    features: dict | None = None,
    openf1_session: dict | None = None,
    live_state: dict | None = None,
) -> dict:
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}

    sprint_error = await _sprint_unavailable(round_num, session)
    if sprint_error:
        return {**sprint_error, "race": race.model_dump(mode="json")}

    profile = profile if profile is not None else await client.get_race_profile(round_num, predictor)
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


async def _weekend_session_result(round_num: int, session: str) -> dict:
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}

    session_code = _canonical_weekend_session(session)
    profile = await client.get_race_profile(round_num, predictor)
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


@router.get("/calendar")
async def get_calendar():
    races = client.get_races()
    if predictor:
        races = predictor.predict_races(races)
    return {"ok": True, "races": [r.model_dump(mode="json") for r in races]}


@router.get("/models")
async def get_f1_models():
    return {"ok": True, "models": F1ModelRegistry.list_models(), "default_model_id": "production_v1"}


@router.get("/models/compare")
async def compare_f1_models(season: int | None = None, include_races: bool = False, allow_partial: bool = False, stage: str = "pre_weekend"):
    backtester = F1BacktestService(client)
    return await backtester.compare_season(season=season, include_races=include_races, allow_partial=allow_partial, stage=stage)


@router.get("/models/compare/summary")
async def compare_f1_models_summary(
    start_season: int = 2023,
    end_season: int | None = None,
    include_races: bool = False,
    allow_partial: bool = False,
    stage: str = "pre_weekend",
):
    backtester = F1BacktestService(client)
    end = end_season if end_season is not None else max(start_season, client.season - 1)
    return await backtester.compare_summary(
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
            "default_model_id": "production_v1",
            "available_models": [model["model_id"] for model in F1ModelRegistry.list_models()],
            "active_version": predictor._version if predictor else None,
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
    }


@router.get("/storage/health")
async def get_f1_storage_health():
    if not storage:
        return {"ok": False, "reason": "storage_unavailable"}
    health = await storage.health()
    return {"ok": True, **health}


@router.get("/backtest")
async def get_f1_backtest(season: int | None = None, include_races: bool = False, allow_partial: bool = False, model_id: str | None = None, stage: str = "pre_weekend"):
    backtester = F1BacktestService(client)
    result = await backtester.backtest_season(season=season, include_races=include_races, allow_partial=allow_partial, model_id=model_id, stage=stage)
    if storage and result.get("ok"):
        result["storage"] = await storage.persist_backtest_result(result)
    return result


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
    return live_engine.get_timeline(race, session=session)


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
async def get_f1_live_probabilities(round_num: int, session: str = "race"):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    profile = await client.get_race_profile(round_num, predictor)
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
    if race.status == "SCHEDULED" and predictor:
        race.prediction = predictor.predict_race(race)

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
    features = {**features, "race_truth": truth, "car_model": truth.get("car_model") or {}}
    sentiment, sentiment_impact = await _sentiment_with_race_impact(round_num, session=session)
    if predictor:
        predictor.load_drivers(client.get_drivers(), client.get_constructors(), features, sentiment)
        race.prediction = predictor.predict_race(race)
    simulation = build_session_simulation(
        race=race,
        drivers=client.get_drivers(),
        constructors=client.get_constructors(),
        prediction=(race.prediction.model_dump(mode="json") if race.prediction else {}),
        features=features,
        qualifying=profile.get("qualifying") or [],
        sprint=profile.get("sprint") or [],
        results=profile.get("results") or [],
        session=session,
        live=True,
    )
    probability_rows = _apply_sentiment_impact_to_rows(simulation.get("simulations") or [], sentiment_impact)
    simulation["simulations"] = probability_rows
    simulation["truth"] = truth
    simulation["context"] = context
    simulation = _apply_probability_audit(simulation, profile, truth, live=True)
    simulation = _apply_live_probability_fields(simulation, truth)
    probability_rows = simulation.get("simulations") or []
    confidence_report = _confidence_report(truth, live_state=live_state, probabilities=probability_rows)
    payload = {
        "ok": True,
        "round": round_num,
        "session": (session or "race").lower(),
        "live_state": live_state,
        "truth": truth,
        "source_mode": truth.get("source_mode"),
        "confidence": truth.get("confidence"),
        "data_age_seconds": truth.get("data_age_seconds"),
        "fallback_reason": truth.get("fallback_reason"),
        "missing_groups": truth.get("missing_groups") or [],
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
    return payload


@router.get("/backtest/summary")
async def get_f1_backtest_summary(
    start_season: int = 2023,
    end_season: int | None = None,
    include_races: bool = False,
    allow_partial: bool = False,
    model_id: str | None = None,
    stage: str = "pre_weekend",
):
    backtester = F1BacktestService(client)
    end = end_season if end_season is not None else max(start_season, client.season - 1)
    result = await backtester.backtest_summary(
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
    backtester = F1BacktestService(client)
    result = await backtester.backtest_race(season=season, round_num=round_num, allow_partial=allow_partial, model_id=model_id, stage=stage)
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
    if race.status == "SCHEDULED" and predictor:
        race.prediction = predictor.predict_race(race)
    return {"ok": True, "race": race.model_dump(mode="json")}


@router.get("/races/{round_num}/profile")
async def get_race_profile(round_num: int):
    return await client.get_race_profile(round_num, predictor)


@router.get("/races/{round_num}/sessions/{session}")
async def get_race_session_result(round_num: int, session: str):
    return await _weekend_session_result(round_num, session)


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
    profile = await client.get_race_profile(round_num, predictor)
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


@router.get("/races/{round_num}/sentiment-impact")
async def get_race_sentiment_impact(round_num: int, session: str = "race"):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    impact = await _race_sentiment_impact_for_round(round_num, session=session)
    return {"ok": True, "race": race.model_dump(mode="json"), **impact}


@router.get("/races/{round_num}/probability-audit")
async def get_race_probability_audit(round_num: int, session: str = "race", stage: str = "auto", live: bool = False):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found", "code": "race_not_found"}
    profile = await client.get_race_profile(round_num, predictor)
    if not profile.get("ok"):
        return profile
    simulation = await get_race_simulation(round_num, session=session, live=live)
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
        "stage": audited.get("stage"),
        "truth": truth,
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
        profile = await client.get_race_profile(round_num, predictor)
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
    profile = await client.get_race_profile(round_num, predictor)
    truth = await _truth_snapshot_for_round(
        round_num,
        session=session,
        live=live,
        profile=profile,
        features=features,
        openf1_session=features.get("openf1_session"),
        live_state=features.get("live_state"),
    )
    features = {**features, "race_truth": truth, "car_model": truth.get("car_model") or {}}
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
async def get_race_simulation(round_num: int, session: str = "race", live: bool = False):
    normalized_session = (session or "race").lower()
    if not live:
        cached = _get_static_cache(_simulation_response_cache, (round_num, normalized_session))
        if cached:
            return cached

    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found"}

    if race.status == "SCHEDULED" and predictor:
        race.prediction = predictor.predict_race(race)

    features = await client.get_prediction_features()
    profile = await client.get_race_profile(round_num, predictor)
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
    features = {**features, "race_truth": truth}
    sentiment, sentiment_impact = await _sentiment_with_race_impact(round_num, session=session)
    if race.status == "SCHEDULED" and predictor:
        predictor.load_drivers(client.get_drivers(), client.get_constructors(), features, sentiment)
        race.prediction = predictor.predict_race(race)

    simulation = build_session_simulation(
        race=race,
        drivers=client.get_drivers(),
        constructors=client.get_constructors(),
        prediction=(race.prediction.model_dump(mode="json") if race.prediction else {}),
        features=features,
        qualifying=profile.get("qualifying") or [],
        sprint=profile.get("sprint") or [],
        results=profile.get("results") or [],
        session=session,
        live=live,
    )
    simulation["context"] = context
    simulation["truth"] = truth
    simulation["source_mode"] = truth.get("source_mode")
    simulation["confidence"] = truth.get("confidence")
    simulation["data_age_seconds"] = truth.get("data_age_seconds")
    simulation["fallback_reason"] = truth.get("fallback_reason")
    simulation["missing_groups"] = truth.get("missing_groups") or []
    simulation["simulations"] = _apply_sentiment_impact_to_rows(simulation.get("simulations") or [], sentiment_impact)
    simulation["probability_explanations"] = _probability_explanations(simulation.get("simulations") or [], truth)
    simulation.update(_sentiment_impact_fields(sentiment_impact))
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
            model_id=simulation.get("model_id") or simulation.get("model_version") or "production_v1",
        )
    if not live and simulation.get("ok", True):
        _set_static_cache(_simulation_response_cache, (round_num, normalized_session), simulation)
        simulation["response_cache"] = {
            "hit": False,
            "age_seconds": 0,
            "ttl_seconds": _STATIC_RESPONSE_CACHE_TTL_SECONDS,
        }
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
        profile = await client.get_race_profile(round_num, predictor)
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


@router.post("/refresh")
async def refresh():
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
    return {
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
