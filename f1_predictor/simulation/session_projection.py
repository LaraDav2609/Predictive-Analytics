"""Formula 1 session simulator for GP-specific probability learning."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from models.f1 import Constructor, Driver, Race
from f1_predictor.config import SESSION_SIMULATION_VERSION
from f1_predictor.features.car_model import build_car_model_analysis
from f1_predictor.features.reliability import ReliabilityFeatureProvider
from f1_predictor.features.tires import TireFeatureProvider
from f1_predictor.features.track import TrackFeatureProvider
from f1_predictor.features.weather import WeatherFeatureProvider
from f1_predictor.live.dynamics import build_live_dynamics
from f1_predictor.scoring.explanations import sentiment_label, simulation_signals
from f1_predictor.scoring.normalization import position_score, prior_score, sentiment_factor
from f1_predictor.simulation.monte_carlo import MonteCarloSimulator


def build_session_projection(
    race: Race,
    drivers: list[Driver],
    constructors: list[Constructor],
    prediction: dict[str, Any],
    features: dict[str, Any],
    qualifying: list[dict],
    sprint: list[dict],
    results: list[dict],
    session: str = "race",
    live: bool = False,
) -> dict[str, Any]:
    session = (session or "race").lower()
    if session not in {"race", "qualifying", "sprint"}:
        session = "race"

    prediction_items = (prediction or {}).get("driver_predictions") or {}
    driver_features = (features or {}).get("drivers") or {}
    constructor_features = (features or {}).get("constructors") or {}
    openf1_session = (features or {}).get("openf1_session") or {}
    race_truth = (features or {}).get("race_truth") or {}
    live_state = (features or {}).get("live_state") or (race_truth if live else {})
    live_positions = (live_state.get("live_positions") or {}) if live else {}
    live_by_driver = (live_state.get("by_driver_id") or {}) if live else {}
    live_confidence = float(live_state.get("confidence") or 0.0) if live else 0.0
    truth_by_driver = race_truth.get("by_driver_id") or {}
    truth_confidence = float(race_truth.get("confidence") or 0.0)
    track_features = TrackFeatureProvider(features).get_features(race)
    weather_features = WeatherFeatureProvider(features).get_features(race, openf1_session, session=session)
    live_chaos = float(((live_state.get("signals") or {}).get("chaos_score") or 0.0) if live_state else 0.0)
    if live_chaos:
        weather_features = {
            **weather_features,
            "chaos_score": max(float(weather_features.get("chaos_score") or 0.0), live_chaos),
            "source": "live_state_openf1_weather",
        }
    tire_features = TireFeatureProvider().get_features(track_features, openf1_session, weather=weather_features)
    reliability_features = ReliabilityFeatureProvider(driver_features).get_features(track_features, weather_features)
    live_dynamics = build_live_dynamics(
        race_truth if race_truth else live_state,
        session=session,
        track=track_features,
        tires=tire_features,
    ) if live else {"ok": False, "reason": "not_live"}
    live_dynamics_by_driver = live_dynamics.get("drivers") or {}
    strategy_state = _strategy_state(track_features, tire_features, weather_features, live_dynamics)
    car_model = (features or {}).get("car_model") or build_car_model_analysis(
        race=race,
        drivers=drivers,
        constructors=constructors,
        features=features,
        session=session,
        track=track_features,
        weather=weather_features,
        tires=tire_features,
        openf1_session=openf1_session,
        sentiment_impact=(features or {}).get("sentiment_impact") or race_truth.get("sentiment_impact"),
    )
    car_model_by_driver = car_model.get("drivers") or {}
    constructors_by_name = {constructor.name.lower(): constructor for constructor in constructors}
    max_points = max((driver.points for driver in drivers), default=1.0) or 1.0
    max_constructor_points = max((constructor.points for constructor in constructors), default=1.0) or 1.0

    quali_by_driver = {item.get("driver_id"): item for item in qualifying if item.get("driver_id")}
    sprint_by_driver = {item.get("driver_id"): item for item in sprint if item.get("driver_id")}
    race_by_driver = {item.get("driver_id"): item for item in results if item.get("driver_id")}

    rows = []
    for driver in drivers:
        base_prediction = prediction_items.get(driver.id) or {}
        feature = driver_features.get(driver.id) or {}
        team_key = (driver.team or "").lower()
        constructor = constructors_by_name.get(team_key)
        constructor_feature = constructor_features.get(team_key) or {}
        car_profile = car_model_by_driver.get(driver.id) or {}

        prior = float(base_prediction.get("win_prob") or 0.0)
        if prior <= 0:
            prior = (driver.points / max_points) / max(len(drivers), 1)

        form = float(feature.get("form_score") or base_prediction.get("form_score") or 0.45)
        reliability = float(feature.get("reliability_score") or base_prediction.get("reliability_score") or 0.75)
        dnf_probability = float((reliability_features.get(driver.id) or {}).get("dnf_probability") or base_prediction.get("dnf_prob") or 0.12)
        qualifying_pace = float(feature.get("qualifying_pace_score") or base_prediction.get("qualifying_pace_score") or form)
        race_pace = float(feature.get("race_pace_score") or base_prediction.get("race_pace_score") or form)
        team = float(constructor_feature.get("team_score") or base_prediction.get("team_score") or ((constructor.points / max_constructor_points) if constructor else 0.45))
        car_model_score = float(car_profile.get("overall_car_score") or base_prediction.get("car_performance_score") or team)
        car_model_modifier = float(car_profile.get("car_modifier") or 1.0)
        sentiment = float(base_prediction.get("sentiment_score") or 0.0)
        label = base_prediction.get("sentiment_label") or sentiment_label(sentiment)
        sentiment_mentions = int(base_prediction.get("sentiment_mentions") or 0)
        personal_news = float(base_prediction.get("personal_news_score") or 0.0)
        team_news = float(base_prediction.get("team_news_score") or 0.0)
        overall_news = float(base_prediction.get("overall_news_score") or 0.0)
        news_win_modifier = float(base_prediction.get("news_win_modifier") or 1.0)
        race_sentiment_impact = float(base_prediction.get("race_sentiment_impact_score") or 0.0)
        race_sentiment_delta = float(base_prediction.get("race_sentiment_delta") or 0.0)
        race_sentiment_confidence = float(base_prediction.get("race_sentiment_confidence") or 0.0)
        race_sentiment_articles = int(base_prediction.get("race_sentiment_articles") or 0)
        race_sentiment_explanations = base_prediction.get("race_sentiment_explanations") or []
        wdc_probability = float(base_prediction.get("wdc_prob") or 0.0)
        wdc_modifier = float(base_prediction.get("wdc_modifier") or 1.0)
        qualifying_score = position_score((quali_by_driver.get(driver.id) or {}).get("position"), default=0.46)
        sprint_score = position_score((sprint_by_driver.get(driver.id) or {}).get("position"), default=0.48)
        truth_driver_state = truth_by_driver.get(driver.id) or {}
        live_driver_state = live_by_driver.get(driver.id) or live_positions.get(driver.id) or truth_driver_state or {}
        live_dynamic = live_dynamics_by_driver.get(driver.id) or {}
        live_position = live_driver_state.get("position") or (race_by_driver.get(driver.id) or {}).get("position")
        live_score = position_score(live_position, default=0.50)
        source_confidence = max(live_confidence, truth_confidence if live else 0.0)
        live_weighted_score = 0.50 + (live_score - 0.50) * max(0.0, min(1.0, source_confidence))
        strategy = _strategy_score(driver, feature, qualifying_score, sprint_score, live_score, live)

        if session == "qualifying":
            strength = (
                0.14 * prior_score(prior, drivers)
                + 0.22 * form
                + 0.22 * team
                + 0.12 * qualifying_score
                + 0.03 * reliability
                + 0.24 * qualifying_pace
                + 0.03 * sentiment_factor(sentiment)
            )
        elif session == "sprint":
            strength = (
                0.12 * prior_score(prior, drivers)
                + 0.20 * form
                + 0.18 * team
                + 0.15 * qualifying_score
                + 0.11 * reliability
                + 0.08 * strategy
                + 0.13 * race_pace
                + 0.03 * sentiment_factor(sentiment)
            )
        elif live and (results or live_positions):
            strength = (
                0.08 * prior_score(prior, drivers)
                + 0.12 * form
                + 0.12 * team
                + 0.10 * qualifying_score
                + 0.34 * live_weighted_score
                + 0.12 * strategy
                + 0.08 * reliability
                + 0.02 * race_pace
                + 0.02 * sentiment_factor(sentiment)
            )
        else:
            strength = (
                0.10 * prior_score(prior, drivers)
                + 0.18 * form
                + 0.17 * team
                + 0.13 * qualifying_score
                + 0.05 * sprint_score
                + 0.08 * reliability
                + 0.18 * race_pace
                + 0.09 * strategy
                + 0.02 * sentiment_factor(sentiment)
            )
        live_strength_multiplier = float(live_dynamic.get("strength_multiplier") or 1.0)
        strength = max(
            0.001,
            strength
            * max(0.60, 1.0 - dnf_probability * 0.55)
            * max(0.94, min(1.06, 1.0 + race_sentiment_delta))
            * max(0.94, min(1.06, car_model_modifier))
            * live_strength_multiplier,
        )

        rows.append({
            "driver_id": driver.id,
            "driver_number": driver.number,
            "driver_code": driver.code,
            "driver_name": f"{driver.first_name} {driver.last_name}".strip(),
            "team": driver.team,
            "sentiment_label": label,
            "sentiment_mentions": sentiment_mentions,
            "personal_news_score": round(personal_news, 4),
            "team_news_score": round(team_news, 4),
            "overall_news_score": round(overall_news, 4),
            "news_win_modifier": round(news_win_modifier, 4),
            "race_sentiment_impact_score": round(race_sentiment_impact, 4),
            "race_sentiment_delta": round(race_sentiment_delta, 4),
            "race_sentiment_confidence": round(race_sentiment_confidence, 4),
            "race_sentiment_articles": race_sentiment_articles,
            "race_sentiment_explanations": race_sentiment_explanations,
            "wdc_probability": round(wdc_probability, 4),
            "wdc_modifier": round(wdc_modifier, 4),
            "source_mode": _source_mode(live, live_driver_state, live_state, race_truth),
            "source_confidence": round(float(live_driver_state.get("confidence") or source_confidence or 0.0), 4) if live else round(truth_confidence, 4),
            "strength": max(strength, 0.001),
            "components": {
                "model_prior": round(prior_score(prior, drivers), 4),
                "raw_model_win_prior": round(prior, 4),
                "championship_position": driver.position,
                "championship_points": round(float(driver.points or 0.0), 2),
                "championship_context_only": True,
                "recent_form": round(form, 4),
                "team_pace": round(team, 4),
                "car_model": round(car_model_score, 4),
                "car_model_modifier": round(max(0.94, min(1.06, car_model_modifier)), 4),
                "car_model_confidence": round(float(car_profile.get("confidence") or 0.0), 4),
                "car_model_scores": car_profile.get("scores") or {},
                "car_model_missing_data": car_profile.get("missing_data") or [],
                "qualifying": round(qualifying_score, 4),
                "sprint": round(sprint_score, 4),
                "live_track_position": round(live_score, 4),
                "live_confidence": round(source_confidence, 4),
                "live_position_gain": round((live_weighted_score - 0.50) * 0.26, 4),
                "live_position_delta": live_dynamic.get("position_delta"),
                "gap_delta": live_dynamic.get("gap_delta"),
                "pace_trend_delta": live_dynamic.get("pace_delta"),
                "tyre_risk_delta": live_dynamic.get("tyre_delta"),
                "pit_strategy_delta": live_dynamic.get("pit_delta"),
                "live_strength_multiplier": round(live_strength_multiplier, 4),
                "live_dynamics_explanations": live_dynamic.get("explanations") or [],
                "live_position": live_position,
                "gap_to_leader": live_driver_state.get("gap_to_leader"),
                "interval": live_driver_state.get("interval"),
                "compound": live_driver_state.get("compound"),
                "tyre_age": live_driver_state.get("tyre_age"),
                "pit_stops": live_driver_state.get("pit_stops"),
                "gap_seconds": live_dynamic.get("gap_seconds"),
                "interval_seconds": live_dynamic.get("interval_seconds"),
                "pace_seconds": live_dynamic.get("pace_seconds"),
                "strategy": round(strategy, 4),
                "reliability": round(reliability, 4),
                "dnf_probability": round(dnf_probability, 4),
                "weather": round(max(0.0, 1.0 - float(weather_features.get("chaos_score") or 0.0)), 4),
                "weather_chaos": round(float(weather_features.get("chaos_score") or 0.0), 4),
                "weather_model_delta_hint": round(float((weather_features.get("weather_model_impact") or {}).get("model_delta_hint") or 0.0), 4),
                "qualifying_pace": round(qualifying_pace, 4),
                "race_pace": round(race_pace, 4),
                "sentiment": round(sentiment, 4),
                "personal_news": round(personal_news, 4),
                "team_news": round(team_news, 4),
                "overall_news": round(overall_news, 4),
                "news_win_modifier": round(news_win_modifier, 4),
                "race_sentiment_impact": round(race_sentiment_impact, 4),
                "race_sentiment_delta": round(race_sentiment_delta, 4),
                "race_sentiment_confidence": round(race_sentiment_confidence, 4),
                "wdc_probability": round(wdc_probability, 4),
                "estimated_source_discount": round(1.0 - source_confidence, 4) if live else round(max(0.0, 1.0 - truth_confidence), 4),
            },
            "signals": simulation_signals(driver, feature, quali_by_driver, sprint_by_driver, race_by_driver, live, label, sentiment_mentions, wdc_probability),
        })

    driver_scores = {row["driver_id"]: row["strength"] for row in rows}
    monte = MonteCarloSimulator(iterations=900, seed=(race.round * 1000 + len(session))).run(
        driver_scores,
        {
            "session": session,
            "laps": track_features.get("laps") or (24 if session == "sprint" else 57),
            "track": track_features,
            "weather": weather_features,
            "tires": tire_features,
            "reliability": reliability_features,
            "live_positions": live_positions,
            "live_confidence": live_confidence,
            "live_dynamics": live_dynamics,
            "race_truth": race_truth,
        },
    )
    monte_by_driver = {item["driver_id"]: item for item in monte.get("drivers", [])}
    total = sum(row["strength"] for row in rows) or 1.0
    rows.sort(key=lambda row: row["strength"], reverse=True)
    for index, row in enumerate(rows, start=1):
        probability = row["strength"] / total
        monte_row = monte_by_driver.get(row["driver_id"]) or {}
        row["rank"] = index
        row["win_probability"] = round(float(monte_row.get("win_probability", probability)), 4)
        row["podium_probability"] = round(float(monte_row.get("podium_probability", min(0.95, probability * 3))), 4)
        points_probability = float(monte_row.get("points_probability", min(0.98, probability * 5)))
        row["top5_probability"] = round(float(monte_row.get("top5_probability", min(0.98, points_probability))), 4)
        row["points_probability"] = round(points_probability, 4)
        row["dnf_probability"] = round(float(monte_row.get("dnf_probability", row["components"].get("dnf_probability", 0.0))), 4)
        row["expected_finish"] = round(float(monte_row.get("expected_finish", index)), 2)
        row["finish_distribution"] = monte_row.get("finish_distribution") or {}
        row.pop("strength", None)

    return {
        "ok": True,
        "race": race.model_dump(mode="json"),
        "session": session,
        "live": live,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": SESSION_SIMULATION_VERSION,
        "monte_carlo": {
            "model_version": monte.get("model_version"),
            "simulation_count": monte.get("simulation_count"),
            "laps": monte.get("laps"),
        },
        "feature_sources": {
            "track": track_features.get("source"),
            "weather": weather_features.get("source"),
            "tires": tire_features.get("source"),
            "car_model": car_model.get("source"),
            "live_state": live_state.get("mode") if live_state else None,
            "live_dynamics": live_dynamics.get("source_mode") if live_dynamics else None,
            "truth": race_truth.get("source_mode"),
        },
        "track_features": track_features,
        "weather": weather_features,
        "weather_source": weather_features.get("source"),
        "weather_confidence": weather_features.get("confidence"),
        "weather_missing": bool(weather_features.get("missing_data")),
        "weather_model_impact": weather_features.get("weather_model_impact") or {},
        "weather_explanations": weather_features.get("weather_explanations") or [],
        "car_model": car_model,
        "car_model_confidence": car_model.get("confidence"),
        "car_model_missing_data": car_model.get("missing_data") or [],
        "live_state": _compact_live_state(live_state),
        "live_dynamics": live_dynamics,
        "strategy_state": strategy_state,
        "truth": _compact_truth(race_truth),
        "source_mode": race_truth.get("source_mode") or (live_state.get("source_mode") if live_state else None),
        "confidence": race_truth.get("confidence") if race_truth else (live_state.get("confidence") if live_state else None),
        "data_age_seconds": race_truth.get("data_age_seconds") if race_truth else (live_state.get("data_age_seconds") if live_state else None),
        "fallback_reason": race_truth.get("fallback_reason") if race_truth else (live_state.get("fallback_reason") if live_state else None),
        "missing_groups": race_truth.get("missing_groups") or [],
        "status": _status(session, live, qualifying, sprint, results, bool(live_positions)),
        "calculation": _calculation_notes(session, live, bool(results or live_positions)),
        "simulations": rows,
    }


def _strategy_score(driver: Driver, feature: dict, qualifying: float, sprint: float, live_score: float, live: bool) -> float:
    reliability = float(feature.get("reliability_score") or 0.72)
    form = float(feature.get("form_score") or 0.45)
    base = 0.45 * reliability + 0.25 * form + 0.18 * qualifying + 0.12 * sprint
    if live:
        base = 0.55 * live_score + 0.25 * reliability + 0.20 * form
    if "ferrari" in (driver.team or "").lower() or "mclaren" in (driver.team or "").lower() or "mercedes" in (driver.team or "").lower():
        base += 0.03
    return max(0.02, min(1.0, base))


def _strategy_state(track: dict[str, Any], tires: dict[str, Any], weather: dict[str, Any], live_dynamics: dict[str, Any]) -> dict[str, Any]:
    tire_stress = float(track.get("tire_stress") or tires.get("degradation_rate") or 0.50)
    pit_loss = float(track.get("pit_loss") or 23.0)
    safety_car = float(track.get("safety_car_probability") or 0.30)
    chaos = float(weather.get("chaos_score") or 0.0)
    undercut = float(tires.get("undercut_strength") or max(0.25, min(0.82, tire_stress + 0.08)))
    overcut = float(tires.get("overcut_strength") or max(0.15, min(0.62, 0.52 - tire_stress * 0.32)))
    expected_stops = int(live_dynamics.get("expected_pit_stops") or (2 if tire_stress >= 0.70 else 1))
    pit_window_strength = max(0.0, min(1.0, undercut * 0.50 + safety_car * 0.30 + chaos * 0.20))
    return {
        "expected_stops": expected_stops,
        "pit_loss": round(pit_loss, 2),
        "undercut_strength": round(undercut, 4),
        "overcut_strength": round(overcut, 4),
        "tyre_degradation": round(float(tires.get("degradation_rate") or tire_stress), 4),
        "safety_car_pit_window": round(min(1.0, safety_car + chaos * 0.35), 4),
        "pit_window_status": "active" if pit_window_strength >= 0.58 else "watch" if pit_window_strength >= 0.40 else "quiet",
        "pit_window_strength": round(pit_window_strength, 4),
        "source": "track_tire_weather_live_dynamics",
    }


def _status(session: str, live: bool, qualifying: list[dict], sprint: list[dict], results: list[dict], has_live_positions: bool = False) -> str:
    if live and has_live_positions:
        return "live_openf1_state"
    if live and results:
        return "live_or_completed_classification"
    if session == "qualifying" and qualifying:
        return "qualifying_result_available"
    if session == "sprint" and sprint:
        return "sprint_result_available"
    if session == "race" and results:
        return "race_result_available"
    return "pre_session_projection"


def _compact_live_state(live_state: dict[str, Any]) -> dict[str, Any] | None:
    if not live_state:
        return None
    return {
        "ok": live_state.get("ok"),
        "mode": live_state.get("mode"),
        "source_mode": live_state.get("source_mode"),
        "confidence": live_state.get("confidence"),
        "data_age_seconds": live_state.get("data_age_seconds"),
        "fallback_reason": live_state.get("fallback_reason"),
        "status": live_state.get("status"),
        "reason": live_state.get("reason"),
        "age_seconds": live_state.get("age_seconds"),
        "stale": live_state.get("stale"),
        "leader": live_state.get("leader"),
        "driver_count": len(live_state.get("drivers") or []),
        "chaos_score": (live_state.get("signals") or {}).get("chaos_score"),
        "session_key": live_state.get("session_key"),
        "meeting_key": live_state.get("meeting_key"),
    }


def _compact_truth(truth: dict[str, Any]) -> dict[str, Any] | None:
    if not truth:
        return None
    return {
        "ok": truth.get("ok"),
        "status": truth.get("status"),
        "source_mode": truth.get("source_mode"),
        "confidence": truth.get("confidence"),
        "data_age_seconds": truth.get("data_age_seconds"),
        "fallback_reason": truth.get("fallback_reason"),
        "missing_groups": truth.get("missing_groups") or [],
        "driver_count": len(truth.get("drivers") or []),
        "chaos_score": (truth.get("signals") or {}).get("chaos_score"),
    }


def _source_mode(live: bool, live_driver_state: dict[str, Any], live_state: dict[str, Any], race_truth: dict[str, Any]) -> str:
    if live:
        return (
            live_driver_state.get("source_mode")
            or live_state.get("source_mode")
            or race_truth.get("source_mode")
            or live_state.get("mode")
            or "live"
        )
    return race_truth.get("source_mode") or "model"


def _calculation_notes(session: str, live: bool, has_results: bool) -> list[dict[str, Any]]:
    if session == "qualifying":
        weights = {"prior": 14, "recent_form": 22, "team_pace": 22, "qualifying_if_available": 12, "reliability": 3, "qualifying_pace": 24, "news_sentiment": 3}
    elif session == "sprint":
        weights = {"prior": 12, "recent_form": 20, "team_pace": 18, "qualifying": 15, "reliability": 11, "strategy": 8, "race_pace": 13, "news_sentiment": 3}
    elif live and has_results:
        weights = {"prior": 8, "recent_form": 12, "team_pace": 12, "qualifying": 10, "live_position": 34, "strategy": 12, "reliability": 8, "race_pace": 2, "news_sentiment": 2}
    else:
        weights = {"prior": 10, "recent_form": 18, "team_pace": 17, "qualifying": 13, "sprint": 5, "reliability": 8, "race_pace": 18, "strategy": 9, "news_sentiment": 2}
    return [{"factor": key, "weight": value} for key, value in weights.items()]
