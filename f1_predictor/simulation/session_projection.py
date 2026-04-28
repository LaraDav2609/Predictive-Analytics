"""Formula 1 session simulator for GP-specific probability learning."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from models.f1 import Constructor, Driver, Race
from f1_predictor.config import SESSION_SIMULATION_VERSION
from f1_predictor.features.reliability import ReliabilityFeatureProvider
from f1_predictor.features.tires import TireFeatureProvider
from f1_predictor.features.track import TrackFeatureProvider
from f1_predictor.features.weather import WeatherFeatureProvider
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
    live_state = (features or {}).get("live_state") or {}
    live_positions = (live_state.get("live_positions") or {}) if live else {}
    live_by_driver = (live_state.get("by_driver_id") or {}) if live else {}
    track_features = TrackFeatureProvider(features).get_features(race)
    weather_features = WeatherFeatureProvider(features).get_features(race, openf1_session)
    live_chaos = float(((live_state.get("signals") or {}).get("chaos_score") or 0.0) if live_state else 0.0)
    if live_chaos:
        weather_features = {
            **weather_features,
            "chaos_score": max(float(weather_features.get("chaos_score") or 0.0), live_chaos),
            "source": "live_state_openf1_weather",
        }
    tire_features = TireFeatureProvider().get_features(track_features, openf1_session)
    reliability_features = ReliabilityFeatureProvider(driver_features).get_features(track_features, weather_features)
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

        prior = float(base_prediction.get("win_prob") or 0.0)
        if prior <= 0:
            prior = (driver.points / max_points) / max(len(drivers), 1)

        form = float(feature.get("form_score") or base_prediction.get("form_score") or 0.45)
        reliability = float(feature.get("reliability_score") or base_prediction.get("reliability_score") or 0.75)
        dnf_probability = float((reliability_features.get(driver.id) or {}).get("dnf_probability") or base_prediction.get("dnf_prob") or 0.12)
        qualifying_pace = float(feature.get("qualifying_pace_score") or base_prediction.get("qualifying_pace_score") or form)
        race_pace = float(feature.get("race_pace_score") or base_prediction.get("race_pace_score") or form)
        team = float(constructor_feature.get("team_score") or base_prediction.get("team_score") or ((constructor.points / max_constructor_points) if constructor else 0.45))
        sentiment = float(base_prediction.get("sentiment_score") or 0.0)
        label = base_prediction.get("sentiment_label") or sentiment_label(sentiment)
        sentiment_mentions = int(base_prediction.get("sentiment_mentions") or 0)
        personal_news = float(base_prediction.get("personal_news_score") or 0.0)
        team_news = float(base_prediction.get("team_news_score") or 0.0)
        overall_news = float(base_prediction.get("overall_news_score") or 0.0)
        news_win_modifier = float(base_prediction.get("news_win_modifier") or 1.0)
        wdc_probability = float(base_prediction.get("wdc_prob") or 0.0)
        wdc_modifier = float(base_prediction.get("wdc_modifier") or 1.0)
        qualifying_score = position_score((quali_by_driver.get(driver.id) or {}).get("position"), default=0.46)
        sprint_score = position_score((sprint_by_driver.get(driver.id) or {}).get("position"), default=0.48)
        live_driver_state = live_by_driver.get(driver.id) or live_positions.get(driver.id) or {}
        live_position = live_driver_state.get("position") or (race_by_driver.get(driver.id) or {}).get("position")
        live_score = position_score(live_position, default=0.50)
        strategy = _strategy_score(driver, feature, qualifying_score, sprint_score, live_score, live)

        if session == "qualifying":
            strength = (
                0.26 * prior_score(prior, drivers)
                + 0.28 * form
                + 0.26 * team
                + 0.12 * qualifying_score
                + 0.03 * reliability
                + 0.02 * qualifying_pace
                + 0.03 * sentiment_factor(sentiment)
            )
        elif session == "sprint":
            strength = (
                0.25 * prior_score(prior, drivers)
                + 0.23 * form
                + 0.20 * team
                + 0.15 * qualifying_score
                + 0.10 * reliability
                + 0.04 * strategy
                + 0.02 * race_pace
                + 0.03 * sentiment_factor(sentiment)
            )
        elif live and (results or live_positions):
            strength = (
                0.16 * prior_score(prior, drivers)
                + 0.16 * form
                + 0.14 * team
                + 0.12 * qualifying_score
                + 0.26 * live_score
                + 0.10 * strategy
                + 0.04 * reliability
                + 0.02 * sentiment_factor(sentiment)
            )
        else:
            strength = (
                0.24 * prior_score(prior, drivers)
                + 0.23 * form
                + 0.19 * team
                + 0.13 * qualifying_score
                + 0.08 * sprint_score
                + 0.05 * reliability
                + 0.03 * race_pace
                + 0.04 * strategy
                + 0.02 * sentiment_factor(sentiment)
            )
        strength = max(0.001, strength * max(0.60, 1.0 - dnf_probability * 0.55))

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
            "wdc_probability": round(wdc_probability, 4),
            "wdc_modifier": round(wdc_modifier, 4),
            "strength": max(strength, 0.001),
            "components": {
                "model_prior": round(prior_score(prior, drivers), 4),
                "recent_form": round(form, 4),
                "team_pace": round(team, 4),
                "qualifying": round(qualifying_score, 4),
                "sprint": round(sprint_score, 4),
                "live_track_position": round(live_score, 4),
                "live_position": live_position,
                "strategy": round(strategy, 4),
                "reliability": round(reliability, 4),
                "dnf_probability": round(dnf_probability, 4),
                "qualifying_pace": round(qualifying_pace, 4),
                "race_pace": round(race_pace, 4),
                "sentiment": round(sentiment, 4),
                "personal_news": round(personal_news, 4),
                "team_news": round(team_news, 4),
                "overall_news": round(overall_news, 4),
                "news_win_modifier": round(news_win_modifier, 4),
                "wdc_probability": round(wdc_probability, 4),
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
        row["top5_probability"] = round(min(0.98, points_probability * 1.15), 4)
        row["points_probability"] = round(points_probability, 4)
        row["dnf_probability"] = round(float(monte_row.get("dnf_probability", row["components"].get("dnf_probability", 0.0))), 4)
        row["expected_finish"] = round(float(monte_row.get("expected_finish", index)), 2)
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
            "live_state": live_state.get("mode") if live_state else None,
        },
        "live_state": _compact_live_state(live_state),
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


def _calculation_notes(session: str, live: bool, has_results: bool) -> list[dict[str, Any]]:
    if session == "qualifying":
        weights = {"prior": 26, "recent_form": 28, "team_pace": 26, "qualifying_if_available": 12, "reliability": 5, "news_sentiment": 3}
    elif session == "sprint":
        weights = {"prior": 25, "recent_form": 23, "team_pace": 20, "qualifying": 15, "reliability": 10, "strategy": 4, "news_sentiment": 3}
    elif live and has_results:
        weights = {"prior": 16, "recent_form": 16, "team_pace": 14, "qualifying": 12, "live_position": 26, "strategy": 10, "reliability": 4, "news_sentiment": 2}
    else:
        weights = {"prior": 24, "recent_form": 23, "team_pace": 19, "qualifying": 13, "sprint": 8, "reliability": 7, "strategy": 4, "news_sentiment": 2}
    return [{"factor": key, "weight": value} for key, value in weights.items()]
