"""Formula 1 session simulator for GP-specific probability learning."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from models.f1 import Constructor, Driver, Race


def build_session_simulation(
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
    """Return explainable win probabilities for a race-weekend session.

    The simulator intentionally returns the component scores. This makes it useful
    for learning how the final winner probability is assembled.
    """
    session = (session or "race").lower()
    if session not in {"race", "qualifying", "sprint"}:
        session = "race"

    prediction_items = (prediction or {}).get("driver_predictions") or {}
    driver_features = (features or {}).get("drivers") or {}
    constructor_features = (features or {}).get("constructors") or {}
    by_driver = {driver.id: driver for driver in drivers}
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
        team = float(constructor_feature.get("team_score") or base_prediction.get("team_score") or ((constructor.points / max_constructor_points) if constructor else 0.45))
        sentiment = float(base_prediction.get("sentiment_score") or 0.0)
        sentiment_label = base_prediction.get("sentiment_label") or _sentiment_label(sentiment)
        sentiment_mentions = int(base_prediction.get("sentiment_mentions") or 0)
        personal_news = float(base_prediction.get("personal_news_score") or 0.0)
        team_news = float(base_prediction.get("team_news_score") or 0.0)
        overall_news = float(base_prediction.get("overall_news_score") or 0.0)
        news_win_modifier = float(base_prediction.get("news_win_modifier") or 1.0)
        wdc_probability = float(base_prediction.get("wdc_prob") or 0.0)
        wdc_modifier = float(base_prediction.get("wdc_modifier") or 1.0)
        qualifying_score = _position_score((quali_by_driver.get(driver.id) or {}).get("position"), default=0.46)
        sprint_score = _position_score((sprint_by_driver.get(driver.id) or {}).get("position"), default=0.48)
        live_score = _position_score((race_by_driver.get(driver.id) or {}).get("position"), default=0.50)
        strategy = _strategy_score(driver, feature, qualifying_score, sprint_score, live_score, live)

        if session == "qualifying":
            strength = (
                0.26 * _prior_score(prior, drivers)
                + 0.28 * form
                + 0.26 * team
                + 0.12 * qualifying_score
                + 0.05 * reliability
                + 0.03 * _sentiment_factor(sentiment)
            )
        elif session == "sprint":
            strength = (
                0.25 * _prior_score(prior, drivers)
                + 0.23 * form
                + 0.20 * team
                + 0.15 * qualifying_score
                + 0.10 * reliability
                + 0.04 * strategy
                + 0.03 * _sentiment_factor(sentiment)
            )
        elif live and results:
            strength = (
                0.16 * _prior_score(prior, drivers)
                + 0.16 * form
                + 0.14 * team
                + 0.12 * qualifying_score
                + 0.26 * live_score
                + 0.10 * strategy
                + 0.04 * reliability
                + 0.02 * _sentiment_factor(sentiment)
            )
        else:
            strength = (
                0.24 * _prior_score(prior, drivers)
                + 0.23 * form
                + 0.19 * team
                + 0.13 * qualifying_score
                + 0.08 * sprint_score
                + 0.07 * reliability
                + 0.04 * strategy
                + 0.02 * _sentiment_factor(sentiment)
            )

        rows.append({
            "driver_id": driver.id,
            "driver_number": driver.number,
            "driver_code": driver.code,
            "driver_name": f"{driver.first_name} {driver.last_name}".strip(),
            "team": driver.team,
            "sentiment_label": sentiment_label,
            "sentiment_mentions": sentiment_mentions,
            "personal_news_score": round(personal_news, 4),
            "team_news_score": round(team_news, 4),
            "overall_news_score": round(overall_news, 4),
            "news_win_modifier": round(news_win_modifier, 4),
            "wdc_probability": round(wdc_probability, 4),
            "wdc_modifier": round(wdc_modifier, 4),
            "strength": max(strength, 0.001),
            "components": {
                "model_prior": round(_prior_score(prior, drivers), 4),
                "recent_form": round(form, 4),
                "team_pace": round(team, 4),
                "qualifying": round(qualifying_score, 4),
                "sprint": round(sprint_score, 4),
                "live_track_position": round(live_score, 4),
                "strategy": round(strategy, 4),
                "reliability": round(reliability, 4),
                "sentiment": round(sentiment, 4),
                "personal_news": round(personal_news, 4),
                "team_news": round(team_news, 4),
                "overall_news": round(overall_news, 4),
                "news_win_modifier": round(news_win_modifier, 4),
                "wdc_probability": round(wdc_probability, 4),
            },
            "signals": _signals(driver, feature, quali_by_driver, sprint_by_driver, race_by_driver, live, sentiment_label, sentiment_mentions, wdc_probability),
        })

    total = sum(row["strength"] for row in rows) or 1.0
    rows.sort(key=lambda row: row["strength"], reverse=True)
    for index, row in enumerate(rows, start=1):
        probability = row["strength"] / total
        row["rank"] = index
        row["win_probability"] = round(probability, 4)
        row["podium_probability"] = round(min(0.95, probability * 3), 4)
        row["top5_probability"] = round(min(0.98, probability * 5), 4)
        row.pop("strength", None)

    return {
        "ok": True,
        "race": race.model_dump(mode="json"),
        "session": session,
        "live": live,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "f1-session-simulator-v1",
        "status": _status(session, live, qualifying, sprint, results),
        "calculation": _calculation_notes(session, live, bool(results)),
        "simulations": rows,
    }


def _prior_score(prior: float, drivers: list[Driver]) -> float:
    return max(0.02, min(1.0, prior * max(len(drivers), 1)))


def _position_score(position: Any, default: float) -> float:
    try:
        value = int(position)
    except (TypeError, ValueError):
        return default
    return max(0.04, min(1.0, (22 - value) / 21))


def _sentiment_factor(sentiment: float) -> float:
    return max(0.0, min(1.0, 0.5 + sentiment / 2))


def _sentiment_label(sentiment: float) -> str:
    if sentiment > 0.08:
        return "Bullish"
    if sentiment < -0.08:
        return "Bearish"
    return "Neutral"


def _strategy_score(driver: Driver, feature: dict, qualifying: float, sprint: float, live_score: float, live: bool) -> float:
    reliability = float(feature.get("reliability_score") or 0.72)
    form = float(feature.get("form_score") or 0.45)
    base = 0.45 * reliability + 0.25 * form + 0.18 * qualifying + 0.12 * sprint
    if live:
        base = 0.55 * live_score + 0.25 * reliability + 0.20 * form
    if "ferrari" in (driver.team or "").lower() or "mclaren" in (driver.team or "").lower() or "mercedes" in (driver.team or "").lower():
        base += 0.03
    return max(0.02, min(1.0, base))


def _signals(
    driver: Driver,
    feature: dict,
    quali: dict,
    sprint: dict,
    race: dict,
    live: bool,
    sentiment_label: str,
    sentiment_mentions: int,
    wdc_probability: float,
) -> list[str]:
    notes = []
    if feature.get("recent_summary"):
        notes.append(str(feature["recent_summary"]))
    if sentiment_mentions:
        notes.append(f"{sentiment_label.lower()} news flow from {sentiment_mentions} signals")
    if wdc_probability:
        notes.append(f"WDC estimate {(wdc_probability * 100):.1f}%")
    if driver.id in quali:
        notes.append(f"qualifying P{quali[driver.id].get('position')}")
    if driver.id in sprint:
        notes.append(f"sprint P{sprint[driver.id].get('position')}")
    if live and driver.id in race:
        notes.append(f"race position P{race[driver.id].get('position')}")
    if not notes:
        notes.append("pre-session estimate from standings, form, team pace, and reliability")
    return notes[:4]


def _status(session: str, live: bool, qualifying: list[dict], sprint: list[dict], results: list[dict]) -> str:
    if live and results:
        return "live_or_completed_classification"
    if session == "qualifying" and qualifying:
        return "qualifying_result_available"
    if session == "sprint" and sprint:
        return "sprint_result_available"
    if session == "race" and results:
        return "race_result_available"
    return "pre_session_projection"


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
