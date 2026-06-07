"""Human-readable explanation builders for F1 predictions."""

from __future__ import annotations

from models.f1 import Driver
from f1_predictor.scoring.normalization import label_from_score


def explain_driver(driver: Driver, component: dict) -> list[str]:
    notes = []
    if component["form_score"] >= 0.72:
        notes.append("strong recent race form")
    elif component["form_score"] <= 0.35:
        notes.append("recent results are limiting the projection")
    if component["team_score"] >= 0.70:
        notes.append(f"{driver.team} has front-running team strength")
    if component.get("track_fit_score", 0) >= 0.70:
        notes.append("track profile matches driver and car strengths")
    if component.get("race_pace_score", 0) >= 0.70:
        notes.append("race pace trend is strong")
    if component["reliability_score"] < 0.65:
        notes.append("reliability risk is pulling the forecast down")
    if component.get("dnf_probability", 0) >= 0.18:
        notes.append("DNF risk is meaningful for this session")
    if component.get("weather_risk_score", 1) <= 0.70:
        notes.append("weather volatility is reducing confidence")
    if component["sentiment_score"] > 0.12:
        notes.append("driver/team news flow is positive")
    elif component["sentiment_score"] < -0.12:
        notes.append("driver/team news flow is negative")
    elif component.get("sentiment_mentions"):
        notes.append(f"{component['sentiment_mentions']} recent news signals folded in")
    return notes[:3]


def simulation_signals(
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
        notes.append("pre-session estimate from race pace, form, team pace, reliability, and bounded championship context")
    return notes[:4]


def sentiment_label(score: float) -> str:
    return label_from_score(score)
