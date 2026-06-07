"""Source reliability and influence policy for F1 prediction inputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourcePolicy:
    source: str
    source_type: str
    reliability: float
    freshness_seconds: int | None
    max_influence: float
    description: str


SOURCE_POLICIES: dict[str, SourcePolicy] = {
    "f1_client_official_results": SourcePolicy("f1_client_official_results", "official_results", 0.96, None, 1.0, "Official completed-session results."),
    "official_results": SourcePolicy("official_results", "official_results", 0.96, None, 1.0, "Official completed-session results."),
    "openf1_session_facts": SourcePolicy("openf1_session_facts", "timing_telemetry", 0.86, 120, 0.75, "OpenF1 timing, laps, stints, and positions."),
    "openf1": SourcePolicy("openf1", "timing_telemetry", 0.86, 120, 0.75, "OpenF1 timing, laps, stints, and positions."),
    "live_session_engine": SourcePolicy("live_session_engine", "live_state", 0.82, 45, 0.80, "Normalized in-process live state."),
    "fastf1_recorded_file": SourcePolicy("fastf1_recorded_file", "recorded_timing", 0.78, None, 0.65, "Locally recorded live timing file."),
    "open_meteo_session_weather": SourcePolicy("open_meteo_session_weather", "weather", 0.72, 21600, 0.08, "Open-Meteo session forecast or historical weather."),
    "openf1_weather": SourcePolicy("openf1_weather", "weather", 0.82, 900, 0.10, "OpenF1 session weather rows."),
    "weather_fallback": SourcePolicy("weather_fallback", "weather", 0.25, None, 0.02, "Neutral weather fallback."),
    "f1_sentiment": SourcePolicy("f1_sentiment", "sentiment", 0.48, 86400, 0.035, "Race-aware RSS/news sentiment."),
    "market_consensus": SourcePolicy("market_consensus", "market", 0.58, 900, 0.045, "Free market consensus signal."),
    "estimated_standings_order": SourcePolicy("estimated_standings_order", "estimated", 0.24, None, 0.20, "Standings/model fallback when facts are missing."),
    "model": SourcePolicy("model", "model", 0.45, None, 0.50, "Internal model projection."),
    "unavailable": SourcePolicy("unavailable", "unavailable", 0.0, None, 0.0, "No usable source."),
}


MODE_TO_SOURCE = {
    "live": "live_session_engine",
    "recent": "openf1_session_facts",
    "recording_pending": "fastf1_recorded_file",
    "recorded": "fastf1_recorded_file",
    "recorded_confident": "fastf1_recorded_file",
    "historical": "f1_client_official_results",
    "estimated": "estimated_standings_order",
    "unavailable": "unavailable",
}


def normalize_source(value: str | None) -> str:
    text = (value or "unavailable").strip().lower().replace(" ", "_").replace("-", "_")
    if text in MODE_TO_SOURCE:
        return MODE_TO_SOURCE[text]
    if text in SOURCE_POLICIES:
        return text
    if "openf1" in text and "weather" in text:
        return "openf1_weather"
    if "openf1" in text:
        return "openf1_session_facts"
    if "meteo" in text:
        return "open_meteo_session_weather"
    if "sentiment" in text or "news" in text:
        return "f1_sentiment"
    if "market" in text or "kalshi" in text or "polymarket" in text:
        return "market_consensus"
    if "estimated" in text:
        return "estimated_standings_order"
    return "model"


def get_source_policy(value: str | None) -> SourcePolicy:
    return SOURCE_POLICIES.get(normalize_source(value), SOURCE_POLICIES["model"])


def policy_payload(value: str | None) -> dict:
    policy = get_source_policy(value)
    return {
        "source": policy.source,
        "type": policy.source_type,
        "reliability": policy.reliability,
        "freshness_seconds": policy.freshness_seconds,
        "max_influence": policy.max_influence,
        "description": policy.description,
    }
