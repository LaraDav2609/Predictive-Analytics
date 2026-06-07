"""Weather feature provider using OpenF1/Open-Meteo enriched snapshots."""

from __future__ import annotations


class WeatherFeatureProvider:
    def __init__(self, features: dict | None = None):
        self._features = features or {}

    def get_features(self, race=None, openf1_session: dict | None = None, session: str = "race") -> dict:
        session_key = _session_key(session)
        openf1_weather = ((openf1_session or {}).get("weather") or {})
        if openf1_weather and not openf1_weather.get("missing_data"):
            chaos = float(openf1_weather.get("chaos_score") or 0.0)
            return _with_model_impact({
                "rain_probability": float(openf1_weather.get("rain_probability") or 0.0),
                "rain_intensity": 1.0 if float(openf1_weather.get("rain_probability") or 0.0) > 0 else 0.0,
                "air_temperature": openf1_weather.get("air_temperature"),
                "track_temperature": openf1_weather.get("track_temperature"),
                "humidity": openf1_weather.get("humidity"),
                "wind_speed": openf1_weather.get("wind_speed"),
                "wind_gusts": openf1_weather.get("wind_gusts"),
                "cloud_cover": openf1_weather.get("cloud_cover"),
                "weather_change_probability": round(min(1.0, chaos * 0.80 + 0.10), 4),
                "chaos_score": chaos,
                "source": "openf1_weather",
                "session": session_key,
                "confidence": 0.82,
                "missing_data": False,
            })

        race_key = str(getattr(race, "round", "") or "")
        session_weather = ((self._features.get("weather_by_round_session") or {}).get(race_key) or {})
        weather = session_weather.get(session_key) or session_weather.get(_session_fallback(session_key)) or {}
        if weather and not weather.get("missing_data"):
            return _with_model_impact({**weather, "session": session_key})

        weather = (self._features.get("weather_by_round") or {}).get(race_key) or {}
        if weather and not weather.get("missing_data"):
            return _with_model_impact({**weather, "session": session_key, "source": weather.get("source") or "open_meteo_race_day"})

        return _with_model_impact({
            "rain_probability": 0.0,
            "rain_intensity": 0.0,
            "air_temperature": None,
            "track_temperature": None,
            "humidity": None,
            "wind_speed": None,
            "wind_gusts": None,
            "cloud_cover": None,
            "weather_change_probability": 0.0,
            "chaos_score": 0.0,
            "source": "weather_fallback",
            "session": session_key,
            "reason": "session_weather_unavailable",
            "confidence": 0.0,
            "missing_data": True,
        })


def _with_model_impact(weather: dict) -> dict:
    rain_probability = _num(weather.get("rain_probability"))
    rain_intensity = _num(weather.get("rain_intensity"))
    wind_gusts = _num(weather.get("wind_gusts"), _num(weather.get("wind_speed")))
    air_temp = _maybe_num(weather.get("air_temperature"))
    track_temp = _maybe_num(weather.get("track_temperature"))
    chaos = _num(weather.get("chaos_score"))
    temp = track_temp if track_temp is not None else air_temp
    hot_tire_stress = max(0.0, min(1.0, ((temp or 0.0) - 30.0) / 18.0)) if temp is not None else 0.0
    cold_warmup = max(0.0, min(1.0, (12.0 - (temp or 12.0)) / 12.0)) if temp is not None else 0.0
    wind_volatility = max(0.0, min(1.0, wind_gusts / 70.0))
    rain_volatility = max(0.0, min(1.0, rain_probability * 0.55 + rain_intensity * 0.35))
    model_delta_hint = min(0.08, chaos * 0.045 + rain_volatility * 0.035 + wind_volatility * 0.02 + hot_tire_stress * 0.015 + cold_warmup * 0.012)
    explanations = []
    if rain_probability >= 0.25 or rain_intensity >= 0.05:
        explanations.append("rain volatility increased incident and strategy risk")
    if hot_tire_stress >= 0.15:
        explanations.append("high temperature increased tire degradation pressure")
    if cold_warmup >= 0.15:
        explanations.append("cold conditions reduced tire warmup confidence")
    if wind_gusts >= 35.0:
        explanations.append("wind gusts increased car stability uncertainty")
    if not explanations:
        explanations.append("weather impact is currently low")

    return {
        **weather,
        "rain_probability": round(rain_probability, 4),
        "rain_intensity": round(rain_intensity, 4),
        "wind_gusts": weather.get("wind_gusts"),
        "chaos_score": round(chaos, 4),
        "weather_model_impact": {
            "rain_volatility": round(rain_volatility, 4),
            "wind_volatility": round(wind_volatility, 4),
            "temperature_tire_stress": round(hot_tire_stress, 4),
            "cold_warmup_risk": round(cold_warmup, 4),
            "model_delta_hint": round(model_delta_hint, 4),
        },
        "weather_explanations": explanations,
    }


def _session_key(session: str | None) -> str:
    value = str(session or "race").lower().replace("-", "_").replace(" ", "_")
    if value.startswith("qual"):
        return "qualifying"
    if value.startswith("sprint_qual"):
        return "sprint_qualifying"
    if value.startswith("sprint"):
        return "sprint"
    if value.startswith("fp1") or "practice_1" in value:
        return "fp1"
    if value.startswith("fp2") or "practice_2" in value:
        return "fp2"
    if value.startswith("fp3") or "practice_3" in value:
        return "fp3"
    return "race"


def _session_fallback(session_key: str) -> str:
    if session_key == "sprint_qualifying":
        return "qualifying"
    if session_key.startswith("fp"):
        return "race"
    return "race"


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _maybe_num(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
