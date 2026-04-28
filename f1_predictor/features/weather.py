"""Weather feature provider using OpenF1/Open-Meteo enriched snapshots."""

from __future__ import annotations


class WeatherFeatureProvider:
    def __init__(self, features: dict | None = None):
        self._features = features or {}

    def get_features(self, race=None, openf1_session: dict | None = None) -> dict:
        openf1_weather = ((openf1_session or {}).get("weather") or {})
        if openf1_weather and not openf1_weather.get("missing_data"):
            chaos = float(openf1_weather.get("chaos_score") or 0.0)
            return {
                "rain_probability": float(openf1_weather.get("rain_probability") or 0.0),
                "rain_intensity": 1.0 if float(openf1_weather.get("rain_probability") or 0.0) > 0 else 0.0,
                "air_temperature": openf1_weather.get("air_temperature"),
                "track_temperature": openf1_weather.get("track_temperature"),
                "humidity": openf1_weather.get("humidity"),
                "wind_speed": openf1_weather.get("wind_speed"),
                "weather_change_probability": round(min(1.0, chaos * 0.80 + 0.10), 4),
                "chaos_score": chaos,
                "source": "openf1_weather",
                "confidence": 0.82,
                "missing_data": False,
            }

        race_key = str(getattr(race, "round", "") or "")
        weather = (self._features.get("weather_by_round") or {}).get(race_key) or {}
        if weather and not weather.get("missing_data"):
            return weather

        return {
            "rain_probability": 0.0,
            "rain_intensity": 0.0,
            "air_temperature": None,
            "track_temperature": None,
            "humidity": None,
            "wind_speed": None,
            "weather_change_probability": 0.0,
            "chaos_score": 0.0,
            "source": "weather_fallback",
            "confidence": 0.0,
            "missing_data": True,
        }
