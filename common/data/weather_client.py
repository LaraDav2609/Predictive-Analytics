"""Small Open-Meteo client used by F1 feature enrichment."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

FORECAST_BASE_URL = "https://api.open-meteo.com/v1"
HISTORICAL_FORECAST_BASE_URL = "https://historical-forecast-api.open-meteo.com/v1"


class OpenMeteoClient:
    def __init__(self) -> None:
        self._forecast = httpx.AsyncClient(base_url=FORECAST_BASE_URL, timeout=20.0)
        self._historical = httpx.AsyncClient(base_url=HISTORICAL_FORECAST_BASE_URL, timeout=25.0)
        self._cache: dict[tuple[float, float, str, str], dict[str, Any]] = {}

    async def close(self) -> None:
        await self._forecast.aclose()
        await self._historical.aclose()

    async def get_race_weather(self, latitude: float | None, longitude: float | None, at: datetime | None) -> dict[str, Any]:
        return await self.get_session_weather(latitude, longitude, at, session="race_day")

    async def get_session_weather(
        self,
        latitude: float | None,
        longitude: float | None,
        at: datetime | None,
        session: str = "race",
    ) -> dict[str, Any]:
        if latitude is None or longitude is None or at is None:
            result = _fallback("missing_coordinates")
            result["session"] = _session_key(session)
            result["requested_at"] = at.isoformat() if at else None
            return result

        at_utc = at.astimezone(timezone.utc) if at.tzinfo else at.replace(tzinfo=timezone.utc)
        date_key = at_utc.date().isoformat()
        hour_key = at_utc.replace(minute=0, second=0, microsecond=0).isoformat()
        cache_key = (round(float(latitude), 4), round(float(longitude), 4), hour_key, _session_key(session))
        if cache_key in self._cache:
            return self._cache[cache_key]

        now = datetime.now(timezone.utc)
        historical = at_utc.date() < now.date()
        client = self._historical if historical else self._forecast
        path = "/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date_key,
            "end_date": date_key,
            "timezone": "UTC",
            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation_probability",
                "precipitation",
                "rain",
                "wind_speed_10m",
                "wind_gusts_10m",
                "cloud_cover",
            ]),
        }
        if not historical:
            params["forecast_days"] = 16
            params.pop("start_date", None)
            params.pop("end_date", None)

        try:
            response = await client.get(path, params=params)
            response.raise_for_status()
            result = _parse_open_meteo(response.json(), at_utc, historical)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("Open-Meteo request failed for %.4f, %.4f at %s: %s", latitude, longitude, date_key, exc)
            result = _fallback("open_meteo_unavailable")

        result["session"] = _session_key(session)
        result["requested_at"] = at_utc.isoformat()
        self._cache[cache_key] = result
        return result


def _parse_open_meteo(data: dict[str, Any], at: datetime, historical: bool) -> dict[str, Any]:
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return _fallback("open_meteo_empty")

    target_hour = at.replace(minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "")
    index = 0
    if target_hour in times:
        index = times.index(target_hour)
    else:
        index = min(range(len(times)), key=lambda idx: abs(_parse_hour(times[idx]) - at.timestamp()))

    def value(key: str, default: float | None = None) -> float | None:
        values = hourly.get(key) or []
        try:
            item = values[index]
            return float(item) if item is not None else default
        except (IndexError, TypeError, ValueError):
            return default

    rain_probability = value("precipitation_probability", 0.0) or 0.0
    precipitation = value("precipitation", 0.0) or 0.0
    rain = value("rain", 0.0) or 0.0
    wind_speed = value("wind_speed_10m", 0.0) or 0.0
    wind_gusts = value("wind_gusts_10m", wind_speed) or wind_speed
    cloud = value("cloud_cover", 0.0) or 0.0
    chaos = min(1.0, (rain_probability / 100.0) * 0.45 + min(1.0, rain / 6.0) * 0.30 + min(1.0, wind_gusts / 55.0) * 0.15 + (cloud / 100.0) * 0.10)
    return {
        "rain_probability": round(rain_probability / 100.0, 4),
        "rain_intensity": round(min(1.0, max(precipitation, rain) / 8.0), 4),
        "air_temperature": value("temperature_2m"),
        "track_temperature": None,
        "humidity": value("relative_humidity_2m"),
        "wind_speed": wind_speed,
        "wind_gusts": wind_gusts,
        "cloud_cover": cloud,
        "weather_change_probability": round(min(1.0, chaos * 0.72 + rain_probability / 100.0 * 0.28), 4),
        "chaos_score": round(chaos, 4),
        "source": "open_meteo_historical_forecast" if historical else "open_meteo_forecast",
        "confidence": 0.74 if historical else 0.62,
        "missing_data": False,
    }


def _parse_hour(value: str) -> float:
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def _fallback(reason: str) -> dict[str, Any]:
    return {
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
        "reason": reason,
        "confidence": 0.0,
        "missing_data": True,
    }


def _session_key(session: str | None) -> str:
    value = str(session or "race").lower().replace("-", "_")
    if value.startswith("qual"):
        return "qualifying"
    if value.startswith("sprint_quali") or value in {"sq", "sprint_shootout"}:
        return "sprint_qualifying"
    if value.startswith("sprint"):
        return "sprint"
    if value.startswith("fp") or value.startswith("practice"):
        return value
    if value == "race_day":
        return value
    return "race"
