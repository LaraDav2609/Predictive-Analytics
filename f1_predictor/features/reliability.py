"""Reliability and DNF risk provider."""

from __future__ import annotations


class ReliabilityFeatureProvider:
    def __init__(self, driver_features: dict | None = None):
        self._driver_features = driver_features or {}

    def get_features(self, track: dict | None = None, weather: dict | None = None) -> dict[str, dict]:
        track = track or {}
        weather = weather or {}
        track_incident = float(track.get("safety_car_probability") or 0.30) * 0.08
        weather_incident = float(weather.get("chaos_score") or 0.0) * 0.12
        rows: dict[str, dict] = {}
        for driver_id, feature in self._driver_features.items():
            reliability = float((feature or {}).get("reliability_score") or 0.75)
            starts = max(1.0, float((feature or {}).get("recent_starts") or 1.0))
            mechanical = float((feature or {}).get("mechanical_dnfs") or 0.0) / starts
            incident = float((feature or {}).get("incident_dnfs") or 0.0) / starts
            mechanical_probability = max(0.01, min(0.22, (1.0 - reliability) * 0.55 + mechanical * 0.35))
            incident_probability = max(0.01, min(0.26, incident * 0.32 + track_incident + weather_incident))
            penalty_probability = max(0.01, min(0.18, 0.04 + incident * 0.14 + weather_incident * 0.35))
            dnf_probability = max(0.01, min(0.45, mechanical_probability + incident_probability * 0.72))
            rows[driver_id] = {
                "finish_probability": round(1.0 - dnf_probability, 4),
                "dnf_probability": round(dnf_probability, 4),
                "mechanical_dnf_probability": round(mechanical_probability, 4),
                "incident_dnf_probability": round(incident_probability, 4),
                "penalty_probability": round(penalty_probability, 4),
                "reliability_score": round(reliability, 4),
                "source": "historical_status_track_weather_reliability",
                "confidence": 0.68 if starts >= 3 else 0.42,
                "missing_data": not bool(feature),
            }
        return rows
