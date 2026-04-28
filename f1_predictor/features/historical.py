"""Adapter over the existing F1Client historical/current-season feature shape."""

from __future__ import annotations


class HistoricalFeatureProvider:
    def __init__(self, features: dict | None):
        self._features = features or {}

    def get_features(self) -> dict:
        drivers = self._features.get("drivers") or {}
        constructors = self._features.get("constructors") or {}
        coverage = self._features.get("source_coverage") or {}
        return {
            "drivers": drivers,
            "constructors": constructors,
            "completed_races": self._features.get("completed_races", 0),
            "total_races": self._features.get("total_races", 0),
            "races": self._features.get("races") or {},
            "track_history": self._features.get("track_history") or {},
            "source_coverage": coverage,
            "updated_at": self._features.get("updated_at"),
            "source": "f1_client_live_multi_season_features",
            "confidence": _coverage_confidence(drivers, constructors, coverage),
            "missing_data": not bool(drivers),
        }


def _coverage_confidence(drivers: dict, constructors: dict, coverage: dict) -> float:
    current = min(1.0, float(coverage.get("current_season_races") or 0) / 8.0)
    historical = min(1.0, float(coverage.get("historical_races") or 0) / 24.0)
    entity = min(1.0, (len(drivers) + len(constructors)) / 30.0)
    return round(0.20 + 0.42 * current + 0.23 * historical + 0.15 * entity, 4)
