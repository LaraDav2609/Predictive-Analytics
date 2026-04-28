"""Tire and degradation assumptions from track traits and OpenF1 stints."""

from __future__ import annotations


class TireFeatureProvider:
    def get_features(self, track: dict | None = None, openf1_session: dict | None = None) -> dict:
        stints = ((openf1_session or {}).get("stints") or {}).get("drivers") or {}
        if stints:
            avg_stints = [float(item.get("avg_stint_laps") or 0) for item in stints.values() if item.get("avg_stint_laps")]
            compounds = sorted({compound for item in stints.values() for compound in (item.get("compounds") or [])})
            avg_len = sum(avg_stints) / len(avg_stints) if avg_stints else 18.0
            deg = max(0.18, min(0.88, 1.0 - avg_len / 38.0))
            return {
                "compound_set": compounds or _compound_set(track),
                "degradation_rate": round(deg, 4),
                "track_temperature_effect": 0.0,
                "undercut_strength": round(max(0.25, min(0.82, deg + 0.12)), 4),
                "overcut_strength": round(max(0.15, min(0.62, 0.58 - deg * 0.40)), 4),
                "avg_stint_laps": round(avg_len, 2),
                "source": "openf1_stints",
                "confidence": 0.76,
                "missing_data": False,
            }

        tire_stress = float((track or {}).get("tire_stress") or 0.50)
        degradation = max(0.18, min(0.90, tire_stress))
        return {
            "compound_set": _compound_set(track),
            "degradation_rate": round(degradation, 4),
            "track_temperature_effect": 0.0,
            "undercut_strength": round(max(0.28, min(0.78, degradation + 0.08)), 4),
            "overcut_strength": round(max(0.16, min(0.56, 0.52 - degradation * 0.32)), 4),
            "source": "track_tire_trait_registry",
            "confidence": float((track or {}).get("confidence") or 0.35),
            "missing_data": not bool(track) or bool((track or {}).get("missing_data")),
        }


def _compound_set(track: dict | None) -> list[str]:
    stress = float((track or {}).get("tire_stress") or 0.50)
    if stress >= 0.68:
        return ["C1", "C2", "C3"]
    if stress <= 0.40:
        return ["C3", "C4", "C5"]
    return ["C2", "C3", "C4"]
