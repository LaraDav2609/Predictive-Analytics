"""Circuit trait provider with a real per-circuit registry and fallbacks."""

from __future__ import annotations

from sports.f1.models.f1 import Race


class TrackFeatureProvider:
    def __init__(self, features: dict | None = None):
        self._features = features or {}

    def get_features(self, race: Race | None) -> dict:
        # Local import avoids a module-load cycle (track_archetype imports TRACK_TRAITS).
        from sports.f1.predictor.features.track_archetype import classify_archetypes
        text = f"{getattr(race, 'name', '')} {getattr(race, 'circuit', '')} {getattr(race, 'country', '')}".lower()
        key = _track_key(text)
        registry = TRACK_TRAITS.get(key) or _fallback_traits(text)
        history = ((self._features.get("track_history") or {}).get(key) or {})
        return {
            **registry,
            "archetypes": classify_archetypes(registry),
            "track_key": key,
            "circuit_id": getattr(race, "circuit_id", None),
            "latitude": getattr(race, "latitude", None),
            "longitude": getattr(race, "longitude", None),
            "history": history,
            "source": "circuit_trait_registry" if key in TRACK_TRAITS else "track_trait_fallback",
            "confidence": 0.78 if key in TRACK_TRAITS else 0.35,
            "missing_data": race is None or key not in TRACK_TRAITS,
        }


TRACK_TRAITS = {
    "albertpark": {"street_circuit": True, "high_speed": False, "length_km": 5.278, "laps": 58, "overtaking_difficulty": 0.58, "qualifying_importance": 0.66, "tire_stress": 0.50, "pit_loss": 20.5, "safety_car_probability": 0.45, "drs_zones": 4},
    "shanghai": {"street_circuit": False, "high_speed": False, "length_km": 5.451, "laps": 56, "overtaking_difficulty": 0.44, "qualifying_importance": 0.55, "tire_stress": 0.62, "pit_loss": 22.0, "safety_car_probability": 0.24, "drs_zones": 2},
    "suzuka": {"street_circuit": False, "high_speed": True, "length_km": 5.807, "laps": 53, "overtaking_difficulty": 0.63, "qualifying_importance": 0.72, "tire_stress": 0.72, "pit_loss": 21.5, "safety_car_probability": 0.30, "drs_zones": 1},
    "miami": {"street_circuit": True, "high_speed": False, "length_km": 5.412, "laps": 57, "overtaking_difficulty": 0.54, "qualifying_importance": 0.62, "tire_stress": 0.47, "pit_loss": 22.0, "safety_car_probability": 0.36, "drs_zones": 3},
    "bahrain": {"street_circuit": False, "high_speed": False, "length_km": 5.412, "laps": 57, "overtaking_difficulty": 0.38, "qualifying_importance": 0.52, "tire_stress": 0.66, "pit_loss": 22.5, "safety_car_probability": 0.24, "drs_zones": 3},
    "jeddah": {"street_circuit": True, "high_speed": True, "length_km": 6.174, "laps": 50, "overtaking_difficulty": 0.50, "qualifying_importance": 0.68, "tire_stress": 0.46, "pit_loss": 20.5, "safety_car_probability": 0.58, "drs_zones": 3},
    "gilles": {"street_circuit": True, "high_speed": True, "length_km": 4.361, "laps": 70, "overtaking_difficulty": 0.40, "qualifying_importance": 0.55, "tire_stress": 0.48, "pit_loss": 18.5, "safety_car_probability": 0.52, "drs_zones": 3},
    "monaco": {"street_circuit": True, "high_speed": False, "length_km": 3.337, "laps": 78, "overtaking_difficulty": 0.94, "qualifying_importance": 0.96, "tire_stress": 0.28, "pit_loss": 19.0, "safety_car_probability": 0.64, "drs_zones": 1},
    "barcelona": {"street_circuit": False, "high_speed": False, "length_km": 4.657, "laps": 66, "overtaking_difficulty": 0.56, "qualifying_importance": 0.66, "tire_stress": 0.69, "pit_loss": 22.0, "safety_car_probability": 0.24, "drs_zones": 2},
    "redbullring": {"street_circuit": False, "high_speed": True, "length_km": 4.318, "laps": 71, "overtaking_difficulty": 0.34, "qualifying_importance": 0.50, "tire_stress": 0.55, "pit_loss": 20.0, "safety_car_probability": 0.32, "drs_zones": 3},
    "monza": {"street_circuit": False, "high_speed": True, "length_km": 5.793, "laps": 53, "overtaking_difficulty": 0.35, "qualifying_importance": 0.54, "tire_stress": 0.48, "pit_loss": 24.0, "safety_car_probability": 0.32, "drs_zones": 2},
    "spa": {"street_circuit": False, "high_speed": True, "length_km": 7.004, "laps": 44, "overtaking_difficulty": 0.42, "qualifying_importance": 0.56, "tire_stress": 0.70, "pit_loss": 21.0, "safety_car_probability": 0.43, "drs_zones": 2},
    "silverstone": {"street_circuit": False, "high_speed": True, "length_km": 5.891, "laps": 52, "overtaking_difficulty": 0.48, "qualifying_importance": 0.60, "tire_stress": 0.74, "pit_loss": 20.5, "safety_car_probability": 0.35, "drs_zones": 2},
    "hungaroring": {"street_circuit": False, "high_speed": False, "length_km": 4.381, "laps": 70, "overtaking_difficulty": 0.76, "qualifying_importance": 0.82, "tire_stress": 0.63, "pit_loss": 21.0, "safety_car_probability": 0.30, "drs_zones": 2},
    "zandvoort": {"street_circuit": False, "high_speed": False, "length_km": 4.259, "laps": 72, "overtaking_difficulty": 0.70, "qualifying_importance": 0.80, "tire_stress": 0.70, "pit_loss": 21.5, "safety_car_probability": 0.48, "drs_zones": 2},
    "baku": {"street_circuit": True, "high_speed": True, "length_km": 6.003, "laps": 51, "overtaking_difficulty": 0.40, "qualifying_importance": 0.67, "tire_stress": 0.38, "pit_loss": 21.0, "safety_car_probability": 0.66, "drs_zones": 2},
    "marinabay": {"street_circuit": True, "high_speed": False, "length_km": 4.94, "laps": 62, "overtaking_difficulty": 0.78, "qualifying_importance": 0.86, "tire_stress": 0.64, "pit_loss": 28.0, "safety_car_probability": 0.78, "drs_zones": 4},
    "lasvegas": {"street_circuit": True, "high_speed": True, "length_km": 6.201, "laps": 50, "overtaking_difficulty": 0.38, "qualifying_importance": 0.54, "tire_stress": 0.42, "pit_loss": 21.5, "safety_car_probability": 0.42, "drs_zones": 2},
    "cota": {"street_circuit": False, "high_speed": False, "length_km": 5.513, "laps": 56, "overtaking_difficulty": 0.42, "qualifying_importance": 0.57, "tire_stress": 0.67, "pit_loss": 20.5, "safety_car_probability": 0.28, "drs_zones": 2},
    "mexico": {"street_circuit": False, "high_speed": False, "length_km": 4.304, "laps": 71, "overtaking_difficulty": 0.46, "qualifying_importance": 0.58, "tire_stress": 0.52, "pit_loss": 22.0, "safety_car_probability": 0.34, "drs_zones": 3},
    "interlagos": {"street_circuit": False, "high_speed": False, "length_km": 4.309, "laps": 71, "overtaking_difficulty": 0.38, "qualifying_importance": 0.52, "tire_stress": 0.60, "pit_loss": 20.0, "safety_car_probability": 0.50, "drs_zones": 2},
    "losail": {"street_circuit": False, "high_speed": True, "length_km": 5.419, "laps": 57, "overtaking_difficulty": 0.58, "qualifying_importance": 0.66, "tire_stress": 0.82, "pit_loss": 24.0, "safety_car_probability": 0.28, "drs_zones": 1},
    "yasmarina": {"street_circuit": False, "high_speed": False, "length_km": 5.281, "laps": 58, "overtaking_difficulty": 0.46, "qualifying_importance": 0.58, "tire_stress": 0.50, "pit_loss": 21.0, "safety_car_probability": 0.28, "drs_zones": 2},
    "imola": {"street_circuit": False, "high_speed": False, "length_km": 4.909, "laps": 63, "overtaking_difficulty": 0.68, "qualifying_importance": 0.76, "tire_stress": 0.62, "pit_loss": 28.0, "safety_car_probability": 0.42, "drs_zones": 1},
}


ALIASES = {
    "albertpark": ["albert park", "melbourne", "australian"],
    "shanghai": ["shanghai", "chinese"],
    "suzuka": ["suzuka", "japanese"],
    "miami": ["miami"],
    "bahrain": ["bahrain", "sakhir"],
    "jeddah": ["jeddah", "saudi"],
    "gilles": ["gilles villeneuve", "montreal", "canadian"],
    "monaco": ["monaco"],
    "barcelona": ["barcelona", "catalunya", "spanish"],
    "redbullring": ["red bull ring", "spielberg", "austrian"],
    "monza": ["monza"],
    "spa": ["spa", "belgian"],
    "silverstone": ["silverstone", "british"],
    "hungaroring": ["hungaroring", "hungarian"],
    "zandvoort": ["zandvoort", "dutch"],
    "baku": ["baku", "azerbaijan"],
    "marinabay": ["marina bay", "singapore"],
    "lasvegas": ["las vegas"],
    "cota": ["circuit of the americas", "austin", "united states"],
    "mexico": ["mexico", "hermanos rodriguez"],
    "interlagos": ["interlagos", "jose carlos pace", "sao paulo", "brazil"],
    "losail": ["losail", "lusail", "qatar"],
    "yasmarina": ["yas marina", "abu dhabi"],
    "imola": ["imola", "emilia romagna"],
}


def _track_key(text: str) -> str:
    for key, aliases in ALIASES.items():
        if any(alias in text for alias in aliases):
            return key
    return "default"


def _fallback_traits(text: str) -> dict:
    street = any(key in text for key in ["jeddah", "miami", "las vegas", "monaco", "baku", "singapore"])
    high_speed = any(key in text for key in ["monza", "spa", "silverstone", "jeddah", "las vegas"])
    return {
        "street_circuit": street,
        "high_speed": high_speed,
        "length_km": None,
        "laps": None,
        "overtaking_difficulty": 0.72 if street else 0.48,
        "qualifying_importance": 0.82 if street else 0.58,
        "tire_stress": 0.62 if high_speed else 0.50,
        "pit_loss": 23.0,
        "safety_car_probability": 0.55 if street else 0.30,
        "drs_zones": None,
    }
