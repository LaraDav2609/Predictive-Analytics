from __future__ import annotations

from datetime import datetime, timezone

from sports.f1.models.f1 import Constructor, Driver, Race
from sports.f1.predictor.agents.car_performance import build_car_performance_agent


def _fixtures():
    race = Race(
        round=6,
        name="Monaco Grand Prix",
        circuit="Circuit de Monaco",
        country="Monaco",
        date=datetime(2026, 6, 7, tzinfo=timezone.utc),
    )
    drivers = [
        Driver(id="russell", number=63, code="RUS", first_name="George", last_name="Russell", nationality="British", team="Mercedes", points=88, position=2),
        Driver(id="antonelli", number=12, code="ANT", first_name="Kimi", last_name="Antonelli", nationality="Italian", team="Mercedes", points=131, position=1),
        Driver(id="leclerc", number=16, code="LEC", first_name="Charles", last_name="Leclerc", nationality="Monegasque", team="Ferrari", points=75, position=3),
        Driver(id="hamilton", number=44, code="HAM", first_name="Lewis", last_name="Hamilton", nationality="British", team="Ferrari", points=72, position=4),
    ]
    constructors = [
        Constructor(id="mercedes", name="Mercedes", nationality="German", points=219, position=1),
        Constructor(id="ferrari", name="Ferrari", nationality="Italian", points=147, position=2),
    ]
    features = {
        "drivers": {
            "russell": {"qualifying_pace_score": 0.78, "race_pace_score": 0.76, "reliability_score": 0.90},
            "antonelli": {"qualifying_pace_score": 0.82, "race_pace_score": 0.80, "reliability_score": 0.86},
            "leclerc": {"qualifying_pace_score": 0.80, "race_pace_score": 0.72, "reliability_score": 0.82},
            "hamilton": {"qualifying_pace_score": 0.77, "race_pace_score": 0.74, "reliability_score": 0.84},
        },
        "constructors": {
            "mercedes": {"team_score": 0.88, "recent_points": 168, "reliability_score": 0.89},
            "ferrari": {"team_score": 0.78, "recent_points": 120, "reliability_score": 0.83},
        },
    }
    return race, drivers, constructors, features


def test_car_performance_agent_returns_key_analysis_groups_with_fallbacks():
    race, drivers, constructors, features = _fixtures()
    payload = build_car_performance_agent(
        race=race,
        drivers=drivers,
        constructors=constructors,
        features=features,
        session="qualifying",
        track={"street_circuit": True, "qualifying_importance": 0.96, "tire_stress": 0.28, "source": "test_track"},
        weather={"chaos_score": 0.05, "source": "test_weather"},
        tires={"degradation_rate": 0.28},
    )

    assert payload["ok"] is True
    assert payload["agent_id"] == "car_performance_analyst_v1"
    assert payload["source_mode"] in {"estimated", "historical"}
    mercedes = next(item for item in payload["constructors"] if item["constructor_id"] == "mercedes")
    assert mercedes["low_speed"]["source"] == "track_traits_fallback"
    assert mercedes["top_speed"]["source"] == "car_model_fallback"
    assert mercedes["sector_strengths"]["source"] == "track_traits_fallback"
    assert mercedes["teammate_deltas"]["available"] is False
    assert "prediction_weights_changed" in payload["constraints"]


def test_car_performance_agent_uses_openf1_stints_car_data_and_teammate_delta():
    race, drivers, constructors, features = _fixtures()
    openf1_session = {
        "laps": {
            "drivers": {
                "63": {"representative_lap": 72.1, "laps": 18, "sector_1": 23.1, "sector_2": 25.0, "sector_3": 24.0},
                "12": {"representative_lap": 72.0, "laps": 20, "sector_1": 23.0, "sector_2": 24.9, "sector_3": 24.1},
                "16": {"representative_lap": 72.9, "laps": 17, "sector_1": 23.5, "sector_2": 25.4, "sector_3": 24.0},
                "44": {"representative_lap": 72.8, "laps": 18, "sector_1": 23.4, "sector_2": 25.2, "sector_3": 24.2},
            }
        },
        "stints": {
            "drivers": {
                "63": {"avg_stint_laps": 24, "compounds": ["MEDIUM"]},
                "12": {"avg_stint_laps": 25, "compounds": ["MEDIUM"]},
                "16": {"avg_stint_laps": 19, "compounds": ["SOFT"]},
                "44": {"avg_stint_laps": 20, "compounds": ["SOFT"]},
            }
        },
        "car_data": {
            "drivers": {
                "63": {"max_speed": 296, "drs_usage": 0.20},
                "12": {"max_speed": 298, "drs_usage": 0.22},
                "16": {"max_speed": 290, "drs_usage": 0.18},
                "44": {"max_speed": 291, "drs_usage": 0.18},
            }
        },
        "raw_counts": {"laps": 80, "stints": 4, "car_data": 400},
    }
    payload = build_car_performance_agent(
        race=race,
        drivers=drivers,
        constructors=constructors,
        features={**features, "openf1_session": openf1_session},
        session="race",
        constructor_id="mercedes",
        track={"street_circuit": True, "qualifying_importance": 0.96, "tire_stress": 0.28, "source": "test_track"},
        weather={"chaos_score": 0.04, "source": "test_weather"},
        tires={"degradation_rate": 0.28},
        openf1_session=openf1_session,
        weekend_evidence={"source_mode": "recent", "confidence": 0.55},
    )

    assert payload["ok"] is True
    assert len(payload["constructors"]) == 1
    mercedes = payload["constructors"][0]
    assert mercedes["top_speed"]["source"] == "openf1_car_data"
    assert mercedes["tyre_degradation"]["source"] == "openf1_stints"
    assert mercedes["teammate_deltas"]["available"] is True
    assert mercedes["teammate_deltas"]["delta_seconds"] == 0.1
    assert payload["source_mode"] == "recent"


def test_car_performance_agent_missing_constructor_is_safe():
    race, drivers, constructors, features = _fixtures()
    payload = build_car_performance_agent(
        race=race,
        drivers=drivers,
        constructors=constructors,
        features=features,
        constructor_id="does-not-exist",
    )

    assert payload["ok"] is False
    assert payload["code"] == "constructor_car_performance_agent_not_found"
