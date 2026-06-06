"""Driver skill, car performance, and combined performance scoring."""

from __future__ import annotations

from models.f1 import Constructor, Driver
from f1_predictor.scoring.normalization import clamp01, num


class PerformanceFeatureProvider:
    def __init__(
        self,
        drivers: list[Driver],
        constructors: list[Constructor],
        driver_features: dict | None = None,
        constructor_features: dict | None = None,
    ):
        self._drivers = drivers
        self._constructors = constructors
        self._driver_features = driver_features or {}
        self._constructor_features = constructor_features or {}

    def get_features(self) -> dict[str, dict]:
        return build_performance_table(
            self._drivers,
            self._constructors,
            self._driver_features,
            self._constructor_features,
        )


def build_performance_table(
    drivers: list[Driver],
    constructors: list[Constructor],
    driver_features: dict | None = None,
    constructor_features: dict | None = None,
) -> dict[str, dict]:
    driver_features = driver_features or {}
    constructor_features = constructor_features or {}
    max_points = max((driver.points for driver in drivers), default=1.0) or 1.0
    max_constructor_points = max((constructor.points for constructor in constructors), default=1.0) or 1.0
    rows: dict[str, dict] = {}

    for driver in drivers:
        feature = driver_features.get(driver.id) or {}
        constructor = next((c for c in constructors if c.name.lower() == driver.team.lower()), None)
        constructor_feature = constructor_features.get(driver.team.lower()) or {}

        points_strength = driver.points / max_points
        position_strength = max(0.0, 1.0 - (driver.position - 1) * 0.08) if driver.position else 0.3
        standing_score = 0.64 * points_strength + 0.36 * position_strength
        form_score = num(feature.get("form_score"), standing_score)
        reliability_score = num(feature.get("reliability_score"), 0.75)
        recent_wins = num(feature.get("recent_wins"), 0.0)
        recent_podiums = num(feature.get("recent_podiums"), 0.0)
        recent_starts = max(1.0, num(feature.get("recent_starts"), 1.0))
        starts = num(feature.get("starts"), 0.0)
        qualifying_pace_score = num(feature.get("qualifying_pace_score"), 0.48)
        race_pace_score = num(feature.get("race_pace_score"), form_score)
        teammate_score = num(feature.get("teammate_score"), 0.50)
        trend_score = num(feature.get("trend_score"), 0.50)

        conversion_score = min(1.0, (recent_wins * 0.28 + recent_podiums * 0.16) / min(4.0, recent_starts))
        experience_score = min(1.0, starts / 80.0)
        driver_skill_score = clamp01(
            0.28 * form_score
            + 0.17 * reliability_score
            + 0.05 * standing_score
            + 0.10 * conversion_score
            + 0.08 * experience_score
            + 0.15 * qualifying_pace_score
            + 0.12 * race_pace_score
            + 0.05 * teammate_score
        )

        constructor_standing = (constructor.points / max_constructor_points) if constructor else standing_score
        team_score = num(constructor_feature.get("team_score"), constructor_standing)
        recent_team_points = num(constructor_feature.get("recent_points"), 0.0)
        team_points_score = min(1.0, recent_team_points / 344.0)
        car_performance_score = clamp01(
            0.50 * team_score
            + 0.14 * constructor_standing
            + 0.36 * team_points_score
        )

        momentum = clamp01(0.50 + (trend_score - 0.50) * 0.50)
        combined_base = (driver_skill_score ** 0.52) * (max(car_performance_score, 0.01) ** 0.38) * (max(momentum, 0.01) ** 0.10)
        performance_score = clamp01(combined_base * (0.90 + 0.10 * reliability_score))
        rows[driver.id] = {
            "driver_id": driver.id,
            "driver_name": f"{driver.first_name} {driver.last_name}",
            "code": driver.code,
            "team": driver.team,
            "position": driver.position,
            "points": driver.points,
            "driver_skill_score": round(driver_skill_score, 4),
            "car_performance_score": round(car_performance_score, 4),
            "performance_score": round(performance_score, 4),
            "form_score": round(form_score, 4),
            "qualifying_pace_score": round(qualifying_pace_score, 4),
            "race_pace_score": round(race_pace_score, 4),
            "teammate_score": round(teammate_score, 4),
            "trend_score": round(trend_score, 4),
            "reliability_score": round(reliability_score, 4),
            "standing_score": round(standing_score, 4),
            "recent_wins": int(recent_wins),
            "recent_podiums": int(recent_podiums),
            "recent_avg_finish": feature.get("recent_avg_finish"),
            "recent_summary": feature.get("recent_summary"),
            "car_recent_points": round(recent_team_points, 2),
            "car_team_score": round(team_score, 4),
        }
    return rows
