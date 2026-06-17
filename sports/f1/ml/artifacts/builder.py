"""Small artifact builders used by CLI smoke paths and tests."""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import median, pstdev
from typing import Any

from sports.f1.ml.artifacts.schema import F1MLArtifactBundle, F1MLDriverArtifact, F1MLModelArtifact
from sports.f1.ml.common.types import Race, SessionType
from sports.f1.ml.providers.base import TelemetryProvider
from sports.f1.ml.providers.synthetic_provider import SyntheticProvider


DEFAULT_FEATURE_COLUMNS = [
    "championship_points",
    "championship_position",
    "form_score",
    "driver_skill_score",
    "car_performance_score",
    "performance_score",
    "race_pace_score",
    "reliability_score",
    "track_laps",
    "tire_stress",
    "rain_probability",
    "weather_chaos",
]


def build_synthetic_artifact_bundle(
    provider: SyntheticProvider | None = None,
    *,
    artifact_id: str = "synthetic-ml-simulator-artifact",
    season_start: int | None = None,
    season_end: int | None = None,
    training_races: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
) -> F1MLArtifactBundle:
    provider = provider or SyntheticProvider()
    pace = provider.driver_pace_table()
    sigma = provider.driver_sigma_table()
    dnf = provider.dnf_rate_table()
    drivers = {
        code: F1MLDriverArtifact(
            driver_code=code,
            pace_mean_seconds=float(pace_value),
            pace_sigma_seconds=float(sigma.get(code, 0.35)),
            dnf_hazard_per_lap=float(dnf.get(code, 0.0008)),
            rating_prior=max(0.05, min(0.95, 0.90 - index * 0.025)),
            confidence=0.82,
            sources=["synthetic_training", "artifact_pace", "artifact_dnf"],
        )
        for index, (code, pace_value) in enumerate(pace.items())
    }
    season = provider.config.season
    now = datetime.now(timezone.utc)
    start = int(season_start or season)
    end = int(season_end or season)
    race_ids = training_races or [f"SYN-{start}-00-TRAIN"]
    return F1MLArtifactBundle(
        artifact_id=artifact_id,
        model_version="ml-simulator-artifact-synthetic-v1",
        created_at=now,
        training_seasons=list(range(start, end + 1)),
        training_races=race_ids,
        training_cutoff=provider.config.start_time.replace(tzinfo=timezone.utc)
        if provider.config.start_time.tzinfo is None
        else provider.config.start_time,
        knowable_as_of=now,
        feature_columns=list(DEFAULT_FEATURE_COLUMNS),
        drivers=drivers,
        validation_metrics=metrics or {"source": "synthetic", "mae_seconds": 0.0},
        source_metadata={"provider": "synthetic", "race_id": provider.config.race_id, "training_races": race_ids},
        leakage_status={"status": "guarded", "future_results_excluded": True},
    )


def build_fastf1_artifact_bundle(
    provider: TelemetryProvider,
    *,
    training_races: list[Race],
    target_race_id: str | None = None,
    knowable_as_of: datetime | None = None,
    artifact_id: str | None = None,
    season_start: int | None = None,
    season_end: int | None = None,
    min_laps_per_driver: int = 3,
) -> F1MLArtifactBundle:
    """Build a portable artifact bundle from provider race laps.

    The function is provider-agnostic but named for the FastF1 training path.
    It deliberately stores compact learned rows instead of serialized runtime
    objects so the production API can load it without extra services or unsafe
    pickle dependencies.
    """
    now = datetime.now(timezone.utc)
    cutoff = _aware(knowable_as_of or now)
    leakage = _leakage_status(training_races, target_race_id=target_race_id, cutoff=cutoff)
    training_ids = [_race_id(race) for race in training_races]
    lap_rows: list[dict[str, Any]] = []
    failed_races: list[dict[str, str]] = []
    for race in training_races:
        try:
            laps = provider.laps(race, SessionType.RACE)
        except Exception as exc:
            failed_races.append({"race_id": _race_id(race), "reason": f"{exc.__class__.__name__}: {exc}"})
            continue
        for lap in laps or []:
            lap_time = _float(getattr(lap, "lap_time_s", None))
            if lap_time is None or lap_time <= 0:
                continue
            if bool(getattr(lap, "pit_in", False)) or bool(getattr(lap, "pit_out", False)):
                continue
            lap_rows.append({
                "race_id": getattr(lap, "race_id", _race_id(race)),
                "driver_code": str(getattr(lap, "driver_code", "") or "").upper(),
                "lap_number": int(getattr(lap, "lap_number", 0) or 0),
                "lap_time_s": lap_time,
                "sector1_s": _float(getattr(lap, "sector1_s", None)),
                "sector2_s": _float(getattr(lap, "sector2_s", None)),
                "sector3_s": _float(getattr(lap, "sector3_s", None)),
                "compound": str(getattr(lap, "compound", "") or ""),
                "tire_age_laps": int(getattr(lap, "tire_age_laps", 0) or 0),
                "position": int(getattr(lap, "position", 0) or 0),
            })

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in lap_rows:
        if row["driver_code"]:
            grouped.setdefault(row["driver_code"], []).append(row)

    drivers: dict[str, F1MLDriverArtifact] = {}
    for index, (code, rows) in enumerate(sorted(grouped.items())):
        usable = _drop_outliers([row["lap_time_s"] for row in rows])
        if len(usable) < min_laps_per_driver:
            continue
        pace_mean = median(usable)
        sigma = pstdev(usable) if len(usable) > 1 else 0.35
        clean_positions = [row["position"] for row in rows if int(row.get("position") or 0) > 0]
        avg_position = sum(clean_positions) / len(clean_positions) if clean_positions else index + 1
        dnf_laps = sum(1 for row in rows if int(row.get("position") or 0) <= 0)
        dnf_rate = max(0.00005, min(0.05, dnf_laps / max(1, len(rows))))
        lap_confidence = min(0.88, 0.48 + len(usable) / 80)
        drivers[code] = F1MLDriverArtifact(
            driver_code=code,
            pace_mean_seconds=round(float(pace_mean), 4),
            pace_sigma_seconds=round(max(0.08, min(2.25, float(sigma))), 4),
            dnf_hazard_per_lap=round(float(dnf_rate), 6),
            rating_prior=round(max(0.05, min(0.95, 1.0 - (avg_position - 1) / max(1, len(grouped)))), 4),
            confidence=round(lap_confidence, 4),
            sources=["fastf1_training_laps", "artifact_pace", "artifact_dnf", "artifact_rating_prior"],
        )

    training_seasons = sorted({race.season for race in training_races})
    start = int(season_start or (min(training_seasons) if training_seasons else now.year))
    end = int(season_end or (max(training_seasons) if training_seasons else start))
    coverage = {
        "training_races": len(training_races),
        "loaded_races": len({row["race_id"] for row in lap_rows}),
        "failed_races": len(failed_races),
        "lap_rows": len(lap_rows),
        "driver_rows": len(drivers),
        "drivers_below_min_laps": max(0, len(grouped) - len(drivers)),
        "min_laps_per_driver": min_laps_per_driver,
    }
    fallback_groups = []
    if failed_races:
        fallback_groups.append("race_load_failures")
    if not drivers:
        fallback_groups.append("missing_driver_payloads")
    if coverage["drivers_below_min_laps"]:
        fallback_groups.append("partial_driver_coverage")
    if leakage["status"] != "guarded":
        fallback_groups.append("leakage_guard_failed")

    artifact = F1MLArtifactBundle(
        artifact_id=artifact_id or f"fastf1-{start}-{end}-ml-simulator",
        model_version="ml-simulator-artifact-fastf1-v1",
        created_at=now,
        training_seasons=list(range(start, end + 1)),
        training_races=training_ids,
        training_cutoff=cutoff,
        knowable_as_of=cutoff,
        feature_columns=[
            *DEFAULT_FEATURE_COLUMNS,
            "lap_number",
            "lap_time_s",
            "sector1_s",
            "sector2_s",
            "sector3_s",
            "compound",
            "tire_age_laps",
            "position",
        ],
        drivers=drivers,
        pace_model=F1MLModelArtifact(
            kind="driver_lap_median_quantile_proxy",
            confidence=_average_confidence(drivers),
            feature_columns=["lap_time_s", "lap_number", "compound", "tire_age_laps", "position"],
            source="fastf1_training_laps",
            payload={"driver_count": len(drivers), "lap_rows": len(lap_rows)},
        ),
        dnf_model=F1MLModelArtifact(
            kind="observed_position_missing_hazard_proxy",
            confidence=max(0.35, _average_confidence(drivers) - 0.08),
            feature_columns=["position", "lap_number"],
            source="fastf1_training_laps",
            payload={"driver_count": len(drivers), "lap_rows": len(lap_rows)},
        ),
        rating_priors={code: float(row.rating_prior or 0.5) for code, row in drivers.items()},
        validation_metrics={
            "provider": "fastf1",
            "training_lap_rows": len(lap_rows),
            "driver_rows": len(drivers),
            "fallback_rate": round(len(fallback_groups) / 4, 4),
            "coverage": coverage,
        },
        source_metadata={
            "provider": "fastf1",
            "target_race_id": target_race_id,
            "knowable_as_of": cutoff.isoformat(),
            "coverage_counts": coverage,
            "model_coverage_counts": {
                "pace_drivers": len([row for row in drivers.values() if row.pace_mean_seconds is not None]),
                "dnf_drivers": len([row for row in drivers.values() if row.dnf_hazard_per_lap is not None]),
                "rating_drivers": len([row for row in drivers.values() if row.rating_prior is not None]),
            },
            "fallback_groups": fallback_groups,
            "failed_races": failed_races[:20],
        },
        leakage_status=leakage,
    )
    return artifact


def _race_id(race: Race) -> str:
    return f"{int(race.season)}-{int(race.round):02d}-{str(race.track_code).upper()}"


def _leakage_status(training_races: list[Race], *, target_race_id: str | None, cutoff: datetime) -> dict[str, Any]:
    target = str(target_race_id or "")
    race_ids = [_race_id(race) for race in training_races]
    future = [
        race_id for race_id, race in zip(race_ids, training_races)
        if _aware(race.scheduled_start) >= cutoff
    ]
    target_hit = target in set(race_ids) if target else False
    if target_hit or future:
        return {
            "status": "failed",
            "future_results_excluded": False,
            "target_race_in_training": target_hit,
            "future_training_races": future,
            "cutoff": cutoff.isoformat(),
        }
    return {
        "status": "guarded",
        "future_results_excluded": True,
        "target_race_in_training": False,
        "future_training_races": [],
        "cutoff": cutoff.isoformat(),
    }


def _drop_outliers(values: list[float], quantile: float = 0.97) -> list[float]:
    if not values:
        return []
    ordered = sorted(float(value) for value in values if value and value > 0)
    if not ordered:
        return []
    cutoff_index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * quantile)))
    cutoff = ordered[cutoff_index]
    return [value for value in ordered if value <= cutoff]


def _average_confidence(drivers: dict[str, F1MLDriverArtifact]) -> float:
    values = [float(row.confidence or 0.0) for row in drivers.values()]
    return round(sum(values) / len(values), 4) if values else 0.0


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
