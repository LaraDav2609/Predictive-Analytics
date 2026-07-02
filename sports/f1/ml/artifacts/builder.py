"""Small artifact builders used by CLI smoke paths and tests."""

from __future__ import annotations

import math
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
    all_lap_times = [float(row["lap_time_s"]) for row in lap_rows if _float(row.get("lap_time_s")) is not None]
    global_pace = median(all_lap_times) if all_lap_times else 80.0
    total_dnf_laps = sum(1 for row in lap_rows if int(row.get("position") or 0) <= 0)
    global_dnf_rate = max(0.00005, min(0.05, total_dnf_laps / max(1, len(lap_rows))))
    for index, (code, rows) in enumerate(sorted(grouped.items())):
        usable = _drop_outliers([row["lap_time_s"] for row in rows])
        if len(usable) < min_laps_per_driver:
            continue
        pace_mean = median(usable)
        sigma = pstdev(usable) if len(usable) > 1 else 0.35
        clean_positions = [row["position"] for row in rows if int(row.get("position") or 0) > 0]
        avg_position = sum(clean_positions) / len(clean_positions) if clean_positions else index + 1
        dnf_laps = sum(1 for row in rows if int(row.get("position") or 0) <= 0)
        dnf_rate = _smoothed_rate(dnf_laps, len(rows), global_dnf_rate, prior_laps=80)
        lap_confidence = _driver_confidence(len(usable), len(rows), sigma, dnf_laps)
        drivers[code] = F1MLDriverArtifact(
            driver_code=code,
            pace_mean_seconds=round(float(pace_mean), 4),
            pace_sigma_seconds=round(max(0.08, min(2.25, float(sigma))), 4),
            dnf_hazard_per_lap=round(float(dnf_rate), 6),
            rating_prior=round(max(0.05, min(0.95, 1.0 - (avg_position - 1) / max(1, len(grouped)))), 4),
            confidence=round(lap_confidence, 4),
            sources=["fastf1_training_laps", "artifact_pace", "artifact_dnf", "artifact_rating_prior"],
        )

    pace_metrics = _pace_validation_metrics(lap_rows, drivers, baseline_seconds=global_pace)
    dnf_metrics = _dnf_validation_metrics(lap_rows, drivers, baseline_rate=global_dnf_rate)
    overtake_metrics = _overtake_validation_metrics(lap_rows)

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
    if pace_metrics.get("trained") and pace_metrics.get("improvement_vs_baseline", 0.0) < 0:
        fallback_groups.append("pace_model_underperformed_baseline")
    if dnf_metrics.get("trained") and dnf_metrics.get("brier_improvement_vs_baseline", 0.0) < 0:
        fallback_groups.append("dnf_model_underperformed_baseline")
    if overtake_metrics.get("trained") and overtake_metrics.get("brier_improvement_vs_baseline", 0.0) < 0:
        fallback_groups.append("overtake_model_underperformed_baseline")
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
            kind="empirical_driver_median_pace_v1",
            confidence=_average_confidence(drivers),
            feature_columns=["lap_time_s", "lap_number", "compound", "tire_age_laps", "position"],
            source="fastf1_training_laps",
            payload={
                "driver_count": len(drivers),
                "lap_rows": len(lap_rows),
                "baseline_pace_seconds": round(float(global_pace), 4),
                "validation": pace_metrics,
            },
        ),
        dnf_model=F1MLModelArtifact(
            kind="smoothed_driver_dnf_hazard_v1",
            confidence=max(0.35, _average_confidence(drivers) - 0.08),
            feature_columns=["position", "lap_number"],
            source="fastf1_training_laps",
            payload={
                "driver_count": len(drivers),
                "lap_rows": len(lap_rows),
                "global_hazard_per_lap": round(float(global_dnf_rate), 6),
                "prior_laps": 80,
                "validation": dnf_metrics,
            },
        ),
        overtake_model=F1MLModelArtifact(
            kind="empirical_position_change_overtake_v1",
            confidence=_overtake_confidence(overtake_metrics),
            feature_columns=["race_id", "driver_code", "lap_number", "position", "position_delta"],
            source="fastf1_training_lap_positions",
            payload={
                "driver_count": len(overtake_metrics.get("driver_priors") or {}),
                "transition_rows": overtake_metrics.get("sample_count", 0),
                "validation": overtake_metrics,
                "driver_priors": overtake_metrics.get("driver_priors") or {},
            },
        ),
        rating_priors={code: float(row.rating_prior or 0.5) for code, row in drivers.items()},
        validation_metrics={
            "provider": "fastf1",
            "training_lap_rows": len(lap_rows),
            "driver_rows": len(drivers),
            "pace_model": pace_metrics,
            "dnf_model": dnf_metrics,
            "overtake_model": {
                key: value
                for key, value in overtake_metrics.items()
                if key != "driver_priors"
            },
            "model_quality": {
                "pace_beats_baseline": bool(pace_metrics.get("trained") and pace_metrics.get("improvement_vs_baseline", 0.0) >= 0),
                "dnf_beats_baseline": bool(dnf_metrics.get("trained") and dnf_metrics.get("brier_improvement_vs_baseline", 0.0) >= 0),
                "overtake_beats_baseline": bool(overtake_metrics.get("trained") and overtake_metrics.get("brier_improvement_vs_baseline", 0.0) >= 0),
            },
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
                "overtake_drivers": len(overtake_metrics.get("driver_priors") or {}),
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


def _pace_validation_metrics(
    lap_rows: list[dict[str, Any]],
    drivers: dict[str, F1MLDriverArtifact],
    *,
    baseline_seconds: float,
) -> dict[str, Any]:
    errors = []
    baseline_errors = []
    covered_rows = 0
    for row in lap_rows:
        code = str(row.get("driver_code") or "").upper()
        driver = drivers.get(code)
        lap_time = _float(row.get("lap_time_s"))
        if driver is None or driver.pace_mean_seconds is None or lap_time is None:
            continue
        covered_rows += 1
        errors.append(abs(float(driver.pace_mean_seconds) - lap_time))
        baseline_errors.append(abs(float(baseline_seconds) - lap_time))
    mae = sum(errors) / len(errors) if errors else None
    baseline_mae = sum(baseline_errors) / len(baseline_errors) if baseline_errors else None
    return {
        "trained": bool(errors),
        "kind": "empirical_driver_median_pace_v1",
        "sample_count": covered_rows,
        "mae_seconds": round(mae, 5) if mae is not None else None,
        "baseline_mae_seconds": round(baseline_mae, 5) if baseline_mae is not None else None,
        "improvement_vs_baseline": round(float(baseline_mae - mae), 5) if mae is not None and baseline_mae is not None else None,
    }


def _dnf_validation_metrics(
    lap_rows: list[dict[str, Any]],
    drivers: dict[str, F1MLDriverArtifact],
    *,
    baseline_rate: float,
) -> dict[str, Any]:
    brier = []
    baseline_brier = []
    log_losses = []
    baseline_log_losses = []
    positives = 0
    for row in lap_rows:
        code = str(row.get("driver_code") or "").upper()
        driver = drivers.get(code)
        if driver is None or driver.dnf_hazard_per_lap is None:
            continue
        actual = 1.0 if int(row.get("position") or 0) <= 0 else 0.0
        positives += int(actual)
        pred = max(1e-6, min(1.0 - 1e-6, float(driver.dnf_hazard_per_lap)))
        base = max(1e-6, min(1.0 - 1e-6, float(baseline_rate)))
        brier.append((pred - actual) ** 2)
        baseline_brier.append((base - actual) ** 2)
        log_losses.append(_binary_log_loss(actual, pred))
        baseline_log_losses.append(_binary_log_loss(actual, base))
    mean_brier = sum(brier) / len(brier) if brier else None
    mean_baseline_brier = sum(baseline_brier) / len(baseline_brier) if baseline_brier else None
    mean_log_loss = sum(log_losses) / len(log_losses) if log_losses else None
    mean_baseline_log_loss = sum(baseline_log_losses) / len(baseline_log_losses) if baseline_log_losses else None
    return {
        "trained": bool(brier),
        "kind": "smoothed_driver_dnf_hazard_v1",
        "sample_count": len(brier),
        "positive_laps": positives,
        "brier": round(mean_brier, 6) if mean_brier is not None else None,
        "baseline_brier": round(mean_baseline_brier, 6) if mean_baseline_brier is not None else None,
        "brier_improvement_vs_baseline": round(float(mean_baseline_brier - mean_brier), 6) if mean_brier is not None and mean_baseline_brier is not None else None,
        "log_loss": round(mean_log_loss, 6) if mean_log_loss is not None else None,
        "baseline_log_loss": round(mean_baseline_log_loss, 6) if mean_baseline_log_loss is not None else None,
    }


def _overtake_validation_metrics(lap_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_driver: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in lap_rows:
        race_id = str(row.get("race_id") or "")
        code = str(row.get("driver_code") or "").upper()
        position = int(row.get("position") or 0)
        lap_number = int(row.get("lap_number") or 0)
        if not race_id or not code or position <= 0 or lap_number <= 0:
            continue
        by_driver.setdefault((race_id, code), []).append(row)

    transitions: list[dict[str, Any]] = []
    driver_events: dict[str, list[float]] = {}
    for (_, code), rows in by_driver.items():
        ordered = sorted(rows, key=lambda item: int(item.get("lap_number") or 0))
        for prev, current in zip(ordered, ordered[1:]):
            prev_position = int(prev.get("position") or 0)
            current_position = int(current.get("position") or 0)
            if prev_position <= 0 or current_position <= 0:
                continue
            actual = 1.0 if current_position < prev_position else 0.0
            transitions.append({"driver_code": code, "actual": actual})
            driver_events.setdefault(code, []).append(actual)

    total_events = sum(item["actual"] for item in transitions)
    baseline_rate = _smoothed_rate(int(total_events), len(transitions), 0.08, prior_laps=100)
    driver_priors = {
        code: round(_smoothed_rate(int(sum(values)), len(values), baseline_rate, prior_laps=40), 6)
        for code, values in sorted(driver_events.items())
    }
    brier = []
    baseline_brier = []
    log_losses = []
    baseline_log_losses = []
    for item in transitions:
        actual = float(item["actual"])
        pred = max(1e-6, min(1.0 - 1e-6, float(driver_priors.get(item["driver_code"], baseline_rate))))
        base = max(1e-6, min(1.0 - 1e-6, float(baseline_rate)))
        brier.append((pred - actual) ** 2)
        baseline_brier.append((base - actual) ** 2)
        log_losses.append(_binary_log_loss(actual, pred))
        baseline_log_losses.append(_binary_log_loss(actual, base))

    mean_brier = sum(brier) / len(brier) if brier else None
    mean_baseline_brier = sum(baseline_brier) / len(baseline_brier) if baseline_brier else None
    mean_log_loss = sum(log_losses) / len(log_losses) if log_losses else None
    mean_baseline_log_loss = sum(baseline_log_losses) / len(baseline_log_losses) if baseline_log_losses else None
    return {
        "trained": bool(transitions),
        "kind": "empirical_position_change_overtake_v1",
        "sample_count": len(transitions),
        "positive_passes": int(total_events),
        "baseline_pass_rate": round(float(baseline_rate), 6),
        "brier": round(mean_brier, 6) if mean_brier is not None else None,
        "baseline_brier": round(mean_baseline_brier, 6) if mean_baseline_brier is not None else None,
        "brier_improvement_vs_baseline": round(float(mean_baseline_brier - mean_brier), 6) if mean_brier is not None and mean_baseline_brier is not None else None,
        "log_loss": round(mean_log_loss, 6) if mean_log_loss is not None else None,
        "baseline_log_loss": round(mean_baseline_log_loss, 6) if mean_baseline_log_loss is not None else None,
        "driver_priors": driver_priors,
    }


def _overtake_confidence(metrics: dict[str, Any]) -> float:
    if not metrics.get("trained"):
        return 0.0
    sample_part = min(0.32, float(metrics.get("sample_count") or 0) / 250)
    positive_part = min(0.14, float(metrics.get("positive_passes") or 0) / 60)
    quality_part = 0.08 if float(metrics.get("brier_improvement_vs_baseline") or 0.0) >= 0 else -0.08
    return round(max(0.30, min(0.82, 0.38 + sample_part + positive_part + quality_part)), 4)


def _binary_log_loss(actual: float, probability: float) -> float:
    return -((actual * math.log(probability)) + ((1.0 - actual) * math.log(1.0 - probability)))


def _smoothed_rate(events: int, samples: int, prior_rate: float, *, prior_laps: int) -> float:
    value = (float(events) + float(prior_laps) * float(prior_rate)) / max(1.0, float(samples + prior_laps))
    return max(0.00005, min(0.05, value))


def _driver_confidence(usable_laps: int, total_laps: int, sigma: float, dnf_laps: int) -> float:
    sample_part = min(0.30, usable_laps / 80)
    stability_part = max(0.0, min(0.10, 0.10 - max(0.0, float(sigma) - 0.45) * 0.04))
    dnf_part = -0.04 if dnf_laps else 0.0
    coverage_part = min(0.08, total_laps / 120)
    return round(max(0.35, min(0.90, 0.46 + sample_part + stability_part + coverage_part + dnf_part)), 4)


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
