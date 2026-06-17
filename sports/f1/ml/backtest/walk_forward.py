"""Walk-forward comparison for the F1 research ML simulator.

Default execution is synthetic and deterministic so CI stays fast. The FastF1
path is lazy/network-backed and only used when requested from the CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.ml.backtest.walk_forward import RaceData
from common.ml.calibration import brier_score, log_loss, reliability_curve
from sports.f1.ml.artifacts.builder import build_fastf1_artifact_bundle, build_synthetic_artifact_bundle
from sports.f1.ml.artifacts.loader import artifact_to_simulator_model_bundle, load_simulator_model_bundle
from sports.f1.ml.artifacts.store import save_artifact_bundle, validate_artifact_bundle
from sports.f1.ml.common.types import Race as MLSimRace
from sports.f1.ml.markets.mapper import dnf_probabilities, podium_probabilities, winner_probabilities
from sports.f1.ml.providers.synthetic_provider import SyntheticProvider, SyntheticRaceConfig, default_grid
from sports.f1.ml.simulator.race_sim import PhysicalParams, SimConfig, simulate_race
from sports.f1.models.f1 import Constructor, Driver, Race
from sports.f1.predictor.service import F1PredictionService


MODEL_CHOICES = ("uniform", "production_v1", "ml_simulator_v1")


@dataclass
class MLBacktestConfig:
    season: int
    provider: str = "synthetic"
    models: tuple[str, ...] = ("uniform",)
    training_seasons: tuple[int, ...] = field(default_factory=tuple)
    min_training_races: int = 20
    refit_every_n_races: int = 1
    output_dir: str = "artifacts/backtest"
    artifact_dir: str = "artifacts/backtest/ml_simulator_v1"
    n_iterations: int = 400
    physical: bool = False
    fastf1_cache: str = ".fastf1_cache"


@dataclass
class MLBacktestResult:
    per_race_metrics: pd.DataFrame
    aggregate_metrics: dict[str, Any]
    model_comparison: dict[str, Any]
    calibration: dict[str, Any]
    output_paths: dict[str, str]


def expand_model_choice(model: str) -> tuple[str, ...]:
    value = (model or "uniform").strip().lower()
    if value == "all":
        return MODEL_CHOICES
    if value not in MODEL_CHOICES:
        raise ValueError(f"unknown model '{model}'")
    return (value,)


def run_backtest(config: MLBacktestConfig) -> MLBacktestResult:
    races = _load_races(config)
    train_seasons = config.training_seasons or tuple(range(config.season - 3, config.season))
    rows: list[dict[str, Any]] = []
    artifact_root = Path(config.artifact_dir)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    test_races = [race for race in sorted(races, key=lambda item: (item.season, item.round)) if race.season == config.season]
    for race in test_races:
        training = [
            item for item in races
            if item.season in train_seasons
            and (item.season, item.round) < (race.season, race.round)
            and item.decision_time < race.decision_time
        ]
        if len(training) < config.min_training_races:
            continue
        for model_id in config.models:
            prediction = _predict_model(model_id, race, training, config, artifact_root)
            rows.append(_score_prediction(model_id, race, prediction))

    frame = pd.DataFrame(rows) if rows else _empty_frame()
    aggregate = _aggregate(frame)
    comparison = _model_comparison(frame)
    calibration = _calibration(frame)
    artifact_manifest = _artifact_manifest(frame)
    paths = _write_outputs(config.season, output_root, frame, aggregate, comparison, calibration, artifact_manifest)
    return MLBacktestResult(
        per_race_metrics=frame,
        aggregate_metrics=aggregate,
        model_comparison=comparison,
        calibration=calibration,
        output_paths=paths,
    )


def _load_races(config: MLBacktestConfig) -> list[RaceData]:
    if config.provider == "synthetic":
        return _synthetic_races(config)
    if config.provider == "fastf1":
        return _fastf1_races(config)
    raise ValueError(f"unknown provider '{config.provider}'")


def _synthetic_races(config: MLBacktestConfig) -> list[RaceData]:
    train_seasons = config.training_seasons or tuple(range(config.season - 3, config.season))
    seasons = (*train_seasons, config.season)
    ordered = [driver.code for driver in sorted(default_grid(), key=lambda item: item.true_pace_s)]
    races: list[RaceData] = []
    for season in seasons:
        for round_num in range(1, 11):
            # Deterministic tiny shake-up so metrics are not completely identical.
            finish = list(ordered)
            if round_num % 5 == 0 and len(finish) >= 2:
                finish[0], finish[1] = finish[1], finish[0]
            races.append(RaceData(
                race_id=f"SYN-{season}-{round_num:02d}",
                season=season,
                round=round_num,
                decision_time=datetime(season, max(1, min(12, round_num)), 1, 12, 0, tzinfo=timezone.utc),
                finish_order=finish,
                dnf_drivers=set(),
            ))
    return races


def _fastf1_races(config: MLBacktestConfig) -> list[RaceData]:
    from sports.f1.ml.common.types import SessionType
    from sports.f1.ml.providers.fastf1_provider import FastF1Provider

    provider = FastF1Provider(cache_dir=config.fastf1_cache)
    seasons = (*config.training_seasons, config.season) if config.training_seasons else tuple(range(config.season - 3, config.season + 1))
    rows: list[RaceData] = []
    for season in seasons:
        for race in provider.list_races(season):
            laps = provider.laps(race, SessionType.RACE)
            if not laps:
                continue
            latest: dict[str, Any] = {}
            for lap in laps:
                latest[lap.driver_code] = lap
            finish = [
                code for code, lap in sorted(
                    latest.items(),
                    key=lambda item: item[1].position if item[1].position > 0 else 99,
                )
            ]
            if not finish:
                continue
            rows.append(RaceData(
                race_id=f"{season}-{race.round:02d}-{race.track_code}",
                season=season,
                round=race.round,
                decision_time=race.scheduled_start if race.scheduled_start.tzinfo else race.scheduled_start.replace(tzinfo=timezone.utc),
                finish_order=finish,
                dnf_drivers={code for code, lap in latest.items() if lap.position <= 0},
            ))
    return rows


def _predict_model(model_id: str, race: RaceData, training: list[RaceData], config: MLBacktestConfig, artifact_root: Path) -> dict[str, Any]:
    if model_id == "uniform":
        return _uniform_prediction(race)
    if model_id == "production_v1":
        return _production_prediction(race)
    if model_id == "ml_simulator_v1":
        return _ml_prediction(race, training, config, artifact_root)
    raise ValueError(f"unknown model '{model_id}'")


def _uniform_prediction(race: RaceData) -> dict[str, Any]:
    drivers = list(race.finish_order)
    count = max(1, len(drivers))
    return {
        "winner": {driver: 1.0 / count for driver in drivers},
        "podium": {driver: min(1.0, 3.0 / count) for driver in drivers},
        "dnf": {driver: 0.03 for driver in drivers},
        "expected_finish": {driver: (count + 1) / 2 for driver in drivers},
        "metadata": {"input_source": "uniform", "leakage_status": "not_applicable"},
    }


def _production_prediction(race: RaceData) -> dict[str, Any]:
    drivers, constructors = _production_entities(race)
    service = F1PredictionService(model_id="production_v1")
    features = {
        "completed_races": max(0, race.round - 1),
        "total_races": 10,
        "drivers": {
            driver.id: {
                "form_score": max(0.05, 1.0 - index / max(1, len(drivers))),
                "race_pace_score": max(0.05, 1.0 - index / max(1, len(drivers))),
                "qualifying_pace_score": max(0.05, 1.0 - index / max(1, len(drivers))),
                "reliability_score": 0.86,
            }
            for index, driver in enumerate(drivers)
        },
    }
    service.load(drivers, constructors, features, sentiment={})
    prediction = service.predict_race(_production_race(race))
    by_code = {driver.id: driver.code for driver in drivers}
    return {
        "winner": {by_code[driver_id]: item.win_prob for driver_id, item in prediction.driver_predictions.items()},
        "podium": {by_code[driver_id]: item.podium_prob for driver_id, item in prediction.driver_predictions.items()},
        "dnf": {by_code[driver_id]: item.dnf_prob for driver_id, item in prediction.driver_predictions.items()},
        "expected_finish": {by_code[driver_id]: item.expected_finish or item.predicted_position or 20 for driver_id, item in prediction.driver_predictions.items()},
        "metadata": {"input_source": "production_v1", "model_id": prediction.model_id},
    }


def _ml_prediction(race: RaceData, training: list[RaceData], config: MLBacktestConfig, artifact_root: Path) -> dict[str, Any]:
    if config.provider == "fastf1":
        return _fastf1_ml_prediction(race, training, config, artifact_root)
    return _synthetic_ml_prediction(race, training, config, artifact_root)


def _synthetic_ml_prediction(race: RaceData, training: list[RaceData], config: MLBacktestConfig, artifact_root: Path) -> dict[str, Any]:
    provider = SyntheticProvider(SyntheticRaceConfig(
        race_id=race.race_id,
        season=race.season,
        round=race.round,
        n_laps=50,
        seed=42 + race.round,
    ))
    training_races = [item.race_id for item in training]
    bundle = build_synthetic_artifact_bundle(
        provider,
        artifact_id=f"walk-forward-{race.race_id}",
        season_start=min(item.season for item in training),
        season_end=max(item.season for item in training),
        training_races=training_races,
        metrics={"source": "walk_forward_synthetic", "training_race_count": len(training)},
    )
    validation = validate_artifact_bundle(bundle, target_race=race.race_id)
    artifact_path = artifact_root / f"{race.race_id}.json"
    if validation.ok:
        save_artifact_bundle(artifact_path, bundle)
        model_bundle = load_simulator_model_bundle(artifact_path, target_race=race.race_id)
    else:
        model_bundle = artifact_to_simulator_model_bundle(bundle, target_race=race.race_id)

    initial_state = {
        "driver_codes": list(provider.driver_pace_table().keys()),
        "driver_mean_pace_s": [81.0] * len(provider.driver_pace_table()),
        "driver_pace_sigma_s": [0.55] * len(provider.driver_pace_table()),
        "driver_dnf_rate_per_lap": [0.001] * len(provider.driver_pace_table()),
        "total_laps": provider.config.n_laps,
    }
    result = simulate_race(
        provider.list_races(provider.config.season)[0],
        initial_state,
        models={"simulator_model_bundle": model_bundle},
        config=SimConfig(
            n_iterations=config.n_iterations,
            seed=202600 + race.round,
            enable_physical=config.physical,
            physical=PhysicalParams(),
        ),
    )
    expected = {
        code: float(result.finish_positions[:, index].mean())
        for index, code in enumerate(result.driver_codes)
    }
    return {
        "winner": winner_probabilities(result),
        "podium": podium_probabilities(result),
        "dnf": dnf_probabilities(result),
        "expected_finish": expected,
        "metadata": {
            "input_source": "ml_simulator_v1",
            "artifact_path": str(artifact_path) if validation.ok else None,
            "artifact_id": bundle.artifact_id,
            "artifact_validation": validation.reason or "ok",
            "leakage_status": "passed" if validation.ok else validation.reason,
            **result.metadata,
        },
    }


def _fastf1_ml_prediction(race: RaceData, training: list[RaceData], config: MLBacktestConfig, artifact_root: Path) -> dict[str, Any]:
    from sports.f1.ml.providers.fastf1_provider import FastF1Provider

    provider = FastF1Provider(cache_dir=config.fastf1_cache)
    training_races = [_ml_race_from_racedata(item) for item in training]
    artifact_path = artifact_root / f"{race.race_id}.json"
    bundle = build_fastf1_artifact_bundle(
        provider,
        training_races=training_races,
        target_race_id=race.race_id,
        knowable_as_of=race.decision_time,
        artifact_id=f"walk-forward-fastf1-{race.race_id}",
        season_start=min((item.season for item in training), default=race.season),
        season_end=max((item.season for item in training), default=race.season),
    )
    validation = validate_artifact_bundle(bundle, target_race=race.race_id)
    if validation.ok:
        save_artifact_bundle(artifact_path, bundle)
        model_bundle = load_simulator_model_bundle(artifact_path, target_race=race.race_id)
    else:
        model_bundle = artifact_to_simulator_model_bundle(bundle, target_race=race.race_id)

    initial_state = _initial_state_from_artifact(bundle, race.finish_order)
    result = simulate_race(
        _ml_race_from_racedata(race),
        initial_state,
        models={"simulator_model_bundle": model_bundle},
        config=SimConfig(
            n_iterations=config.n_iterations,
            seed=202700 + race.round,
            enable_physical=config.physical,
            physical=PhysicalParams(),
        ),
    )
    expected = {
        code: float(result.finish_positions[:, index].mean())
        for index, code in enumerate(result.driver_codes)
    }
    return {
        "winner": winner_probabilities(result),
        "podium": podium_probabilities(result),
        "dnf": dnf_probabilities(result),
        "expected_finish": expected,
        "metadata": {
            "input_source": "ml_simulator_v1",
            "provider": "fastf1",
            "artifact_path": str(artifact_path) if validation.ok else None,
            "artifact_id": bundle.artifact_id,
            "artifact_validation": validation.reason or "ok",
            "leakage_status": "passed" if validation.ok else validation.reason,
            "leakage_guard_status": bundle.leakage_status,
            "coverage_counts": bundle.source_metadata.get("coverage_counts") or {},
            "model_coverage_counts": bundle.source_metadata.get("model_coverage_counts") or {},
            "fallback_groups": bundle.source_metadata.get("fallback_groups") or [],
            **result.metadata,
        },
    }


def _score_prediction(model_id: str, race: RaceData, prediction: dict[str, Any]) -> dict[str, Any]:
    drivers = list(race.finish_order)
    winner_actual = race.finish_order[0] if race.finish_order else None
    podium_actual = set(race.finish_order[:3])
    dnf_actual = set(race.dnf_drivers or set())
    winner_probs = _complete_probs(prediction.get("winner") or {}, drivers)
    podium_probs = _complete_probs(prediction.get("podium") or {}, drivers)
    dnf_probs = _complete_probs(prediction.get("dnf") or {}, drivers)
    winner_y = np.array([1.0 if driver == winner_actual else 0.0 for driver in drivers])
    podium_y = np.array([1.0 if driver in podium_actual else 0.0 for driver in drivers])
    dnf_y = np.array([1.0 if driver in dnf_actual else 0.0 for driver in drivers])
    winner_p = np.array([winner_probs[driver] for driver in drivers])
    podium_p = np.array([podium_probs[driver] for driver in drivers])
    dnf_p = np.array([dnf_probs[driver] for driver in drivers])
    predicted_winner = max(winner_probs.items(), key=lambda item: item[1])[0] if winner_probs else None
    predicted_top3 = [driver for driver, _ in sorted(winner_probs.items(), key=lambda item: item[1], reverse=True)[:3]]
    expected = prediction.get("expected_finish") or {}
    finish_errors = [
        abs(float(expected.get(driver, index + 1)) - float(index + 1))
        for index, driver in enumerate(race.finish_order)
        if driver in expected
    ]
    top_probability = float(winner_probs.get(predicted_winner or "", 0.0))
    metadata = prediction.get("metadata") or {}
    return {
        "race_id": race.race_id,
        "season": race.season,
        "round": race.round,
        "model_id": model_id,
        "actual_winner": winner_actual,
        "predicted_winner": predicted_winner,
        "winner_hit": predicted_winner == winner_actual,
        "winner_brier": brier_score(winner_p, winner_y),
        "winner_log_loss": -float(np.log(max(min(winner_probs.get(winner_actual or "", 0.0), 1.0 - 1e-12), 1e-12))),
        "podium_brier": brier_score(podium_p, podium_y),
        "podium_log_loss": log_loss(podium_p, podium_y),
        "dnf_brier": brier_score(dnf_p, dnf_y),
        "dnf_log_loss": log_loss(dnf_p, dnf_y),
        "top_pick_probability": top_probability,
        "podium_hit_rate": len(set(predicted_top3).intersection(podium_actual)) / max(1, len(podium_actual)),
        "expected_finish_error": float(sum(finish_errors) / len(finish_errors)) if finish_errors else None,
        "winner_probabilities": json.dumps(winner_probs, sort_keys=True),
        "podium_probabilities": json.dumps(podium_probs, sort_keys=True),
        "dnf_probabilities": json.dumps(dnf_probs, sort_keys=True),
        "model_metadata": json.dumps(metadata, sort_keys=True, default=str),
        "leakage_status": metadata.get("leakage_status") or metadata.get("artifact_validation") or "unknown",
        "fallback_reason": metadata.get("ml_model_fallback_reason") or metadata.get("artifact_validation"),
    }


def _complete_probs(raw: dict[str, float], drivers: list[str]) -> dict[str, float]:
    return {driver: max(0.0, min(1.0, float(raw.get(driver, 0.0)))) for driver in drivers}


def _aggregate(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"race_count": 0, "models": {}}
    models = {}
    for model_id, group in frame.groupby("model_id"):
        models[str(model_id)] = {
            "race_count": int(len(group)),
            "winner_accuracy": float(group["winner_hit"].mean()),
            "winner_brier": float(group["winner_brier"].mean()),
            "winner_log_loss": float(group["winner_log_loss"].mean()),
            "podium_brier": float(group["podium_brier"].mean()),
            "podium_log_loss": float(group["podium_log_loss"].mean()),
            "dnf_brier": float(group["dnf_brier"].mean()),
            "dnf_log_loss": float(group["dnf_log_loss"].mean()),
            "podium_hit_rate": float(group["podium_hit_rate"].mean()),
            "expected_finish_error": float(group["expected_finish_error"].dropna().mean()) if group["expected_finish_error"].notna().any() else None,
            "fallback_rate": _fallback_rate(group),
        }
    return {"race_count": int(frame["race_id"].nunique()), "models": models}


def _fallback_rate(group: pd.DataFrame) -> float:
    if "fallback_reason" not in group:
        return 0.0
    values = group["fallback_reason"].fillna("").astype(str).str.lower()
    fallback = values.apply(lambda item: item not in {"", "none", "ok", "passed"})
    return float(fallback.mean()) if len(fallback) else 0.0


def _model_comparison(frame: pd.DataFrame) -> dict[str, Any]:
    aggregate = _aggregate(frame).get("models") or {}
    if not aggregate:
        return {"best_model": None, "models": {}}
    best = sorted(
        aggregate.items(),
        key=lambda item: (
            item[1].get("winner_log_loss", 999.0),
            item[1].get("winner_brier", 999.0),
            -item[1].get("winner_accuracy", 0.0),
        ),
    )[0][0]
    return {"best_model": best, "selection_rule": "winner_log_loss,winner_brier,winner_accuracy", "models": aggregate}


def _calibration(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"models": {}}
    by_model = {}
    for model_id, group in frame.groupby("model_id"):
        probs = group["top_pick_probability"].astype(float).to_numpy()
        hits = group["winner_hit"].astype(float).to_numpy()
        pred, obs, counts = reliability_curve(probs, hits, n_bins=5)
        overconfidence = float(np.nanmean(pred - obs)) if len(pred) else 0.0
        by_model[str(model_id)] = {
            "top_pick_buckets": [
                {"bucket": index, "avg_predicted": float(pred[index]), "actual_rate": float(obs[index]), "count": int(counts[index])}
                for index in range(len(counts))
                if counts[index] > 0
            ],
            "overconfidence": overconfidence,
            "diagnosis": "overconfident" if overconfidence > 0.05 else "underconfident" if overconfidence < -0.05 else "balanced",
        }
    return {"models": by_model}


def _artifact_manifest(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "model_metadata" not in frame.columns:
        return {"artifacts": [], "fallback_rate": 0.0}
    artifacts: list[dict[str, Any]] = []
    fallback_count = 0
    artifact_rows = 0
    for _, row in frame.iterrows():
        if str(row.get("model_id")) != "ml_simulator_v1":
            continue
        artifact_rows += 1
        try:
            metadata = json.loads(row.get("model_metadata") or "{}")
        except (TypeError, ValueError):
            metadata = {}
        fallback_groups = metadata.get("fallback_groups") or []
        validation = metadata.get("artifact_validation")
        if validation and validation != "ok":
            fallback_count += 1
        elif fallback_groups:
            fallback_count += 1
        artifacts.append({
            "race_id": row.get("race_id"),
            "season": int(row.get("season")) if row.get("season") is not None else None,
            "round": int(row.get("round")) if row.get("round") is not None else None,
            "artifact_id": metadata.get("artifact_id"),
            "artifact_path": metadata.get("artifact_path"),
            "artifact_validation": validation,
            "leakage_status": metadata.get("leakage_status"),
            "leakage_guard_status": metadata.get("leakage_guard_status"),
            "coverage_counts": metadata.get("coverage_counts") or {},
            "model_coverage_counts": metadata.get("model_coverage_counts") or {},
            "fallback_groups": fallback_groups,
        })
    return {
        "artifact_count": len(artifacts),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / artifact_rows, 4) if artifact_rows else 0.0,
        "artifacts": artifacts,
    }


def _write_outputs(
    season: int,
    output_root: Path,
    frame: pd.DataFrame,
    aggregate: dict[str, Any],
    comparison: dict[str, Any],
    calibration: dict[str, Any],
    artifact_manifest: dict[str, Any],
) -> dict[str, str]:
    paths = {
        "per_race_metrics": str(output_root / f"per_race_metrics_{season}.csv"),
        "aggregate_metrics": str(output_root / f"aggregate_metrics_{season}.json"),
        "model_comparison": str(output_root / f"model_comparison_{season}.json"),
        "calibration": str(output_root / f"calibration_{season}.json"),
        "artifact_manifest": str(output_root / f"artifact_manifest_{season}.json"),
    }
    frame.to_csv(paths["per_race_metrics"], index=False)
    for key, payload in (
        ("aggregate_metrics", aggregate),
        ("model_comparison", comparison),
        ("calibration", calibration),
        ("artifact_manifest", artifact_manifest),
    ):
        Path(paths[key]).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def _ml_race_from_racedata(race: RaceData) -> MLSimRace:
    parts = str(race.race_id).split("-")
    track_code = parts[2] if len(parts) >= 3 else f"R{race.round:02d}"
    return MLSimRace(
        season=int(race.season),
        round=int(race.round),
        track_code=str(track_code).upper(),
        name=str(race.race_id),
        scheduled_start=race.decision_time,
    )


def _initial_state_from_artifact(bundle, finish_order: list[str]) -> dict[str, Any]:
    rows = bundle.drivers or {}
    available_means = [
        float(row.pace_mean_seconds)
        for row in rows.values()
        if row.pace_mean_seconds is not None
    ]
    available_sigmas = [
        float(row.pace_sigma_seconds)
        for row in rows.values()
        if row.pace_sigma_seconds is not None
    ]
    fallback_mean = float(np.median(available_means)) if available_means else 81.0
    fallback_sigma = float(np.median(available_sigmas)) if available_sigmas else 0.55
    codes = list(finish_order) or list(rows.keys())
    means = []
    sigmas = []
    dnfs = []
    for code in codes:
        row = rows.get(code)
        means.append(float(row.pace_mean_seconds) if row and row.pace_mean_seconds is not None else fallback_mean)
        sigmas.append(max(0.08, min(2.25, float(row.pace_sigma_seconds) if row and row.pace_sigma_seconds is not None else fallback_sigma)))
        dnfs.append(max(0.00005, min(0.05, float(row.dnf_hazard_per_lap) if row and row.dnf_hazard_per_lap is not None else 0.001)))
    return {
        "driver_codes": codes,
        "driver_mean_pace_s": means,
        "driver_pace_sigma_s": sigmas,
        "driver_dnf_rate_per_lap": dnfs,
        "total_laps": 50,
        "driver_starting_compound": ["MEDIUM"] * len(codes),
        "driver_pit_lap": [None] * len(codes),
        "driver_pit_compound": [None] * len(codes),
    }


def _production_entities(race: RaceData) -> tuple[list[Driver], list[Constructor]]:
    drivers = [
        Driver(
            id=code.lower(),
            number=index + 1,
            code=code,
            first_name=code,
            last_name="Synthetic",
            nationality="Test",
            team=f"Team {index // 2 + 1}",
            points=max(0, 100 - index * 4),
            position=index + 1,
        )
        for index, code in enumerate(race.finish_order)
    ]
    constructors = [
        Constructor(id=f"team-{index + 1}", name=f"Team {index + 1}", nationality="Test", points=max(0, 100 - index * 8), position=index + 1)
        for index in range(max(1, (len(drivers) + 1) // 2))
    ]
    return drivers, constructors


def _production_race(race: RaceData) -> Race:
    return Race(
        round=race.round,
        name=f"Synthetic Round {race.round}",
        circuit="Synthetic",
        country="Synthetic",
        date=race.decision_time,
    )


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "race_id",
        "season",
        "round",
        "model_id",
        "actual_winner",
        "predicted_winner",
        "winner_hit",
        "winner_brier",
        "winner_log_loss",
        "podium_brier",
        "podium_log_loss",
        "dnf_brier",
        "dnf_log_loss",
        "top_pick_probability",
        "podium_hit_rate",
        "expected_finish_error",
        "winner_probabilities",
        "podium_probabilities",
        "dnf_probabilities",
        "model_metadata",
        "leakage_status",
        "fallback_reason",
    ])
