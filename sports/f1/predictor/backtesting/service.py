"""Backtesting service for the F1 prediction engine."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sports.f1.predictor.backtesting.calibration import recommend_weight_adjustments
from sports.f1.predictor.backtesting.loader import HistoricalRaceLoader
from sports.f1.predictor.backtesting.metrics import evaluate_race, summarize_races
from sports.f1.predictor.backtesting.replay import RaceReplayBuilder
from sports.f1.predictor.models.configs import PRODUCTION_MODEL_ID
from sports.f1.predictor.models.registry import F1ModelRegistry
from sports.f1.predictor.probability import enrich_probability_payload
from sports.f1.predictor.service import F1PredictionService
from sports.f1.predictor.simulation.session_projection import build_session_projection
from sports.f1.models.f1 import DriverRacePrediction, RacePrediction


def _cache_policy(value: str | None) -> str:
    policy = str(value or "read_through").strip().lower().replace("-", "_").replace(" ", "_")
    if policy in {"cache", "cacheonly", "offline"}:
        return "cache_only"
    if policy in {"force_refresh", "reload"}:
        return "refresh"
    if policy not in {"read_through", "cache_only", "refresh"}:
        return "read_through"
    return policy


class F1BacktestService:
    def __init__(
        self,
        client,
        lookback_races: int = 8,
        cache_dir: str | None = None,
        use_disk_cache: bool = True,
        evidence_cache_dir: str | None = None,
        use_evidence_cache: bool = True,
    ):
        self._client = client
        disk_cache_enabled = use_disk_cache and bool(getattr(client, "backtest_disk_cache", True))
        self._loader = HistoricalRaceLoader(
            client,
            cache_dir=cache_dir,
            use_disk_cache=disk_cache_enabled,
            evidence_cache_dir=evidence_cache_dir,
            use_evidence_cache=use_evidence_cache,
        )
        self._builder = RaceReplayBuilder(lookback_races=lookback_races)
        self._cache: dict[tuple, dict[str, Any]] = {}

    async def backtest_season(
        self,
        season: int | None = None,
        include_races: bool = False,
        allow_partial: bool = False,
        model_id: str | None = None,
        stage: str = "pre_weekend",
        _compact: bool = False,
    ) -> dict[str, Any]:
        model_id = model_id or PRODUCTION_MODEL_ID
        stage = _stage_key(stage)
        season = self._default_season() if season is None else int(season)
        partial = season >= int(getattr(self._client, "season", datetime.now(timezone.utc).year))
        if partial and not allow_partial:
            return {
                "ok": False,
                "reason": "Current in-progress season backtests require allow_partial=true",
                "season": season,
                "partial": True,
            }

        cache_key = ("season", season, include_races, allow_partial, model_id, stage, _compact)
        if cache_key in self._cache:
            return self._cache[cache_key]

        races = await self._loader.load_season(season)
        completed = [race for race in races if race.get("Results")]
        race_rows = await asyncio.to_thread(
            self._backtest_completed_races,
            season,
            races,
            completed,
            model_id,
            stage,
            _compact,
        )
        summary = summarize_races(race_rows)
        result = {
            "ok": True,
            "season": season,
            "partial": partial,
            "model_version": _model_version(race_rows),
            "model_id": model_id,
            "stage": stage,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **summary,
            "fallback_coverage": _fallback_coverage_summary(race_rows),
            "leakage_status": _leakage_status_summary(race_rows),
            "recommended_weights": recommend_weight_adjustments(summary),
        }
        if include_races:
            result["races"] = race_rows
        self._cache[cache_key] = result
        return result

    async def backtest_summary(
        self,
        start_season: int,
        end_season: int,
        include_races: bool = False,
        allow_partial: bool = False,
        model_id: str | None = None,
        stage: str = "pre_weekend",
    ) -> dict[str, Any]:
        model_id = model_id or PRODUCTION_MODEL_ID
        stage = _stage_key(stage)
        start_season = int(start_season)
        end_season = int(end_season)
        if end_season < start_season:
            start_season, end_season = end_season, start_season
        seasons = []
        all_races = []
        for season in range(start_season, end_season + 1):
            season_result = await self.backtest_season(
                season,
                include_races=True,
                allow_partial=allow_partial,
                model_id=model_id,
                stage=stage,
                _compact=not include_races,
            )
            if not season_result.get("ok"):
                seasons.append(season_result)
                continue
            races = season_result.get("races") or []
            all_races.extend(races)
            season_compact = {key: value for key, value in season_result.items() if key != "races"}
            if include_races:
                season_compact["races"] = races
            seasons.append(season_compact)

        summary = summarize_races(all_races)
        return {
            "ok": True,
            "start_season": start_season,
            "end_season": end_season,
            "model_id": model_id,
            "stage": stage,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **summary,
            "fallback_coverage": _fallback_coverage_summary(all_races),
            "leakage_status": _leakage_status_summary(all_races),
            "recommended_weights": recommend_weight_adjustments(summary),
            "seasons": seasons,
        }

    async def backtest_race(self, season: int, round_num: int, allow_partial: bool = True, model_id: str | None = None, stage: str = "pre_weekend") -> dict[str, Any]:
        model_id = model_id or PRODUCTION_MODEL_ID
        stage = _stage_key(stage)
        season = int(season)
        if season >= int(getattr(self._client, "season", datetime.now(timezone.utc).year)) and not allow_partial:
            return {
                "ok": False,
                "reason": "Current in-progress season race backtests require allow_partial=true",
                "season": season,
                "round": round_num,
                "partial": True,
            }
        races = await self._loader.load_season(season)
        try:
            return {"ok": True, **self.backtest_race_from_rows(season, races, int(round_num), model_id=model_id, stage=stage)}
        except ValueError as exc:
            return {"ok": False, "season": season, "round": round_num, "reason": str(exc)}

    async def compare_season(self, season: int | None = None, include_races: bool = False, allow_partial: bool = False, stage: str = "pre_weekend") -> dict[str, Any]:
        season = self._default_season() if season is None else int(season)
        stage = _stage_key(stage)
        models = []
        for model in F1ModelRegistry.list_models():
            result = await self.backtest_season(
                season=season,
                include_races=include_races,
                allow_partial=allow_partial,
                model_id=model["model_id"],
                stage=stage,
                _compact=not include_races,
            )
            models.append(result)
        return _comparison_result({"season": season}, models)

    async def compare_summary(
        self,
        start_season: int,
        end_season: int,
        include_races: bool = False,
        allow_partial: bool = False,
        stage: str = "pre_weekend",
    ) -> dict[str, Any]:
        stage = _stage_key(stage)
        start_season = int(start_season)
        end_season = int(end_season)
        if end_season < start_season:
            start_season, end_season = end_season, start_season
        cache_key = ("compare_summary", start_season, end_season, include_races, allow_partial, stage)
        if cache_key in self._cache:
            return self._cache[cache_key]

        current_season = int(getattr(self._client, "season", datetime.now(timezone.utc).year))
        season_rows: dict[int, list[dict[str, Any]]] = {}
        season_errors: list[dict[str, Any]] = []
        for season in range(start_season, end_season + 1):
            if season >= current_season and not allow_partial:
                season_errors.append({
                    "ok": False,
                    "season": season,
                    "partial": True,
                    "reason": "Current in-progress season backtests require allow_partial=true",
                })
                continue
            season_rows[season] = await self._loader.load_season(season)

        models = []
        for model in F1ModelRegistry.list_models():
            model_id = model["model_id"]
            seasons = list(season_errors)
            all_races: list[dict[str, Any]] = []
            for season, races in season_rows.items():
                completed = [race for race in races if race.get("Results")]
                race_rows = await asyncio.to_thread(
                    self._backtest_completed_races,
                    season,
                    races,
                    completed,
                    model_id,
                    stage,
                    not include_races,
                )
                all_races.extend(race_rows)
                summary = summarize_races(race_rows)
                season_result = {
                    "ok": True,
                    "season": season,
                    "partial": False,
                    "model_version": _model_version(race_rows),
                    "model_id": model_id,
                    "stage": stage,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    **summary,
                    "fallback_coverage": _fallback_coverage_summary(race_rows),
                    "leakage_status": _leakage_status_summary(race_rows),
                    "recommended_weights": recommend_weight_adjustments(summary),
                }
                if include_races:
                    season_result["races"] = race_rows
                seasons.append(season_result)

            summary = summarize_races(all_races)
            models.append({
                "ok": True,
                "start_season": start_season,
                "end_season": end_season,
                "model_id": model_id,
                "stage": stage,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                **summary,
                "fallback_coverage": _fallback_coverage_summary(all_races),
                "leakage_status": _leakage_status_summary(all_races),
                "recommended_weights": recommend_weight_adjustments(summary),
                "seasons": seasons,
            })

        result = _comparison_result({"start_season": start_season, "end_season": end_season}, models)
        self._cache[cache_key] = result
        return result

    def _backtest_completed_races(
        self,
        season: int,
        races: list[dict[str, Any]],
        completed: list[dict[str, Any]],
        model_id: str,
        stage: str,
        compact: bool = False,
    ) -> list[dict[str, Any]]:
        return [
            self.backtest_race_from_rows(
                season,
                races,
                int(race.get("round") or 0),
                model_id=model_id,
                stage=stage,
                compact=compact,
            )
            for race in completed
        ]

    async def deep_backtest(
        self,
        start_season: int,
        end_season: int,
        *,
        stages: list[str] | None = None,
        model_ids: list[str] | None = None,
        include_races: bool = False,
        allow_partial: bool = False,
        include_ablations: bool = True,
        export_artifact: bool = False,
        cache_policy: str = "read_through",
    ) -> dict[str, Any]:
        """Run a multi-model, multi-stage backtest from one cached data load.

        This intentionally avoids calling ``backtest_summary`` repeatedly because
        a deep run can fan out into many model/stage combinations. The loader is
        called once per season, then every replay is built from the in-memory
        season rows. That keeps the analysis broader without repeatedly hitting
        Jolpica/OpenF1 limits.
        """
        start_season = int(start_season)
        end_season = int(end_season)
        if end_season < start_season:
            start_season, end_season = end_season, start_season
        stages = _normalized_stage_list(stages)
        model_ids = _normalized_model_ids(model_ids)
        cache_policy = _cache_policy(cache_policy)

        cache_key = (
            "deep",
            start_season,
            end_season,
            tuple(stages),
            tuple(model_ids),
            include_races,
            allow_partial,
            include_ablations,
            export_artifact,
            cache_policy,
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        current_season = int(getattr(self._client, "season", datetime.now(timezone.utc).year))
        season_rows: dict[int, list[dict[str, Any]]] = {}
        load_errors: list[dict[str, Any]] = []
        for season in range(start_season, end_season + 1):
            if season >= current_season and not allow_partial:
                load_errors.append({
                    "season": season,
                    "reason": "current_in_progress_requires_allow_partial",
                    "partial": True,
                })
                continue
            try:
                season_rows[season] = await self._loader.load_season(season, cache_policy=cache_policy)
            except Exception as exc:
                load_errors.append({"season": season, "reason": str(exc), "partial": False})

        runs: list[dict[str, Any]] = []
        for model_id in model_ids:
            for stage in stages:
                race_rows: list[dict[str, Any]] = []
                skipped: list[dict[str, Any]] = []
                for season, races in season_rows.items():
                    for race in [item for item in races if item.get("Results")]:
                        round_num = int(race.get("round") or 0)
                        try:
                            race_rows.append(self.backtest_race_from_rows(
                                season,
                                races,
                                round_num,
                                model_id=model_id,
                                stage=stage,
                            ))
                        except Exception as exc:
                            skipped.append({
                                "season": season,
                                "round": round_num,
                                "reason": str(exc),
                            })
                summary = summarize_races(race_rows)
                run = {
                    "ok": True,
                    "model_id": model_id,
                    "stage": stage,
                    "start_season": start_season,
                    "end_season": end_season,
                    "season_count": len(season_rows),
                    "skipped_races": skipped,
                    **summary,
                    "recommended_weights": recommend_weight_adjustments(summary),
                    "track_segment_summary": _track_segment_summary(race_rows),
                }
                if include_races:
                    run["races"] = race_rows
                runs.append(run)

        best = _best_model(runs)
        stage_matrix = _stage_matrix(runs)
        model_matrix = _model_matrix(runs)
        track_segment_matrix = _track_segment_matrix(runs)
        track_segment_ablations = _track_segment_ablations(track_segment_matrix) if include_ablations else []
        coverage = _coverage_report(season_rows, stages)
        result = {
            "ok": True,
            "mode": "deep_backtest",
            "start_season": start_season,
            "end_season": end_season,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "requested": {
                "stages": stages,
                "model_ids": model_ids,
                "include_races": include_races,
                "allow_partial": allow_partial,
                "include_ablations": include_ablations,
                "export_artifact": export_artifact,
                "cache_policy": cache_policy,
            },
            "load_strategy": {
                "season_rows_loaded_once": True,
                "upstream_calls_minimized": True,
                "cache_policy": cache_policy,
                "loader_cache_scope": "per deep backtest request plus F1Client season caches",
                "completed_races_loaded": sum(
                    1 for races in season_rows.values() for race in races if race.get("Results")
                ),
                "seasons_loaded": sorted(season_rows),
                "load_errors": load_errors,
                "historical_cache": self._loader.cache_stats,
            },
            "coverage": coverage,
            "best_model_id": best.get("model_id") if best else None,
            "best_stage": best.get("stage") if best else None,
            "best_run": best,
            "stage_matrix": stage_matrix,
            "model_matrix": model_matrix,
            "track_segment_matrix": track_segment_matrix,
            "evidence_ablations": _evidence_ablations(stage_matrix) if include_ablations else [],
            "track_segment_ablations": track_segment_ablations,
            "limitations": _deep_limitations(season_rows, load_errors, stages, coverage),
            "recommendations": _deep_recommendations(runs, coverage, track_segment_matrix, track_segment_ablations),
            "compact_report": _compact_report(best, stage_matrix, model_matrix, coverage, track_segment_matrix),
            "runs": runs,
            "selection_rule": "lowest log loss, then lowest Brier score, then highest winner accuracy",
        }
        if export_artifact:
            result["artifact"] = _write_deep_backtest_artifact(result)
        self._cache[cache_key] = result
        return result

    def backtest_race_from_rows(self, season: int, races: list[dict[str, Any]], round_num: int, model_id: str | None = None, stage: str = "pre_weekend", compact: bool = False) -> dict[str, Any]:
        model_id = model_id or PRODUCTION_MODEL_ID
        stage = _stage_key(stage)
        replay = self._builder.build(season, races, round_num, stage=stage)
        service = F1PredictionService(model_id=model_id)
        service.load(replay.drivers, replay.constructors, replay.features, sentiment={})
        baseline_prediction = service.predict_race(replay.race)
        if compact:
            metrics = evaluate_race(baseline_prediction, replay.actual_results)
            return {
                "season": season,
                "round": round_num,
                "model_id": model_id,
                "stage": replay.stage,
                "race_name": replay.race.name,
                "circuit": replay.race.circuit,
                "country": replay.race.country,
                "track_key": replay.race.circuit_id,
                "track_segments": [],
                "track_traits": {},
                "completed_races_before": replay.features.get("completed_races", 0),
                "replay_evidence": {
                    "stage": replay.stage,
                    "source_coverage": replay.features.get("source_coverage") or {},
                    "backtest": replay.features.get("backtest") or {},
                },
                "actual_winner": metrics["actual_winner"],
                "predicted_winner": metrics["predicted_winner"],
                "actual_podium": metrics["actual_podium"],
                "predicted_top3": metrics["predicted_top3"],
                "probability_distribution": _probability_distribution(baseline_prediction),
                "probability_audit": {},
                "calibration_profile": {},
                "probability_governance": {},
                "model_confidence": baseline_prediction.confidence,
                "model_input_metadata": _model_input_metadata(baseline_prediction),
                "fallback_coverage": _fallback_coverage_race(baseline_prediction),
                "leakage_status": _leakage_status_race(replay),
                "missing_feature_groups": [],
                "feature_sources": {"sentiment": "disabled_for_backtest_v1"},
                "component_scores": _component_audit(baseline_prediction),
                "metrics": metrics,
                "model_version": baseline_prediction.model_version,
                "baseline_model_version": baseline_prediction.model_version,
                "compact_backtest": True,
            }
        snapshot = service.build_features(replay.race, "race")
        profile = _replay_profile(replay)
        truth = _replay_truth(replay)
        simulation = build_session_projection(
            race=replay.race,
            drivers=replay.drivers,
            constructors=replay.constructors,
            prediction=baseline_prediction.model_dump(mode="json"),
            features=replay.features,
            qualifying=(replay.features.get("replay_profile") or {}).get("qualifying") or [],
            sprint=[],
            results=[],
            session="race",
            live=replay.stage == "live",
        )
        simulation["model_id"] = model_id
        simulation["truth"] = truth
        simulation = enrich_probability_payload(
            simulation,
            profile=profile,
            truth=truth,
            stage=replay.stage,
            live=replay.stage == "live",
            model_id=model_id,
        )
        prediction = _prediction_from_simulation(simulation, baseline_prediction)
        metrics = evaluate_race(prediction, replay.actual_results)
        distribution = _probability_distribution(prediction)
        components = _component_audit_from_rows(simulation.get("simulations") or [], baseline_prediction)
        track_traits = snapshot.track or {}
        track_segments = _track_segments(track_traits)
        return {
            "season": season,
            "round": round_num,
            "model_id": model_id,
            "stage": replay.stage,
            "race_name": replay.race.name,
            "circuit": replay.race.circuit,
            "country": replay.race.country,
            "track_key": track_traits.get("track_key"),
            "track_segments": track_segments,
            "track_traits": {
                "street_circuit": track_traits.get("street_circuit"),
                "high_speed": track_traits.get("high_speed"),
                "overtaking_difficulty": track_traits.get("overtaking_difficulty"),
                "qualifying_importance": track_traits.get("qualifying_importance"),
                "tire_stress": track_traits.get("tire_stress"),
                "safety_car_probability": track_traits.get("safety_car_probability"),
                "source": track_traits.get("source"),
            },
            "completed_races_before": replay.features.get("completed_races", 0),
            "replay_evidence": {
                "stage": replay.stage,
                "source_coverage": replay.features.get("source_coverage") or {},
                "backtest": replay.features.get("backtest") or {},
            },
            "actual_winner": metrics["actual_winner"],
            "predicted_winner": metrics["predicted_winner"],
            "actual_podium": metrics["actual_podium"],
            "predicted_top3": metrics["predicted_top3"],
            "probability_distribution": distribution,
            "probability_audit": simulation.get("probability_audit") or {},
            "calibration_profile": simulation.get("calibration_profile") or {},
            "probability_governance": simulation.get("probability_governance") or {},
            "model_confidence": prediction.confidence,
            "model_input_metadata": _model_input_metadata(baseline_prediction),
            "fallback_coverage": _fallback_coverage_race(baseline_prediction),
            "leakage_status": _leakage_status_race(replay),
            "missing_feature_groups": snapshot.missing_data,
            "feature_sources": {
                "track": snapshot.track.get("source"),
                "weather": snapshot.weather.get("source"),
                "tires": snapshot.tires.get("source"),
                "sentiment": "disabled_for_backtest_v1",
            },
            "component_scores": components,
            "metrics": metrics,
            "model_version": prediction.model_version,
            "baseline_model_version": baseline_prediction.model_version,
        }

    def _default_season(self) -> int:
        return int(getattr(self._client, "season", datetime.now(timezone.utc).year)) - 1


def _probability_distribution(prediction) -> list[dict[str, Any]]:
    rows = []
    for item in (prediction.driver_predictions or {}).values():
        rows.append({
            "driver_id": item.driver_id,
            "driver_name": item.driver_name,
            "win_probability": item.win_prob,
            "podium_probability": item.podium_prob,
            "top5_probability": item.top5_prob,
            "wdc_probability": item.wdc_prob,
            "predicted_position": item.predicted_position,
        })
    return sorted(rows, key=lambda item: item["win_probability"], reverse=True)


def _prediction_from_simulation(simulation: dict[str, Any], baseline: RacePrediction) -> RacePrediction:
    baseline_by_driver = baseline.driver_predictions or {}
    predictions = {}
    for row in simulation.get("simulations") or simulation.get("probabilities") or []:
        driver_id = row.get("driver_id")
        if not driver_id:
            continue
        prior = baseline_by_driver.get(driver_id)
        components = row.get("components") or {}
        predicted_position = row.get("rank")
        if predicted_position is None and prior:
            predicted_position = prior.predicted_position
        predictions[driver_id] = DriverRacePrediction(
            driver_id=driver_id,
            driver_name=row.get("driver_name") or (prior.driver_name if prior else driver_id),
            win_prob=float(row.get("win_probability") or row.get("calibrated_probability") or 0.0),
            podium_prob=float(row.get("podium_probability") or 0.0),
            top5_prob=float(row.get("top5_probability") or 0.0),
            predicted_position=_safe_int(predicted_position),
            expected_finish=float(row.get("expected_finish") or 0.0) or None,
            form_score=components.get("recent_form") if components else (prior.form_score if prior else None),
            team_score=components.get("team_pace") if components else (prior.team_score if prior else None),
            driver_skill_score=prior.driver_skill_score if prior else None,
            car_performance_score=components.get("car_model") if components else (prior.car_performance_score if prior else None),
            performance_score=prior.performance_score if prior else None,
            qualifying_pace_score=components.get("qualifying_pace") if components else (prior.qualifying_pace_score if prior else None),
            race_pace_score=components.get("race_pace") if components else (prior.race_pace_score if prior else None),
            reliability_score=components.get("reliability") if components else (prior.reliability_score if prior else None),
            dnf_prob=float(row.get("dnf_probability") or components.get("dnf_probability") or 0.0),
            sentiment_score=components.get("sentiment") if components else (prior.sentiment_score if prior else None),
            wdc_prob=components.get("wdc_probability") if components else (prior.wdc_prob if prior else None),
        )
    return RacePrediction(
        driver_predictions=predictions,
        model_version=f"{baseline.model_version}+{simulation.get('model_version') or 'session_projection'}+calibrated",
        generated_at=datetime.now(timezone.utc),
        data_sources=[
            "backtest_replay_no_future_race_results",
            "session_projection_monte_carlo",
            "stage_probability_calibration",
        ],
        confidence=simulation.get("confidence"),
    )


def _model_input_metadata(prediction: RacePrediction) -> dict[str, Any]:
    return {
        "model_id": prediction.model_id,
        "ml_input_source": prediction.ml_input_source,
        "ml_provider_sources": prediction.ml_provider_sources,
        "ml_fallback_reason": prediction.ml_fallback_reason,
        "ml_confidence": prediction.ml_confidence,
        "trained_artifacts_used": prediction.trained_artifacts_used,
        "evidence_groups_used": prediction.evidence_groups_used,
        "simulator_iterations": prediction.simulator_iterations,
        "ml_artifact_id": prediction.ml_artifact_id,
        "ml_artifact_version": prediction.ml_artifact_version,
        "ml_artifact_readiness": prediction.ml_artifact_readiness,
        "ml_model_contract_used": prediction.ml_model_contract_used,
        "ml_model_adapters_used": prediction.ml_model_adapters_used,
        "ml_model_fallback_reason": prediction.ml_model_fallback_reason,
        "pace_adapter_source": prediction.pace_adapter_source,
        "dnf_adapter_source": prediction.dnf_adapter_source,
        "rating_adapter_source": prediction.rating_adapter_source,
    }


def _fallback_coverage_race(prediction: RacePrediction) -> dict[str, Any]:
    ml_source = prediction.ml_input_source
    is_ml = prediction.model_id == "ml_simulator_v1" or bool(ml_source)
    return {
        "model_id": prediction.model_id,
        "is_ml_simulator": is_ml,
        "ml_input_source": ml_source,
        "trained_artifacts_used": bool(prediction.trained_artifacts_used),
        "fallback_used": bool(is_ml and prediction.ml_fallback_reason),
        "fallback_reason": prediction.ml_fallback_reason,
        "evidence_groups_used": prediction.evidence_groups_used or [],
    }


def _fallback_coverage_summary(races: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(races)
    fallback_count = sum(1 for race in races if (race.get("fallback_coverage") or {}).get("fallback_used"))
    trained_count = sum(1 for race in races if (race.get("fallback_coverage") or {}).get("trained_artifacts_used"))
    by_source: dict[str, int] = {}
    for race in races:
        source = str((race.get("fallback_coverage") or {}).get("ml_input_source") or "non_ml_or_unknown")
        by_source[source] = by_source.get(source, 0) + 1
    return {
        "race_count": total,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / total, 4) if total else 0.0,
        "trained_artifact_count": trained_count,
        "trained_artifact_rate": round(trained_count / total, 4) if total else 0.0,
        "by_input_source": by_source,
    }


def _leakage_status_race(replay) -> dict[str, Any]:
    backtest_meta = replay.features.get("backtest") or {}
    return {
        "status": "guarded",
        "stage": replay.stage,
        "future_results_excluded": True,
        "target_practice_used": bool(backtest_meta.get("target_practice_used")),
        "target_qualifying_used": bool(backtest_meta.get("target_qualifying_used")),
        "target_race_inputs_used": bool(backtest_meta.get("target_race_inputs_used")),
        "notes": "Replay builder exposes only facts available for the selected stage.",
    }


def _leakage_status_summary(races: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [race.get("leakage_status") or {} for race in races]
    return {
        "status": "guarded" if all((item.get("status") == "guarded") for item in statuses) else "unknown",
        "race_count": len(races),
        "future_results_excluded": all(item.get("future_results_excluded") for item in statuses) if statuses else True,
        "stage_usage": {
            "practice": sum(1 for item in statuses if item.get("target_practice_used")),
            "qualifying": sum(1 for item in statuses if item.get("target_qualifying_used")),
            "race_inputs": sum(1 for item in statuses if item.get("target_race_inputs_used")),
        },
    }


def _component_audit(prediction) -> dict[str, dict[str, Any]]:
    rows = {}
    for item in (prediction.driver_predictions or {}).values():
        rows[item.driver_id] = {
            "form": item.form_score,
            "team": item.team_score,
            "performance": item.performance_score,
            "driver_skill": item.driver_skill_score,
            "car_performance": item.car_performance_score,
            "qualifying_pace": item.qualifying_pace_score,
            "race_pace": item.race_pace_score,
            "track_fit": item.track_fit_score,
            "tire_strategy": item.tire_strategy_score,
            "weather_risk": item.weather_risk_score,
            "reliability": item.reliability_score,
            "dnf_probability": item.dnf_prob,
            "sentiment": item.sentiment_score,
            "explanation": item.explanation,
        }
    return rows


def _component_audit_from_rows(rows: list[dict[str, Any]], baseline: RacePrediction) -> dict[str, dict[str, Any]]:
    if not rows:
        return _component_audit(baseline)
    output = {}
    for row in rows:
        driver_id = row.get("driver_id")
        if not driver_id:
            continue
        components = row.get("components") or {}
        output[driver_id] = {
            "form": components.get("recent_form"),
            "team": components.get("team_pace"),
            "performance": row.get("calibrated_probability") or row.get("win_probability"),
            "driver_skill": components.get("qualifying_pace"),
            "car_performance": components.get("car_model"),
            "qualifying_pace": components.get("qualifying_pace"),
            "race_pace": components.get("race_pace"),
            "track_fit": components.get("race_start_grid") or components.get("qualifying"),
            "grid_position": components.get("grid_position"),
            "grid_penalty": components.get("grid_penalty"),
            "grid_modifier": components.get("grid_modifier"),
            "tire_strategy": components.get("strategy"),
            "weather_risk": components.get("weather"),
            "reliability": components.get("reliability"),
            "dnf_probability": row.get("dnf_probability") or components.get("dnf_probability"),
            "sentiment": components.get("sentiment"),
            "explanation": row.get("signals") or [],
        }
    return output


def _replay_profile(replay) -> dict[str, Any]:
    profile = replay.features.get("replay_profile") or {}
    qualifying = profile.get("qualifying") or []
    sessions = []
    if replay.stage in {"practice_available", "post_qualifying", "live", "completed"}:
        sessions.append({"name": "Practice", "status": "completed"})
    if replay.stage in {"post_qualifying", "live", "completed"} and qualifying:
        sessions.append({"name": "Qualifying", "status": "completed"})
    return {
        "ok": True,
        "race": replay.race.model_dump(mode="json"),
        "sessions": sessions,
        "qualifying": qualifying,
        "results": [],
        "context": {
            "has_results": False,
            "has_qualifying": bool(qualifying),
            "completed_sessions": len(sessions),
        },
    }


def _replay_truth(replay) -> dict[str, Any]:
    coverage = replay.features.get("source_coverage") or {}
    mode = "historical" if replay.stage in {"post_qualifying", "completed"} and coverage.get("qualifying_sessions") else "estimated"
    if replay.stage == "practice_available" and coverage.get("practice_sessions"):
        mode = "recent"
    confidence = 0.74 if mode == "historical" else 0.56 if mode == "recent" else 0.24
    missing = []
    if not coverage.get("practice_sessions") and replay.stage in {"practice_available", "post_qualifying", "live", "completed"}:
        missing.append("practice_laps")
    if not coverage.get("qualifying_sessions") and replay.stage in {"post_qualifying", "live", "completed"}:
        missing.append("qualifying_grid")
    return {
        "ok": True,
        "source_mode": mode,
        "confidence": confidence,
        "missing_groups": missing,
        "fallback_reason": "Backtest replay uses only stage-available historical facts",
        "data_age_seconds": None,
        "drivers": [],
        "by_driver_id": {},
    }


def _model_version(race_rows: list[dict[str, Any]]) -> str | None:
    return next((row.get("model_version") for row in race_rows if row.get("model_version")), None)


def _stage_key(stage: str | None) -> str:
    value = str(stage or "pre_weekend").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pre": "pre_weekend",
        "pre_race": "pre_weekend",
        "practice": "practice_available",
        "post_practice": "practice_available",
        "fp": "practice_available",
        "post_fp": "practice_available",
        "quali": "post_qualifying",
        "qualifying": "post_qualifying",
        "post_quali": "post_qualifying",
        "race_live": "live",
        "replay": "live",
        "complete": "completed",
    }
    value = aliases.get(value, value)
    if value not in {"pre_weekend", "practice_available", "post_qualifying", "live", "completed"}:
        return "pre_weekend"
    return value


def _normalized_stage_list(stages: list[str] | None) -> list[str]:
    values = [_stage_key(stage) for stage in (stages or []) if str(stage or "").strip()]
    if not values:
        values = ["pre_weekend", "practice_available", "post_qualifying"]
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _normalized_model_ids(model_ids: list[str] | None) -> list[str]:
    known = {model["model_id"] for model in F1ModelRegistry.list_models()}
    output = []
    for model_id in model_ids or []:
        value = str(model_id or "").strip()
        if value and value in known and value not in output:
            output.append(value)
    if not output:
        output = [model["model_id"] for model in F1ModelRegistry.list_models()]
    return output


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _comparison_result(scope: dict[str, Any], models: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [model for model in models if model.get("ok")]
    best = _best_model(successful)
    return {
        "ok": True,
        **scope,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_model_id": best.get("model_id") if best else None,
        "best_model": best,
        "models": models,
        "selection_rule": "lowest log loss, then lowest Brier score, then highest winner accuracy",
    }


def _best_model(models: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not models:
        return None
    return sorted(
        models,
        key=lambda model: (
            float(model.get("avg_log_loss") or 999.0),
            float(model.get("avg_brier_score") or 999.0),
            -float(model.get("winner_accuracy") or 0.0),
        ),
    )[0]


def _stage_matrix(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        if run.get("ok"):
            by_stage.setdefault(str(run.get("stage") or "unknown"), []).append(run)
    for stage, items in sorted(by_stage.items()):
        best = _best_model(items)
        rows.append({
            "stage": stage,
            "run_count": len(items),
            "best_model_id": best.get("model_id") if best else None,
            "best_log_loss": best.get("avg_log_loss") if best else None,
            "best_brier_score": best.get("avg_brier_score") if best else None,
            "best_winner_accuracy": best.get("winner_accuracy") if best else None,
            "avg_race_count": _avg([float(item.get("race_count") or 0.0) for item in items]),
            "models": [
                {
                    "model_id": item.get("model_id"),
                    "race_count": item.get("race_count"),
                    "winner_accuracy": item.get("winner_accuracy"),
                    "avg_podium_hit_rate": item.get("avg_podium_hit_rate"),
                    "avg_points_hit_rate": item.get("avg_points_hit_rate"),
                    "avg_brier_score": item.get("avg_brier_score"),
                    "avg_log_loss": item.get("avg_log_loss"),
                    "avg_expected_finish_error": item.get("avg_expected_finish_error"),
                }
                for item in sorted(items, key=lambda row: str(row.get("model_id") or ""))
            ],
        })
    return rows


def _model_matrix(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        if run.get("ok"):
            by_model.setdefault(str(run.get("model_id") or "unknown"), []).append(run)
    for model_id, items in sorted(by_model.items()):
        best = _best_model(items)
        rows.append({
            "model_id": model_id,
            "run_count": len(items),
            "best_stage": best.get("stage") if best else None,
            "best_log_loss": best.get("avg_log_loss") if best else None,
            "best_brier_score": best.get("avg_brier_score") if best else None,
            "best_winner_accuracy": best.get("winner_accuracy") if best else None,
            "stages": [
                {
                    "stage": item.get("stage"),
                    "race_count": item.get("race_count"),
                    "winner_accuracy": item.get("winner_accuracy"),
                    "avg_podium_hit_rate": item.get("avg_podium_hit_rate"),
                    "avg_points_hit_rate": item.get("avg_points_hit_rate"),
                    "avg_brier_score": item.get("avg_brier_score"),
                    "avg_log_loss": item.get("avg_log_loss"),
                    "avg_expected_finish_error": item.get("avg_expected_finish_error"),
                }
                for item in sorted(items, key=lambda row: str(row.get("stage") or ""))
            ],
        })
    return rows


def _track_segment_summary(race_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize metrics by circuit trait tags.

    A race can belong to several tags, for example Monaco is both a street
    circuit and a track-position circuit. Multi-tag grouping is intentional:
    it tells calibration where a model is fragile without forcing tracks into
    one artificial bucket.
    """
    by_segment: dict[str, list[dict[str, Any]]] = {}
    for row in race_rows:
        segments = row.get("track_segments") or ["uncategorized"]
        for segment in segments:
            by_segment.setdefault(str(segment), []).append(row)

    output = []
    for segment, rows in sorted(by_segment.items()):
        summary = summarize_races(rows)
        metrics = [row.get("metrics") or {} for row in rows]
        top_probabilities = [float(item.get("top_probability") or 0.0) for item in metrics]
        winner_probabilities = [float(item.get("winner_probability") or 0.0) for item in metrics]
        output.append({
            "segment": segment,
            "race_count": summary.get("race_count"),
            "winner_accuracy": summary.get("winner_accuracy"),
            "avg_podium_hit_rate": summary.get("avg_podium_hit_rate"),
            "avg_points_hit_rate": summary.get("avg_points_hit_rate"),
            "avg_brier_score": summary.get("avg_brier_score"),
            "avg_log_loss": summary.get("avg_log_loss"),
            "avg_expected_finish_error": summary.get("avg_expected_finish_error"),
            "avg_top_probability": _avg(top_probabilities),
            "avg_winner_probability": _avg(winner_probabilities),
            "overconfidence_gap": _overconfidence_gap(top_probabilities, metrics),
            "sample_warning": len(rows) < 8,
        })
    return output


def _track_segment_matrix(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_segment: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        if not run.get("ok"):
            continue
        for row in run.get("track_segment_summary") or []:
            by_segment.setdefault(str(row.get("segment") or "uncategorized"), []).append({
                **row,
                "model_id": run.get("model_id"),
                "stage": run.get("stage"),
            })

    output = []
    for segment, rows in sorted(by_segment.items()):
        best = _best_segment_row(rows)
        output.append({
            "segment": segment,
            "run_count": len(rows),
            "best_model_id": best.get("model_id") if best else None,
            "best_stage": best.get("stage") if best else None,
            "best_log_loss": best.get("avg_log_loss") if best else None,
            "best_brier_score": best.get("avg_brier_score") if best else None,
            "best_winner_accuracy": best.get("winner_accuracy") if best else None,
            "best_race_count": best.get("race_count") if best else None,
            "sample_warning": any(bool(row.get("sample_warning")) for row in rows),
            "models": [
                {
                    "model_id": row.get("model_id"),
                    "stage": row.get("stage"),
                    "race_count": row.get("race_count"),
                    "winner_accuracy": row.get("winner_accuracy"),
                    "avg_podium_hit_rate": row.get("avg_podium_hit_rate"),
                    "avg_points_hit_rate": row.get("avg_points_hit_rate"),
                    "avg_brier_score": row.get("avg_brier_score"),
                    "avg_log_loss": row.get("avg_log_loss"),
                    "avg_expected_finish_error": row.get("avg_expected_finish_error"),
                    "overconfidence_gap": row.get("overconfidence_gap"),
                    "sample_warning": row.get("sample_warning"),
                }
                for row in sorted(rows, key=lambda item: (str(item.get("stage") or ""), str(item.get("model_id") or "")))
            ],
        })
    return output


def _track_segment_ablations(track_segment_matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for segment in track_segment_matrix:
        rows = segment.get("models") or []
        pre = _best_segment_stage_row(rows, "pre_weekend")
        if not pre:
            continue
        for stage, label in [
            ("practice_available", "practice_evidence"),
            ("post_qualifying", "qualifying_grid_evidence"),
            ("live", "live_replay_evidence"),
            ("completed", "completed_audit_evidence"),
        ]:
            comparison = _best_segment_stage_row(rows, stage)
            if not comparison:
                continue
            pre_count = int(pre.get("race_count") or 0)
            comparison_count = int(comparison.get("race_count") or 0)
            output.append({
                "segment": segment.get("segment"),
                "evidence_group": label,
                "baseline_stage": "pre_weekend",
                "comparison_stage": stage,
                "baseline_model_id": pre.get("model_id"),
                "comparison_model_id": comparison.get("model_id"),
                "race_count": comparison_count,
                "baseline_race_count": pre_count,
                "sample_warning": pre_count < 8 or comparison_count < 8,
                "delta_log_loss": _delta(comparison.get("avg_log_loss"), pre.get("avg_log_loss"), lower_is_better=True),
                "delta_brier_score": _delta(comparison.get("avg_brier_score"), pre.get("avg_brier_score"), lower_is_better=True),
                "delta_winner_accuracy": _delta(comparison.get("winner_accuracy"), pre.get("winner_accuracy")),
                "delta_expected_finish_error": _delta(
                    comparison.get("avg_expected_finish_error"),
                    pre.get("avg_expected_finish_error"),
                    lower_is_better=True,
                ),
                "interpretation": _ablation_interpretation(_segment_ablation_row(comparison), _segment_ablation_row(pre)),
            })
    return output


def _best_segment_stage_row(rows: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("stage") == stage]
    return _best_segment_row(candidates)


def _segment_ablation_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "best_log_loss": row.get("avg_log_loss"),
        "best_brier_score": row.get("avg_brier_score"),
        "best_winner_accuracy": row.get("winner_accuracy"),
    }


def _best_segment_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("avg_log_loss") or 999.0),
            float(row.get("avg_brier_score") or 999.0),
            -float(row.get("winner_accuracy") or 0.0),
            -float(row.get("race_count") or 0.0),
        ),
    )[0]


def _track_segments(track: dict[str, Any]) -> list[str]:
    if not track:
        return ["uncategorized"]
    segments = []
    if track.get("street_circuit"):
        segments.append("street")
    if track.get("high_speed"):
        segments.append("high_speed")
    overtaking = _safe_float(track.get("overtaking_difficulty"))
    qualifying = _safe_float(track.get("qualifying_importance"))
    tire_stress = _safe_float(track.get("tire_stress"))
    safety_car = _safe_float(track.get("safety_car_probability"))
    if overtaking is not None and overtaking >= 0.68 or qualifying is not None and qualifying >= 0.76:
        segments.append("track_position")
    if overtaking is not None and overtaking <= 0.44:
        segments.append("overtaking_friendly")
    if tire_stress is not None and tire_stress >= 0.66:
        segments.append("tyre_stress")
    if safety_car is not None and safety_car >= 0.55:
        segments.append("high_safety_car")
    if not segments:
        segments.append("balanced")
    return segments


def _overconfidence_gap(top_probabilities: list[float], metrics: list[dict[str, Any]]) -> float | None:
    if not top_probabilities or not metrics:
        return None
    hit_rate = sum(1 for item in metrics if item.get("winner_hit")) / max(1, len(metrics))
    return round(_avg(top_probabilities) - hit_rate, 4)


def _evidence_ablations(stage_matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report stage deltas as an evidence-ablation proxy.

    The replay builder only exposes evidence by stage in v1. Comparing
    pre-weekend to practice/post-qualifying lets us quantify whether adding
    those evidence groups helped or hurt without extra upstream fetches.
    """
    by_stage = {row.get("stage"): row for row in stage_matrix}
    pre = by_stage.get("pre_weekend")
    if not pre:
        return []
    output = []
    for stage, label in [
        ("practice_available", "practice_evidence"),
        ("post_qualifying", "qualifying_grid_evidence"),
        ("live", "live_replay_evidence"),
        ("completed", "completed_audit_evidence"),
    ]:
        row = by_stage.get(stage)
        if not row:
            continue
        output.append({
            "evidence_group": label,
            "baseline_stage": "pre_weekend",
            "comparison_stage": stage,
            "best_model_id": row.get("best_model_id"),
            "delta_log_loss": _delta(row.get("best_log_loss"), pre.get("best_log_loss"), lower_is_better=True),
            "delta_brier_score": _delta(row.get("best_brier_score"), pre.get("best_brier_score"), lower_is_better=True),
            "delta_winner_accuracy": _delta(row.get("best_winner_accuracy"), pre.get("best_winner_accuracy")),
            "interpretation": _ablation_interpretation(row, pre),
        })
    return output


def _coverage_report(season_rows: dict[int, list[dict[str, Any]]], stages: list[str]) -> dict[str, Any]:
    seasons = []
    totals = {
        "races": 0,
        "completed_races": 0,
        "qualifying_races": 0,
        "practice_races": 0,
        "practice_detail_races": 0,
        "practice_quality_races": 0,
        "practice_distribution_races": 0,
        "race_input_races": 0,
        "race_stint_races": 0,
        "sprint_races": 0,
        "result_driver_rows": 0,
        "qualifying_driver_rows": 0,
        "practice_driver_rows": 0,
        "practice_detail_driver_rows": 0,
        "practice_quality_driver_rows": 0,
        "practice_distribution_driver_rows": 0,
        "race_input_driver_rows": 0,
        "race_stint_driver_rows": 0,
    }
    for season, races in sorted(season_rows.items()):
        row = {
            "season": season,
            "races": len(races),
            "completed_races": sum(1 for race in races if race.get("Results")),
            "qualifying_races": sum(1 for race in races if race.get("QualifyingResults") or race.get("qualifying")),
            "practice_races": sum(1 for race in races if _practice_available(race)),
            "practice_detail_races": sum(1 for race in races if _practice_detail_available(race)),
            "practice_quality_races": sum(1 for race in races if _practice_quality_available(race)),
            "practice_distribution_races": sum(1 for race in races if _practice_distribution_available(race)),
            "race_input_races": sum(1 for race in races if _race_inputs_available(race)),
            "race_stint_races": sum(1 for race in races if _race_stint_available(race)),
            "sprint_races": sum(1 for race in races if race.get("SprintResults") or race.get("Sprint")),
            "result_driver_rows": sum(len(race.get("Results") or []) for race in races),
            "qualifying_driver_rows": sum(len(race.get("QualifyingResults") or race.get("qualifying") or []) for race in races),
            "practice_driver_rows": sum(len(_practice_rows(race)) for race in races),
            "practice_detail_driver_rows": sum(_practice_detail_row_count(race) for race in races),
            "practice_quality_driver_rows": sum(_practice_quality_row_count(race) for race in races),
            "practice_distribution_driver_rows": sum(_practice_distribution_row_count(race) for race in races),
            "race_input_driver_rows": sum(_race_input_row_count(race) for race in races),
            "race_stint_driver_rows": sum(_race_stint_row_count(race) for race in races),
        }
        for key in totals:
            totals[key] += int(row.get(key) or 0)
        row["qualifying_coverage"] = round(row["qualifying_races"] / max(1, row["completed_races"]), 4)
        row["practice_coverage"] = round(row["practice_races"] / max(1, row["completed_races"]), 4)
        seasons.append(row)

    completed = totals["completed_races"]
    evidence_groups = {
        "results": {
            "available_races": totals["completed_races"],
            "coverage": round(totals["completed_races"] / max(1, totals["races"]), 4),
            "driver_rows": totals["result_driver_rows"],
        },
        "qualifying": {
            "available_races": totals["qualifying_races"],
            "coverage": round(totals["qualifying_races"] / max(1, completed), 4),
            "driver_rows": totals["qualifying_driver_rows"],
            "requested": any(stage in {"post_qualifying", "live", "completed"} for stage in stages),
        },
        "practice": {
            "available_races": totals["practice_races"],
            "coverage": round(totals["practice_races"] / max(1, completed), 4),
            "driver_rows": totals["practice_driver_rows"],
            "detail_available_races": totals["practice_detail_races"],
            "detail_coverage": round(totals["practice_detail_races"] / max(1, completed), 4),
            "detail_driver_rows": totals["practice_detail_driver_rows"],
            "quality_available_races": totals["practice_quality_races"],
            "quality_coverage": round(totals["practice_quality_races"] / max(1, completed), 4),
            "quality_driver_rows": totals["practice_quality_driver_rows"],
            "distribution_available_races": totals["practice_distribution_races"],
            "distribution_coverage": round(totals["practice_distribution_races"] / max(1, completed), 4),
            "distribution_driver_rows": totals["practice_distribution_driver_rows"],
            "avg_telemetry_quality": _avg_practice_quality(season_rows),
            "requested": any(stage in {"practice_available", "live", "completed"} for stage in stages),
            "optional_for": [
                stage for stage in stages
                if stage == "post_qualifying"
            ],
        },
        "sprint": {
            "available_races": totals["sprint_races"],
            "coverage": round(totals["sprint_races"] / max(1, completed), 4),
            "requested": any(stage == "sprint" for stage in stages),
        },
        "race_inputs": {
            "available_races": totals["race_input_races"],
            "coverage": round(totals["race_input_races"] / max(1, completed), 4),
            "driver_rows": totals["race_input_driver_rows"],
            "stint_available_races": totals["race_stint_races"],
            "stint_coverage": round(totals["race_stint_races"] / max(1, completed), 4),
            "stint_driver_rows": totals["race_stint_driver_rows"],
            "requested": any(stage in {"live", "completed"} for stage in stages),
        },
    }
    return {
        "totals": totals,
        "seasons": seasons,
        "evidence_groups": evidence_groups,
        "coverage_grade": _coverage_grade(completed, evidence_groups),
    }


def _coverage_grade(completed_races: int, evidence_groups: dict[str, dict[str, Any]]) -> str:
    qualifying = float((evidence_groups.get("qualifying") or {}).get("coverage") or 0.0)
    practice = float((evidence_groups.get("practice") or {}).get("coverage") or 0.0)
    if completed_races >= 60 and qualifying >= 0.80 and practice >= 0.50:
        return "strong"
    if completed_races >= 30 and qualifying >= 0.60:
        return "usable"
    if completed_races >= 12:
        return "thin"
    return "very_thin"


def _deep_limitations(
    season_rows: dict[int, list[dict[str, Any]]],
    load_errors: list[dict[str, Any]],
    stages: list[str],
    coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    limitations = []
    if load_errors:
        limitations.append({
            "code": "season_load_errors",
            "severity": "medium",
            "message": "Some requested seasons were skipped.",
            "details": load_errors,
        })
    completed = int(((coverage.get("totals") or {}).get("completed_races")) or 0)
    if completed < 40:
        limitations.append({
            "code": "small_sample",
            "severity": "high",
            "message": "Backtest race count is still small for serious calibration.",
            "details": {"completed_races": completed},
        })
    evidence_groups = coverage.get("evidence_groups") or {}
    if any(stage in {"post_qualifying", "live", "completed"} for stage in stages):
        qualifying = evidence_groups.get("qualifying") or {}
        if float(qualifying.get("coverage") or 0.0) < 0.75:
            limitations.append({
                "code": "qualifying_coverage_low",
                "severity": "medium",
                "message": "Post-qualifying comparisons are constrained by incomplete qualifying/grid coverage.",
                "details": qualifying,
            })
    if any(stage in {"practice_available", "live", "completed"} for stage in stages):
        practice = evidence_groups.get("practice") or {}
        if float(practice.get("coverage") or 0.0) < 0.50:
            limitations.append({
                "code": "practice_coverage_low",
                "severity": "high",
                "message": "Practice-stage backtests are mostly proxy evidence until real FP lap rows are loaded.",
                "details": practice,
            })
    if any(stage in {"live", "completed"} for stage in stages):
        race_inputs = evidence_groups.get("race_inputs") or {}
        race_coverage = float(race_inputs.get("coverage") or 0.0)
        stint_coverage = float(race_inputs.get("stint_coverage") or 0.0)
        if race_coverage >= 0.75 and stint_coverage >= 0.75:
            limitations.append({
                "code": "race_compact_stint_distribution",
                "severity": "low",
                "message": "Live/completed replay has compact race order, gap, pit and stint distributions; full lap-by-lap race telemetry is still not replayed directly.",
                "details": {"requested_stages": stages, "race_inputs": race_inputs},
            })
        else:
            limitations.append({
                "code": "live_or_completed_telemetry_proxy",
                "severity": "medium",
                "message": "Live/completed stage evidence is replayed from compact historical rows, not full FastF1/OpenF1 lap telemetry yet.",
                "details": {"requested_stages": stages, "race_inputs": race_inputs},
            })
    elif any(stage == "practice_available" for stage in stages):
        practice = evidence_groups.get("practice") or {}
        detail_coverage = float(practice.get("detail_coverage") or 0.0)
        quality_coverage = float(practice.get("quality_coverage") or 0.0)
        distribution_coverage = float(practice.get("distribution_coverage") or 0.0)
        avg_quality = float(practice.get("avg_telemetry_quality") or 0.0)
        practice_coverage = float(practice.get("coverage") or 0.0)
        if (
            practice_coverage >= 0.75
            and detail_coverage >= 0.75
            and quality_coverage >= 0.75
            and distribution_coverage >= 0.75
            and avg_quality >= 0.55
        ):
            limitations.append({
                "code": "practice_compact_distribution",
                "severity": "low",
                "message": "Practice-stage evidence uses compact clean-lap distributions plus sector/stability summaries; raw lap streams are still not replayed directly.",
                "details": {"requested_stages": stages, "practice": practice},
            })
        elif practice_coverage >= 0.75 and detail_coverage >= 0.75 and quality_coverage >= 0.75 and avg_quality >= 0.55:
            limitations.append({
                "code": "practice_compact_summary",
                "severity": "low",
                "message": "Practice-stage evidence uses enriched compact FP summaries with sector/stability detail where cached, not raw lap-by-lap telemetry.",
                "details": {"requested_stages": stages, "practice": practice},
            })
        elif practice_coverage >= 0.75 and detail_coverage >= 0.75:
            limitations.append({
                "code": "practice_quality_incomplete",
                "severity": "medium",
                "message": "Practice rows have sector/stability detail, but clean-lap telemetry quality is incomplete for calibration-grade practice modeling.",
                "details": {"requested_stages": stages, "practice": practice},
            })
        else:
            limitations.append({
                "code": "practice_telemetry_proxy",
                "severity": "medium",
                "message": "Practice-stage evidence uses compact FP summaries where available; sector/stability detail is incomplete until evidence cache files are refreshed.",
                "details": {"requested_stages": stages, "practice": practice},
            })
    limitations.append({
        "code": "upstream_calls_minimized",
        "severity": "info",
        "message": "Each season is loaded once per deep run and reused across models/stages.",
        "details": {"season_count": len(season_rows)},
    })
    return limitations


def _deep_recommendations(
    runs: list[dict[str, Any]],
    coverage: dict[str, Any],
    track_segment_matrix: list[dict[str, Any]] | None = None,
    track_segment_ablations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    recommendations = []
    grade = coverage.get("coverage_grade")
    evidence_groups = coverage.get("evidence_groups") or {}
    if grade in {"very_thin", "thin"}:
        recommendations.append({
            "priority": "high",
            "target": "sample_depth",
            "action": "Extend the backtest window and persist downloaded season rows before tuning model weights.",
            "reason": f"Coverage grade is {grade}. Metrics are directionally useful but not calibration-grade.",
        })
    practice = evidence_groups.get("practice") or {}
    if practice.get("requested") and float(practice.get("coverage") or 0.0) < 0.50:
        recommendations.append({
            "priority": "high",
            "target": "practice_ingestion",
            "action": "Wire FastF1/OpenF1 FP1-FP3 lap rows into replay fixtures before trusting practice-stage deltas.",
            "reason": "Practice evidence coverage is low.",
        })
    qualifying = evidence_groups.get("qualifying") or {}
    if float(qualifying.get("coverage") or 0.0) < 0.75:
        recommendations.append({
            "priority": "medium",
            "target": "grid_truth",
            "action": "Backfill qualifying/grid/penalty rows for seasons with missing qualifying coverage.",
            "reason": "Post-qualifying analysis is only as good as the grid evidence.",
        })
    best = _best_model([run for run in runs if run.get("ok")])
    if best and float(best.get("winner_accuracy") or 0.0) < 0.25:
        recommendations.append({
            "priority": "high",
            "target": "winner_selection",
            "action": "Add learned pace/live-session evidence before using winner accuracy for betting decisions.",
            "reason": "Best-run winner accuracy is below 25%.",
        })
    for segment in track_segment_matrix or []:
        race_count = int(segment.get("best_race_count") or 0)
        winner_accuracy = float(segment.get("best_winner_accuracy") or 0.0)
        log_loss = float(segment.get("best_log_loss") or 0.0)
        if race_count < 8:
            continue
        if winner_accuracy < 0.45 or log_loss > 1.65:
            recommendations.append({
                "priority": "medium",
                "target": f"track_segment:{segment.get('segment')}",
                "action": "Tune calibration and feature weights for this circuit trait group before trusting generic probabilities there.",
                "reason": (
                    f"{segment.get('segment')} tracks have {winner_accuracy:.1%} winner accuracy "
                    f"and log loss {log_loss:.4f} over {race_count} races."
                ),
            })
    for ablation in track_segment_ablations or []:
        if ablation.get("sample_warning"):
            continue
        if ablation.get("interpretation") == "evidence_hurt_or_overfit":
            recommendations.append({
                "priority": "medium",
                "target": f"segment_ablation:{ablation.get('segment')}:{ablation.get('evidence_group')}",
                "action": "Review feature weighting for this evidence group on this track segment; it is reducing backtest quality.",
                "reason": (
                    f"{ablation.get('evidence_group')} on {ablation.get('segment')} tracks changed "
                    f"log-loss by {ablation.get('delta_log_loss')} and winner accuracy by "
                    f"{ablation.get('delta_winner_accuracy')} over {ablation.get('race_count')} races."
                ),
            })
    return recommendations


def _compact_report(
    best: dict[str, Any] | None,
    stage_matrix: list[dict[str, Any]],
    model_matrix: list[dict[str, Any]],
    coverage: dict[str, Any],
    track_segment_matrix: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    weakest_segment = _weakest_track_segment(track_segment_matrix or [])
    return {
        "headline": (
            f"Best run is {best.get('model_id')} at {best.get('stage')} "
            f"with log loss {best.get('avg_log_loss')}."
            if best else "No successful backtest runs."
        ),
        "coverage_grade": coverage.get("coverage_grade"),
        "completed_races": (coverage.get("totals") or {}).get("completed_races"),
        "best_model_id": best.get("model_id") if best else None,
        "best_stage": best.get("stage") if best else None,
        "best_metrics": {
            "winner_accuracy": best.get("winner_accuracy") if best else None,
            "avg_podium_hit_rate": best.get("avg_podium_hit_rate") if best else None,
            "avg_points_hit_rate": best.get("avg_points_hit_rate") if best else None,
            "avg_brier_score": best.get("avg_brier_score") if best else None,
            "avg_log_loss": best.get("avg_log_loss") if best else None,
            "avg_expected_finish_error": best.get("avg_expected_finish_error") if best else None,
        },
        "stage_count": len(stage_matrix),
        "model_count": len(model_matrix),
        "track_segment_count": len(track_segment_matrix or []),
        "weakest_track_segment": weakest_segment,
    }


def _weakest_track_segment(track_segment_matrix: list[dict[str, Any]]) -> dict[str, Any] | None:
    reliable = [row for row in track_segment_matrix if int(row.get("best_race_count") or 0) >= 8]
    if not reliable:
        return None
    weakest = sorted(
        reliable,
        key=lambda row: (
            float(row.get("best_winner_accuracy") or 0.0),
            -float(row.get("best_log_loss") or 0.0),
        ),
    )[0]
    return {
        "segment": weakest.get("segment"),
        "race_count": weakest.get("best_race_count"),
        "winner_accuracy": weakest.get("best_winner_accuracy"),
        "avg_log_loss": weakest.get("best_log_loss"),
        "best_stage": weakest.get("best_stage"),
        "best_model_id": weakest.get("best_model_id"),
    }


def _write_deep_backtest_artifact(result: dict[str, Any]) -> dict[str, Any]:
    artifact_dir = Path.cwd() / "artifacts" / "f1_backtest_reports"
    requested = result.get("requested") or {}
    scope = f"{result.get('start_season')}_{result.get('end_season')}"
    stages = "-".join(requested.get("stages") or []) or "stages"
    models = "-".join(requested.get("model_ids") or []) or "models"
    timestamp = _artifact_timestamp(result.get("generated_at"))
    stem = _safe_filename(f"deep_{scope}_{stages}_{models}_{timestamp}")
    full_path = artifact_dir / f"{stem}.json"
    compact_path = artifact_dir / f"{stem}.compact.json"
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        full_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        compact_payload = {
            "ok": result.get("ok"),
            "mode": result.get("mode"),
            "start_season": result.get("start_season"),
            "end_season": result.get("end_season"),
            "generated_at": result.get("generated_at"),
            "requested": requested,
            "compact_report": result.get("compact_report"),
            "coverage": result.get("coverage"),
            "best_model_id": result.get("best_model_id"),
            "best_stage": result.get("best_stage"),
            "stage_matrix": result.get("stage_matrix"),
            "model_matrix": result.get("model_matrix"),
            "track_segment_matrix": result.get("track_segment_matrix"),
            "evidence_ablations": result.get("evidence_ablations"),
            "track_segment_ablations": result.get("track_segment_ablations"),
            "limitations": result.get("limitations"),
            "recommendations": result.get("recommendations"),
            "selection_rule": result.get("selection_rule"),
        }
        compact_path.write_text(json.dumps(compact_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return {
            "ok": True,
            "artifact_dir": str(artifact_dir),
            "full_path": str(full_path),
            "compact_path": str(compact_path),
            "format": "json",
            "full_bytes": full_path.stat().st_size,
            "compact_bytes": compact_path.stat().st_size,
        }
    except Exception as exc:
        return {
            "ok": False,
            "artifact_dir": str(artifact_dir),
            "reason": str(exc),
        }


def _artifact_timestamp(value: Any) -> str:
    text = str(value or datetime.now(timezone.utc).isoformat())
    return re.sub(r"[^0-9T]+", "", text)[:15] or "now"


def _safe_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return text[:180].strip("._-") or "deep_backtest"


def _practice_available(race: dict[str, Any]) -> bool:
    return bool(_practice_rows(race))


def _practice_detail_available(race: dict[str, Any]) -> bool:
    return _practice_detail_row_count(race) > 0


def _practice_quality_available(race: dict[str, Any]) -> bool:
    return _practice_quality_row_count(race) > 0


def _practice_distribution_available(race: dict[str, Any]) -> bool:
    return _practice_distribution_row_count(race) > 0


def _race_inputs_available(race: dict[str, Any]) -> bool:
    return _race_input_row_count(race) > 0


def _race_stint_available(race: dict[str, Any]) -> bool:
    return _race_stint_row_count(race) > 0


def _practice_detail_row_count(race: dict[str, Any]) -> int:
    count = 0
    for row in _practice_rows(race):
        if not isinstance(row, dict):
            continue
        has_sector = any(row.get(f"representative_sector_{idx}") or row.get(f"best_sector_{idx}") for idx in (1, 2, 3))
        has_stability = row.get("pace_stability") is not None or row.get("lap_time_stddev") is not None
        if has_sector or has_stability:
            count += 1
    return count


def _practice_quality_row_count(race: dict[str, Any]) -> int:
    return sum(1 for row in _practice_rows(race) if _practice_quality(row) >= 0.55)


def _practice_distribution_row_count(race: dict[str, Any]) -> int:
    return sum(1 for row in _practice_rows(race) if _practice_distribution_present(row))


def _race_input_row_count(race: dict[str, Any]) -> int:
    return len(_race_input_rows(race))


def _race_stint_row_count(race: dict[str, Any]) -> int:
    return sum(1 for row in _race_input_rows(race) if _race_stint_present(row))


def _avg_practice_quality(season_rows: dict[int, list[dict[str, Any]]]) -> float:
    values = [
        _practice_quality(row)
        for races in season_rows.values()
        for race in races
        for row in _practice_rows(race)
        if _practice_quality(row) > 0.0
    ]
    return round(sum(values) / len(values), 4) if values else 0.0


def _practice_quality(row: dict[str, Any]) -> float:
    if not isinstance(row, dict):
        return 0.0
    try:
        explicit = row.get("telemetry_quality")
        if explicit is not None:
            return max(0.0, min(1.0, float(explicit)))
    except (TypeError, ValueError):
        pass
    sector_count = sum(1 for idx in (1, 2, 3) if row.get(f"representative_sector_{idx}") or row.get(f"best_sector_{idx}"))
    sector_coverage = sector_count / 3.0
    lap_count = _safe_int(row.get("usable_laps")) or _safe_int(row.get("laps")) or 0
    has_stability = row.get("pace_stability") is not None or row.get("lap_time_stddev") is not None
    return round(max(0.0, min(1.0, 0.45 * min(lap_count, 12) / 12.0 + 0.35 * sector_coverage + (0.20 if has_stability else 0.0))), 4)


def _practice_distribution_present(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    distribution = row.get("lap_distribution")
    if not isinstance(distribution, dict):
        return False
    return bool(distribution.get("sample_size") and (
        distribution.get("spread_p90_p10") is not None
        or distribution.get("p10") is not None
        or distribution.get("p90") is not None
    ))


def _practice_rows(race: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(race.get("PracticeResults"), list):
        return list(race.get("PracticeResults") or [])
    rows = []
    for session in race.get("PracticeSessions") or race.get("Practice") or []:
        rows.extend(session.get("Results") or session.get("results") or [])
    return rows


def _race_input_rows(race: dict[str, Any]) -> list[dict[str, Any]]:
    payload = race.get("RaceInputs") or race.get("race_inputs") or {}
    if not isinstance(payload, dict):
        return []
    drivers = payload.get("drivers") or {}
    if isinstance(drivers, dict):
        return [row for row in drivers.values() if isinstance(row, dict)]
    if isinstance(drivers, list):
        return [row for row in drivers if isinstance(row, dict)]
    return []


def _race_stint_present(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    distribution = row.get("stint_lap_distribution")
    has_distribution = isinstance(distribution, dict) and int(distribution.get("sample_size") or 0) > 0
    return bool(
        has_distribution
        or row.get("compound_sequence")
        or row.get("avg_stint_laps") is not None
        or row.get("max_stint_laps") is not None
        or row.get("final_stint_laps") is not None
        or row.get("stints") is not None
    )


def _delta(value: Any, baseline: Any, lower_is_better: bool = False) -> float | None:
    try:
        raw = float(value) - float(baseline)
    except (TypeError, ValueError):
        return None
    if lower_is_better:
        raw *= -1.0
    return round(raw, 4)


def _ablation_interpretation(row: dict[str, Any], baseline: dict[str, Any]) -> str:
    log_delta = _delta(row.get("best_log_loss"), baseline.get("best_log_loss"), lower_is_better=True)
    accuracy_delta = _delta(row.get("best_winner_accuracy"), baseline.get("best_winner_accuracy"))
    if log_delta is None and accuracy_delta is None:
        return "insufficient_metrics"
    if (log_delta or 0.0) > 0.02 or (accuracy_delta or 0.0) > 0.02:
        return "evidence_helped"
    if (log_delta or 0.0) < -0.02 or (accuracy_delta or 0.0) < -0.02:
        return "evidence_hurt_or_overfit"
    return "neutral"


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
