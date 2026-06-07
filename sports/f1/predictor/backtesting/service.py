"""Backtesting service for the F1 prediction engine."""

from __future__ import annotations

from datetime import datetime, timezone
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


class F1BacktestService:
    def __init__(self, client, lookback_races: int = 8):
        self._client = client
        self._loader = HistoricalRaceLoader(client)
        self._builder = RaceReplayBuilder(lookback_races=lookback_races)
        self._cache: dict[tuple, dict[str, Any]] = {}

    async def backtest_season(
        self,
        season: int | None = None,
        include_races: bool = False,
        allow_partial: bool = False,
        model_id: str | None = None,
        stage: str = "pre_weekend",
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

        cache_key = ("season", season, include_races, allow_partial, model_id, stage)
        if cache_key in self._cache:
            return self._cache[cache_key]

        races = await self._loader.load_season(season)
        completed = [race for race in races if race.get("Results")]
        race_rows = [self.backtest_race_from_rows(season, races, int(race.get("round") or 0), model_id=model_id, stage=stage) for race in completed]
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
            season_result = await self.backtest_season(season, include_races=True, allow_partial=allow_partial, model_id=model_id, stage=stage)
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
        models = []
        for model in F1ModelRegistry.list_models():
            result = await self.backtest_summary(
                start_season=start_season,
                end_season=end_season,
                include_races=include_races,
                allow_partial=allow_partial,
                model_id=model["model_id"],
                stage=stage,
            )
            models.append(result)
        return _comparison_result({"start_season": start_season, "end_season": end_season}, models)

    def backtest_race_from_rows(self, season: int, races: list[dict[str, Any]], round_num: int, model_id: str | None = None, stage: str = "pre_weekend") -> dict[str, Any]:
        model_id = model_id or PRODUCTION_MODEL_ID
        stage = _stage_key(stage)
        replay = self._builder.build(season, races, round_num, stage=stage)
        service = F1PredictionService(model_id=model_id)
        service.load(replay.drivers, replay.constructors, replay.features, sentiment={})
        baseline_prediction = service.predict_race(replay.race)
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
        return {
            "season": season,
            "round": round_num,
            "model_id": model_id,
            "stage": replay.stage,
            "race_name": replay.race.name,
            "circuit": replay.race.circuit,
            "country": replay.race.country,
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


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
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
