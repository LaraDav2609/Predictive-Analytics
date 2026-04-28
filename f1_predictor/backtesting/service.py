"""Backtesting service for the F1 prediction engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from f1_predictor.backtesting.calibration import recommend_weight_adjustments
from f1_predictor.backtesting.loader import HistoricalRaceLoader
from f1_predictor.backtesting.metrics import evaluate_race, summarize_races
from f1_predictor.backtesting.replay import RaceReplayBuilder
from f1_predictor.models.configs import PRODUCTION_MODEL_ID
from f1_predictor.models.registry import F1ModelRegistry
from f1_predictor.service import F1PredictionService


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
    ) -> dict[str, Any]:
        model_id = model_id or PRODUCTION_MODEL_ID
        season = self._default_season() if season is None else int(season)
        partial = season >= int(getattr(self._client, "season", datetime.now(timezone.utc).year))
        if partial and not allow_partial:
            return {
                "ok": False,
                "reason": "Current in-progress season backtests require allow_partial=true",
                "season": season,
                "partial": True,
            }

        cache_key = ("season", season, include_races, allow_partial, model_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        races = await self._loader.load_season(season)
        completed = [race for race in races if race.get("Results")]
        race_rows = [self.backtest_race_from_rows(season, races, int(race.get("round") or 0), model_id=model_id) for race in completed]
        summary = summarize_races(race_rows)
        result = {
            "ok": True,
            "season": season,
            "partial": partial,
            "model_version": _model_version(race_rows),
            "model_id": model_id,
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
    ) -> dict[str, Any]:
        model_id = model_id or PRODUCTION_MODEL_ID
        start_season = int(start_season)
        end_season = int(end_season)
        if end_season < start_season:
            start_season, end_season = end_season, start_season
        seasons = []
        all_races = []
        for season in range(start_season, end_season + 1):
            season_result = await self.backtest_season(season, include_races=True, allow_partial=allow_partial, model_id=model_id)
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
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **summary,
            "recommended_weights": recommend_weight_adjustments(summary),
            "seasons": seasons,
        }

    async def backtest_race(self, season: int, round_num: int, allow_partial: bool = True, model_id: str | None = None) -> dict[str, Any]:
        model_id = model_id or PRODUCTION_MODEL_ID
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
            return {"ok": True, **self.backtest_race_from_rows(season, races, int(round_num), model_id=model_id)}
        except ValueError as exc:
            return {"ok": False, "season": season, "round": round_num, "reason": str(exc)}

    async def compare_season(self, season: int | None = None, include_races: bool = False, allow_partial: bool = False) -> dict[str, Any]:
        season = self._default_season() if season is None else int(season)
        models = []
        for model in F1ModelRegistry.list_models():
            result = await self.backtest_season(
                season=season,
                include_races=include_races,
                allow_partial=allow_partial,
                model_id=model["model_id"],
            )
            models.append(result)
        return _comparison_result({"season": season}, models)

    async def compare_summary(
        self,
        start_season: int,
        end_season: int,
        include_races: bool = False,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        models = []
        for model in F1ModelRegistry.list_models():
            result = await self.backtest_summary(
                start_season=start_season,
                end_season=end_season,
                include_races=include_races,
                allow_partial=allow_partial,
                model_id=model["model_id"],
            )
            models.append(result)
        return _comparison_result({"start_season": start_season, "end_season": end_season}, models)

    def backtest_race_from_rows(self, season: int, races: list[dict[str, Any]], round_num: int, model_id: str | None = None) -> dict[str, Any]:
        model_id = model_id or PRODUCTION_MODEL_ID
        replay = self._builder.build(season, races, round_num)
        service = F1PredictionService(model_id=model_id)
        service.load(replay.drivers, replay.constructors, replay.features, sentiment={})
        prediction = service.predict_race(replay.race)
        snapshot = service.build_features(replay.race, "race")
        metrics = evaluate_race(prediction, replay.actual_results)
        distribution = _probability_distribution(prediction)
        components = _component_audit(prediction)
        return {
            "season": season,
            "round": round_num,
            "model_id": model_id,
            "race_name": replay.race.name,
            "circuit": replay.race.circuit,
            "country": replay.race.country,
            "completed_races_before": replay.features.get("completed_races", 0),
            "actual_winner": metrics["actual_winner"],
            "predicted_winner": metrics["predicted_winner"],
            "actual_podium": metrics["actual_podium"],
            "predicted_top3": metrics["predicted_top3"],
            "probability_distribution": distribution,
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


def _model_version(race_rows: list[dict[str, Any]]) -> str | None:
    return next((row.get("model_version") for row in race_rows if row.get("model_version")), None)


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
