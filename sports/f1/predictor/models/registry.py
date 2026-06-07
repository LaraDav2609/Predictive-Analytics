"""Model registry for F1 prediction submodels."""

from __future__ import annotations

from sports.f1.models.f1 import Constructor, Driver, Race, RacePrediction
from sports.f1.predictor.models.baseline import BaselineRaceModel
from sports.f1.predictor.models.configs import PRODUCTION_MODEL_ID, get_model_config, list_model_configs


class F1ModelRegistry:
    def __init__(self, baseline: BaselineRaceModel | None = None, model_id: str | None = None):
        self._model_id = model_id or PRODUCTION_MODEL_ID
        self._baseline = baseline or BaselineRaceModel(config=get_model_config(self._model_id))

    @property
    def model_id(self) -> str:
        return self._model_id

    @staticmethod
    def list_models() -> list[dict]:
        return list_model_configs()

    def with_model(self, model_id: str | None):
        return F1ModelRegistry(model_id=(model_id or PRODUCTION_MODEL_ID))

    def predict_race(
        self,
        race: Race,
        drivers: list[Driver],
        constructors: list[Constructor],
        features: dict,
        sentiment: dict,
    ) -> RacePrediction:
        return self._baseline.predict(race, drivers, constructors, features, sentiment)
