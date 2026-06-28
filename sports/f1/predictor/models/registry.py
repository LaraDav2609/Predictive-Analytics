"""Model registry for F1 prediction submodels."""

from __future__ import annotations

from sports.f1.models.f1 import Constructor, Driver, Race, RacePrediction
from sports.f1.predictor.models.baseline import BaselineRaceModel
from sports.f1.predictor.models.configs import PRODUCTION_MODEL_ID, get_model_config, list_model_configs
from sports.f1.predictor.models.ml_simulator import MLSimulatorRaceModel
from sports.f1.predictor.models.telemetry_simulator import TelemetrySimulatorRaceModel


class F1ModelRegistry:
    def __init__(self, baseline: BaselineRaceModel | None = None, model_id: str | None = None):
        self._model_id = model_id or PRODUCTION_MODEL_ID
        self._model_config = get_model_config(self._model_id)
        self._model = self._build_model(baseline)

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
        return self._model.predict(race, drivers, constructors, features, sentiment)

    def _build_model(self, baseline: BaselineRaceModel | None):
        if self._model_id == "telemetry_simulator_v1":
            return TelemetrySimulatorRaceModel(config=self._model_config)
        if self._model_id == "ml_simulator_v1":
            return MLSimulatorRaceModel(config=self._model_config)
        return baseline or BaselineRaceModel(config=self._model_config)
