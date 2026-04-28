"""Service facade for semi-standalone F1 prediction modules."""

from __future__ import annotations

from models.f1 import Constructor, Driver, Race, RacePrediction
from f1_predictor.config import MODEL_VERSION
from f1_predictor.features.builder import F1FeatureBuilder
from f1_predictor.features.performance import build_performance_table
from f1_predictor.models.configs import PRODUCTION_MODEL_ID, get_model_config
from f1_predictor.models.registry import F1ModelRegistry
from f1_predictor.simulation.session_projection import build_session_projection


class F1PredictionService:
    def __init__(self, model_version: str = MODEL_VERSION, model_id: str | None = None):
        self._drivers: list[Driver] = []
        self._constructors: list[Constructor] = []
        self._features: dict = {}
        self._sentiment: dict = {}
        self._model_id = model_id or PRODUCTION_MODEL_ID
        self._model_config = get_model_config(self._model_id)
        self._model_version = model_version if model_id is None else self._model_config.model_version
        self._registry = F1ModelRegistry(model_id=self._model_id)

    @property
    def features(self) -> dict:
        return self._features

    @property
    def sentiment(self) -> dict:
        return self._sentiment

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_id(self) -> str:
        return self._model_id

    def load(
        self,
        drivers: list[Driver],
        constructors: list[Constructor] | None = None,
        features: dict | None = None,
        sentiment: dict | None = None,
    ) -> None:
        self._drivers = sorted(drivers, key=lambda driver: driver.points, reverse=True)
        self._constructors = constructors or self._constructors
        if features is not None:
            self._features = features
        if sentiment is not None:
            self._sentiment = sentiment

    def load_features(self, features: dict) -> None:
        self._features = features or {}

    def load_sentiment(self, sentiment: dict) -> None:
        self._sentiment = sentiment or {}

    def build_features(self, race: Race | None = None, session_stage: str = "race"):
        return F1FeatureBuilder(self._drivers, self._constructors, self._features, self._sentiment).build(race, session_stage)

    def predict_race(self, race: Race) -> RacePrediction:
        return self._registry.predict_race(race, self._drivers, self._constructors, self._features, self._sentiment)

    def predict_races(self, races: list[Race]) -> list[Race]:
        for race in races:
            if race.status == "SCHEDULED":
                race.prediction = self.predict_race(race)
        return races

    def get_performance_intelligence(self) -> dict:
        driver_features = self._features.get("drivers") or {}
        constructor_features = self._features.get("constructors") or {}
        rows = build_performance_table(self._drivers, self._constructors, driver_features, constructor_features)
        ranked = sorted(rows.values(), key=lambda item: item["performance_score"], reverse=True)
        for index, row in enumerate(ranked, start=1):
            row["rank"] = index
        return {
            "ok": True,
            "model_version": self._model_version,
            "updated_at": self._features.get("updated_at"),
            "completed_races": self._features.get("completed_races", 0),
            "total_races": self._features.get("total_races", 0),
            "drivers": ranked,
            "method": {
                "driver_skill": "recent form, reliability, standings signal, podium/win conversion, and F1 experience",
                "car_performance": "constructor pace from standings, recent team points, and average finish",
                "performance_score": "geometric blend of driver skill and car performance with reliability damping",
            },
        }

    def build_session_simulation(
        self,
        race: Race,
        prediction: dict,
        qualifying: list[dict],
        sprint: list[dict],
        results: list[dict],
        session: str = "race",
        live: bool = False,
    ) -> dict:
        return build_session_projection(
            race=race,
            drivers=self._drivers,
            constructors=self._constructors,
            prediction=prediction,
            features=self._features,
            qualifying=qualifying,
            sprint=sprint,
            results=results,
            session=session,
            live=live,
        )
