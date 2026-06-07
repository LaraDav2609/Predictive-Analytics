"""Compatibility wrapper for the semi-standalone F1 predictor package."""

from __future__ import annotations

import logging
from typing import Any

from sports.f1.models.f1 import Constructor, Driver, Race, RacePrediction
from sports.f1.predictor.config import MODEL_VERSION
from sports.f1.predictor.features.performance import build_performance_table
from sports.f1.predictor.scoring.explanations import explain_driver
from sports.f1.predictor.scoring.normalization import clamp01, label_from_score, num
from sports.f1.predictor.service import F1PredictionService

logger = logging.getLogger(__name__)


class F1Predictor:
    """Preserves the historical analytics API while delegating to sports.f1.predictor."""

    def __init__(self):
        self._drivers: list[Driver] = []
        self._constructors: list[Constructor] = []
        self._features: dict = {}
        self._sentiment: dict = {}
        self._version = MODEL_VERSION
        self._service = F1PredictionService(self._version)

    def load_drivers(
        self,
        drivers: list[Driver],
        constructors: list[Constructor] | None = None,
        features: dict | None = None,
        sentiment: dict | None = None,
    ) -> None:
        self._drivers = sorted(drivers, key=lambda d: d.points, reverse=True)
        self._constructors = constructors or self._constructors
        if features is not None:
            self._features = features
        if sentiment is not None:
            self._sentiment = sentiment
        self._service.load(self._drivers, self._constructors, self._features, self._sentiment)
        logger.info(
            "F1 predictor loaded %d drivers, %d constructors, %d feature drivers",
            len(self._drivers),
            len(self._constructors),
            len((self._features.get("drivers") or {})),
        )

    def load_features(self, features: dict) -> None:
        self._features = features or {}
        self._service.load_features(self._features)

    def load_sentiment(self, sentiment: dict) -> None:
        self._sentiment = sentiment or {}
        self._service.load_sentiment(self._sentiment)

    def get_sentiment(self) -> dict:
        return self._sentiment or {}

    def predict_race(self, race: Race) -> RacePrediction:
        return self._service.predict_race(race)

    def get_performance_intelligence(self) -> dict:
        return self._service.get_performance_intelligence()

    def predict_races(self, races: list[Race]) -> list[Race]:
        return self._service.predict_races(races)

    def build_feature_snapshot(self, race: Race | None = None, session_stage: str = "race"):
        return self._service.build_features(race, session_stage)

    def _build_performance_table(self, driver_features: dict, constructor_features: dict) -> dict[str, dict]:
        return build_performance_table(self._drivers, self._constructors, driver_features, constructor_features)

    @staticmethod
    def _prediction_confidence(driver_features: dict, sentiment_drivers: dict) -> float:
        history_part = min(0.45, len(driver_features) / 20 * 0.45)
        sentiment_mentions = sum(1 for item in sentiment_drivers.values() if int((item or {}).get("mentions") or 0) > 0)
        sentiment_part = min(0.25, sentiment_mentions / 20 * 0.25)
        return round(0.30 + history_part + sentiment_part, 2)

    @staticmethod
    def _clamp01(value: float) -> float:
        return clamp01(value)

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        return num(value, default)

    @staticmethod
    def _explain_driver(driver: Driver, component: dict) -> list[str]:
        return explain_driver(driver, component)


def _label_from_score(score: float) -> str:
    return label_from_score(score)
