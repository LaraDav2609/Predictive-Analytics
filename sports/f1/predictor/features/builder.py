"""Central F1 feature snapshot builder."""

from __future__ import annotations

from datetime import datetime, timezone

from sports.f1.models.f1 import Constructor, Driver, Race
from sports.f1.predictor.features.historical import HistoricalFeatureProvider
from sports.f1.predictor.features.car_model import build_car_model_analysis
from sports.f1.predictor.features.metadata import MetadataFeatureProvider
from sports.f1.predictor.features.performance import PerformanceFeatureProvider
from sports.f1.predictor.features.practice import apply_practice_pace_adjustments
from sports.f1.predictor.features.reliability import ReliabilityFeatureProvider
from sports.f1.predictor.features.sentiment import SentimentFeatureProvider
from sports.f1.predictor.features.tires import TireFeatureProvider
from sports.f1.predictor.features.track import TrackFeatureProvider
from sports.f1.predictor.features.weather import WeatherFeatureProvider
from sports.f1.predictor.schemas import FeatureSnapshot


class F1FeatureBuilder:
    def __init__(
        self,
        drivers: list[Driver],
        constructors: list[Constructor],
        features: dict | None = None,
        sentiment: dict | None = None,
    ):
        self._drivers = drivers
        self._constructors = constructors
        self._features = features or {}
        self._sentiment = sentiment or {}

    def build(self, race: Race | None = None, session_stage: str = "race") -> FeatureSnapshot:
        historical = HistoricalFeatureProvider(self._features).get_features()
        metadata = MetadataFeatureProvider(self._drivers, self._constructors).get_features()
        driver_features = historical.get("drivers") or {}
        constructor_features = historical.get("constructors") or {}
        openf1_session = self._features.get("openf1_session") or {}
        driver_features = apply_practice_pace_adjustments(
            self._drivers,
            driver_features,
            openf1_session,
            session_stage=session_stage,
        )
        performance = PerformanceFeatureProvider(
            self._drivers,
            self._constructors,
            driver_features,
            constructor_features,
        ).get_features()
        sentiment = SentimentFeatureProvider(self._sentiment).get_features()
        track = TrackFeatureProvider(self._features).get_features(race)
        weather = WeatherFeatureProvider(self._features).get_features(race, openf1_session, session=session_stage)
        tires = TireFeatureProvider().get_features(track, openf1_session, weather=weather)
        car_model = build_car_model_analysis(
            race=race,
            drivers=self._drivers,
            constructors=self._constructors,
            features=self._features,
            sentiment_impact=(self._sentiment.get("race_sentiment_impact") or self._features.get("sentiment_impact")),
            session=session_stage,
            track=track,
            weather=weather,
            tires=tires,
            openf1_session=openf1_session,
        )
        reliability = ReliabilityFeatureProvider(driver_features).get_features(track, weather)

        merged_drivers = {}
        for driver in self._drivers:
            merged_drivers[driver.id] = {
                **(metadata.get("drivers", {}).get(driver.id) or {}),
                **(driver_features.get(driver.id) or {}),
                "performance": performance.get(driver.id) or {},
                "reliability": reliability.get(driver.id) or {},
                "sentiment": (sentiment.get("drivers") or {}).get(driver.id) or {},
            }

        missing_data = []
        if historical.get("missing_data"):
            missing_data.append("historical")
        if weather.get("missing_data"):
            missing_data.append("weather")
        if tires.get("missing_data"):
            missing_data.append("tires")
        if track.get("missing_data"):
            missing_data.append("track")
        if not sentiment.get("sentiment_available"):
            missing_data.append("sentiment")

        return FeatureSnapshot(
            race_id=getattr(race, "round", None),
            session_stage=session_stage,
            generated_at=datetime.now(timezone.utc),
            drivers=merged_drivers,
            constructors=metadata.get("constructors") or {},
            track=track,
            weather=weather,
            tires=tires,
            car_model=car_model,
            reliability=reliability,
            sentiment=sentiment.get("drivers") or {},
            missing_data=missing_data,
        )
