"""Vendor / team-data provider stub — slot in higher-resolution telemetry later.

Examples: SportsRadar, GeniusSports, direct team feed under partnership. Higher Hz,
adds engine/tire-temp/fuel-flow channels not present in FastF1's broadcast feed.

Implement when access is acquired; downstream feature code already consumes the
normalized TelemetrySample shape, so swap is non-invasive.
"""

from __future__ import annotations

from collections.abc import Iterator

from sports.f1.ml.common.types import Lap, Race, SessionType, TelemetrySample, WeatherFrame
from sports.f1.ml.providers.base import TelemetryProvider


class VendorProvider(TelemetryProvider):
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url

    def list_races(self, season: int) -> list[Race]:
        raise NotImplementedError("vendor REST: GET /seasons/{season}/races")

    def stream_telemetry(self, race: Race, session: SessionType) -> Iterator[TelemetrySample]:
        raise NotImplementedError("vendor websocket / chunked HTTP → normalized samples")

    def laps(self, race: Race, session: SessionType) -> list[Lap]:
        raise NotImplementedError

    def weather(self, race: Race) -> list[WeatherFrame]:
        raise NotImplementedError

    def is_live(self) -> bool:
        # Vendor feeds typically support both replay and live; expose via constructor flag.
        raise NotImplementedError
