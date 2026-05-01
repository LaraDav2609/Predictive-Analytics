"""Abstract TelemetryProvider — the seam that lets us swap FastF1 for a vendor feed
later without touching feature / model code.

All concrete providers normalize to TelemetrySample / Lap / WeatherFrame from common.types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from f1_ml.common.types import Lap, Race, SessionType, TelemetrySample, WeatherFrame


class TelemetryProvider(ABC):
    """Source-agnostic telemetry feed. Implementations: FastF1Provider, VendorProvider."""

    @abstractmethod
    def list_races(self, season: int) -> list[Race]:
        ...

    @abstractmethod
    def stream_telemetry(self, race: Race, session: SessionType) -> Iterator[TelemetrySample]:
        """Iterate ticks in chronological order. Lazy — for live mode this is unbounded."""
        ...

    @abstractmethod
    def laps(self, race: Race, session: SessionType) -> list[Lap]:
        ...

    @abstractmethod
    def weather(self, race: Race) -> list[WeatherFrame]:
        ...

    @abstractmethod
    def is_live(self) -> bool:
        """True for live providers (v2 in-race), False for replayed historical (v1)."""
        ...
