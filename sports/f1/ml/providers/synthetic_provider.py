"""Synthetic provider — generates a fake race with known driver true pace.

Used for thin-slice end-to-end testing: we control the ground truth (top-pace
driver should win ~most simulations), so calibration and pipeline correctness
are easy to assert without depending on FastF1 / external data.

Each driver has a configured `true_pace_s` (mean lap time) and `pace_sigma_s`.
Sampled lap times are Gaussian around the true pace. DNFs are sampled at a
constant per-lap hazard.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from sports.f1.ml.common.types import (
    Lap,
    Race,
    SessionType,
    TelemetrySample,
    TireCompound,
    WeatherFrame,
)
from sports.f1.ml.providers.base import TelemetryProvider


@dataclass
class SyntheticDriver:
    code: str
    team_code: str
    true_pace_s: float        # mean lap time in seconds
    pace_sigma_s: float = 0.3
    dnf_rate_per_lap: float = 0.0008


@dataclass
class SyntheticRaceConfig:
    race_id: str = "SYN-2026-01"
    track_code: str = "SYNTH"
    season: int = 2026
    round: int = 1
    n_laps: int = 50
    seed: int = 42
    drivers: list[SyntheticDriver] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime(2026, 4, 26, 13, 0))


def default_grid() -> list[SyntheticDriver]:
    """A 20-car grid with realistic pace spread (~1.5 s pole-to-back)."""
    teams = ["RBR", "MCL", "FER", "MER", "AST", "ALP", "WIL", "RBR2", "STA", "HAA"]
    base_pace = 80.0  # 1:20.000 baseline
    grid: list[SyntheticDriver] = []
    rng = np.random.default_rng(42)
    for i, team in enumerate(teams):
        team_pace = base_pace + i * 0.15  # team gap
        for j in range(2):
            driver_offset = rng.normal(0.0, 0.10)  # intra-team driver effect
            grid.append(SyntheticDriver(
                code=f"{team}{j+1}",
                team_code=team,
                true_pace_s=team_pace + driver_offset,
                pace_sigma_s=0.30,
            ))
    return grid


class SyntheticProvider(TelemetryProvider):
    """Implements TelemetryProvider against a configured fake race."""

    def __init__(self, config: SyntheticRaceConfig | None = None) -> None:
        self.config = config or SyntheticRaceConfig(drivers=default_grid())
        if not self.config.drivers:
            self.config.drivers = default_grid()
        self._rng = np.random.default_rng(self.config.seed)

    # ----- TelemetryProvider interface -----

    def list_races(self, season: int) -> list[Race]:
        if season != self.config.season:
            return []
        return [self._race()]

    def stream_telemetry(self, race: Race, session: SessionType) -> Iterator[TelemetrySample]:
        # Thin slice: don't generate per-tick GPS — just synthesize one summary
        # tick per lap per driver so consumers can iterate something.
        for lap in self.laps(race, session):
            yield TelemetrySample(
                race_id=lap.race_id,
                driver_code=lap.driver_code,
                timestamp=self.config.start_time + timedelta(seconds=lap.lap_number * 80.0),
                lap=lap.lap_number,
                s_coord=0.0,
                speed_kph=200.0,
                accel_long_g=0.0,
                accel_lat_g=0.0,
            )

    def laps(self, race: Race, session: SessionType) -> list[Lap]:
        if session != SessionType.RACE:
            return []
        return self._sample_laps()

    def weather(self, race: Race) -> list[WeatherFrame]:
        # Single dry-weather frame at race start.
        return [WeatherFrame(
            race_id=self._race_id(),
            timestamp=self.config.start_time,
            air_temp_c=22.0,
            track_temp_c=35.0,
            humidity_pct=45.0,
            wind_kph=8.0,
            rain_intensity=0.0,
        )]

    def is_live(self) -> bool:
        return False

    # ----- thin-slice helpers -----

    def driver_pace_table(self) -> dict[str, float]:
        """Ground-truth pace table — used by the simulator's thin-slice mode."""
        return {d.code: d.true_pace_s for d in self.config.drivers}

    def driver_sigma_table(self) -> dict[str, float]:
        return {d.code: d.pace_sigma_s for d in self.config.drivers}

    def dnf_rate_table(self) -> dict[str, float]:
        return {d.code: d.dnf_rate_per_lap for d in self.config.drivers}

    # ----- internals -----

    def _race(self) -> Race:
        return Race(
            season=self.config.season,
            round=self.config.round,
            track_code=self.config.track_code,
            name=f"Synthetic Round {self.config.round}",
            scheduled_start=self.config.start_time,
        )

    def _race_id(self) -> str:
        return self.config.race_id

    def _sample_laps(self) -> list[Lap]:
        """Sample one realization of the race for use as 'historical' lap data."""
        laps: list[Lap] = []
        n = len(self.config.drivers)
        cum_time = np.zeros(n)
        active = np.ones(n, dtype=bool)
        for lap_n in range(1, self.config.n_laps + 1):
            for idx, drv in enumerate(self.config.drivers):
                if not active[idx]:
                    continue
                t = self._rng.normal(drv.true_pace_s, drv.pace_sigma_s)
                if self._rng.random() < drv.dnf_rate_per_lap:
                    active[idx] = False
                    continue
                cum_time[idx] += t
                # Position computed from current cum_time among actives
                ranks = np.argsort(np.where(active, cum_time, np.inf))
                pos = int(np.where(ranks == idx)[0][0]) + 1
                laps.append(Lap(
                    race_id=self._race_id(),
                    driver_code=drv.code,
                    lap_number=lap_n,
                    lap_time_s=float(t),
                    sector1_s=float(t / 3),
                    sector2_s=float(t / 3),
                    sector3_s=float(t / 3),
                    compound=TireCompound.MEDIUM,
                    tire_age_laps=lap_n - 1,
                    position=pos,
                    gap_to_leader_s=float(cum_time[idx] - cum_time[ranks[0]]) if active[ranks[0]] else None,
                ))
        return laps
