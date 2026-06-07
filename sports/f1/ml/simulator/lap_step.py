"""Per-lap state update — the inner loop of the Monte Carlo simulator.

Order of operations (matters for correctness):
1. Strategy module decides if any drivers pit this lap.
2. Sample base pace per driver from pace model.
3. Apply tire-deg, fuel-correction, track-evolution, dirty-air adjustments.
4. Add pit-stop loss for drivers who pitted.
5. Sample DNF events (hazard model) — DNF'd drivers are removed.
6. Update positions / gaps from cumulative race time.
7. For adjacent pairs within DRS range, sample overtake outcomes.
8. Sample safety car / VSC event; if triggered, bunch the field.

This module exposes a simpler one-lap-at-a-time API used by:
  - In-race replay (v2): step the simulator after each new live lap arrives.
  - Unit tests: assert per-step behavior without rolling out a full race.

The main batch simulator (`simulator.race_sim`) is vectorized over MC iterations
and inlines this loop for performance; this module is the readable reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sports.f1.ml.simulator.restart_logic import RestartConfig, bunch_field


@dataclass
class LapState:
    lap: int
    cumulative_time_s: list[float]
    tire_age: list[int]
    compounds: list[str]
    dnf: list[bool]
    sc_active: bool
    fuel_kg: list[float]


@dataclass
class LapStepConfig:
    fuel_burn_per_lap_kg: float = 1.6
    fuel_penalty_s_per_kg: float = 0.030
    pit_loss_s: float = 23.0
    dirty_air_threshold_s: float = 1.5
    dirty_air_penalty_per_s_close: float = 0.5
    dirty_air_max_penalty_s: float = 1.0
    deg_per_lap_per_compound: dict[str, float] = field(default_factory=lambda: {
        "SOFT": 0.10, "MEDIUM": 0.05, "HARD": 0.03,
        "INTERMEDIATE": 0.07, "WET": 0.06,
    })
    sc_per_lap_rate: float = 0.0
    restart: RestartConfig = field(default_factory=RestartConfig)


def step(
    state: LapState,
    pace_means: list[float],
    pace_sigmas: list[float],
    dnf_per_lap: list[float],
    pit_decisions: list[bool],
    pit_compounds: list[str | None],
    config: LapStepConfig,
    rng: np.random.Generator,
) -> LapState:
    """Advance LapState by one lap. Returns a new LapState (does not mutate)."""
    n = len(state.cumulative_time_s)
    if not all(len(x) == n for x in (
        pace_means, pace_sigmas, dnf_per_lap, pit_decisions, pit_compounds,
        state.tire_age, state.compounds, state.dnf, state.fuel_kg
    )):
        raise ValueError("all per-driver arrays must have the same length")

    cum_time = list(state.cumulative_time_s)
    tire_age = list(state.tire_age)
    compounds = list(state.compounds)
    dnf = list(state.dnf)
    fuel_kg = list(state.fuel_kg)
    sc_active = state.sc_active

    # 1. Sample raw pace per active driver.
    lap_times = [
        float(rng.normal(pace_means[i], pace_sigmas[i])) if not dnf[i] else 0.0
        for i in range(n)
    ]

    # 2. Fuel correction.
    for i in range(n):
        if dnf[i]:
            continue
        lap_times[i] += fuel_kg[i] * config.fuel_penalty_s_per_kg
        fuel_kg[i] = max(0.0, fuel_kg[i] - config.fuel_burn_per_lap_kg)

    # 3. Tire degradation.
    for i in range(n):
        if dnf[i]:
            continue
        slope = config.deg_per_lap_per_compound.get(
            compounds[i], config.deg_per_lap_per_compound["MEDIUM"]
        )
        lap_times[i] += slope * tire_age[i]

    # 4. Pit stops: pay the loss, swap compound, reset tire age.
    for i in range(n):
        if dnf[i] or not pit_decisions[i]:
            continue
        loss = config.pit_loss_s
        if sc_active:
            loss *= config.restart.pit_loss_under_sc_factor
        lap_times[i] += loss
        tire_age[i] = 0
        if pit_compounds[i] is not None:
            compounds[i] = pit_compounds[i]

    # 5. Dirty air penalties (computed against tentative cum_time).
    tentative_cum = [
        cum_time[i] + lap_times[i] if not dnf[i] else float("inf")
        for i in range(n)
    ]
    if config.dirty_air_penalty_per_s_close > 0:
        ranking = sorted(range(n), key=lambda i: tentative_cum[i])
        prev_time = float("-inf")
        for rank, idx in enumerate(ranking):
            if dnf[idx]:
                continue
            if rank == 0 or prev_time == float("-inf"):
                gap_ahead = float("inf")
            else:
                gap_ahead = tentative_cum[idx] - prev_time
            closeness = max(0.0, config.dirty_air_threshold_s - gap_ahead)
            penalty = min(
                closeness * config.dirty_air_penalty_per_s_close,
                config.dirty_air_max_penalty_s,
            )
            lap_times[idx] += penalty
            prev_time = tentative_cum[idx]

    # 6. DNF roll for this lap.
    for i in range(n):
        if dnf[i]:
            continue
        if rng.uniform() < dnf_per_lap[i]:
            dnf[i] = True

    # 7. Commit cum_time + age tires (cars that pitted this lap do NOT age:
    #    their reset to 0 carries into the next lap).
    for i in range(n):
        if dnf[i]:
            continue
        cum_time[i] += lap_times[i]
        if not pit_decisions[i]:
            tire_age[i] += 1

    # 8. Safety car roll. If triggered, bunch the field on the next step.
    if config.sc_per_lap_rate > 0 and not sc_active:
        if rng.uniform() < (1.0 - np.exp(-config.sc_per_lap_rate)):
            leader_idx = int(np.argmin([
                t if not dnf[i] else float("inf") for i, t in enumerate(cum_time)
            ]))
            cum_time = bunch_field(cum_time, leader_idx, config.restart, dnf)
            sc_active = True
    elif sc_active:
        # Simple model: SC ends after one lap of bunched racing.
        sc_active = False

    return LapState(
        lap=state.lap + 1,
        cumulative_time_s=cum_time,
        tire_age=tire_age,
        compounds=compounds,
        dnf=dnf,
        sc_active=sc_active,
        fuel_kg=fuel_kg,
    )
