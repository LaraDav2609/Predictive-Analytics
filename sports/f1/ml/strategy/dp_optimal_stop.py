"""Dynamic-programming optimal pit strategy — single-driver, deterministic baseline.

State: (current_lap, current_compound, tire_age, has_pitted_for_compound[…])
Action: pit (and choose new compound) or stay out.
Cost: deterministic lap-time function (base_pace + tire_deg(age, compound) + pit_loss_if_box).

Solve by backward induction. Sub-second per driver; recompute each lap as new
information arrives.

This is the simulator's default strategy module: each driver picks the DP-optimal
plan against the current pace + deg estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sports.f1.ml.common.types import TireCompound


@dataclass
class StrategyPlan:
    pit_laps: list[int]
    compounds_to_fit: list[TireCompound]
    expected_total_time_s: float


def solve(
    total_laps: int,
    current_compound: TireCompound,
    pace_fn: Callable[[TireCompound, int, int], float],  # (compound, lap_in_race, tire_age) → lap time
    pit_loss_s: float = 23.0,
    available_compounds: tuple[TireCompound, ...] = (
        TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
    ),
    current_lap: int = 1,
    current_tire_age: int = 0,
    must_use_two_compounds: bool = True,
) -> StrategyPlan:
    """Find the time-minimizing one-pit strategy by brute-force search over
    (pit_lap, new_compound) pairs.

    For F1 v1, one pit covers ~95% of dry-race strategies. Multi-pit DP is a
    v2 enhancement (state grows by O(n_compounds * total_laps) per additional
    stop, still tractable but more complex).

    Inputs:
      pace_fn(compound, lap_in_race, tire_age) → expected lap time in seconds.
        The caller should bake fuel correction and tire degradation into this
        function — solve() treats lap_time as a black box.

    Output: StrategyPlan with pit_laps (list of ints) and compounds_to_fit
    (list of TireCompound). For 1-pit strategies these have length 0 or 1.
    """
    if total_laps <= current_lap:
        return StrategyPlan(pit_laps=[], compounds_to_fit=[], expected_total_time_s=0.0)

    # Helper: total lap time on `compound` from `start_lap` (inclusive, race-lap number)
    # to `end_lap` (inclusive), starting at `start_age` in that compound.
    def stint_time(compound: TireCompound, start_lap: int, end_lap: int, start_age: int) -> float:
        return sum(
            pace_fn(compound, lap, start_age + (lap - start_lap))
            for lap in range(start_lap, end_lap + 1)
        )

    best_total = float("inf")
    best_plan: tuple[list[int], list[TireCompound]] = ([], [])

    # Option 0: stay out (only valid if 2-compound rule is disabled, or already
    # satisfied — we don't track satisfaction in this 1-pit search).
    if not must_use_two_compounds:
        no_pit_total = stint_time(current_compound, current_lap, total_laps, current_tire_age)
        if no_pit_total < best_total:
            best_total = no_pit_total
            best_plan = ([], [])

    # Option 1: one pit at lap k, switch to compound c (c != current_compound).
    for pit_lap in range(current_lap, total_laps):
        pre_pit = stint_time(current_compound, current_lap, pit_lap, current_tire_age)
        for new_compound in available_compounds:
            if new_compound == current_compound:
                continue
            post_pit = stint_time(new_compound, pit_lap + 1, total_laps, 0)
            total = pre_pit + pit_loss_s + post_pit
            if total < best_total:
                best_total = total
                best_plan = ([pit_lap], [new_compound])

    return StrategyPlan(
        pit_laps=best_plan[0],
        compounds_to_fit=best_plan[1],
        expected_total_time_s=best_total,
    )


def make_pace_fn(
    base_pace_s: float,
    deg_per_lap_per_compound: dict[str, float],
    fuel_start_kg: float = 110.0,
    fuel_burn_per_lap_kg: float = 1.6,
    fuel_penalty_s_per_kg: float = 0.030,
) -> Callable[[TireCompound, int, int], float]:
    """Convenience: build a pace_fn from per-compound degradation rates and
    standard fuel parameters. Matches the simulator's PhysicalParams defaults."""

    def pace_fn(compound: TireCompound, lap_in_race: int, tire_age: int) -> float:
        compound_key = compound.value if hasattr(compound, "value") else str(compound)
        deg_slope = deg_per_lap_per_compound.get(
            compound_key, deg_per_lap_per_compound.get("MEDIUM", 0.05)
        )
        fuel_remaining_kg = max(0.0, fuel_start_kg - (lap_in_race - 1) * fuel_burn_per_lap_kg)
        return base_pace_s + deg_slope * tire_age + fuel_remaining_kg * fuel_penalty_s_per_kg

    return pace_fn
