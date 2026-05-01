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
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LapState:
    lap: int
    cumulative_time_s: list[float]
    tire_age: list[int]
    compounds: list[str]
    dnf: list[bool]
    sc_active: bool
    fuel_kg: list[float]


def step(state: LapState, models: dict, rng) -> LapState:
    """One lap of one MC iteration. Mutates and returns state."""
    raise NotImplementedError(
        "ordered per-lap operations; called n_laps times per MC iteration"
    )
