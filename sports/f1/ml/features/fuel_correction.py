"""Fuel mass correction — strip the ~0.03 s/lap-per-kg fuel-burn effect from raw lap times.

A clean pace signal is essential before fitting tire degradation or pace estimation:
otherwise the fuel-burn linear trend confounds compound-specific deg.

F1 cars start with ~110 kg fuel; consumption ~1.4–1.8 kg/lap depending on track.
"""

from __future__ import annotations

import pandas as pd

# Approximate fuel-burn lap-time penalty per kg of fuel onboard. Track-tunable.
DEFAULT_FUEL_PENALTY_S_PER_KG = 0.030


def fuel_correct(
    laps: pd.DataFrame,
    start_fuel_kg: float = 110.0,
    burn_per_lap_kg: float = 1.6,
    penalty_s_per_kg: float = DEFAULT_FUEL_PENALTY_S_PER_KG,
    lap_col: str = "lap_number",
    lap_time_col: str = "lap_time_s",
) -> pd.DataFrame:
    """Add a `fuel_corrected_lap_time_s` column (lap_time minus fuel offset).

    The correction is the lap-time penalty attributable to the fuel still
    onboard, computed assuming linear burn from the start. Subtracting it
    yields a "as if running on fumes" pace that's comparable across laps.
    """
    out = laps.copy()
    fuel_remaining = (start_fuel_kg - (out[lap_col] - 1) * burn_per_lap_kg).clip(lower=0.0)
    out["fuel_kg_remaining"] = fuel_remaining
    out["fuel_correction_s"] = fuel_remaining * penalty_s_per_kg
    out["fuel_corrected_lap_time_s"] = out[lap_time_col] - out["fuel_correction_s"]
    return out


def fuel_correction_at_lap(lap_number: int, start_fuel_kg: float = 110.0,
                            burn_per_lap_kg: float = 1.6,
                            penalty_s_per_kg: float = DEFAULT_FUEL_PENALTY_S_PER_KG) -> float:
    """Convenience: lap-time penalty (seconds) attributable to fuel mass at the
    start of `lap_number` (1-indexed). Used by the simulator's lap loop."""
    fuel_remaining = max(0.0, start_fuel_kg - (lap_number - 1) * burn_per_lap_kg)
    return fuel_remaining * penalty_s_per_kg
