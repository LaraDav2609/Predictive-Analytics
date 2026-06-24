"""Compound-aware stint-strategy analysis (analytic, not a re-sim).

Estimates the 1-stop optimal pit window, undercut/overcut value, safety-car pit
value, and the 1-stop-vs-2-stop trade-off from the real per-circuit pit-loss /
tire-stress / lap-count (``TRACK_TRAITS``) and the tire-degradation features.
Produces a race-level strategy summary plus per-driver strategy deltas.

Deliberately analytic (closed-form tire-deg integrals) rather than re-running the
Monte-Carlo simulator: fast, deterministic, and testable, and good enough to rank
undercut vs overcut vs hold and 1 vs 2 stops. A full counterfactual re-sim (sweep
pit laps through ``race_sim``) is a heavier follow-up.
"""
from __future__ import annotations

from typing import Any


def _deg_slope_s(degradation_rate: float, tire_stress: float) -> float:
    """Per-lap pace loss (seconds) from normalized degradation severity."""
    sev = max(0.0, min(1.0, 0.6 * float(degradation_rate or 0.5) + 0.4 * float(tire_stress or 0.5)))
    return round(0.035 + sev * 0.11, 4)   # ~0.035–0.145 s/lap


def _stint_deg_cost(stint_laps: float, slope: float) -> float:
    """Cumulative time lost to degradation over a stint of n laps (triangular sum)."""
    n = max(0.0, float(stint_laps))
    return slope * n * (n - 1) / 2.0


def analyze_race_strategy(
    *,
    laps: int,
    pit_loss_s: float,
    tire_stress: float = 0.5,
    degradation_rate: float = 0.5,
    undercut_strength: float = 0.5,
    overcut_strength: float = 0.4,
    safety_car_probability: float = 0.3,
    overtaking_difficulty: float = 0.5,
    compound_set: list[str] | None = None,
) -> dict[str, Any]:
    laps = int(laps or 0) or 57
    pit_loss_s = float(pit_loss_s or 22.0)
    slope = _deg_slope_s(degradation_rate, tire_stress)

    # 1-stop vs 2-stop: total degradation cost + pit time.
    one_stop = _stint_deg_cost(laps / 2.0, slope) * 2 + pit_loss_s
    two_stop = _stint_deg_cost(laps / 3.0, slope) * 3 + pit_loss_s * 2
    two_stop_delta = round(two_stop - one_stop, 2)          # < 0 → 2-stop faster
    recommended_stops = 2 if two_stop_delta < 0 else 1

    # Optimal 1-stop window: ideal ~mid-distance, pulled earlier by undercut power.
    ideal = laps / 2.0 - (float(undercut_strength) - 0.5) * laps * 0.10
    ideal = max(8.0, min(laps - 6.0, ideal))
    spread = max(2.0, laps * 0.07)
    early, late = int(round(ideal - spread)), int(round(ideal + spread))

    # Undercut / overcut value (seconds), scaled by the track's strengths.
    typical_age = laps / 3.0
    undercut_value = round(slope * typical_age * float(undercut_strength) - pit_loss_s * 0.04, 2)
    overcut_value = round(slope * (typical_age * 0.5) * float(overcut_strength), 2)

    # A safety-car stop saves ~half the green-flag pit loss; weight by SC probability.
    sc_pit_save = round(pit_loss_s * 0.55, 2)
    sc_expected_value = round(sc_pit_save * float(safety_car_probability), 2)

    return {
        "laps": laps,
        "pit_loss_s": round(pit_loss_s, 2),
        "deg_slope_s_per_lap": slope,
        "recommended_stops": recommended_stops,
        "two_stop_delta_s": two_stop_delta,
        "optimal_pit_window": {"early": early, "ideal": int(round(ideal)), "late": late},
        "undercut_value_s": undercut_value,
        "overcut_value_s": overcut_value,
        "safety_car_pit_save_s": sc_pit_save,
        "safety_car_expected_value_s": sc_expected_value,
        "preferred_tactic": "undercut" if undercut_value >= overcut_value else "overcut",
        "compounds": compound_set or [],
        "mandatory_two_compounds": True,
        "high_overtaking_difficulty": float(overtaking_difficulty) >= 0.70,
        "notes": "Analytic tire-deg model over real per-circuit pit-loss/tire-stress; not a re-sim.",
    }


def driver_strategy_delta(base: dict[str, Any], *, grid_position: int | None = None,
                          tire_management: float = 0.5) -> dict[str, Any]:
    """Per-driver tactic recommendation. The undercut is most valuable in the midfield
    pack (track position to gain); the overcut suits strong tire-management cars in
    clear air. Returns the recommended tactic + an estimated position delta."""
    undercut = float(base.get("undercut_value_s") or 0.0)
    overcut = float(base.get("overcut_value_s") or 0.0)
    grid = int(grid_position) if grid_position else 10
    midfield_boost = 1.0 + (0.15 if 4 <= grid <= 14 else 0.0)
    undercut_eff = undercut * midfield_boost
    overcut_eff = overcut * (0.8 + 0.4 * float(tire_management))
    if undercut_eff >= overcut_eff and undercut_eff > 0:
        tactic, value = "undercut", undercut_eff
    elif overcut_eff > 0:
        tactic, value = "overcut", overcut_eff
    else:
        tactic, value = "hold", 0.0
    # Rough heuristic: ~1 place per ~1.2 s of strategic gain on a normal track.
    return {
        "grid_position": grid,
        "recommended_tactic": tactic,
        "estimated_gain_s": round(value, 2),
        "estimated_position_delta": round(value / 1.2, 2),
        "tire_management": round(float(tire_management), 3),
    }
