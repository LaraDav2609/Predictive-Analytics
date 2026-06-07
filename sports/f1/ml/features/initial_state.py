"""Build a simulator `initial_state` dict from a TelemetryProvider.

This is the bridge between the data layer (FastF1 / vendor / synthetic) and the
simulator. It pulls historical laps for the target track, fuel-corrects them,
and emits per-driver pace stats + per-compound tire-deg fits in the shape
`simulate_race(initial_state=...)` expects.

Entry point:
    build_initial_state_from_provider(provider, race_id, total_laps, ...)

Race ID format: `YYYY-RR-TRACK`, e.g. `2024-14-MONZA`. Parsed by `parse_race_id`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median
from typing import Iterable

import pandas as pd

from sports.f1.ml.common.types import Lap, Race, SessionType, TireCompound
from sports.f1.ml.features.fuel_correction import fuel_correct
from sports.f1.ml.features.tire_degradation import DegradationFit, fit_linear
from sports.f1.ml.providers.base import TelemetryProvider
from sports.f1.ml.simulator.race_sim import DEFAULT_DEG_PER_LAP_PER_COMPOUND
from sports.f1.ml.strategy.dp_optimal_stop import make_pace_fn, solve


_RACE_ID_RE = re.compile(r"^(\d{4})-(\d{1,2})-([A-Z0-9_]+)$")


@dataclass
class ParsedRaceId:
    season: int
    round: int
    track_code: str


def parse_race_id(race_id: str) -> ParsedRaceId:
    """`2024-14-MONZA` → ParsedRaceId(season=2024, round=14, track_code='MONZA')."""
    m = _RACE_ID_RE.match(race_id)
    if not m:
        raise ValueError(f"race_id must be 'YYYY-RR-TRACK' (got {race_id!r})")
    return ParsedRaceId(season=int(m.group(1)), round=int(m.group(2)), track_code=m.group(3))


@dataclass
class InitialStateBuild:
    initial_state: dict
    diagnostics: dict
    deg_fits: dict[str, DegradationFit]


def build_initial_state_from_provider(
    provider: TelemetryProvider,
    race_id: str,
    total_laps: int,
    history_seasons: int = 1,
    default_pace_sigma_s: float = 0.30,
    default_dnf_rate_per_lap: float = 0.001,
    pit_lap: int | None = None,
    pit_compound: str = "HARD",
    starting_compound: str = "MEDIUM",
    optimize_strategy: bool = True,
    pit_loss_s: float = 23.0,
    available_compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD"),
) -> InitialStateBuild:
    """Pull historical laps for the same track in the previous `history_seasons`
    seasons and convert into a simulator-ready initial_state dict.

    Returns the dict plus diagnostics (drivers loaded, fallback counts, tire-deg
    fits) so the caller can log what actually went into the prediction.
    """
    parsed = parse_race_id(race_id)

    # Collect laps from prior season(s) at the same track.
    historical_laps: list[Lap] = []
    history_races_used: list[Race] = []
    for offset in range(1, history_seasons + 1):
        try:
            prior_season_races = provider.list_races(parsed.season - offset)
        except Exception:
            continue
        match = next((r for r in prior_season_races if r.track_code == parsed.track_code), None)
        if match is None:
            continue
        try:
            laps = provider.laps(match, SessionType.RACE)
        except Exception:
            continue
        if laps:
            historical_laps.extend(laps)
            history_races_used.append(match)

    if not historical_laps:
        raise ValueError(
            f"no historical race laps found for track '{parsed.track_code}' in "
            f"the {history_seasons} season(s) before {parsed.season}"
        )

    laps_df = _laps_to_df(historical_laps)
    laps_df = _drop_pit_and_outlier_laps(laps_df)
    laps_df = fuel_correct(
        laps_df, lap_col="lap_number", lap_time_col="lap_time_s",
    )

    # Per-driver pace stats.
    per_driver = (
        laps_df.groupby("driver_code")["fuel_corrected_lap_time_s"]
        .agg(["median", "std", "count"])
        .rename(columns={"median": "median_pace_s", "std": "pace_sigma_s", "count": "n_laps"})
        .fillna({"pace_sigma_s": default_pace_sigma_s})
    )
    # Floor sigma to avoid pathological deterministic drivers when we have few laps.
    per_driver["pace_sigma_s"] = per_driver["pace_sigma_s"].clip(lower=default_pace_sigma_s / 2)

    drivers = per_driver.index.tolist()

    # Per-compound tire-deg fits (cross-driver average — used as fallback).
    deg_fits: dict[str, DegradationFit] = {}
    for compound, group in laps_df.groupby("compound"):
        if len(group) < 5:
            continue
        group_for_fit = group.assign(driver_code="_global", track_code=parsed.track_code).copy()
        deg_fits[str(compound)] = fit_linear(group_for_fit)

    # Per-(driver, compound) fits — surface natural strategy diversity. Drivers
    # with too few laps on a compound fall back to the global average above.
    per_driver_compound_fits: dict[tuple[str, str], DegradationFit] = {}
    for (drv, compound), group in laps_df.groupby(["driver_code", "compound"]):
        if len(group) < 5:
            continue
        group_for_fit = group.assign(track_code=parsed.track_code).copy()
        try:
            per_driver_compound_fits[(str(drv), str(compound))] = fit_linear(group_for_fit)
        except Exception:
            continue

    # Default DNF rate from observed retirements (very rough — use season avg
    # if FastF1 doesn't expose explicit DNF flags). Fallback to a flat default.
    dnf_rate = default_dnf_rate_per_lap

    pace_list = per_driver["median_pace_s"].tolist()

    # --- per-driver strategy ---
    # Global per-compound fallback table (used when a driver lacks data on a compound).
    global_deg_table = {c: f.deg_per_lap_s for c, f in deg_fits.items()}
    for c in available_compounds:
        global_deg_table.setdefault(c, DEFAULT_DEG_PER_LAP_PER_COMPOUND.get(c, 0.05))

    # Per-driver per-compound deg table with fallback hierarchy:
    #   1. driver-specific fit on this compound (if ≥5 laps observed)
    #   2. cross-driver fit on this compound from the same race
    #   3. physics-realistic default (SOFT > MEDIUM > HARD)
    def _slope_for(driver_code: str, compound: str) -> float:
        if (driver_code, compound) in per_driver_compound_fits:
            return per_driver_compound_fits[(driver_code, compound)].deg_per_lap_s
        return global_deg_table[compound]

    driver_compound_table: dict[str, dict[str, float]] = {
        drv: {c: _slope_for(drv, c) for c in available_compounds} for drv in drivers
    }

    strategy_diagnostics: list[dict] = []
    if pit_lap is None and optimize_strategy:
        per_driver_pit_laps: list[int | None] = []
        per_driver_pit_compounds: list[str | None] = []
        compound_objs = tuple(TireCompound(c) for c in available_compounds if c in {x.value for x in TireCompound})
        for driver_idx, base_pace in enumerate(pace_list):
            driver = drivers[driver_idx]
            # Use this driver's per-compound slopes (with fallbacks already applied).
            pace_fn = make_pace_fn(
                base_pace_s=float(base_pace),
                deg_per_lap_per_compound=driver_compound_table[driver],
            )
            plan = solve(
                total_laps=total_laps,
                current_compound=TireCompound(starting_compound),
                pace_fn=pace_fn,
                pit_loss_s=pit_loss_s,
                available_compounds=compound_objs,
                must_use_two_compounds=True,
            )
            chosen_lap = plan.pit_laps[0] if plan.pit_laps else None
            chosen_compound = plan.compounds_to_fit[0].value if plan.compounds_to_fit else None
            per_driver_pit_laps.append(chosen_lap)
            per_driver_pit_compounds.append(chosen_compound)
            strategy_diagnostics.append({
                "driver": driver,
                "pit_lap": chosen_lap,
                "pit_compound": chosen_compound,
                "expected_race_time_s": round(plan.expected_total_time_s, 2),
            })
    else:
        per_driver_pit_laps = [pit_lap] * len(drivers)
        per_driver_pit_compounds = [pit_compound] * len(drivers)

    # Per-driver tire-deg slopes for the simulator's lap loop. Falls back to
    # compound-level / default if simulator-side PhysicalParams are used instead.
    driver_starting_deg_slope = [
        driver_compound_table[drv][starting_compound] for drv in drivers
    ]
    driver_pit_deg_slope = [
        driver_compound_table[drv][pc] if pc else driver_compound_table[drv][starting_compound]
        for drv, pc in zip(drivers, per_driver_pit_compounds)
    ]

    initial_state = {
        "driver_codes": drivers,
        "driver_mean_pace_s": pace_list,
        "driver_pace_sigma_s": per_driver["pace_sigma_s"].tolist(),
        "driver_dnf_rate_per_lap": [dnf_rate] * len(drivers),
        "total_laps": total_laps,
        "driver_starting_compound": [starting_compound] * len(drivers),
        "driver_pit_lap": per_driver_pit_laps,
        "driver_pit_compound": per_driver_pit_compounds,
        # Per-driver tire-deg slopes (simulator honors these when present).
        "driver_starting_deg_slope": driver_starting_deg_slope,
        "driver_pit_deg_slope": driver_pit_deg_slope,
    }

    diagnostics = {
        "race_id": race_id,
        "history_races_used": [f"{r.season}-{r.round:02d}-{r.track_code}" for r in history_races_used],
        "n_drivers_loaded": len(drivers),
        "n_historical_laps": int(len(laps_df)),
        "median_lap_seconds": float(laps_df["fuel_corrected_lap_time_s"].median()),
        "compound_deg_slopes": {
            c: round(fit.deg_per_lap_s, 4) for c, fit in deg_fits.items()
        },
        "n_per_driver_compound_fits": len(per_driver_compound_fits),
        "n_possible_per_driver_compound_fits": len(drivers) * len(available_compounds),
        "strategy_optimized": pit_lap is None and optimize_strategy,
        "per_driver_strategy": strategy_diagnostics,
    }

    return InitialStateBuild(initial_state=initial_state, diagnostics=diagnostics, deg_fits=deg_fits)


# ----- helpers -----


def _laps_to_df(laps: Iterable[Lap]) -> pd.DataFrame:
    rows = []
    for lap in laps:
        rows.append({
            "race_id": lap.race_id,
            "driver_code": lap.driver_code,
            "lap_number": lap.lap_number,
            "lap_time_s": lap.lap_time_s,
            "compound": lap.compound.value if hasattr(lap.compound, "value") else str(lap.compound),
            "tire_age_laps": lap.tire_age_laps,
            "position": lap.position,
            "pit_in": lap.pit_in,
            "pit_out": lap.pit_out,
        })
    return pd.DataFrame(rows)


def _drop_pit_and_outlier_laps(df: pd.DataFrame, outlier_quantile: float = 0.97) -> pd.DataFrame:
    """Drop pit in/out laps and lap-time outliers (typically yellow-flag laps,
    safety-car laps, or sensor blips). Keeps the median pace estimate clean."""
    if df.empty:
        return df
    df = df[(~df["pit_in"]) & (~df["pit_out"])].copy()
    if df.empty:
        return df
    cutoff = df["lap_time_s"].quantile(outlier_quantile)
    return df[df["lap_time_s"] <= cutoff].reset_index(drop=True)
