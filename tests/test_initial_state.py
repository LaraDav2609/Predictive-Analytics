"""Tests for features/initial_state.py — uses a fake TelemetryProvider so tests
run without hitting the FastF1 API."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import pytest

from f1_ml.common.types import Lap, Race, SessionType, TelemetrySample, TireCompound, WeatherFrame
from f1_ml.providers.base import TelemetryProvider


# ---------------------------------------------------------------- fake provider

class FakeProvider(TelemetryProvider):
    """In-memory provider with canned data; lets us exercise initial_state code
    paths without touching the network or FastF1's cache."""

    def __init__(self, races_by_season: dict[int, list[Race]], laps_by_race: dict[str, list[Lap]]):
        self.races_by_season = races_by_season
        self.laps_by_race = laps_by_race

    def list_races(self, season: int) -> list[Race]:
        return self.races_by_season.get(season, [])

    def stream_telemetry(self, race: Race, session: SessionType) -> Iterator[TelemetrySample]:
        return iter(())

    def laps(self, race: Race, session: SessionType) -> list[Lap]:
        return self.laps_by_race.get(_key(race), [])

    def weather(self, race: Race) -> list[WeatherFrame]:
        return []

    def is_live(self) -> bool:
        return False


def _key(race: Race) -> str:
    return f"{race.season}-{race.round:02d}-{race.track_code}"


def _build_synthetic_history() -> FakeProvider:
    """Build a 2023 Monza race with 4 drivers, 50 laps each, known pace gaps
    + a known tire-deg slope. Used as historical input for predicting 2024 Monza."""
    monza_2023 = Race(season=2023, round=14, track_code="MONZA", name="Italy 2023",
                     scheduled_start=datetime(2023, 9, 3))

    # (code, base_pace, sigma, medium_deg_mult, hard_deg_mult) — separate
    # multipliers per compound so MEDIUM:HARD ratio differs across drivers
    # (real-world: some drivers are gentle on hards but rough on softs, etc.).
    # Asymmetry is essential for DP to pick different pit laps per driver.
    drivers = [
        ("VER", 80.0, 0.20, 1.0, 1.0),   # baseline / baseline
        ("HAM", 80.4, 0.20, 1.5, 2.0),   # rough on both, especially hard
        ("LEC", 80.7, 0.25, 0.5, 0.2),   # gentle, esp. on hards
        ("RUS", 80.9, 0.30, 1.2, 1.0),   # slightly rougher on mediums
    ]
    deg_per_lap_medium_base = 0.04
    deg_per_lap_hard_base = 0.024  # 60% of MEDIUM
    laps: list[Lap] = []
    # Simulate realistic fuel-burn so fuel correction cancels cleanly when the
    # builder applies it. fuel_correct() defaults: 110 kg start, 1.6 kg/lap, 0.030 s/kg.
    fuel_start_kg, burn_per_lap_kg, penalty_s_per_kg = 110.0, 1.6, 0.030
    for code, base_pace, sigma, medium_mult, hard_mult in drivers:
        # 25-lap stint on MEDIUM, then pit, then 25-lap stint on HARD.
        compound_slope = [
            (TireCompound.MEDIUM, deg_per_lap_medium_base * medium_mult),
            (TireCompound.HARD, deg_per_lap_hard_base * hard_mult),
        ]
        for stint_idx, (compound, slope) in enumerate(compound_slope):
            for stint_lap in range(25):
                lap_n = stint_idx * 25 + stint_lap + 1
                fuel_remaining = max(0.0, fuel_start_kg - (lap_n - 1) * burn_per_lap_kg)
                fuel_effect_s = fuel_remaining * penalty_s_per_kg
                lap_time = base_pace + slope * stint_lap + fuel_effect_s + (sigma * 0.05)
                laps.append(Lap(
                    race_id=_key(monza_2023),
                    driver_code=code,
                    lap_number=lap_n,
                    lap_time_s=lap_time,
                    sector1_s=lap_time / 3,
                    sector2_s=lap_time / 3,
                    sector3_s=lap_time / 3,
                    compound=compound,
                    tire_age_laps=stint_lap,
                    position=1,  # not used by builder
                    pit_in=(stint_idx == 0 and stint_lap == 24),
                    pit_out=(stint_idx == 1 and stint_lap == 0),
                ))
    return FakeProvider(
        races_by_season={2023: [monza_2023]},
        laps_by_race={_key(monza_2023): laps},
    )


# ------------------------------------------------------------------ tests

def test_parse_race_id_valid_formats():
    from f1_ml.features.initial_state import parse_race_id
    p = parse_race_id("2024-14-MONZA")
    assert p.season == 2024 and p.round == 14 and p.track_code == "MONZA"

    p2 = parse_race_id("2026-01-BAHRAIN")
    assert p2.season == 2026 and p2.round == 1 and p2.track_code == "BAHRAIN"


def test_parse_race_id_rejects_invalid():
    from f1_ml.features.initial_state import parse_race_id
    with pytest.raises(ValueError):
        parse_race_id("garbage")
    with pytest.raises(ValueError):
        parse_race_id("monza-2024")


def test_initial_state_recovers_known_pace_ranking():
    """The builder should produce per-driver pace estimates whose ranking
    reflects the combined effect of base pace and tire degradation behavior.

    With the asymmetric per-driver deg multipliers in the fixture (HAM rough
    on both compounds, LEC gentle on both, esp. hards), median fuel-corrected
    pace orders as VER < LEC < HAM < RUS — LEC's gentle deg compounds the
    advantage over a 50-lap stint set."""
    from f1_ml.features.initial_state import build_initial_state_from_provider

    provider = _build_synthetic_history()
    build = build_initial_state_from_provider(
        provider, race_id="2024-14-MONZA", total_laps=53, history_seasons=1,
    )
    state = build.initial_state

    assert state["total_laps"] == 53
    drivers = state["driver_codes"]
    paces = state["driver_mean_pace_s"]
    by_pace = sorted(zip(drivers, paces), key=lambda dp: dp[1])
    # VER has fastest base pace and baseline deg → fastest median.
    # RUS has slowest base pace and rough deg → slowest median.
    assert by_pace[0][0] == "VER"
    assert by_pace[-1][0] == "RUS"


def test_initial_state_fits_compound_deg():
    """Per-compound tire deg fit should recover the ground-truth slope (with tolerance)."""
    from f1_ml.features.initial_state import build_initial_state_from_provider

    provider = _build_synthetic_history()
    build = build_initial_state_from_provider(
        provider, race_id="2024-14-MONZA", total_laps=50, history_seasons=1,
    )

    assert "MEDIUM" in build.deg_fits
    assert "HARD" in build.deg_fits
    medium_slope = build.deg_fits["MEDIUM"].deg_per_lap_s
    hard_slope = build.deg_fits["HARD"].deg_per_lap_s
    # Synthetic ground truth: MEDIUM = 0.04, HARD = 0.024. Fits should be in the right ballpark.
    assert 0.02 <= medium_slope <= 0.06
    assert 0.01 <= hard_slope <= 0.05
    # MEDIUM should degrade faster than HARD.
    assert medium_slope > hard_slope


def test_initial_state_diagnostics_populated():
    from f1_ml.features.initial_state import build_initial_state_from_provider

    provider = _build_synthetic_history()
    build = build_initial_state_from_provider(
        provider, race_id="2024-14-MONZA", total_laps=53, history_seasons=1,
    )
    diag = build.diagnostics
    assert diag["race_id"] == "2024-14-MONZA"
    assert diag["history_races_used"] == ["2023-14-MONZA"]
    assert diag["n_drivers_loaded"] == 4
    assert diag["n_historical_laps"] > 100
    assert "MEDIUM" in diag["compound_deg_slopes"]


def test_initial_state_raises_when_no_history_available():
    from f1_ml.features.initial_state import build_initial_state_from_provider

    provider = FakeProvider(races_by_season={}, laps_by_race={})
    with pytest.raises(ValueError, match="no historical race laps"):
        build_initial_state_from_provider(provider, race_id="2024-14-MONZA", total_laps=50)


def test_dp_solve_picks_earlier_pit_when_starting_compound_is_high_deg():
    """Starting on a high-deg compound and switching to a low-deg one should
    push the optimal pit lap earlier than the symmetric reverse."""
    from f1_ml.common.types import TireCompound
    from f1_ml.strategy.dp_optimal_stop import make_pace_fn, solve

    fast_pace = make_pace_fn(
        base_pace_s=80.0,
        deg_per_lap_per_compound={"SOFT": 0.20, "MEDIUM": 0.05, "HARD": 0.02},
    )

    plan_starting_soft = solve(
        total_laps=50,
        current_compound=TireCompound.SOFT,
        pace_fn=fast_pace,
        pit_loss_s=23.0,
        available_compounds=(TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD),
    )
    plan_starting_hard = solve(
        total_laps=50,
        current_compound=TireCompound.HARD,
        pace_fn=fast_pace,
        pit_loss_s=23.0,
        available_compounds=(TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD),
    )

    assert plan_starting_soft.pit_laps and plan_starting_hard.pit_laps
    # Starting on soft (high deg) → ditch it earlier.
    assert plan_starting_soft.pit_laps[0] < plan_starting_hard.pit_laps[0]
    # Starting on soft → switch to a slower-deg compound.
    assert plan_starting_soft.compounds_to_fit[0] in (TireCompound.MEDIUM, TireCompound.HARD)


def test_dp_solve_must_use_two_compounds_forces_a_pit():
    """Even if not pitting would be faster, must_use_two_compounds=True forces one stop."""
    from f1_ml.common.types import TireCompound
    from f1_ml.strategy.dp_optimal_stop import make_pace_fn, solve

    # Zero deg → no incentive to pit; the rule should still force one.
    pace_fn = make_pace_fn(
        base_pace_s=80.0,
        deg_per_lap_per_compound={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
    )
    plan = solve(
        total_laps=30, current_compound=TireCompound.MEDIUM, pace_fn=pace_fn,
        pit_loss_s=23.0, must_use_two_compounds=True,
    )
    assert len(plan.pit_laps) == 1


def test_dp_solve_no_pit_when_rule_disabled_and_no_deg():
    """With must_use_two_compounds=False and no deg, optimal is to never pit."""
    from f1_ml.common.types import TireCompound
    from f1_ml.strategy.dp_optimal_stop import make_pace_fn, solve

    pace_fn = make_pace_fn(
        base_pace_s=80.0,
        deg_per_lap_per_compound={"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
        fuel_start_kg=0.0,  # remove fuel effect to keep "constant pace" pure
    )
    plan = solve(
        total_laps=30, current_compound=TireCompound.MEDIUM, pace_fn=pace_fn,
        pit_loss_s=23.0, must_use_two_compounds=False,
    )
    assert plan.pit_laps == []


def test_initial_state_optimize_strategy_produces_per_driver_picks():
    """When optimize_strategy=True, each driver's pit lap is determined by DP.
    Different drivers may converge on the same pit lap, but the pipeline must
    populate per-driver fields and surface them in diagnostics."""
    from f1_ml.features.initial_state import build_initial_state_from_provider

    provider = _build_synthetic_history()
    build = build_initial_state_from_provider(
        provider, race_id="2024-14-MONZA", total_laps=53, optimize_strategy=True,
    )
    state = build.initial_state
    diag = build.diagnostics

    assert diag["strategy_optimized"] is True
    assert len(diag["per_driver_strategy"]) == len(state["driver_codes"])
    # Every driver should have a pit_lap set (must_use_two_compounds enforced).
    assert all(p is not None for p in state["driver_pit_lap"])
    assert all(c in {"SOFT", "MEDIUM", "HARD"} for c in state["driver_pit_compound"])


def test_initial_state_fits_per_driver_compound_slopes():
    """The synthetic fixture has known per-driver deg multipliers (1.0, 1.5, 0.5, 1.2).
    Per-driver fits should recover those — HAM (mult=1.5) > VER (mult=1.0) > LEC (mult=0.5)."""
    from f1_ml.features.initial_state import build_initial_state_from_provider

    provider = _build_synthetic_history()
    build = build_initial_state_from_provider(
        provider, race_id="2024-14-MONZA", total_laps=53, optimize_strategy=True,
    )
    diag = build.diagnostics

    # We expect fits for each (driver, compound) pair where laps >= 5.
    # 4 drivers × 2 compounds = 8 fits possible.
    assert diag["n_per_driver_compound_fits"] >= 6, (
        f"expected most per-driver fits to succeed; got {diag['n_per_driver_compound_fits']}/8"
    )

    # Driver-specific starting deg slopes (everyone starts MEDIUM in the fixture):
    starting_slopes = build.initial_state["driver_starting_deg_slope"]
    drivers = build.initial_state["driver_codes"]
    slope_by_driver = dict(zip(drivers, starting_slopes))

    # Ground truth: MEDIUM × multiplier. HAM=0.06, VER=0.04, RUS=0.048, LEC=0.02.
    assert slope_by_driver["HAM"] > slope_by_driver["VER"], (
        f"HAM should have steeper deg than VER; got HAM={slope_by_driver['HAM']:.4f}, VER={slope_by_driver['VER']:.4f}"
    )
    assert slope_by_driver["VER"] > slope_by_driver["LEC"], (
        f"VER should have steeper deg than LEC; got VER={slope_by_driver['VER']:.4f}, LEC={slope_by_driver['LEC']:.4f}"
    )


def test_initial_state_per_driver_deg_drives_strategy_diversity():
    """With asymmetric per-driver deg slopes (HAM rough on hards, LEC gentle
    on hards), DP should pick measurably different pit laps per driver.

    The direction depends on the *gain* from switching compounds, not the
    starting-compound deg alone:
      - LEC's HARD is 4× better than her MEDIUM → pits early to maximize
        the long second stint on the much-better compound.
      - HAM's HARD is barely better than his MEDIUM → little incentive to
        rush the switch; pits late.
    """
    from f1_ml.features.initial_state import build_initial_state_from_provider

    provider = _build_synthetic_history()
    build = build_initial_state_from_provider(
        provider, race_id="2024-14-MONZA", total_laps=53, optimize_strategy=True,
    )
    state = build.initial_state
    drivers = state["driver_codes"]
    pit_lap_by_driver = dict(zip(drivers, state["driver_pit_lap"]))

    assert pit_lap_by_driver["HAM"] is not None and pit_lap_by_driver["LEC"] is not None
    # Bigger compound-switch gain → earlier pit. LEC's huge gain → very early; HAM's tiny gain → late.
    assert pit_lap_by_driver["LEC"] < pit_lap_by_driver["HAM"], (
        f"LEC (big switch gain) should pit earlier than HAM (small switch gain); "
        f"got LEC={pit_lap_by_driver['LEC']}, HAM={pit_lap_by_driver['HAM']}"
    )

    # The per-driver picks should produce real diversity (not all the same lap).
    unique_pit_laps = set(state["driver_pit_lap"])
    assert len(unique_pit_laps) >= 2, (
        f"expected at least 2 distinct pit laps; got {state['driver_pit_lap']}"
    )


def test_initial_state_runs_through_simulator_end_to_end():
    """Hot path validation: state from builder → simulate_race → mapper.
    Catches schema mismatches between the builder output and what the simulator
    expects."""
    from f1_ml.features.initial_state import build_initial_state_from_provider
    from f1_ml.markets.mapper import winner_probabilities
    from f1_ml.simulator.race_sim import SimConfig, simulate_race

    provider = _build_synthetic_history()
    build = build_initial_state_from_provider(
        provider, race_id="2024-14-MONZA", total_laps=53, history_seasons=1,
    )

    # Find the Race object the builder used. (Provider's list_races returns a list;
    # builder finds its own; we just construct one for the simulator's signature.)
    race_obj = provider.list_races(2023)[0]

    result = simulate_race(
        race=race_obj,
        initial_state=build.initial_state,
        models={},
        config=SimConfig(n_iterations=500, seed=0),
    )

    winner = winner_probabilities(result)
    assert sum(winner.values()) == pytest.approx(1.0, abs=1e-6)
    # VER has the fastest historical pace → should win the most often.
    assert max(winner, key=winner.get) == "VER"
