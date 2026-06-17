"""Main Monte Carlo race simulator.

Runs N (default 50,000) lap-by-lap rollouts of the race, each sampling pace,
DNFs, safety cars, pit decisions, and overtakes. Outputs the empirical joint
distribution over finishing positions per driver.

Designed so v1 (pre-race batch) and v2 (in-race lap-by-lap re-run) call the
same engine. In-race mode passes a pre-warmed initial state instead of starting
from grid.

Two pace modes:
  - **Thin-slice (default):** lap_time ~ N(driver_mean, driver_sigma), per-lap
    DNF as Bernoulli. No fuel / tire / dirty air / pit. Fast.
  - **Physical (`config.enable_physical=True`):** thin-slice pace becomes the
    "clean-air dry pace" reference; on top we apply
       fuel_correction(lap)
     + tire_degradation(compound, tire_age)
     + dirty_air_penalty(gap_ahead)
     + pit_loss on the lap a driver pits (resets tire_age, swaps compound).
    Vectorized across MC iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sports.f1.ml.common.types import Race
from sports.f1.ml.simulator.model_contract import coerce_simulator_model_bundle


# --- Compound deg defaults (s/lap). Track-tunable; override in PhysicalParams.
DEFAULT_DEG_PER_LAP_PER_COMPOUND: dict[str, float] = {
    "SOFT": 0.10,
    "MEDIUM": 0.05,
    "HARD": 0.03,
    "INTERMEDIATE": 0.07,
    "WET": 0.06,
}


@dataclass
class PhysicalParams:
    """Knobs that turn the synthetic Gaussian sim into a physically realistic one."""
    fuel_start_kg: float = 110.0
    fuel_burn_per_lap_kg: float = 1.6
    fuel_penalty_s_per_kg: float = 0.030
    pit_loss_s: float = 23.0
    dirty_air_threshold_s: float = 1.5
    dirty_air_penalty_per_s_close: float = 0.5  # s of lap-time inflation per s of gap closure
    dirty_air_max_penalty_s: float = 1.0
    deg_per_lap_per_compound: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_DEG_PER_LAP_PER_COMPOUND)
    )


@dataclass
class SimConfig:
    n_iterations: int = 50_000
    seed: int = 0
    enable_safety_cars: bool = True
    enable_dnfs: bool = True
    enable_overtakes: bool = True
    enable_physical: bool = False           # opt-in: use physics layer below
    physical: PhysicalParams = field(default_factory=PhysicalParams)
    strategy_module: str = "strategy.dp_optimal_stop"  # registry name


@dataclass
class SimResult:
    finish_positions: np.ndarray  # shape (n_iterations, n_drivers); int positions 1..n
    dnf_mask: np.ndarray          # shape (n_iterations, n_drivers); bool
    safety_car_counts: np.ndarray # shape (n_iterations,); int
    fastest_lap_driver: np.ndarray  # shape (n_iterations,); driver_idx
    driver_codes: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def simulate_race(
    race: Race,
    initial_state: dict,  # grid order, weather forecast, tire allocations
    models: dict,         # built from registry: pace, dnf, sc, overtake, pit_time, strategy
    config: SimConfig = SimConfig(),
) -> SimResult:
    """Run the full Monte Carlo. Vectorized over Monte Carlo iterations for speed.

    Required `initial_state` keys for the thin slice:
        driver_codes:           list[str]
        driver_mean_pace_s:     list[float]   length n_drivers (clean-air dry pace ref)
        driver_pace_sigma_s:    list[float]   length n_drivers
        driver_dnf_rate_per_lap:list[float]   length n_drivers
        total_laps:             int

    Additional keys consumed when `config.enable_physical=True` (all optional;
    defaults applied per driver if missing):
        driver_starting_compound: list[str]   default "MEDIUM"
        driver_pit_lap:           list[int|None]   default None (no pit)
        driver_pit_compound:      list[str|None]   default opposite of starting
    """
    rng = np.random.default_rng(config.seed)

    driver_codes: list[str] = list(initial_state["driver_codes"])
    n = len(driver_codes)
    n_laps = int(initial_state["total_laps"])
    n_iter = int(config.n_iterations)

    pace_means = np.asarray(initial_state["driver_mean_pace_s"], dtype=float)
    pace_sigmas = np.asarray(initial_state["driver_pace_sigma_s"], dtype=float)
    dnf_per_lap = (
        np.asarray(initial_state["driver_dnf_rate_per_lap"], dtype=float)
        if config.enable_dnfs
        else np.zeros(n)
    )
    model_bundle = coerce_simulator_model_bundle(models)
    model_runtime = model_bundle.runtime if model_bundle is not None else None
    if model_runtime is not None:
        pace_means, pace_sigmas = model_runtime.apply_rating_priors(driver_codes, pace_means, pace_sigmas)

    # Cumulative race time, best-lap-so-far, active mask per (iteration, driver).
    cum_time = np.zeros((n_iter, n))
    best_lap = np.full((n_iter, n), np.inf)
    active = np.ones((n_iter, n), dtype=bool)
    sc_counts = np.zeros(n_iter, dtype=int)

    if config.enable_physical:
        return _simulate_physical(
            rng, driver_codes, n, n_laps, n_iter,
            pace_means, pace_sigmas, dnf_per_lap,
            cum_time, best_lap, active, sc_counts,
            initial_state, config, model_runtime,
        )

    # ----- Thin-slice fast path -----
    for lap in range(1, n_laps + 1):
        lap_means = pace_means
        lap_sigmas = pace_sigmas
        lap_dnf = dnf_per_lap
        if model_runtime is not None:
            state = {"cum_time": cum_time, "active": active, "best_lap": best_lap}
            features = {"initial_state": initial_state, "race": race, "mode": "thin_slice"}
            lap_means, lap_sigmas = model_runtime.pace_arrays(driver_codes, lap, pace_means, pace_sigmas, state, features)
            if config.enable_dnfs:
                lap_dnf = model_runtime.dnf_array(driver_codes, lap, dnf_per_lap, state, features)

        lap_times = rng.normal(lap_means, lap_sigmas, size=(n_iter, n))

        if config.enable_dnfs and lap_dnf.max() > 0:
            dnf_now = rng.random(size=(n_iter, n)) < lap_dnf
            active &= ~dnf_now

        cum_time = np.where(active, cum_time + lap_times, cum_time)
        best_lap = np.where(active & (lap_times < best_lap), lap_times, best_lap)

    return _build_result(cum_time, best_lap, active, sc_counts, driver_codes, _model_metadata(model_runtime))


def _simulate_physical(
    rng, driver_codes, n, n_laps, n_iter,
    pace_means, pace_sigmas, dnf_per_lap,
    cum_time, best_lap, active, sc_counts,
    initial_state, config, model_runtime=None,
) -> SimResult:
    p = config.physical

    # Per-driver compound start (string code) → deg slope (s/lap)
    starting_compound = list(initial_state.get("driver_starting_compound", ["MEDIUM"] * n))
    pit_lap_raw = list(initial_state.get("driver_pit_lap", [None] * n))
    pit_compound = list(initial_state.get("driver_pit_compound", [None] * n))

    # Per-driver tire-deg slopes (precomputed from history). When absent, fall back
    # to PhysicalParams' compound-level table.
    explicit_start_slope = initial_state.get("driver_starting_deg_slope")
    explicit_pit_slope = initial_state.get("driver_pit_deg_slope")

    if explicit_start_slope is not None:
        deg_slope = np.asarray(explicit_start_slope, dtype=float)
    else:
        deg_slope = np.array(
            [p.deg_per_lap_per_compound.get(c, p.deg_per_lap_per_compound["MEDIUM"]) for c in starting_compound],
            dtype=float,
        )
    pit_lap = np.array([(-1 if x is None else int(x)) for x in pit_lap_raw], dtype=int)

    if explicit_pit_slope is not None:
        deg_slope_after_pit = np.asarray(explicit_pit_slope, dtype=float)
    else:
        deg_slope_after_pit = np.array(
            [
                p.deg_per_lap_per_compound.get(pc, p.deg_per_lap_per_compound["MEDIUM"])
                if pc is not None
                else _opposite_deg(starting_compound[i], p)
                for i, pc in enumerate(pit_compound)
            ],
            dtype=float,
        )

    # Per (iter, driver) state.
    tire_age = np.zeros((n_iter, n), dtype=int)
    current_deg_slope = np.broadcast_to(deg_slope, (n_iter, n)).copy()
    has_pitted = np.zeros((n_iter, n), dtype=bool)

    # Reference clean-air dry pace mean per driver (broadcast each lap).
    pace_means_2d = np.broadcast_to(pace_means, (n_iter, n))

    for lap in range(1, n_laps + 1):
        lap_means = pace_means
        lap_sigmas = pace_sigmas
        lap_dnf = dnf_per_lap
        if model_runtime is not None:
            state = {
                "cum_time": cum_time,
                "active": active,
                "best_lap": best_lap,
                "tire_age": tire_age,
                "has_pitted": has_pitted,
            }
            features = {"initial_state": initial_state, "race_mode": "physical", "lap": lap}
            lap_means, lap_sigmas = model_runtime.pace_arrays(driver_codes, lap, pace_means, pace_sigmas, state, features)
            if config.enable_dnfs:
                lap_dnf = model_runtime.dnf_array(driver_codes, lap, dnf_per_lap, state, features)

        # 1. Sample raw clean-air pace.
        pace_means_2d = np.broadcast_to(lap_means, (n_iter, n))
        lap_times = rng.normal(pace_means_2d, lap_sigmas, size=(n_iter, n))

        # 2. Fuel: lap-time inflation from remaining fuel mass.
        fuel_remaining = max(0.0, p.fuel_start_kg - (lap - 1) * p.fuel_burn_per_lap_kg)
        lap_times += fuel_remaining * p.fuel_penalty_s_per_kg

        # 3. Tire degradation: per (iter, driver), slope × age of current stint.
        lap_times += current_deg_slope * tire_age

        # 4. Pit stop: any driver whose pit_lap == lap pays pit_loss_s and resets tire age + slope.
        # Per-iteration `has_pitted` ensures we don't double-pit when v2 multi-stop strategy lands.
        pitting_per_driver = (pit_lap == lap)  # shape (n,)
        pitting_2d = np.broadcast_to(pitting_per_driver, (n_iter, n)) & ~has_pitted & active
        if pitting_2d.any():
            lap_times = np.where(pitting_2d, lap_times + p.pit_loss_s, lap_times)
            tire_age = np.where(pitting_2d, 0, tire_age)
            current_deg_slope = np.where(pitting_2d, deg_slope_after_pit, current_deg_slope)
            has_pitted |= pitting_2d

        # 5. DNFs (Bernoulli per iter,driver).
        if config.enable_dnfs and lap_dnf.max() > 0:
            dnf_now = rng.random(size=(n_iter, n)) < lap_dnf
            active &= ~dnf_now

        # 6. Tentative cum_time (before dirty air; gaps computed against this).
        tentative_cum = np.where(active, cum_time + lap_times, np.inf)

        # 7. Dirty air: penalize close-following cars based on gap_ahead at this point in race.
        if p.dirty_air_penalty_per_s_close > 0:
            order = np.argsort(tentative_cum, axis=1)  # (n_iter, n) — driver index per finishing position
            sorted_cum = np.take_along_axis(tentative_cum, order, axis=1)
            # `inf - inf = nan` for back-of-grid DNF'd drivers; suppress the warning
            # since the isfinite mask below zeroes those penalties anyway.
            with np.errstate(invalid="ignore"):
                gaps_by_position = np.diff(sorted_cum, axis=1, prepend=np.inf)
            gap_ahead = np.empty_like(tentative_cum)
            np.put_along_axis(gap_ahead, order, gaps_by_position, axis=1)

            closeness = np.maximum(0.0, p.dirty_air_threshold_s - gap_ahead)
            penalty = np.minimum(closeness * p.dirty_air_penalty_per_s_close, p.dirty_air_max_penalty_s)
            penalty = np.where(np.isfinite(gap_ahead), penalty, 0.0)
            lap_times = np.where(active, lap_times + penalty, lap_times)

        # 8. Commit: accumulate time, update best lap, age tires.
        cum_time = np.where(active, cum_time + lap_times, cum_time)
        best_lap = np.where(active & (lap_times < best_lap), lap_times, best_lap)
        tire_age = np.where(active, tire_age + 1, tire_age)

    return _build_result(cum_time, best_lap, active, sc_counts, driver_codes, _model_metadata(model_runtime))


def _opposite_deg(starting: str, p: PhysicalParams) -> float:
    """Default post-pit compound: pick the next harder compound for sensible alt-strategy.
    Ensures the simulator runs even when caller didn't specify a pit_compound."""
    order = ["SOFT", "MEDIUM", "HARD"]
    if starting in order:
        idx = order.index(starting)
        next_compound = order[min(idx + 1, len(order) - 1)]
    else:
        next_compound = "MEDIUM"
    return p.deg_per_lap_per_compound.get(next_compound, p.deg_per_lap_per_compound["MEDIUM"])


def _build_result(
    cum_time: np.ndarray,
    best_lap: np.ndarray,
    active: np.ndarray,
    sc_counts: np.ndarray,
    driver_codes: list[str],
    metadata: dict | None = None,
) -> SimResult:
    n_iter, n = cum_time.shape

    final_time = np.where(active, cum_time, np.inf)
    order = np.argsort(final_time, axis=1)
    positions = np.empty_like(order)
    rank_indices = np.broadcast_to(np.arange(1, n + 1), (n_iter, n))
    np.put_along_axis(positions, order, rank_indices, axis=1)

    fastest_lap_driver = np.argmin(np.where(active, best_lap, np.inf), axis=1)

    return SimResult(
        finish_positions=positions,
        dnf_mask=~active,
        safety_car_counts=sc_counts,
        fastest_lap_driver=fastest_lap_driver,
        driver_codes=driver_codes,
        metadata=metadata or {},
    )


def _model_metadata(model_runtime) -> dict:
    if model_runtime is None:
        return {"ml_model_contract_used": False}
    return model_runtime.metadata()
