"""Tests for simulator helpers — restart_logic + lap_step."""

from __future__ import annotations

import numpy as np
import pytest

from f1_ml.simulator.lap_step import LapState, LapStepConfig, step
from f1_ml.simulator.restart_logic import (
    RestartConfig,
    bunch_field,
    pit_loss_under_sc,
    restart_overtake_modifier,
)


# -------------------------------------------------------------- restart_logic

def test_bunch_field_collapses_gaps_to_legal_minimum():
    """After bunching, every non-DNF car should be `bunching_gap_s` behind the
    car ahead (in current order)."""
    times = [100.0, 105.0, 112.0, 120.0]
    config = RestartConfig(bunching_gap_s=0.5)
    new = bunch_field(times, leader_idx=0, config=config)
    assert new[0] == 100.0
    assert new[1] == pytest.approx(100.5)
    assert new[2] == pytest.approx(101.0)
    assert new[3] == pytest.approx(101.5)


def test_bunch_field_preserves_running_order():
    """The bunched ordering must match the pre-bunch ordering by cum time."""
    times = [120.0, 100.0, 105.0, 112.0]  # idx 1 is leader
    new = bunch_field(times, leader_idx=1, config=RestartConfig())
    # New ordering by cum time = [1, 2, 3, 0].
    assert sorted(range(4), key=lambda i: new[i]) == [1, 2, 3, 0]


def test_bunch_field_skips_dnf_cars():
    times = [100.0, 105.0, 999.0, 120.0]
    dnf = [False, False, True, False]
    new = bunch_field(times, leader_idx=0, config=RestartConfig(bunching_gap_s=0.5), dnf_mask=dnf)
    assert new[0] == 100.0
    assert new[1] == pytest.approx(100.5)
    assert new[2] == 999.0  # untouched
    assert new[3] == pytest.approx(101.0)


def test_bunch_field_handles_empty_input():
    assert bunch_field([], leader_idx=0, config=RestartConfig()) == []


def test_bunch_field_invalid_leader_raises():
    with pytest.raises(IndexError):
        bunch_field([100.0, 105.0], leader_idx=5, config=RestartConfig())


def test_bunch_field_dnf_leader_raises():
    with pytest.raises(ValueError):
        bunch_field([100.0, 105.0], leader_idx=0, config=RestartConfig(),
                    dnf_mask=[True, False])


def test_restart_overtake_modifier_caps_at_one():
    config = RestartConfig(restart_overtake_boost=2.0)
    assert restart_overtake_modifier(0.6, config) == 1.0


def test_restart_overtake_modifier_boosts_low_prob():
    config = RestartConfig(restart_overtake_boost=1.5)
    assert restart_overtake_modifier(0.4, config) == pytest.approx(0.6)


def test_restart_overtake_modifier_rejects_invalid_input():
    with pytest.raises(ValueError):
        restart_overtake_modifier(1.5, RestartConfig())


def test_pit_loss_under_sc_halves_the_loss():
    config = RestartConfig(pit_loss_under_sc_factor=0.5)
    assert pit_loss_under_sc(20.0, config) == pytest.approx(10.0)


def test_pit_loss_under_sc_rejects_negative():
    with pytest.raises(ValueError):
        pit_loss_under_sc(-1.0, RestartConfig())


# ------------------------------------------------------------------ lap_step

def _make_state(n: int = 3) -> LapState:
    return LapState(
        lap=0,
        cumulative_time_s=[0.0] * n,
        tire_age=[0] * n,
        compounds=["MEDIUM"] * n,
        dnf=[False] * n,
        sc_active=False,
        fuel_kg=[100.0] * n,
    )


def test_lap_step_increments_lap_and_cum_time():
    rng = np.random.default_rng(0)
    state = _make_state()
    new_state = step(
        state,
        pace_means=[80.0, 80.5, 81.0],
        pace_sigmas=[0.1, 0.1, 0.1],
        dnf_per_lap=[0.0, 0.0, 0.0],
        pit_decisions=[False, False, False],
        pit_compounds=[None, None, None],
        config=LapStepConfig(),
        rng=rng,
    )
    assert new_state.lap == 1
    # Cum time should have grown by ~80s + fuel correction (100kg * 0.030 = 3s).
    for t in new_state.cumulative_time_s:
        assert 80.0 < t < 90.0


def test_lap_step_burns_fuel_each_lap():
    rng = np.random.default_rng(0)
    state = _make_state()
    new_state = step(
        state,
        pace_means=[80.0, 80.0, 80.0],
        pace_sigmas=[0.0, 0.0, 0.0],
        dnf_per_lap=[0.0, 0.0, 0.0],
        pit_decisions=[False, False, False],
        pit_compounds=[None, None, None],
        config=LapStepConfig(fuel_burn_per_lap_kg=2.0),
        rng=rng,
    )
    for f in new_state.fuel_kg:
        assert f == pytest.approx(98.0)


def test_lap_step_pit_decision_resets_tire_age():
    rng = np.random.default_rng(0)
    state = _make_state()
    state.tire_age = [10, 10, 10]
    new_state = step(
        state,
        pace_means=[80.0, 80.0, 80.0],
        pace_sigmas=[0.0, 0.0, 0.0],
        dnf_per_lap=[0.0, 0.0, 0.0],
        pit_decisions=[True, False, False],
        pit_compounds=["HARD", None, None],
        config=LapStepConfig(),
        rng=rng,
    )
    assert new_state.tire_age[0] == 0
    assert new_state.compounds[0] == "HARD"
    assert new_state.tire_age[1] == 11  # not pitting → ages normally


def test_lap_step_dnf_freezes_cum_time():
    rng = np.random.default_rng(0)
    state = _make_state()
    state.dnf = [True, False, False]
    state.cumulative_time_s = [200.0, 100.0, 100.0]
    new_state = step(
        state,
        pace_means=[80.0, 80.0, 80.0],
        pace_sigmas=[0.0, 0.0, 0.0],
        dnf_per_lap=[0.0, 0.0, 0.0],
        pit_decisions=[False, False, False],
        pit_compounds=[None, None, None],
        config=LapStepConfig(),
        rng=rng,
    )
    # DNF'd car retains its cum time; others advance.
    assert new_state.cumulative_time_s[0] == 200.0
    assert new_state.cumulative_time_s[1] > 100.0


def test_lap_step_pit_under_sc_reduces_loss():
    rng = np.random.default_rng(0)
    state = _make_state()
    state.sc_active = True
    new_state = step(
        state,
        pace_means=[80.0, 80.0, 80.0],
        pace_sigmas=[0.0, 0.0, 0.0],
        dnf_per_lap=[0.0, 0.0, 0.0],
        pit_decisions=[True, False, False],
        pit_compounds=[None, None, None],
        config=LapStepConfig(
            pit_loss_s=20.0,
            restart=RestartConfig(pit_loss_under_sc_factor=0.5),
            dirty_air_penalty_per_s_close=0.0,
        ),
        rng=rng,
    )
    # Pit during SC: pit_loss = 20 * 0.5 = 10, so cum_time ≈ 80 + 100 * 0.030 + 10 = 93
    diff = new_state.cumulative_time_s[0] - new_state.cumulative_time_s[1]
    assert 9.5 < diff < 10.5


def test_lap_step_dnf_per_lap_kills_drivers_eventually():
    """Run with high DNF rate; expect at least one car retired after a few laps."""
    rng = np.random.default_rng(0)
    state = _make_state(n=5)
    for _ in range(20):
        state = step(
            state,
            pace_means=[80.0] * 5,
            pace_sigmas=[0.1] * 5,
            dnf_per_lap=[0.5] * 5,  # 50% per lap
            pit_decisions=[False] * 5,
            pit_compounds=[None] * 5,
            config=LapStepConfig(),
            rng=rng,
        )
    assert any(state.dnf)


def test_lap_step_validates_input_lengths():
    rng = np.random.default_rng(0)
    state = _make_state(n=3)
    with pytest.raises(ValueError):
        step(
            state,
            pace_means=[80.0, 80.0],  # length mismatch
            pace_sigmas=[0.1] * 3,
            dnf_per_lap=[0.0] * 3,
            pit_decisions=[False] * 3,
            pit_compounds=[None] * 3,
            config=LapStepConfig(),
            rng=rng,
        )
