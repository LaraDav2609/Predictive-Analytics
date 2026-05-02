"""Tests for f1_ml.features — pace_gp, track_evolution, mini_sectors, driver_form."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from f1_ml.common.types import TelemetrySample
from f1_ml.features import driver_form, mini_sectors, track_evolution
from f1_ml.features.pace_gp import PaceGP


# -------------------------------------------------------------------- pace_gp

def test_pace_gp_recovers_smooth_signal():
    rng = np.random.default_rng(0)
    laps = np.arange(1, 41)
    truth = 80.0 + 0.05 * laps  # linear trend
    obs = truth + rng.normal(0, 0.1, size=len(laps))

    gp = PaceGP(length_scale_laps=8.0, noise_s=0.1).fit(laps, obs)
    mean, std = gp.predict(laps)
    assert mean.shape == laps.shape
    assert std.shape == laps.shape
    # MAE between GP mean and ground truth should be small.
    assert np.mean(np.abs(mean - truth)) < 0.15


def test_pace_gp_uncertainty_grows_off_training_support():
    laps = np.arange(1, 21)
    obs = 80.0 + np.zeros_like(laps, dtype=float)
    gp = PaceGP(length_scale_laps=3.0, noise_s=0.1).fit(laps, obs)
    _, std_in = gp.predict(np.array([10]))
    _, std_out = gp.predict(np.array([60]))
    assert std_out[0] > std_in[0]


def test_pace_gp_predict_without_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        PaceGP().predict(np.array([1, 2, 3]))


def test_pace_gp_rejects_invalid_hyperparams():
    with pytest.raises(ValueError):
        PaceGP(length_scale_laps=0)
    with pytest.raises(ValueError):
        PaceGP(noise_s=-0.1)


def test_pace_gp_rejects_empty_fit():
    with pytest.raises(ValueError):
        PaceGP().fit(np.array([]), np.array([]))


# ----------------------------------------------------------- track_evolution

def _make_evolution_history() -> pd.DataFrame:
    """Race ~ 80 s; FP3 1.005×, FP2 1.01×, FP1 1.02×, Q 0.985×."""
    rows = []
    for r in range(8):
        rows.append({"track_code": "MONZA", "session": "FP1",
                     "median_pace_s": 80.0 * 1.020, "race_id": r})
        rows.append({"track_code": "MONZA", "session": "FP2",
                     "median_pace_s": 80.0 * 1.010, "race_id": r})
        rows.append({"track_code": "MONZA", "session": "FP3",
                     "median_pace_s": 80.0 * 1.005, "race_id": r})
        rows.append({"track_code": "MONZA", "session": "Q",
                     "median_pace_s": 80.0 * 0.985, "race_id": r})
        rows.append({"track_code": "MONZA", "session": "R",
                     "median_pace_s": 80.0, "race_id": r})
    return pd.DataFrame(rows)


def test_track_evolution_factors_are_ordered():
    fit = track_evolution.fit(_make_evolution_history())
    assert fit.fp1_factor > fit.fp2_factor > fit.fp3_factor > fit.race_factor > fit.quali_factor


def test_track_evolution_normalizes_to_race_correctly():
    fit = track_evolution.fit(_make_evolution_history())
    # FP3 lap of 80.4 should normalize to ~80.0 race-equivalent (factor 1.005).
    race_equiv = track_evolution.normalize_to_race(80.4, "FP3", fit)
    assert race_equiv == pytest.approx(80.4 / 1.005, rel=1e-6)


def test_track_evolution_unknown_session_raises():
    fit = track_evolution.fit(_make_evolution_history())
    with pytest.raises(ValueError, match="unknown session"):
        track_evolution.normalize_to_race(80.0, "GIBBERISH", fit)


def test_track_evolution_requires_race_baseline():
    rows = [{"track_code": "X", "session": "FP1", "median_pace_s": 80.0}]
    with pytest.raises(ValueError, match="race-session"):
        track_evolution.fit(pd.DataFrame(rows))


def test_track_evolution_rejects_empty_input():
    with pytest.raises(ValueError):
        track_evolution.fit(pd.DataFrame(columns=["track_code", "session", "median_pace_s"]))


def test_track_evolution_intra_session_grip_up_monotone_decreasing():
    """Lap times should decrease as the session progresses (more grip)."""
    p1 = track_evolution.evolve_within_session(80.0, lap_in_session=1, laps_in_session=20)
    p10 = track_evolution.evolve_within_session(80.0, lap_in_session=10, laps_in_session=20)
    p20 = track_evolution.evolve_within_session(80.0, lap_in_session=20, laps_in_session=20)
    assert p1 >= p10 >= p20


# ----------------------------------------------------------- mini_sectors

def _make_telemetry_samples(
    n_laps: int = 3, n_ticks_per_lap: int = 50, lap_length_m: float = 5000.0
) -> list[TelemetrySample]:
    """Synthetic ticks: constant speed driver running n_laps laps."""
    samples: list[TelemetrySample] = []
    base = datetime(2024, 9, 1, 14, 0, 0)
    for lap in range(1, n_laps + 1):
        # Each lap takes 80 s; tick spacing ~ 80/n_ticks_per_lap seconds.
        for i in range(n_ticks_per_lap):
            frac = i / max(n_ticks_per_lap - 1, 1)
            samples.append(TelemetrySample(
                race_id="R1",
                driver_code="VER",
                timestamp=base + timedelta(seconds=(lap - 1) * 80 + frac * 80),
                lap=lap,
                s_coord=frac * lap_length_m,
                speed_kph=200.0,
                accel_long_g=0.0,
                accel_lat_g=0.0,
            ))
    return samples


def test_mini_sectors_build_reference_line_returns_n_segments():
    samples = _make_telemetry_samples()
    ref = mini_sectors.build_reference_line(samples, n_segments=10)
    assert len(ref) == 10
    assert (ref["s_end"] > ref["s_start"]).all()


def test_mini_sectors_decompose_assigns_one_row_per_lap_sector():
    samples = _make_telemetry_samples(n_laps=2, n_ticks_per_lap=50)
    ref = mini_sectors.build_reference_line(samples, n_segments=5)
    out = mini_sectors.decompose(samples, ref)
    # 2 laps × 5 sectors expected.
    assert {"driver_code", "lap", "mini_sector_idx", "time_s", "avg_speed_kph"}.issubset(out.columns)
    assert len(out) == 2 * 5
    # All times positive (each sector spans real samples).
    assert (out["time_s"] > 0).all()


def test_mini_sectors_decompose_handles_empty_samples():
    out = mini_sectors.decompose([], pd.DataFrame(columns=["segment_idx", "s_start", "s_end"]))
    assert out.empty


def test_mini_sectors_build_reference_handles_empty_samples():
    ref = mini_sectors.build_reference_line([], n_segments=5)
    assert ref.empty


def test_mini_sectors_invalid_n_segments_raises():
    with pytest.raises(ValueError):
        mini_sectors.build_reference_line(_make_telemetry_samples(), n_segments=0)


# ----------------------------------------------------------- driver_form

def _make_results_history() -> pd.DataFrame:
    """A & B alternate between 1st and 2nd."""
    rows = []
    for race in range(10):
        if race % 2 == 0:
            rows.append({"driver_code": "A", "race": race, "position": 1, "team_code": "RBR",
                         "quali_time_s": 80.0})
            rows.append({"driver_code": "B", "race": race, "position": 2, "team_code": "RBR",
                         "quali_time_s": 80.2})
        else:
            rows.append({"driver_code": "A", "race": race, "position": 2, "team_code": "RBR",
                         "quali_time_s": 80.1})
            rows.append({"driver_code": "B", "race": race, "position": 1, "team_code": "RBR",
                         "quali_time_s": 79.9})
    return pd.DataFrame(rows)


def test_rolling_finish_position_average_around_1_5():
    """Both A and B alternate 1st/2nd so EWMA → ~1.5."""
    out = driver_form.rolling_finish_position(_make_results_history(), halflife_races=2.0)
    final_a = out[out["driver_code"] == "A"].iloc[-1]["rolling_finish_position"]
    final_b = out[out["driver_code"] == "B"].iloc[-1]["rolling_finish_position"]
    assert 1.2 < final_a < 1.8
    assert 1.2 < final_b < 1.8


def test_teammate_quali_gap_is_symmetric_per_race():
    """For each race, the teammate gap must be equal-and-opposite (symmetric)."""
    out = driver_form.teammate_quali_gap(_make_results_history(), halflife_races=10.0)
    # On each race, A's EWMA value should equal -B's EWMA value (the gap is
    # antisymmetric within a teammate pair).
    pivot = out.pivot(index="race", columns="driver_code", values="ewma_teammate_quali_gap_s")
    np.testing.assert_allclose(pivot["A"].values, -pivot["B"].values, atol=1e-9)


def test_rolling_finish_position_handles_empty_input():
    out = driver_form.rolling_finish_position(
        pd.DataFrame(columns=["driver_code", "race", "position"])
    )
    assert out.empty


def test_teammate_quali_gap_handles_empty_input():
    out = driver_form.teammate_quali_gap(
        pd.DataFrame(columns=["driver_code", "race", "team_code", "quali_time_s"])
    )
    assert out.empty


def test_championship_pressure_leader_under_pressure_late_season():
    standings = pd.DataFrame([
        {"season": 2024, "round": 20, "driver_code": "VER", "points": 400},
        {"season": 2024, "round": 20, "driver_code": "NOR", "points": 380},
        {"season": 2024, "round": 20, "driver_code": "PIA", "points": 200},
    ])
    out = driver_form.championship_pressure(standings, race_round=20, season=2024, n_total_rounds=24)
    pressure = dict(zip(out["driver_code"], out["championship_pressure"]))
    # Title-fighters: positive pressure (more conservative).
    assert pressure["VER"] > 0
    assert pressure["NOR"] > 0
    # PIA is 200 pts back: out of contention, takes risks.
    assert pressure["PIA"] < 0


def test_championship_pressure_handles_empty_input():
    out = driver_form.championship_pressure(
        pd.DataFrame(columns=["season", "round", "driver_code", "points"]),
        race_round=10,
        season=2024,
    )
    assert out.empty
