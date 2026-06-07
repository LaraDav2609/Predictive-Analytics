"""Tests for the real (non-stub) feature and ratings modules.

Marks:
  - default: fast unit tests (synthetic data, deterministic)
  - slow: hierarchical Bayes (pymc sampling — ~30 s)
  - network: FastF1 integration (requires internet)

Run only fast tests:        pytest tests/test_real_models.py
Run including slow:         pytest tests/test_real_models.py -m "slow or not slow"
Run everything:             pytest tests/test_real_models.py -m ""
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------- fuel_correction

def test_fuel_correction_shrinks_with_lap_number():
    """Fuel correction should be largest at lap 1 and smallest at the end."""
    from sports.f1.ml.features.fuel_correction import fuel_correct
    laps = pd.DataFrame({
        "lap_number": [1, 25, 50],
        "lap_time_s": [85.0, 85.0, 85.0],  # same observed time → corrected times reveal fuel
    })
    out = fuel_correct(laps, start_fuel_kg=110.0, burn_per_lap_kg=1.6, penalty_s_per_kg=0.030)
    # Lap 1 has full fuel → biggest correction → lowest corrected time.
    assert out["fuel_corrected_lap_time_s"].iloc[0] < out["fuel_corrected_lap_time_s"].iloc[2]
    # Sanity on absolute value: lap 1 correction ≈ 110 * 0.030 = 3.3 s.
    assert out["fuel_correction_s"].iloc[0] == pytest.approx(110.0 * 0.030, rel=1e-6)


def test_fuel_correction_round_trip():
    """Adding the correction back should recover the original lap time."""
    from sports.f1.ml.features.fuel_correction import fuel_correct
    laps = pd.DataFrame({"lap_number": [1, 5, 10, 20], "lap_time_s": [85.1, 84.9, 84.7, 84.3]})
    out = fuel_correct(laps)
    recovered = out["fuel_corrected_lap_time_s"] + out["fuel_correction_s"]
    np.testing.assert_allclose(recovered.values, laps["lap_time_s"].values)


# -------------------------------------------------------------- tire_degradation

def test_tire_degradation_recovers_known_slope():
    """Synthetic stint with known degradation slope; fit should recover it."""
    from sports.f1.ml.features.tire_degradation import fit_linear

    rng = np.random.default_rng(0)
    tire_age = np.arange(1, 21)
    true_base = 80.0
    true_slope = 0.04
    pace = true_base + true_slope * tire_age + rng.normal(0, 0.05, size=20)

    laps = pd.DataFrame({
        "tire_age_laps": tire_age,
        "fuel_corrected_lap_time_s": pace,
        "driver_code": "TEST",
        "compound": "MEDIUM",
        "track_code": "TEST_TRACK",
    })
    fit = fit_linear(laps)
    assert fit.deg_per_lap_s == pytest.approx(true_slope, abs=0.01)
    assert fit.base_pace_s == pytest.approx(true_base, abs=0.1)
    assert fit.n_observations == 20


def test_tire_degradation_cliff_search_finds_synthetic_cliff():
    """Stint with clear pre/post slope shift — search should locate the breakpoint."""
    from sports.f1.ml.features.tire_degradation import fit_linear_with_cliff

    rng = np.random.default_rng(0)
    pre_age = np.arange(1, 11)
    post_age = np.arange(11, 26)
    pre_pace = 80.0 + 0.03 * pre_age + rng.normal(0, 0.02, size=10)
    post_pace = 80.0 + 0.03 * 10 + 0.25 * (post_age - 10) + rng.normal(0, 0.02, size=15)
    laps = pd.DataFrame({
        "tire_age_laps": np.concatenate([pre_age, post_age]),
        "fuel_corrected_lap_time_s": np.concatenate([pre_pace, post_pace]),
        "driver_code": "TEST",
        "compound": "SOFT",
        "track_code": "T",
    })
    fit = fit_linear_with_cliff(laps)
    # Cliff should be detected somewhere near tire_age = 10.
    assert fit.cliff_lap is not None, "expected a cliff to be detected"
    assert 7 <= fit.cliff_lap <= 13, f"cliff at age {fit.cliff_lap}, expected ~10"
    # And the cliff factor should be substantial (post-slope much higher than pre).
    assert fit.cliff_factor > 3.0, f"cliff_factor {fit.cliff_factor} should reflect ~8x slope shift"


def test_tire_degradation_predict_pace_at_age():
    from sports.f1.ml.features.tire_degradation import DegradationFit, predict_pace_at_age
    fit = DegradationFit(
        driver_code="X", compound="M", track_code="T",
        base_pace_s=80.0, deg_per_lap_s=0.05, cliff_lap=10, cliff_factor=4.0, n_observations=20,
    )
    # Pre-cliff: linear at base + slope * age.
    assert predict_pace_at_age(fit, 5) == pytest.approx(80.25, abs=1e-6)
    assert predict_pace_at_age(fit, 10) == pytest.approx(80.50, abs=1e-6)
    # Post-cliff: pre_cliff_pace + 0.05 * 4 * (age - 10).
    assert predict_pace_at_age(fit, 15) == pytest.approx(80.50 + 0.20 * 5, abs=1e-6)


# ----------------------------------------------------------------- dirty_air

def test_dirty_air_recovers_known_penalty():
    """Synthesize laps with clean baseline + known penalty in dirty air; fit recovers it."""
    from sports.f1.ml.features.dirty_air import apply_penalty, fit

    rng = np.random.default_rng(0)
    rows = []
    # Clean-air laps for several driver/compound/age buckets.
    for drv in ["A", "B", "C"]:
        for age_bucket_start in [0, 3, 6]:
            for offset in range(8):
                rows.append({
                    "driver_code": drv,
                    "compound": "MEDIUM",
                    "tire_age_laps": age_bucket_start + (offset % 3),
                    "lap_time_s": 80.0 + rng.normal(0, 0.03),
                    "gap_ahead_s": 5.0,  # clear air
                    "track_code": "T",
                })
    # Dirty-air laps with known penalty: 1.0 s/s of closeness.
    true_slope = 1.0
    for drv in ["A", "B", "C"]:
        for age_bucket_start in [0, 3]:
            for gap in [0.3, 0.6, 0.9, 1.2]:
                penalty = (1.5 - gap) * true_slope
                rows.append({
                    "driver_code": drv,
                    "compound": "MEDIUM",
                    "tire_age_laps": age_bucket_start,
                    "lap_time_s": 80.0 + penalty + rng.normal(0, 0.05),
                    "gap_ahead_s": gap,
                    "track_code": "T",
                })
    laps = pd.DataFrame(rows)
    model = fit(laps, gap_threshold_s=1.5)
    # Recovered slope should be close to the truth.
    assert model.penalty_per_s_close == pytest.approx(true_slope, abs=0.3), (
        f"expected ~{true_slope} s/s, got {model.penalty_per_s_close}"
    )
    # apply_penalty monotonicity: closer ahead → bigger penalty.
    assert apply_penalty(model, 0.3) > apply_penalty(model, 1.0) > apply_penalty(model, 1.5)
    # Outside threshold → zero penalty.
    assert apply_penalty(model, 2.0) == 0.0


# ------------------------------------------------------------ hierarchical_bayes

@pytest.mark.slow
def test_hierarchical_bayes_recovers_team_ranking():
    """With a small synthetic dataset of known team / driver effects, the
    posterior should rank teams correctly. Uses few draws / tune for speed."""
    from sports.f1.ml.ratings.hierarchical_bayes import HierarchicalBayesStrength

    rng = np.random.default_rng(7)
    # 4 teams, 2 drivers each, 8 races. Team effects spread by 0.5 s.
    team_effects = {"T0": 0.0, "T1": 0.15, "T2": 0.30, "T3": 0.45}
    driver_effects = {f"D{i}": (i % 2 - 0.5) * 0.1 for i in range(8)}
    drivers_to_teams = {f"D{i}": f"T{i // 2}" for i in range(8)}

    rows = []
    for race in range(8):
        for drv, team in drivers_to_teams.items():
            base = 90.0  # era baseline
            obs = base + team_effects[team] + driver_effects[drv] + rng.normal(0, 0.05)
            rows.append({"season": 2024, "race": race, "driver_code": drv, "team_code": team, "mean_pace_s": obs})
    df = pd.DataFrame(rows)

    model = HierarchicalBayesStrength(n_samples=200)
    model.fit(df, n_tune=200, target_accept=0.95)

    summary = model.driver_strength_summary()
    # Driver effects should approximately recover the truth (loose tolerance for small sample).
    for _, row in summary.iterrows():
        truth = driver_effects[row["driver_code"]]
        assert abs(row["driver_effect_mean"] - truth) < 0.15, (
            f"driver {row['driver_code']}: posterior mean {row['driver_effect_mean']} vs truth {truth}"
        )

    # sample_pace should produce values near the team+driver+era expectation.
    samples = model.sample_pace("D0", "T0", n=500)
    assert 89.5 < samples.mean() < 90.5, f"sampled mean {samples.mean()} outside expected range"


# --------------------------------------------------------------- fastf1_provider (no network)

def test_fastf1_compound_normalization():
    from sports.f1.ml.providers.fastf1_provider import _normalize_compound
    from sports.f1.ml.common.types import TireCompound

    assert _normalize_compound("SOFT") == TireCompound.SOFT
    assert _normalize_compound("medium") == TireCompound.MEDIUM
    assert _normalize_compound("Hard") == TireCompound.HARD
    assert _normalize_compound("INTERMEDIATE") == TireCompound.INTERMEDIATE
    assert _normalize_compound("WET") == TireCompound.WET
    # Unknown / missing → default MEDIUM (don't crash on no-tire-data laps).
    assert _normalize_compound("UNKNOWN") == TireCompound.MEDIUM
    assert _normalize_compound(None) == TireCompound.MEDIUM


def test_fastf1_race_id_format():
    from sports.f1.ml.providers.fastf1_provider import _race_id
    assert _race_id(2024, 14, "MONZA") == "2024-14-MONZA"
    assert _race_id(2026, 1, "BAHRAIN") == "2026-01-BAHRAIN"


def test_fastf1_seconds_or_zero_handles_missing():
    from sports.f1.ml.providers.fastf1_provider import _seconds_or_zero
    assert _seconds_or_zero(None) == 0.0
    assert _seconds_or_zero(5.5) == 5.5
    # Object that exposes total_seconds (e.g., timedelta).
    import datetime
    assert _seconds_or_zero(datetime.timedelta(seconds=12.3)) == 12.3


@pytest.mark.network
def test_fastf1_list_races_real():
    """Hits the real FastF1 API. Skipped by default; opt-in with `pytest -m network`."""
    import tempfile
    from sports.f1.ml.providers.fastf1_provider import FastF1Provider

    with tempfile.TemporaryDirectory() as tmp:
        provider = FastF1Provider(cache_dir=tmp)
        races = provider.list_races(2024)
        assert len(races) >= 20  # a full season
        assert all(r.season == 2024 for r in races)
