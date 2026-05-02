"""Tests for f1_ml.events — survival models + Poisson SC + pit-time + overtake."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_ml.events import dnf_cox, dnf_weibull, pit_time, safety_car_poisson
from f1_ml.events.overtake import OvertakeFeatures, OvertakeModel


# -------------------------------------------------------------------- weibull

def _make_dnf_stints(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Stint-level data: longer stints + lower mech_concern → fewer DNFs.
    Failure time scales as exp(-0.5 * mech_concern)."""
    rng = np.random.default_rng(seed)
    mech_concern = rng.uniform(0, 2, size=n)
    engine_age = rng.integers(0, 8, size=n)

    # Latent failure time: Weibull with shape 1.5, scale modulated by covariates.
    scale = 60.0 * np.exp(-0.5 * mech_concern - 0.05 * engine_age)
    u = rng.uniform(0, 1, size=n)
    fail_lap = scale * (-np.log(1 - u)) ** (1.0 / 1.5)
    race_length = 60
    laps_completed = np.minimum(fail_lap, race_length).astype(int) + 1
    dnf = (fail_lap < race_length).astype(int)

    return pd.DataFrame({
        "laps_completed": laps_completed,
        "dnf": dnf,
        "mech_concern": mech_concern,
        "engine_age": engine_age,
    })


def test_weibull_aft_fits_and_returns_positive_hazard():
    df = _make_dnf_stints(n=400)
    fit = dnf_weibull.fit(df)
    assert fit.scale_lambda > 0
    assert fit.shape_k > 0
    h = dnf_weibull.hazard_per_lap(fit, lap=10, covariates={"mech_concern": 1.0, "engine_age": 3})
    assert 0.0 <= h <= 1.0


def test_weibull_aft_higher_concern_means_higher_hazard():
    df = _make_dnf_stints(n=500)
    fit = dnf_weibull.fit(df)
    low = dnf_weibull.hazard_per_lap(fit, lap=20, covariates={"mech_concern": 0.1, "engine_age": 1})
    high = dnf_weibull.hazard_per_lap(fit, lap=20, covariates={"mech_concern": 1.8, "engine_age": 6})
    assert high > low


def test_weibull_aft_rejects_empty_input():
    with pytest.raises(ValueError):
        dnf_weibull.fit(pd.DataFrame(columns=["laps_completed", "dnf"]))


def test_weibull_aft_hazard_at_lap_zero_is_zero():
    df = _make_dnf_stints(n=200)
    fit = dnf_weibull.fit(df)
    assert dnf_weibull.hazard_per_lap(fit, lap=0, covariates={"mech_concern": 1.0, "engine_age": 1}) == 0.0


# ------------------------------------------------------------------- cox

def test_cox_ph_fits_and_returns_positive_hazard():
    df = _make_dnf_stints(n=400)
    fit = dnf_cox.fit(df)
    assert "mech_concern" in fit.coef
    h = dnf_cox.hazard_per_lap(fit, lap=15, covariates={"mech_concern": 1.0, "engine_age": 3})
    assert 0.0 <= h <= 1.0


def test_cox_ph_concern_coefficient_positive():
    """Higher mech_concern → higher hazard ⇒ positive Cox coefficient."""
    df = _make_dnf_stints(n=600)
    fit = dnf_cox.fit(df)
    assert fit.coef["mech_concern"] > 0


def test_cox_ph_baseline_cumulative_hazard_is_monotone():
    df = _make_dnf_stints(n=300)
    fit = dnf_cox.fit(df)
    bch = fit.baseline_cumulative_hazard.values
    diffs = np.diff(bch)
    # Monotone non-decreasing (Breslow estimator).
    assert (diffs >= -1e-9).all()


# ---------------------------------------------------------- safety car poisson

def _make_sc_history() -> pd.DataFrame:
    """3 tracks with very different SC rates."""
    rows = []
    # Monaco: SC every other race
    for r in range(10):
        rows.append({"track_code": "MON", "race_laps": 78,
                     "sc_events": 1 if r % 2 == 0 else 0,
                     "sc_events_rain": 0, "rain_laps": 0,
                     "sc_events_late": 1 if r % 2 == 0 else 0, "late_laps": 20,
                     "avg_sc_duration_laps": 5})
    # Spa: SC rare
    for r in range(10):
        rows.append({"track_code": "SPA", "race_laps": 44,
                     "sc_events": 1 if r == 0 else 0,
                     "sc_events_rain": 0, "rain_laps": 0,
                     "sc_events_late": 0, "late_laps": 11,
                     "avg_sc_duration_laps": 3})
    # Silverstone: in between
    for r in range(10):
        rows.append({"track_code": "SIL", "race_laps": 52,
                     "sc_events": 1 if r % 4 == 0 else 0,
                     "sc_events_rain": 0, "rain_laps": 0,
                     "sc_events_late": 0, "late_laps": 13,
                     "avg_sc_duration_laps": 4})
    return pd.DataFrame(rows)


def test_safety_car_rates_are_track_specific():
    rates = safety_car_poisson.fit(_make_sc_history())
    assert "MON" in rates and "SPA" in rates
    # Monaco should be the highest base rate.
    assert rates["MON"].base_rate_per_lap > rates["SPA"].base_rate_per_lap


def test_safety_car_sample_event_returns_bool():
    rates = safety_car_poisson.fit(_make_sc_history())
    rng = np.random.default_rng(0)
    out = safety_car_poisson.sample_event(rates["MON"], lap=10, total_laps=78, rain_intensity=0.0, rng=rng)
    assert isinstance(out, bool)


def test_safety_car_expected_events_increases_with_rate():
    rates = safety_car_poisson.fit(_make_sc_history())
    e_mon = safety_car_poisson.expected_events(rates["MON"], total_laps=78)
    e_spa = safety_car_poisson.expected_events(rates["SPA"], total_laps=44)
    assert e_mon > e_spa


def test_safety_car_handles_empty_input():
    rates = safety_car_poisson.fit(pd.DataFrame(columns=["track_code", "race_laps", "sc_events"]))
    assert rates == {}


def test_safety_car_late_race_inflation():
    """At a track where late events outnumber early, the late multiplier > 1."""
    rng = np.random.default_rng(0)
    rows = []
    for r in range(20):
        rows.append({"track_code": "X", "race_laps": 60, "sc_events": 1,
                     "sc_events_rain": 0, "rain_laps": 0,
                     "sc_events_late": 1, "late_laps": 15,  # all SC events late
                     "avg_sc_duration_laps": 4})
    rates = safety_car_poisson.fit(pd.DataFrame(rows))
    assert rates["X"].late_race_multiplier > 1.0


# ------------------------------------------------------------------ pit time

def _make_pit_history(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    # Red Bull: tight 2.3 ± 0.2, 2% botches at +5s
    for _ in range(80):
        if rng.uniform() < 0.02:
            t = float(rng.normal(2.3, 0.2)) + 5.0
        else:
            t = float(rng.normal(2.3, 0.2))
        rows.append({"team_code": "RBR", "stationary_time_s": max(t, 1.5)})
    # Williams: looser 3.0 ± 0.4, 10% botches at +4s
    for _ in range(80):
        if rng.uniform() < 0.10:
            t = float(rng.normal(3.0, 0.4)) + 4.0
        else:
            t = float(rng.normal(3.0, 0.4))
        rows.append({"team_code": "WIL", "stationary_time_s": max(t, 2.0)})
    return pd.DataFrame(rows)


def test_pit_time_fit_recovers_team_ordering():
    dists = pit_time.fit(_make_pit_history())
    assert dists["RBR"].mean_clean_s < dists["WIL"].mean_clean_s


def test_pit_time_sample_returns_finite_positive_seconds():
    dists = pit_time.fit(_make_pit_history())
    rng = np.random.default_rng(0)
    samples = [pit_time.sample(dists["RBR"], rng=rng) for _ in range(100)]
    assert all(np.isfinite(samples))
    assert np.mean(samples) > 1.5  # plausible pit-time floor


def test_pit_time_botch_probability_in_unit_interval():
    dists = pit_time.fit(_make_pit_history())
    for d in dists.values():
        assert 0.0 <= d.botch_probability <= 1.0


def test_pit_time_handles_empty_input():
    dists = pit_time.fit(pd.DataFrame(columns=["team_code", "stationary_time_s"]))
    assert dists == {}


def test_pit_time_low_sample_team_uses_defaults():
    """Teams with fewer than min_observations get the global default."""
    df = pd.DataFrame([{"team_code": "ALP", "stationary_time_s": 2.5}])
    dists = pit_time.fit(df, min_observations=5)
    assert dists["ALP"].mean_clean_s == 2.5  # default
    assert dists["ALP"].botch_probability == 0.05


def test_pit_time_expected_loss_includes_botch():
    dists = pit_time.fit(_make_pit_history())
    williams = dists["WIL"]
    e_loss = pit_time.expected_loss(williams)
    assert e_loss > williams.mean_clean_s  # botch adds to expectation


# ------------------------------------------------------------------ overtake

def _make_overtake_dataset(n: int = 600, seed: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "gap_ahead_s": rng.uniform(0.1, 2.0, size=n),
        "pace_delta_s": rng.uniform(-0.5, 1.5, size=n),
        "drs_available": rng.integers(0, 2, size=n),
        "tire_age_delta_laps": rng.integers(-15, 16, size=n),
        "compound_advantage": rng.integers(-1, 2, size=n),
        "track_overtake_index": rng.uniform(0.05, 0.7, size=n),
    })
    score = (X["pace_delta_s"]
             + 0.3 * X["drs_available"]
             - 0.5 * X["gap_ahead_s"]
             + 0.5 * X["track_overtake_index"])
    y = (rng.uniform(0, 1, size=n) < 1.0 / (1.0 + np.exp(-score * 3))).astype(int).to_numpy()
    return X, y


def test_overtake_model_fits_and_predicts_with_features_object():
    X, y = _make_overtake_dataset(n=600)
    model = OvertakeModel(n_estimators=200).fit(X, y)
    feats = OvertakeFeatures(
        gap_ahead_s=0.3, pace_delta_s=1.2, drs_available=True,
        tire_age_delta_laps=-5, compound_advantage=1, track_overtake_index=0.6,
    )
    p = model.predict_proba(feats)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_overtake_model_higher_pace_means_higher_proba():
    X, y = _make_overtake_dataset(n=800)
    model = OvertakeModel(n_estimators=200).fit(X, y)
    fast = OvertakeFeatures(
        gap_ahead_s=0.3, pace_delta_s=1.2, drs_available=True,
        tire_age_delta_laps=-5, compound_advantage=1, track_overtake_index=0.6,
    )
    slow = OvertakeFeatures(
        gap_ahead_s=0.3, pace_delta_s=-0.4, drs_available=False,
        tire_age_delta_laps=2, compound_advantage=0, track_overtake_index=0.2,
    )
    assert model.predict_proba(fast) > model.predict_proba(slow)


def test_overtake_model_predict_without_fit_raises():
    feats = OvertakeFeatures(
        gap_ahead_s=0.5, pace_delta_s=0.0, drs_available=False,
        tire_age_delta_laps=0, compound_advantage=0, track_overtake_index=0.3,
    )
    with pytest.raises(RuntimeError, match="not fitted"):
        OvertakeModel().predict_proba(feats)


def test_overtake_model_predict_dataframe_returns_array():
    X, y = _make_overtake_dataset(n=400)
    model = OvertakeModel(n_estimators=100).fit(X, y)
    probs = model.predict_proba(X.iloc[:20])
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (20,)
    assert ((probs >= 0) & (probs <= 1)).all()
