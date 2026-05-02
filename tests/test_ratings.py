"""Tests for the simple rating models in f1_ml.ratings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------- Bradley-Terry

def test_bradley_terry_recovers_dominant_driver():
    """Driver A wins every head-to-head against B → skill_A > skill_B."""
    from f1_ml.ratings.bradley_terry import fit, predict_h2h

    rows = []
    for race in range(20):
        rows.append({"race": race, "driver_code": "A", "position": 1})
        rows.append({"race": race, "driver_code": "B", "position": 2})
    df = pd.DataFrame(rows)
    f = fit(df)
    a_idx = f.driver_codes.index("A")
    b_idx = f.driver_codes.index("B")
    assert f.skills[a_idx] > f.skills[b_idx]
    assert predict_h2h(f, "A", "B") > 0.95
    assert predict_h2h(f, "B", "A") < 0.05


def test_bradley_terry_h2h_probabilities_sum_to_one():
    from f1_ml.ratings.bradley_terry import fit, predict_h2h

    rows = []
    for race in range(10):
        for pos, drv in enumerate(["A", "B", "C"], start=1):
            rows.append({"race": race, "driver_code": drv, "position": pos})
    f = fit(pd.DataFrame(rows))
    p_ab = predict_h2h(f, "A", "B")
    p_ba = predict_h2h(f, "B", "A")
    assert p_ab + p_ba == pytest.approx(1.0, abs=1e-6)


def test_bradley_terry_orders_three_drivers_correctly():
    from f1_ml.ratings.bradley_terry import fit

    # Strict total order: A > B > C in every race.
    rows = []
    for race in range(15):
        for pos, drv in enumerate(["A", "B", "C"], start=1):
            rows.append({"race": race, "driver_code": drv, "position": pos})
    f = fit(pd.DataFrame(rows))
    skills = dict(zip(f.driver_codes, f.skills))
    assert skills["A"] > skills["B"] > skills["C"]


def test_bradley_terry_unknown_driver_raises():
    from f1_ml.ratings.bradley_terry import fit, predict_h2h

    rows = [{"race": 0, "driver_code": "A", "position": 1},
            {"race": 0, "driver_code": "B", "position": 2}]
    f = fit(pd.DataFrame(rows))
    with pytest.raises(KeyError):
        predict_h2h(f, "A", "Z")


def test_bradley_terry_handles_empty_input():
    from f1_ml.ratings.bradley_terry import fit
    f = fit(pd.DataFrame(columns=["race", "driver_code", "position"]))
    assert f.driver_codes == []
    assert len(f.skills) == 0


# ---------------------------------------------------------------- Plackett-Luce

def test_plackett_luce_recovers_strict_ordering():
    from f1_ml.ratings.plackett_luce import fit, race_win_probabilities

    rows = []
    for race in range(20):
        for pos, drv in enumerate(["A", "B", "C", "D"], start=1):
            rows.append({"race": race, "driver_code": drv, "position": pos})
    f = fit(pd.DataFrame(rows))
    skills = dict(zip(f.driver_codes, f.skills))
    assert skills["A"] > skills["B"] > skills["C"] > skills["D"]
    probs = race_win_probabilities(f, ["A", "B", "C", "D"])
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
    assert probs["A"] > 0.5


def test_plackett_luce_log_likelihood_finite():
    from f1_ml.ratings.plackett_luce import fit

    rng = np.random.default_rng(0)
    rows = []
    drivers = ["VER", "HAM", "LEC", "NOR", "PIA"]
    for race in range(10):
        order = rng.permutation(drivers).tolist()
        for pos, drv in enumerate(order, start=1):
            rows.append({"race": race, "driver_code": drv, "position": pos})
    f = fit(pd.DataFrame(rows), max_iter=100)
    assert np.isfinite(f.log_likelihood)


def test_plackett_luce_race_win_probabilities_drops_unknown():
    from f1_ml.ratings.plackett_luce import fit, race_win_probabilities

    rows = []
    for race in range(5):
        for pos, drv in enumerate(["A", "B"], start=1):
            rows.append({"race": race, "driver_code": drv, "position": pos})
    f = fit(pd.DataFrame(rows))
    probs = race_win_probabilities(f, ["A", "B", "ROOKIE"])
    assert "ROOKIE" not in probs
    assert sum(probs.values()) == pytest.approx(1.0)


def test_plackett_luce_handles_empty_input():
    from f1_ml.ratings.plackett_luce import fit
    f = fit(pd.DataFrame(columns=["race", "driver_code", "position"]))
    assert f.driver_codes == []
    assert f.log_likelihood == 0.0


def test_plackett_luce_skills_are_centered():
    """Centering for interpretability: log-skills should average to ~0."""
    from f1_ml.ratings.plackett_luce import fit

    rows = []
    for race in range(10):
        for pos, drv in enumerate(["A", "B", "C", "D", "E"], start=1):
            rows.append({"race": race, "driver_code": drv, "position": pos})
    f = fit(pd.DataFrame(rows))
    assert abs(float(f.skills.mean())) < 1e-6
