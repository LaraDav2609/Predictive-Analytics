"""Tests for the CS2 calibrated ensemble model."""

from dataclasses import replace

import pytest

from games.csgo.analytics.model import CsgoEnsembleModel
from games.csgo.features import MatchFeatures, TeamFeatures


def _mf(best_of=3, t1_over=None, t2_over=None):
    t1 = TeamFeatures(team_id=1, rating=1700, rating_deviation=60, recent_form=0.65,
                      map_strength={"Mirage": 0.7, "Inferno": 0.6}, roster_stability=1.0)
    t2 = TeamFeatures(team_id=2, rating=1500, rating_deviation=60, recent_form=0.45,
                      map_strength={"Mirage": 0.5, "Inferno": 0.5}, roster_stability=1.0)
    if t1_over:
        t1 = replace(t1, **t1_over)
    if t2_over:
        t2 = replace(t2, **t2_over)
    return MatchFeatures(
        match_id="m", team1=t1, team2=t2,
        rating_diff=t1.rating - t2.rating, form_diff=t1.recent_form - t2.recent_form,
        best_of=best_of, likely_maps=["Mirage", "Inferno"], provenance={"ratings": "glicko2"},
    )


def _swap(mf: MatchFeatures) -> MatchFeatures:
    return MatchFeatures(
        match_id="m", team1=mf.team2, team2=mf.team1,
        rating_diff=-mf.rating_diff, form_diff=-mf.form_diff, best_of=mf.best_of,
        likely_maps=mf.likely_maps, provenance=mf.provenance,
    )


def test_favorite_has_series_and_map_edge():
    out = CsgoEnsembleModel().predict(_mf())
    assert out.winner_prob > 0.5
    assert out.map1_team1_prob > 0.5
    assert out.model_version == "csgo-ensemble-v1"


def test_stand_in_reduces_favorite_probability():
    m = CsgoEnsembleModel()
    base = m.predict(_mf()).winner_prob
    with_standin = m.predict(_mf(t1_over={"roster_stability": 0.8, "stand_in_count": 1})).winner_prob
    assert with_standin < base


def test_over_2_5_maps_only_for_bo3():
    m = CsgoEnsembleModel()
    bo3 = m.predict(_mf(best_of=3))
    assert bo3.over_2_5_maps_prob is not None
    assert 0.0 < bo3.over_2_5_maps_prob <= 0.5
    assert m.predict(_mf(best_of=1)).over_2_5_maps_prob is None


def test_feature_vector_exposes_training_surface():
    v = CsgoEnsembleModel().feature_vector(_mf())
    assert {"base_per_map", "form_diff", "map_adv", "roster_diff", "h2h_centered", "best_of"} <= set(v)


def test_swapping_teams_inverts_map_probability():
    m = CsgoEnsembleModel()
    mf = _mf()
    p = m.predict(mf).map1_team1_prob
    p_swapped = m.predict(_swap(mf)).map1_team1_prob
    assert p + p_swapped == pytest.approx(1.0, abs=1e-6)
