"""Tests that the live /matches path uses the full ensemble pipeline."""

import asyncio

from games.csgo.analytics.pipeline import CsgoModelPipeline
from games.csgo.api import csgo_routes
from games.csgo.data.csgo_client import StubCsgoClient


def _fitted_pipeline(client):
    pipe = CsgoModelPipeline()
    pipe.fit(client.get_past_matches(), client.get_teams())
    return pipe


def test_pipeline_predicts_with_ensemble_outputs():
    client = StubCsgoClient()
    matches = _fitted_pipeline(client).predict_matches(client.get_matches())
    preds = [m.prediction for m in matches if m.prediction]
    assert preds
    p = preds[0]
    assert p.model_version == "csgo-ensemble-v1"
    assert abs(p.team1_win_prob + p.team2_win_prob - 1.0) < 1e-6
    assert p.over_2_5_maps_prob is not None        # stub matches are bo3
    assert 0.0 < p.confidence <= 1.0


def test_matches_route_uses_pipeline():
    client = StubCsgoClient()
    csgo_routes.init(client, _fitted_pipeline(client))
    res = asyncio.run(csgo_routes.get_matches())
    assert res["ok"] is True
    assert res["matches"][0]["prediction"]["model_version"] == "csgo-ensemble-v1"


def test_refresh_route_fits_pipeline():
    client = StubCsgoClient()
    csgo_routes.init(client, CsgoModelPipeline())
    res = asyncio.run(csgo_routes.refresh())
    assert res["ok"] is True
    # after fit, matches carry ensemble predictions
    matches = asyncio.run(csgo_routes.get_matches())
    assert matches["matches"][0]["prediction"]["model_version"] == "csgo-ensemble-v1"
