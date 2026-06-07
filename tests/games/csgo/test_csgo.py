"""CSGO scaffold tests — baseline predictor, shared-core reuse, and API routes.

These also serve as the reuse proof: the game-domain predictor pulls edge/kelly
from common.ml and serializes to the shared OutcomeProbability + Redis bridge.
"""

import asyncio

from common.ml.bridge.outcome_publisher import InMemoryOutcomePublisher
from common.ml.markets.edge import EdgeOpportunity
from common.ml.markets.kelly import KellySize
from common.ml.types import OutcomeProbability
from games.csgo.analytics.csgo_predictor import CsgoPredictor
from games.csgo.api import csgo_routes
from games.csgo.data.csgo_client import StubCsgoClient


def _loaded():
    client = StubCsgoClient()
    pred = CsgoPredictor()
    pred.load_teams(client.get_teams())
    return client, pred


def test_stub_client_has_sample_data():
    client = StubCsgoClient()
    assert len(client.get_teams()) >= 4
    matches = client.get_matches()
    assert len(matches) >= 1
    assert client.get_match(matches[0].id).id == matches[0].id
    assert client.get_match("does-not-exist") is None


def test_higher_rated_team_is_favored():
    client, pred = _loaded()
    match = next(m for m in client.get_matches()
                 if m.team1_abbrev == "NAVI" and m.team2_abbrev == "FAZE")
    p = pred.predict(match)
    assert p.team1_win_prob > p.team2_win_prob
    assert abs(p.team1_win_prob + p.team2_win_prob - 1.0) < 1e-6


def test_best_of_amplifies_favorite():
    pred = CsgoPredictor()
    assert pred._best_of_win_prob(0.6, 1) == 0.6
    assert pred._best_of_win_prob(0.6, 3) > 0.6  # a bo3 favors the stronger side more


def test_predict_matches_sets_predictions():
    client, pred = _loaded()
    matches = pred.predict_matches(client.get_matches())
    assert all(m.prediction is not None for m in matches)


def test_outcome_probabilities_use_shared_contract():
    client, pred = _loaded()
    match = client.get_matches()[0]
    probs = pred.to_outcome_probabilities(match)
    assert len(probs) == 2
    assert all(isinstance(p, OutcomeProbability) for p in probs)
    assert all(p.domain == "csgo" and p.market == "winner" for p in probs)
    assert all(p.entity_id == match.id for p in probs)


def test_publish_uses_shared_bridge_with_csgo_namespace():
    client, pred = _loaded()
    pub = InMemoryOutcomePublisher()
    n = pred.publish(client.get_matches(), pub)
    assert n == 2 * len(client.get_matches())
    assert pub.channel_messages
    assert all(ch.startswith("csgo:prob:") for ch, _ in pub.channel_messages)
    assert all(key.startswith("csgo:snapshot:") for key in pub.snapshots)


def test_edge_and_stake_reuse_shared_core():
    edge, stake = CsgoPredictor.edge_and_stake(0.65, 0.50, bankroll_usd=1000.0)
    assert isinstance(stake, KellySize)
    assert isinstance(edge, EdgeOpportunity)
    assert edge.direction == "YES"
    assert stake.notional_usd > 0


def test_routes_return_teams_and_predicted_matches():
    client, pred = _loaded()
    csgo_routes.init(client, pred)
    matches = asyncio.run(csgo_routes.get_matches())
    assert matches["ok"] is True
    assert len(matches["matches"]) >= 1
    assert matches["matches"][0]["prediction"] is not None
    teams = asyncio.run(csgo_routes.get_teams())
    assert teams["ok"] is True and len(teams["teams"]) >= 4
