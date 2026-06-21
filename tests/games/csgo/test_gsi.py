"""Tests for CS2 GSI → LiveMatchState mapping and the ingest route."""

import asyncio

from common.ml.bridge.outcome_publisher import InMemoryOutcomePublisher
from games.csgo.analytics.pipeline import CsgoModelPipeline
from games.csgo.api import csgo_routes
from games.csgo.data.csgo_client import StubCsgoClient
from games.csgo.gsi import gsi_to_live_state

_PAYLOAD = {
    "provider": {"name": "Counter-Strike: Global Offensive", "timestamp": 1700000000},
    "map": {
        "mode": "competitive", "name": "de_mirage", "phase": "live", "round": 15,
        "num_matches_to_win_series": 2,
        "team_ct": {"name": "Natus Vincere", "score": 9, "matches_won_this_series": 1},
        "team_t": {"name": "FaZe Clan", "score": 6, "matches_won_this_series": 0},
    },
    "round": {"phase": "live", "bomb": "planted"},
    "allplayers": {
        "1": {"team": "CT", "state": {"health": 100, "equip_value": 4000}},
        "2": {"team": "CT", "state": {"health": 0, "equip_value": 3000}},
        "3": {"team": "CT", "state": {"health": 50, "equip_value": 2000}},
        "4": {"team": "T", "state": {"health": 100, "equip_value": 4500}},
        "5": {"team": "T", "state": {"health": 100, "equip_value": 4500}},
    },
}


def test_gsi_maps_state_with_ct_team1():
    state, n1, n2 = gsi_to_live_state(_PAYLOAD, team1_name="Natus Vincere")
    assert (n1, n2) == ("Natus Vincere", "FaZe Clan")
    assert state.maps_won_team1 == 1 and state.maps_won_team2 == 0
    assert state.team1_rounds == 9 and state.team2_rounds == 6
    assert state.team1_side == "CT"
    assert state.bomb_planted and state.bomb_planter_team == 2   # T side (team2) planted
    assert state.players_alive_team1 == 2 and state.players_alive_team2 == 2
    assert state.best_of == 3 and state.current_map == "de_mirage"


def test_gsi_resolves_team1_on_t_side():
    state, n1, n2 = gsi_to_live_state(_PAYLOAD, team1_name="FaZe Clan")
    assert (n1, n2) == ("FaZe Clan", "Natus Vincere")
    assert state.team1_side == "T"
    assert state.team1_rounds == 6 and state.team2_rounds == 9
    assert state.bomb_planter_team == 1   # team1 is now the T side


def test_gsi_route_ingests_and_publishes():
    client = StubCsgoClient()                       # has an upcoming NAVI vs FaZe match
    csgo_routes.init(client, CsgoModelPipeline())
    pub = InMemoryOutcomePublisher()
    csgo_routes.set_publisher(pub)
    try:
        res = asyncio.run(csgo_routes.gsi(_PAYLOAD))
        assert res["ok"] is True
        assert 0.0 <= res["live"]["team1_win_prob"] <= 1.0
        assert pub.channel_messages
        assert all(ch.startswith("csgo:prob:") for ch, _ in pub.channel_messages)

        latest = asyncio.run(csgo_routes.gsi_live())
        assert latest["ok"] is True and latest["match_id"] == res["match_id"]
    finally:
        csgo_routes.set_publisher(None)
