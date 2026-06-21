"""Tests for CS2 team/player detail endpoints."""

import asyncio

from games.csgo.analytics.pipeline import CsgoModelPipeline
from games.csgo.api import csgo_routes
from games.csgo.data.csgo_client import StubCsgoClient


def test_stub_team_has_five_players():
    client = StubCsgoClient()
    team = client.get_teams()[0]
    players = client.get_players(team.id)
    assert len(players) == 5
    assert all(p.team_id == team.id for p in players)
    assert client.get_player(players[0].id).id == players[0].id
    assert client.get_player(999_999) is None


def test_team_roster_matches_players():
    client = StubCsgoClient()
    team = client.get_teams()[0]
    assert set(team.roster) == {p.id for p in client.get_players(team.id)}


def test_team_and_player_routes():
    client = StubCsgoClient()
    csgo_routes.init(client, CsgoModelPipeline())
    team = client.get_teams()[0]

    team_res = asyncio.run(csgo_routes.get_team(team.id))
    assert team_res["ok"] and team_res["team"]["id"] == team.id
    assert len(team_res["players"]) == 5

    players_res = asyncio.run(csgo_routes.get_team_players(team.id))
    assert len(players_res["players"]) == 5

    pid = team_res["players"][0]["id"]
    player_res = asyncio.run(csgo_routes.get_player(pid))
    assert player_res["ok"] and player_res["player"]["id"] == pid
