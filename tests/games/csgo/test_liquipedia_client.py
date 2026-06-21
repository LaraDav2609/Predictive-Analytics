"""Liquipedia CS2 client tests — offline via httpx.MockTransport (no network/key)."""

import asyncio

import httpx

from games.csgo.analytics.ratings import rate_matches
from games.csgo.data.csgo_client import StubCsgoClient
from games.csgo.data.factory import build_csgo_client
from games.csgo.data.liquipedia_client import LiquipediaCsgoClient

_PAST = [
    {"match2id": "m1", "date": "2026-01-10 17:00:00", "finished": "1", "bestof": 3, "winner": "1",
     "tournament": "IEM Katowice 2026",
     "match2opponents": [{"name": "Natus Vincere", "score": 2}, {"name": "FaZe Clan", "score": 1}],
     "match2games": [{"map": "Mirage", "winner": 1}, {"map": "Inferno", "winner": 2}, {"map": "Nuke", "winner": 1}]},
    {"match2id": "m2", "date": "2026-01-11 17:00:00", "finished": "1", "bestof": 3, "winner": "2",
     "tournament": "IEM Katowice 2026",
     "match2opponents": [{"name": "G2 Esports", "score": 0}, {"name": "Team Vitality", "score": 2}],
     "match2games": [{"map": "Ancient", "winner": 2}, {"map": "Anubis", "winner": 2}]},
]
_UPCOMING = [
    {"match2id": "u1", "date": "2026-07-01 17:00:00", "finished": "0", "bestof": 3,
     "tournament": "IEM Cologne 2026",
     "match2opponents": [{"name": "Natus Vincere"}, {"name": "Team Vitality"}]},
]


def _handler(request: httpx.Request) -> httpx.Response:
    cond = request.url.params.get("conditions", "")
    return httpx.Response(200, json={"result": _UPCOMING if "finished::false" in cond else _PAST})


def _client() -> LiquipediaCsgoClient:
    mock = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    return LiquipediaCsgoClient("test-key", user_agent="test/1.0 (me@example.com)", client=mock, rate_limit_s=0.0)


def test_refresh_maps_results_maps_and_upcoming():
    c = _client()
    asyncio.run(c.refresh())
    past = c.get_past_matches()
    assert len(past) == 2

    m = next(p for p in past if p.id == "m1")
    assert (m.team1, m.team2) == ("Natus Vincere", "FaZe Clan")
    assert m.team1_score == 2 and m.team2_score == 1
    assert m.winner_id == m.team1_id
    assert m.best_of == 3 and m.event == "IEM Katowice 2026"
    assert [ms.map_name for ms in m.map_scores] == ["Mirage", "Inferno", "Nuke"]
    assert m.map_scores[0].winner_id == m.team1_id

    up = c.get_matches()
    assert len(up) == 1 and up[0].status == "SCHEDULED"
    assert {t.name for t in c.get_teams()} >= {"Natus Vincere", "FaZe Clan", "G2 Esports", "Team Vitality"}


def test_winner_index_two_maps_to_team2():
    c = _client()
    asyncio.run(c.refresh())
    m2 = next(p for p in c.get_past_matches() if p.id == "m2")
    assert m2.winner_id == m2.team2_id


def test_ratings_run_on_liquipedia_data():
    c = _client()
    asyncio.run(c.refresh())
    assert rate_matches(c.get_past_matches())


def test_factory_requires_api_key(monkeypatch):
    monkeypatch.setenv("CSGO_DATA_PROVIDER", "liquipedia")
    monkeypatch.delenv("LIQUIPEDIA_API_KEY", raising=False)
    assert isinstance(build_csgo_client(), StubCsgoClient)

    monkeypatch.setenv("LIQUIPEDIA_API_KEY", "abc123")
    client = build_csgo_client()
    assert isinstance(client, LiquipediaCsgoClient)
    asyncio.run(client.close())
