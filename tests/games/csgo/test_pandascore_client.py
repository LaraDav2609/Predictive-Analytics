"""PandaScore CSGO client tests — offline via httpx.MockTransport.

Verifies team/match mapping, provisional Elo from recent results, and the
env-driven provider factory. No network (safe under the default test profile).
"""

import asyncio

import httpx

from games.csgo.data.csgo_client import StubCsgoClient
from games.csgo.data.factory import build_csgo_client
from games.csgo.data.pandascore_client import PandaScoreCsgoClient

_TEAMS = [
    {"id": 1, "name": "Natus Vincere", "acronym": "NAVI", "location": "UA"},
    {"id": 2, "name": "FaZe Clan", "acronym": "FAZE", "location": "EU"},
    {"id": 3, "name": "Team Vitality", "acronym": "VIT", "location": "EU"},
]

_UPCOMING = [
    {
        "id": 1001, "name": "NAVI vs FAZE", "status": "not_started",
        "begin_at": "2026-06-20T17:00:00Z", "number_of_games": 3,
        "serie": {"full_name": "IEM Katowice 2026"},
        "opponents": [
            {"opponent": {"id": 1, "name": "Natus Vincere", "acronym": "NAVI"}},
            {"opponent": {"id": 2, "name": "FaZe Clan", "acronym": "FAZE"}},
        ],
        "results": [],
    },
    # A TBD entry (single opponent) must be skipped, not crash.
    {
        "id": 1002, "name": "TBD", "status": "not_started",
        "begin_at": "2026-06-21T17:00:00Z", "number_of_games": 3,
        "opponents": [{"opponent": {"id": 1, "name": "Natus Vincere", "acronym": "NAVI"}}],
        "results": [],
    },
]

_PAST = [
    {
        "id": 900, "status": "finished", "begin_at": "2026-06-10T17:00:00Z",
        "number_of_games": 3, "serie": {"full_name": "BLAST Premier"},
        "opponents": [
            {"opponent": {"id": 1, "name": "Natus Vincere", "acronym": "NAVI"}},
            {"opponent": {"id": 3, "name": "Team Vitality", "acronym": "VIT"}},
        ],
        "results": [{"team_id": 1, "score": 2}, {"team_id": 3, "score": 0}],
    },
    {
        "id": 901, "status": "finished", "begin_at": "2026-06-09T17:00:00Z",
        "number_of_games": 3,
        "opponents": [
            {"opponent": {"id": 2, "name": "FaZe Clan", "acronym": "FAZE"}},
            {"opponent": {"id": 3, "name": "Team Vitality", "acronym": "VIT"}},
        ],
        "results": [{"team_id": 2, "score": 2}, {"team_id": 3, "score": 1}],
    },
]


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/teams"):
        return httpx.Response(200, json=_TEAMS)
    if "/matches/upcoming" in path:
        return httpx.Response(200, json=_UPCOMING)
    if "/matches/running" in path:
        return httpx.Response(200, json=[])
    if "/matches/past" in path:
        return httpx.Response(200, json=_PAST)
    return httpx.Response(200, json=[])


def _client() -> PandaScoreCsgoClient:
    mock = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    return PandaScoreCsgoClient(token="test-token", client=mock)


def test_refresh_maps_teams_and_matches():
    c = _client()
    asyncio.run(c.refresh())
    teams = c.get_teams()
    assert {t.abbreviation for t in teams} == {"NAVI", "FAZE", "VIT"}

    matches = c.get_matches()
    assert len(matches) == 1  # TBD entry skipped, no live matches
    m = matches[0]
    assert m.id == "1001"
    assert m.status == "SCHEDULED"
    assert m.best_of == 3
    assert m.event == "IEM Katowice 2026"
    assert {m.team1_abbrev, m.team2_abbrev} == {"NAVI", "FAZE"}


def test_provisional_elo_reflects_results():
    c = _client()
    asyncio.run(c.refresh())
    by_id = {t.id: t for t in c.get_teams()}
    # NAVI (won) > 1500 > VIT (lost twice); FAZE (won) > VIT.
    assert by_id[1].rating > 1500.0
    assert by_id[3].rating < 1500.0
    assert by_id[2].rating > by_id[3].rating


def test_get_match_and_past():
    c = _client()
    asyncio.run(c.refresh())
    assert c.get_match("1001") is not None       # upcoming
    assert c.get_match("900") is not None         # past (results recorded)
    assert c.get_match("nope") is None
    past = c.get_past_matches()
    assert len(past) == 2
    assert all(p.team1_score is not None for p in past)


def test_factory_falls_back_to_stub_without_token(monkeypatch):
    monkeypatch.delenv("PANDASCORE_TOKEN", raising=False)
    monkeypatch.setenv("CSGO_DATA_PROVIDER", "pandascore")
    assert isinstance(build_csgo_client(), StubCsgoClient)

    monkeypatch.setenv("CSGO_DATA_PROVIDER", "stub")
    assert isinstance(build_csgo_client(), StubCsgoClient)


def test_factory_builds_pandascore_with_token(monkeypatch):
    monkeypatch.setenv("CSGO_DATA_PROVIDER", "pandascore")
    monkeypatch.setenv("PANDASCORE_TOKEN", "abc123")
    client = build_csgo_client()
    assert isinstance(client, PandaScoreCsgoClient)
    asyncio.run(client.close())
