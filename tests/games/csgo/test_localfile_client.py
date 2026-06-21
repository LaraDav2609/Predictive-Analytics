"""Tests for the local-file CS2 history backfill adapter."""

import itertools
import json

from games.csgo.analytics.backtest import run_backtest
from games.csgo.analytics.ratings import rate_matches
from games.csgo.data.csgo_client import StubCsgoClient
from games.csgo.data.factory import build_csgo_client
from games.csgo.data.localfile_client import LocalHistoryCsgoClient, _stable_team_id

_CSV = """date,event,team1,team2,best_of,team1_score,team2_score
2026-01-01,IEM Katowice,NAVI,FaZe,3,2,0
2026-01-02,IEM Katowice,FaZe,Vitality,3,2,1
2026-01-03,IEM Katowice,NAVI,Vitality,3,2,0
"""


def _write(tmp_path, text, name="hist.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_csv_backfill_loads_results(tmp_path):
    c = LocalHistoryCsgoClient(_write(tmp_path, _CSV))
    assert c.is_available()
    past = c.get_past_matches()
    assert len(past) == 3
    assert all(m.winner_id is not None and m.status == "FINAL" for m in past)
    assert {t.name for t in c.get_teams()} == {"NAVI", "FaZe", "Vitality"}


def test_stable_team_ids():
    assert _stable_team_id("42", "X") == 42
    assert _stable_team_id(None, "FaZe Clan") == _stable_team_id(None, "faze clan")  # name-normalized, stable


def test_ratings_and_backtest_run_on_backfill(tmp_path):
    rows = ["date,event,team1,team2,best_of,team1_score,team2_score"]
    teams = ["NAVI", "FaZe", "Vitality", "G2", "Spirit"]
    day = 1
    for _ in range(4):
        for a, b in itertools.combinations(teams, 2):
            rows.append(f"2026-02-{day:02d},Event,{a},{b},3,2,1")
            day = day % 27 + 1
    c = LocalHistoryCsgoClient(_write(tmp_path, "\n".join(rows)))
    past = c.get_past_matches()
    assert len(past) >= 20
    assert rate_matches(past)                       # ratings fit
    assert run_backtest(past, min_history=5)["scored"] > 0


def test_json_backfill_supports_map_scores(tmp_path):
    data = [{
        "id": "m1", "team1": "NAVI", "team2": "FaZe", "team1_id": 1, "team2_id": 2,
        "date": "2026-01-01T17:00:00Z", "best_of": 3, "team1_score": 2, "team2_score": 1, "winner_id": 1,
        "map_scores": [{"order": 1, "map_name": "Mirage", "winner_id": 1}],
    }]
    p = tmp_path / "h.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    c = LocalHistoryCsgoClient(str(p))
    past = c.get_past_matches()
    assert len(past) == 1 and past[0].winner_id == 1
    assert past[0].map_scores[0].map_name == "Mirage"


def test_factory_selects_localfile(tmp_path, monkeypatch):
    path = _write(tmp_path, _CSV)
    monkeypatch.setenv("CSGO_DATA_PROVIDER", "localfile")
    monkeypatch.setenv("CSGO_HISTORY_FILE", path)
    assert isinstance(build_csgo_client(), LocalHistoryCsgoClient)
    monkeypatch.delenv("CSGO_HISTORY_FILE")
    assert isinstance(build_csgo_client(), StubCsgoClient)   # missing file → stub


def test_missing_file_is_graceful(tmp_path):
    c = LocalHistoryCsgoClient(str(tmp_path / "does-not-exist.csv"))
    assert not c.is_available()
    assert c.get_past_matches() == []
