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


def test_factory_tags_data_provenance(tmp_path, monkeypatch):
    # A real provider reports itself as non-synthetic.
    path = _write(tmp_path, _CSV)
    monkeypatch.setenv("CSGO_DATA_PROVIDER", "localfile")
    monkeypatch.setenv("CSGO_HISTORY_FILE", path)
    real = build_csgo_client()
    assert real.provider_name == "localfile" and real.is_synthetic is False

    # A stub fallback (missing file) reports synthetic, so the dashboard can flag it.
    monkeypatch.delenv("CSGO_HISTORY_FILE")
    fell_back = build_csgo_client()
    assert fell_back.provider_name == "stub" and fell_back.is_synthetic is True

    # Default (no provider configured) is synthetic too.
    monkeypatch.delenv("CSGO_DATA_PROVIDER")
    assert build_csgo_client().is_synthetic is True


def test_missing_file_is_graceful(tmp_path):
    c = LocalHistoryCsgoClient(str(tmp_path / "does-not-exist.csv"))
    assert not c.is_available()
    assert c.get_past_matches() == []


# ── map-level (Kaggle "CS:GO Professional Matches" results.csv) ────────────────
_MAP_LEVEL_CSV = """date,team_1,team_2,_map,result_1,result_2,map_winner,event_id,match_id,rank_1,rank_2,map_wins_1,map_wins_2,match_winner
2020-03-01,NAVI,FaZe,Dust2,16,10,1,100,9001,1,5,2,1,1
2020-03-01,NAVI,FaZe,Mirage,11,16,2,100,9001,1,5,2,1,1
2020-03-01,NAVI,FaZe,Inferno,16,12,1,100,9001,1,5,2,1,1
2020-03-02,G2,Vitality,Nuke,16,14,1,101,9002,3,2,1,0,1
2020-03-03,Spirit,G2,Overpass,10,16,2,102,9003,8,3,0,2,2
2020-03-03,Spirit,G2,Ancient,13,16,2,102,9003,8,3,0,2,2
"""


def test_map_level_csv_aggregates_by_match_id(tmp_path):
    c = LocalHistoryCsgoClient(_write(tmp_path, _MAP_LEVEL_CSV))
    past = {m.id: m for m in c.get_past_matches()}
    assert set(past) == {"9001", "9002", "9003"}

    navi = past["9001"]
    assert navi.team1 == "NAVI" and navi.team2 == "FaZe"
    assert (navi.team1_score, navi.team2_score) == (2, 1)
    assert navi.best_of == 3                       # inferred from 2-1
    assert navi.winner_id == navi.team1_id         # match_winner=1 -> NAVI
    assert len(navi.map_scores) == 3
    assert navi.map_scores[0].map_name == "Dust2"
    assert navi.map_scores[0].winner_id == navi.team1_id   # map_winner=1
    assert navi.map_scores[1].winner_id == navi.team2_id   # map_winner=2
    assert (navi.map_scores[0].team1_rounds, navi.map_scores[0].team2_rounds) == (16, 10)

    assert past["9002"].best_of == 1               # single map, 1-0 -> bo1
    assert past["9003"].winner_id == past["9003"].team2_id  # match_winner=2 -> G2
    assert past["9003"].best_of == 3


def test_map_level_backtest_runs(tmp_path):
    header = "date,team_1,team_2,_map,result_1,result_2,map_winner,match_id,map_wins_1,map_wins_2,match_winner"
    rows = [header]
    teams = ["NAVI", "FaZe", "Vitality", "G2", "Spirit"]
    mid, day = 9000, 1
    for _ in range(3):
        for a, b in itertools.combinations(teams, 2):
            rows.append(f"2020-04-{day:02d},{a},{b},Dust2,16,10,1,{mid},2,1,1")
            rows.append(f"2020-04-{day:02d},{a},{b},Mirage,10,16,2,{mid},2,1,1")
            rows.append(f"2020-04-{day:02d},{a},{b},Inferno,16,12,1,{mid},2,1,1")
            mid, day = mid + 1, day % 27 + 1
    c = LocalHistoryCsgoClient(_write(tmp_path, "\n".join(rows)))
    past = c.get_past_matches()
    assert len(past) >= 20
    assert all(len(m.map_scores) == 3 and m.best_of == 3 for m in past)
    assert run_backtest(past, min_history=5)["scored"] > 0
