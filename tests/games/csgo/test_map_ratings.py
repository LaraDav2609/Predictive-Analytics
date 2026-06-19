"""Tests for per-map Glicko ratings and the map-edge feature."""

from datetime import datetime, timedelta, timezone

from games.csgo.analytics.ratings import Rating, get_map_rating, rate_maps
from games.csgo.features import FeatureExtractor
from games.csgo.models.csgo import CsgoMatch, MapScore


def _match(i, t1, t2, day, maps):
    wins_t1 = sum(1 for ms in maps if ms.winner_id == t1)
    return CsgoMatch(
        id=str(i), team1=f"T{t1}", team2=f"T{t2}", team1_id=t1, team2_id=t2,
        date=datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(days=day),
        best_of=3, winner_id=t1 if wins_t1 > len(maps) / 2 else t2, map_scores=maps,
    )


def test_rate_maps_specializes_by_map():
    matches = []
    for d in range(6):
        matches.append(_match(len(matches), 1, 2, d, [
            MapScore(order=1, map_name="Mirage", winner_id=1),
            MapScore(order=2, map_name="Inferno", winner_id=2),
            MapScore(order=3, map_name="Mirage", winner_id=1),
        ]))
    mr = rate_maps(matches)
    assert mr[(1, "Mirage")].rating > mr.get((1, "Inferno"), Rating()).rating
    assert mr[(2, "Inferno")].rating > mr.get((2, "Mirage"), Rating()).rating


def test_get_map_rating_falls_back_to_global():
    rating, rd = get_map_rating({}, team_id=1, map_name="Mirage", fallback=1700)
    assert rating == 1700 and rd == 350.0


def test_feature_extractor_emits_positive_map_edge_on_strong_map():
    matches = [
        _match(i, 1, 2, i, [MapScore(order=1, map_name="Mirage", winner_id=1),
                            MapScore(order=2, map_name="Mirage", winner_id=1)])
        for i in range(6)
    ]
    mr = rate_maps(matches)
    upcoming = CsgoMatch(id="u", team1="T1", team2="T2", team1_id=1, team2_id=2,
                         date=datetime(2026, 7, 1, tzinfo=timezone.utc), best_of=1)
    feats = FeatureExtractor(matches, map_ratings=mr).extract(upcoming)
    assert feats.map_edge > 0  # team1 dominates the likely map
