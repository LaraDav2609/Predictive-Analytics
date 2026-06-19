"""Tests for CS2 pre-game feature extraction."""

from datetime import datetime, timedelta, timezone

from games.csgo.analytics.ratings import rate_matches
from games.csgo.features import FeatureExtractor, classify_event_tier
from games.csgo.models.csgo import CsgoMatch, CsgoTeam, MapScore


def _team(tid: int, roster: list[int]) -> CsgoTeam:
    return CsgoTeam(id=tid, name=f"T{tid}", abbreviation=f"T{tid}", roster=roster)


def _match(i, t1, t2, winner, day, maps=None, r1=None, r2=None, event="ESL Pro League"):
    return CsgoMatch(
        id=str(i), team1=f"T{t1}", team2=f"T{t2}", team1_id=t1, team2_id=t2,
        date=datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(days=day),
        winner_id=winner, best_of=3, event=event,
        map_scores=maps or [], team1_roster=r1 or [], team2_roster=r2 or [],
    )


def _fixture():
    teams = {
        1: _team(1, [101, 102, 103, 104, 105]),
        2: _team(2, [201, 202, 203, 204, 205]),
        3: _team(3, [301, 302, 303, 304, 305]),
    }
    past, day = [], 0
    for _ in range(5):
        past.append(_match(len(past), 1, 2, 1, day,
                           maps=[MapScore(order=1, map_name="Mirage", winner_id=1),
                                 MapScore(order=2, map_name="Inferno", winner_id=1)],
                           r1=[101, 102, 103, 104, 105], r2=[201, 202, 203, 204, 205])); day += 1
        past.append(_match(len(past), 1, 3, 1, day,
                           maps=[MapScore(order=1, map_name="Mirage", winner_id=1)],
                           r1=[101, 102, 103, 104, 105])); day += 1
        past.append(_match(len(past), 2, 3, 2, day)); day += 1
    return teams, past, rate_matches(past)


def test_event_tier_classification():
    assert classify_event_tier("IEM Katowice 2026") == ("S", 1.0)
    assert classify_event_tier("ESL Pro League Season 21")[0] == "A"
    assert classify_event_tier("Random Regional Cup") == ("B", 0.65)


def test_features_capture_strength_form_maps_roster_h2h():
    teams, past, ratings = _fixture()
    fx = FeatureExtractor(past, ratings, teams)

    # Upcoming: team1 fields a stand-in (999 not in core); top-tier event.
    upcoming = _match(999, 1, 2, winner=None, day=99,
                      r1=[101, 102, 103, 104, 999], r2=[201, 202, 203, 204, 205],
                      event="IEM Katowice 2026")
    mf = fx.extract(upcoming)

    # Strength + form
    assert mf.rating_diff > 0
    assert mf.team1.recent_form > 0.9          # team1 won everything
    assert mf.team2.recent_form < 0.5          # opponent-weighted: losses to strong team1

    # Map strength
    assert mf.team1.map_strength.get("Mirage") == 1.0

    # Roster stability / stand-in detection
    assert mf.team1.stand_in_count == 1
    assert mf.team1.roster_stability == 0.8
    assert mf.team2.stand_in_count == 0

    # Event tier + H2H + format
    assert mf.event_tier == "S"
    assert mf.h2h_sample == 5 and mf.h2h_team1_winrate == 1.0
    assert mf.format_amplification > 0.6       # bo3 amplifies the favorite
    assert mf.likely_maps                       # non-empty
    assert mf.provenance["ratings"] == "glicko2"
