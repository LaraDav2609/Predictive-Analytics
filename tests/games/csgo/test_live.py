"""Tests for the CS2 live win-probability updater (no Redis required)."""

from common.ml.bridge.outcome_publisher import InMemoryOutcomePublisher
from games.csgo.live import LiveMatchState, publish_live, update_live_probability


def test_fresh_state_tracks_pregame_edge():
    assert update_live_probability(0.55, LiveMatchState()).team1_win_prob > 0.5
    even = update_live_probability(0.50, LiveMatchState()).team1_win_prob
    assert abs(even - 0.5) < 0.05


def test_winning_first_map_boosts_series():
    base = update_live_probability(0.55, LiveMatchState()).team1_win_prob
    after = update_live_probability(0.55, LiveMatchState(maps_won_team1=1, map_in_progress=False)).team1_win_prob
    assert after > base
    assert after > 0.7


def test_big_map_lead_raises_current_map_and_series():
    fresh = update_live_probability(0.5, LiveMatchState(current_map="Mirage")).team1_win_prob
    lead = update_live_probability(0.5, LiveMatchState(current_map="Mirage", team1_rounds=10, team2_rounds=2))
    assert lead.current_map_team1_prob > 0.8
    assert lead.team1_win_prob > fresh


def test_man_advantage_drives_round_probability():
    state = LiveMatchState(current_map="Mirage", players_alive_team1=5, players_alive_team2=0)
    out = update_live_probability(0.5, state)
    assert out.round_team1_prob > 0.9
    assert out.drivers.get("man_advantage") == 5


def test_clinched_series_is_certain():
    out = update_live_probability(0.4, LiveMatchState(best_of=3, maps_won_team1=2, map_in_progress=False))
    assert out.team1_win_prob == 1.0


def test_publish_uses_csgo_namespace():
    pub = InMemoryOutcomePublisher()
    live = update_live_probability(0.55, LiveMatchState())
    n = publish_live("m1", "NAVI", "FAZE", live, pub)
    assert n == 2
    assert all(ch.startswith("csgo:prob:") for ch, _ in pub.channel_messages)
