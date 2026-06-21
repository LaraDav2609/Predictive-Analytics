"""Tests for the bounded CS2 sentiment modifier."""

from games.csgo.sentiment import MAX_TOTAL_SHIFT, NewsItem, apply_sentiment


def test_standin_on_team1_lowers_team1():
    adj = apply_sentiment(0.55, [NewsItem(kind="stand_in", team=1, severity=1.0)])
    assert adj.adjusted_prob < 0.55
    assert "stand_in:team1" in adj.reasons


def test_bad_news_on_team2_helps_team1():
    adj = apply_sentiment(0.50, [NewsItem(kind="illness", team=2, severity=1.0)])
    assert adj.adjusted_prob > 0.50


def test_total_shift_is_capped():
    items = [NewsItem(kind="stand_in", team=1, severity=1.0) for _ in range(5)]
    adj = apply_sentiment(0.60, items)
    assert abs(adj.delta) <= MAX_TOTAL_SHIFT + 1e-9   # cannot dominate the model


def test_no_items_no_change():
    adj = apply_sentiment(0.55, [])
    assert adj.adjusted_prob == 0.55
    assert adj.reasons == []


def test_patch_is_neutral():
    adj = apply_sentiment(0.55, [NewsItem(kind="patch", team=None, severity=1.0)])
    assert adj.adjusted_prob == 0.55
