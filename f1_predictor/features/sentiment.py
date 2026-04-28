"""Optional adapter over completed F1 sentiment outputs."""

from __future__ import annotations

from f1_predictor.scoring.normalization import label_from_score, num


class SentimentFeatureProvider:
    def __init__(self, sentiment: dict | None):
        self._sentiment = sentiment or {}

    def get_features(self) -> dict:
        drivers = self._sentiment.get("drivers") or {}
        teams = self._sentiment.get("teams") or {}
        source_items = int(self._sentiment.get("source_items") or self._sentiment.get("published_items") or 0)
        source_count = int(self._sentiment.get("configured_sources") or 0)
        return {
            "drivers": {
                driver_id: {
                    **(item or {}),
                    "coverage_score": _coverage_score(item),
                    "source": "f1_sentiment_driver_adapter",
                    "missing_data": int((item or {}).get("mentions") or 0) == 0,
                }
                for driver_id, item in drivers.items()
            },
            "teams": {
                team_id: {
                    **(item or {}),
                    "coverage_score": _coverage_score(item),
                    "source": "f1_sentiment_constructor_adapter",
                    "missing_data": int((item or {}).get("mentions") or 0) == 0,
                }
                for team_id, item in teams.items()
            },
            "composite": self._sentiment.get("composite") or {},
            "source_items": source_items,
            "configured_sources": source_count,
            "confidence": round(min(0.82, 0.25 + min(1.0, source_items / 80.0) * 0.40 + min(1.0, source_count / 10.0) * 0.17), 4),
            "sentiment_available": bool(drivers or teams),
        }


def driver_sentiment_component(driver_id: str, team: str, sentiment: dict | None) -> dict:
    sentiment = sentiment or {}
    sentiment_drivers = sentiment.get("drivers") or {}
    sentiment_teams = sentiment.get("teams") or {}
    global_sentiment = float((sentiment.get("composite") or {}).get("Composite") or 0.0)
    driver_sentiment_data = sentiment_drivers.get(driver_id) or {}
    team_sentiment_data = sentiment_teams.get((team or "").lower()) or {}

    personal_news_score = num(
        driver_sentiment_data.get("personal_news_score", driver_sentiment_data.get("personal_score")),
        driver_sentiment_data.get("score", 0.0),
    )
    team_news_score = num(
        driver_sentiment_data.get("team_news_score", driver_sentiment_data.get("team_score")),
        team_sentiment_data.get("score", 0.0),
    )
    overall_news_score = num(
        driver_sentiment_data.get("overall_news_score", driver_sentiment_data.get("overall_score")),
        global_sentiment,
    )
    sentiment_score = max(
        -1.0,
        min(
            1.0,
            num(
                driver_sentiment_data.get("score"),
                0.55 * personal_news_score + 0.30 * team_news_score + 0.15 * overall_news_score,
            ),
        ),
    )
    news_win_modifier = max(
        0.90,
        min(1.10, num(driver_sentiment_data.get("win_probability_modifier"), 1.0 + sentiment_score * 0.10)),
    )
    wdc_modifier = max(
        0.86,
        min(1.14, num(driver_sentiment_data.get("wdc_probability_modifier"), 1.0 + sentiment_score * 0.14)),
    )
    return {
        "sentiment_score": sentiment_score,
        "sentiment_label": driver_sentiment_data.get("label") or label_from_score(sentiment_score),
        "sentiment_mentions": int(driver_sentiment_data.get("mentions") or 0),
        "personal_news_score": personal_news_score,
        "personal_news_mentions": int(driver_sentiment_data.get("personal_mentions") or 0),
        "team_news_score": team_news_score,
        "team_news_mentions": int(driver_sentiment_data.get("team_mentions") or 0),
        "overall_news_score": overall_news_score,
        "news_win_modifier": news_win_modifier,
        "wdc_modifier": wdc_modifier,
        "source_breakdown": driver_sentiment_data.get("source_breakdown") or {},
        "topic_scores": driver_sentiment_data.get("topic_scores") or {},
        "coverage_score": _coverage_score(driver_sentiment_data),
    }


def _coverage_score(item: dict | None) -> float:
    item = item or {}
    mentions = int(item.get("mentions") or item.get("personal_mentions") or 0)
    sources = len(item.get("source_breakdown") or {})
    return round(min(1.0, mentions / 18.0 * 0.70 + sources / 6.0 * 0.30), 4)
