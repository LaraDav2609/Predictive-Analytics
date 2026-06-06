import unittest
from datetime import datetime, timedelta, timezone

from data.f1_sentiment import DEFAULT_RSS_FEEDS, _dedupe_items, _enrich_scored_item, _is_social_prediction_signal, _score_item, _source_from_feed
from f1_predictor.features.sentiment import build_race_sentiment_impact
from models.f1 import Constructor, Driver, Race


class F1RaceSentimentImpactTests(unittest.TestCase):
    def setUp(self):
        self.drivers = [
            Driver(id="leclerc", number=16, code="LEC", first_name="Charles", last_name="Leclerc", nationality="Monegasque", team="Ferrari", points=75, wins=1, position=3),
            Driver(id="hamilton", number=44, code="HAM", first_name="Lewis", last_name="Hamilton", nationality="British", team="Ferrari", points=72, wins=0, position=4),
            Driver(id="verstappen", number=1, code="VER", first_name="Max", last_name="Verstappen", nationality="Dutch", team="Red Bull Racing", points=43, wins=0, position=7),
        ]
        self.constructors = [
            Constructor(id="ferrari", name="Ferrari", nationality="Italian", points=147, wins=1, position=2),
            Constructor(id="red_bull", name="Red Bull Racing", nationality="Austrian", points=43, wins=0, position=4),
        ]
        self.monaco = Race(round=6, name="Monaco Grand Prix", circuit="Circuit de Monaco", country="Monaco", date=datetime.now(timezone.utc))

    def test_exact_driver_mention_maps_only_to_that_driver(self):
        sentiment = self._sentiment([
            self._item("Charles Leclerc confident of Ferrari pole pace in Monaco", "Leclerc says qualifying pace is strong after clean practice.", "formula1"),
        ])

        impact = build_race_sentiment_impact(self.monaco, self.drivers, self.constructors, sentiment, session="qualifying")

        self.assertGreater(impact["drivers"]["leclerc"]["article_count"], 0)
        self.assertEqual(0, impact["drivers"]["verstappen"]["article_count"])
        self.assertGreater(impact["drivers"]["leclerc"]["qualifying_impact"], 0)

    def test_team_article_affects_both_team_drivers_but_lower_than_driver_specific(self):
        sentiment = self._sentiment([
            self._item("Ferrari upgrade package improves Monaco race pace", "Ferrari brings a new floor and strong long run pace.", "autosport.com"),
            self._item("Charles Leclerc shows fastest Monaco pole pace breakthrough", "Leclerc is confident after fastest clean qualifying simulations and strong track fit.", "formula1"),
        ])

        impact = build_race_sentiment_impact(self.monaco, self.drivers, self.constructors, sentiment, session="race")

        self.assertGreater(impact["drivers"]["hamilton"]["article_count"], 0)
        self.assertGreater(impact["drivers"]["leclerc"]["driver_impact_score"], impact["drivers"]["hamilton"]["driver_impact_score"])
        self.assertGreater(impact["constructors"]["ferrari"]["article_count"], 0)

    def test_monaco_weights_qualifying_and_track_position(self):
        sentiment = self._sentiment([
            self._item("Leclerc says Monaco track position makes qualifying vital", "Pole and grid position are crucial with overtaking difficult.", "racefans.net"),
        ])

        impact = build_race_sentiment_impact(self.monaco, self.drivers, self.constructors, sentiment, session="race")

        leclerc = impact["drivers"]["leclerc"]
        self.assertGreater(leclerc["qualifying_impact"], 0)
        self.assertGreater(leclerc["driver_impact_score"], 0)
        self.assertTrue(any("qualifying" in explanation.lower() or "track" in explanation.lower() for explanation in leclerc["explanations"]))

    def test_low_quality_market_noise_gets_lower_confidence(self):
        rumor = self._sentiment([
            self._item("Ferrari driver market rumour grows before Monaco", "Silly season seat switch rumour adds noise.", "mirror.co.uk"),
        ])
        official = self._sentiment([
            self._item("Ferrari confirms upgrade package for Monaco", "Team confirms improved package and confident race pace.", "formula1"),
        ])

        rumor_impact = build_race_sentiment_impact(self.monaco, self.drivers, self.constructors, rumor, session="race")
        official_impact = build_race_sentiment_impact(self.monaco, self.drivers, self.constructors, official, session="race")

        self.assertLess(rumor_impact["drivers"]["leclerc"]["confidence"], official_impact["drivers"]["leclerc"]["confidence"])

    def test_stale_articles_decay_toward_neutral(self):
        fresh = self._item("Leclerc has strong Monaco qualifying pace", "Pole pace and confidence are strong.", "formula1")
        stale = self._item("Leclerc has strong Monaco qualifying pace", "Pole pace and confidence are strong.", "formula1", days_old=30)

        fresh_impact = build_race_sentiment_impact(self.monaco, self.drivers, self.constructors, self._sentiment([fresh]), session="qualifying")
        stale_impact = build_race_sentiment_impact(self.monaco, self.drivers, self.constructors, self._sentiment([stale]), session="qualifying")

        self.assertGreater(
            abs(fresh_impact["drivers"]["leclerc"]["driver_impact_score"]),
            abs(stale_impact["drivers"]["leclerc"]["driver_impact_score"]),
        )

    def test_penalty_regulation_news_creates_negative_impact(self):
        sentiment = self._sentiment([
            self._item("Verstappen faces FIA grid drop penalty in Monaco", "Stewards issue a grid drop after regulation breach.", "fia"),
        ])

        impact = build_race_sentiment_impact(self.monaco, self.drivers, self.constructors, sentiment, session="race")

        self.assertLess(impact["drivers"]["verstappen"]["regulation_impact"], 0)
        self.assertLess(impact["drivers"]["verstappen"]["prediction_delta"], 0)

    def test_public_social_sources_are_configured_but_identified_as_reddit(self):
        self.assertGreaterEqual(len(DEFAULT_RSS_FEEDS), 22)
        self.assertIn("https://www.reddit.com/r/formula1/.rss", DEFAULT_RSS_FEEDS)
        self.assertEqual("reddit-f1technical", _source_from_feed("https://www.reddit.com/r/F1Technical/.rss"))
        self.assertNotIn("https://www.reddit.com/r/F1FeederSeries/.rss", DEFAULT_RSS_FEEDS)

    def test_social_rumor_is_capped_below_official_confirmation(self):
        social = self._item(
            "Reddit insider rumour says Ferrari upgrade solves Monaco pace",
            "Unverified paddock whisper claims Ferrari has a breakthrough package.",
            "reddit-formula1",
        )
        official = self._item(
            "Ferrari confirms Monaco upgrade package improves race pace",
            "Official team confirmation says the new floor improves long run pace.",
            "formula1",
        )

        self.assertIn("social_source", social["BiasFlags"])
        self.assertIn("rumor_or_unverified", social["BiasFlags"])
        self.assertLess(social["SentimentConfidence"], official["SentimentConfidence"])
        self.assertLess(abs(social["PredictionImpactScore"]), abs(official["PredictionImpactScore"]))
        self.assertLess(social["PredictionImpactCap"], official["PredictionImpactCap"])

    def test_social_repost_chain_dedupes_to_one_story(self):
        first = self._item(
            "Ferrari upgrade package improves Monaco race pace",
            "Ferrari brings a new floor and strong long run pace.",
            "reddit-formula1",
        )
        repost = {
            **first,
            "Url": "https://www.reddit.com/r/formula1/comments/example/?utm_source=share",
            "NativeId": "different-native-id",
        }
        official = self._item(
            "Ferrari confirms Monaco upgrade package improves race pace",
            "Team confirmation adds source diversity.",
            "formula1",
        )

        deduped = _dedupe_items([first, repost, official])

        self.assertEqual(2, len(deduped))

    def test_social_filter_rejects_generic_chatter_but_keeps_strategy_posts(self):
        self.assertFalse(_is_social_prediction_signal("r/F1Strategy Lounge", "General chat and introductions"))
        self.assertFalse(_is_social_prediction_signal("6 months ago I posted my telemetry tool", "Thanks for the feedback"))
        self.assertFalse(_is_social_prediction_signal("How do Mercedes and Ferrari share CAD data?", "Generic car design question"))
        self.assertFalse(_is_social_prediction_signal("Apart from boost, do drivers control battery deployment?", "General technical discussion"))
        self.assertFalse(_is_social_prediction_signal("New driving4answers video on opposed piston two strokes", "General engineering video"))
        self.assertTrue(_is_social_prediction_signal("Canadian Grand Prix race strategy recap", "Tyre degradation and pit stop windows"))
        self.assertTrue(_is_social_prediction_signal("FIA publishes updated technical regulations", "Floor and wing changes"))

    def _sentiment(self, items):
        return {
            "items": items,
            "drivers": {},
            "teams": {},
            "source_items": len(items),
            "configured_sources": 3,
            "composite": {"Composite": 0.0},
        }

    def _item(self, title, content, source, days_old=0):
        published = datetime.now(timezone.utc) - timedelta(days=days_old)
        item = {
            "Title": title,
            "Content": content,
            "Url": f"https://example.test/{abs(hash(title))}",
            "Source": source,
            "Feed": source,
            "MarketCategory": "f1",
            "PublishedUtc": published.isoformat().replace("+00:00", "Z"),
            "Timestamp": published.isoformat().replace("+00:00", "Z"),
            "Topics": [],
        }
        return _enrich_scored_item(_score_item(item), self.drivers, self.constructors, season=2026)


if __name__ == "__main__":
    unittest.main()
