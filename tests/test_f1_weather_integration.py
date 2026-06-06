import unittest
from datetime import datetime, timezone

from f1_predictor.features.tires import TireFeatureProvider
from f1_predictor.features.weather import WeatherFeatureProvider
from f1_predictor.truth import build_race_truth_snapshot
from models.f1 import Driver, Race


class F1WeatherIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.race = Race(
            round=6,
            name="Monaco Grand Prix",
            circuit="Circuit de Monaco",
            country="Monaco",
            date=datetime(2026, 6, 7, 13, 0, tzinfo=timezone.utc),
        )
        self.drivers = [
            Driver(id="leclerc", number=16, code="LEC", first_name="Charles", last_name="Leclerc", nationality="Monegasque", team="Ferrari", position=1),
        ]

    def test_session_weather_overrides_race_day_weather(self):
        features = {
            "weather_by_round": {
                "6": {
                    "rain_probability": 0.05,
                    "rain_intensity": 0.0,
                    "air_temperature": 24.0,
                    "wind_speed": 8.0,
                    "chaos_score": 0.04,
                    "source": "race_day",
                    "confidence": 0.5,
                    "missing_data": False,
                }
            },
            "weather_by_round_session": {
                "6": {
                    "qualifying": {
                        "rain_probability": 0.55,
                        "rain_intensity": 0.35,
                        "air_temperature": 18.0,
                        "wind_speed": 12.0,
                        "wind_gusts": 39.0,
                        "chaos_score": 0.42,
                        "source": "open_meteo_forecast",
                        "confidence": 0.62,
                        "missing_data": False,
                    }
                }
            },
        }

        weather = WeatherFeatureProvider(features).get_features(self.race, session="qualifying")

        self.assertEqual("open_meteo_forecast", weather["source"])
        self.assertEqual(0.55, weather["rain_probability"])
        self.assertGreater(weather["weather_model_impact"]["model_delta_hint"], 0)
        self.assertIn("rain volatility increased incident and strategy risk", weather["weather_explanations"])

    def test_openf1_weather_beats_open_meteo_session_weather(self):
        features = {
            "weather_by_round_session": {
                "6": {
                    "race": {
                        "rain_probability": 0.9,
                        "chaos_score": 0.8,
                        "source": "open_meteo_forecast",
                        "confidence": 0.62,
                        "missing_data": False,
                    }
                }
            }
        }
        openf1_session = {
            "weather": {
                "rain_probability": 0.1,
                "air_temperature": 21.0,
                "wind_speed": 7.0,
                "chaos_score": 0.12,
                "missing_data": False,
            }
        }

        weather = WeatherFeatureProvider(features).get_features(self.race, openf1_session, session="race")

        self.assertEqual("openf1_weather", weather["source"])
        self.assertEqual(0.1, weather["rain_probability"])

    def test_temperature_increases_tire_degradation(self):
        track = {"tire_stress": 0.5, "confidence": 0.8}

        cool = TireFeatureProvider().get_features(track, weather={"air_temperature": 22.0})
        hot = TireFeatureProvider().get_features(track, weather={"air_temperature": 38.0})

        self.assertGreater(hot["degradation_rate"], cool["degradation_rate"])
        self.assertGreater(hot["track_temperature_effect"], 0)

    def test_truth_uses_open_meteo_weather_when_openf1_weather_missing(self):
        truth = build_race_truth_snapshot(
            self.race,
            self.drivers,
            profile={"ok": True},
            openf1_session={"ok": True, "raw_counts": {"weather": 0}},
            weather={
                "rain_probability": 0.3,
                "chaos_score": 0.2,
                "source": "open_meteo_forecast",
                "confidence": 0.62,
                "missing_data": False,
            },
        )

        self.assertEqual("open_meteo_forecast", truth["signals"]["weather"]["source"])
        self.assertNotIn("openf1_weather", truth["missing_groups"])


if __name__ == "__main__":
    unittest.main()
