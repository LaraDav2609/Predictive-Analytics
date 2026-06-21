import unittest
from datetime import datetime, timezone

import httpx

from common.data.weather_client import OpenMeteoClient


class OpenMeteoClientCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_forecast_payload_across_sessions(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                json={
                    "hourly": {
                        "time": ["2026-06-28T12:00", "2026-06-28T13:00"],
                        "temperature_2m": [22.0, 24.0],
                        "relative_humidity_2m": [55.0, 56.0],
                        "precipitation_probability": [10.0, 20.0],
                        "precipitation": [0.0, 0.1],
                        "rain": [0.0, 0.0],
                        "wind_speed_10m": [8.0, 9.0],
                        "wind_gusts_10m": [12.0, 13.0],
                        "cloud_cover": [25.0, 30.0],
                    }
                },
            )

        client = OpenMeteoClient()
        await client._forecast.aclose()
        await client._historical.aclose()
        client._forecast = httpx.AsyncClient(
            base_url="https://api.open-meteo.com/v1",
            transport=httpx.MockTransport(handler),
        )
        client._historical = httpx.AsyncClient(
            base_url="https://historical-forecast-api.open-meteo.com/v1",
            transport=httpx.MockTransport(handler),
        )

        try:
            at = datetime(2026, 6, 28, 12, 30, tzinfo=timezone.utc)
            race = await client.get_session_weather(47.2197, 14.7647, at, session="race")
            qualifying = await client.get_session_weather(47.2197, 14.7647, at, session="qualifying")

            self.assertEqual(1, calls)
            self.assertEqual("race", race["session"])
            self.assertEqual("qualifying", qualifying["session"])
            self.assertEqual(race["rain_probability"], qualifying["rain_probability"])
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
