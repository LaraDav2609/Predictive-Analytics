import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from data.mlb_client import MLBClient


class MLBClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_schedule_requests_today_through_next_7_days(self):
        client = MLBClient()
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"dates": []},
        )
        client._client = SimpleNamespace(get=AsyncMock(return_value=response))

        await client._fetch_schedule()

        today = date.today()
        client._client.get.assert_awaited_once_with("/schedule", params={
            "sportId": 1,
            "startDate": today.isoformat(),
            "endDate": (today + timedelta(days=7)).isoformat(),
            "hydrate": "probablePitcher,team",
        })


if __name__ == "__main__":
    unittest.main()
