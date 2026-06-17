import unittest
from datetime import datetime, timezone

import httpx

from sports.f1.data.openf1_client import OpenF1Client
from sports.f1.data.f1_client import F1Client
from sports.f1.models.f1 import Race


class F1ClientPaginationTests(unittest.IsolatedAsyncioTestCase):
    async def test_historical_results_use_paginated_fetch(self):
        client = F1Client()
        client._client = _PagedClient([
            _page(total=3, offset=0, limit=2, races=[_race(1), _race(2)]),
            _page(total=3, offset=2, limit=2, races=[_race(3)]),
        ])

        rows = await client.get_historical_race_results(2024)

        self.assertEqual(["1", "2", "3"], [row["round"] for row in rows])
        self.assertEqual([
            "/2024/results.json?limit=100&offset=0",
            "/2024/results.json?limit=100&offset=2",
        ], client._client.paths)

    async def test_historical_qualifying_uses_paginated_fetch(self):
        client = F1Client()
        client._client = _PagedClient([
            _page(total=3, offset=0, limit=2, races=[_race(1), _race(2)]),
            _page(total=3, offset=2, limit=2, races=[_race(3)]),
        ])

        rows = await client.get_historical_qualifying_results(2024)

        self.assertEqual(["1", "2", "3"], [row["round"] for row in rows])
        self.assertEqual([
            "/2024/qualifying.json?limit=100&offset=0",
            "/2024/qualifying.json?limit=100&offset=2",
        ], client._client.paths)

    async def test_openf1_session_data_reports_rate_limit_reason(self):
        client = OpenF1Client()
        client._client = _RateLimitedClient()
        race = Race(
            round=1,
            name="Bahrain Grand Prix",
            circuit="Bahrain International Circuit",
            country="Bahrain",
            date=datetime(2024, 3, 2, 15, 0, tzinfo=timezone.utc),
        )

        result = await client.get_session_data(race, "fp1")

        self.assertFalse(result["ok"])
        self.assertEqual("openf1_rate_limited", result["reason"])
        self.assertEqual(429, result["last_error"]["status_code"])


class _PagedClient:
    def __init__(self, pages):
        self.pages = list(pages)
        self.paths = []

    async def get(self, path):
        self.paths.append(path)
        return _Response(self.pages.pop(0))


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _RateLimitedClient:
    async def get(self, path, params=None, **kwargs):
        request = httpx.Request("GET", f"https://api.openf1.org/v1{path}")
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("rate limited", request=request, response=response)


def _page(total, offset, limit, races):
    return {
        "MRData": {
            "total": str(total),
            "offset": str(offset),
            "limit": str(limit),
            "RaceTable": {"Races": races},
        }
    }


def _race(round_num):
    return {"round": str(round_num), "raceName": f"Race {round_num}"}


if __name__ == "__main__":
    unittest.main()
