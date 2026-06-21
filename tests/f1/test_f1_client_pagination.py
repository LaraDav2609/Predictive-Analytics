import asyncio
import unittest
from datetime import datetime, timezone

import httpx

from sports.f1.data.openf1_client import OpenF1Client
from sports.f1.data.f1_client import F1Client
from sports.f1.models.f1 import Driver, Race


class F1ClientPaginationTests(unittest.IsolatedAsyncioTestCase):
    async def test_known_driver_photo_uses_curated_fallback_before_openf1(self):
        client = F1Client()
        calls = 0

        async def fail_openf1(number, code):
            nonlocal calls
            calls += 1
            return None

        client._resolve_openf1_photo = fail_openf1

        photo = await client._resolve_driver_photo("leclerc", None, 16, "LEC")

        self.assertEqual(0, calls)
        self.assertIn("CHALEC01_Charles_Leclerc", photo)

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

    async def test_openf1_missing_endpoint_is_negative_cached(self):
        client = OpenF1Client()
        client._client = _MissingEndpointClient()

        first = await client.get_laps(11315)
        first_error = dict(client._last_error or {})
        second = await client.get_laps(11315)
        second_error = dict(client._last_error or {})

        self.assertEqual([], first)
        self.assertEqual([], second)
        self.assertEqual(1, client._client.calls)
        self.assertEqual(404, first_error["status_code"])
        self.assertEqual(404, second_error["status_code"])
        self.assertTrue(second_error["cached"])

    async def test_openf1_duplicate_requests_share_inflight_call(self):
        client = OpenF1Client()
        client._client = _SlowMissingEndpointClient()

        results = await asyncio.gather(
            client.get_laps(11315),
            client.get_laps(11315),
            client.get_laps(11315),
        )

        self.assertEqual([[], [], []], results)
        self.assertEqual(1, client._client.calls)
        self.assertEqual(404, client._last_error["status_code"])

    async def test_openf1_live_session_search_uses_current_year_only(self):
        client = OpenF1Client()
        client._client = _SessionYearRecorder()
        race = Race(
            round=8,
            name="Spanish Grand Prix",
            circuit="Circuit de Barcelona-Catalunya",
            country="Spain",
            date=datetime(2026, 6, 14, 15, 0, tzinfo=timezone.utc),
        )

        await client.get_session_data(race, "race", live=True)

        self.assertEqual([2026], client._client.years)

    async def test_openf1_session_features_use_race_year_only(self):
        client = OpenF1Client()
        client._client = _SessionFeatureYearRecorder()
        race = Race(
            round=8,
            name="Spanish Grand Prix",
            circuit="Circuit de Barcelona-Catalunya",
            country="Spain",
            date=datetime(2026, 6, 14, 15, 0, tzinfo=timezone.utc),
        )

        result = await client.get_session_features(race, "race", live=False)

        self.assertTrue(result["ok"])
        self.assertEqual([2026], client._client.years)

    async def test_openf1_session_lookup_uses_positive_cache(self):
        client = OpenF1Client()
        client._client = _SessionFeatureYearRecorder()
        race = Race(
            round=8,
            name="Spanish Grand Prix",
            circuit="Circuit de Barcelona-Catalunya",
            country="Spain",
            date=datetime(2026, 6, 14, 15, 0, tzinfo=timezone.utc),
        )

        first = await client.get_session_data(race, "race", live=False)
        client._session_cache.clear()
        second = await client.get_session_data(race, "race", live=False)

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertEqual([2026], client._client.years)

    async def test_openf1_session_features_stops_after_repeated_missing_details(self):
        client = OpenF1Client()
        client._client = _MissingDetailClient()
        race = Race(
            round=8,
            name="Spanish Grand Prix",
            circuit="Circuit de Barcelona-Catalunya",
            country="Spain",
            date=datetime(2026, 6, 14, 15, 0, tzinfo=timezone.utc),
        )

        result = await client.get_session_features(race, "race", live=True)

        self.assertTrue(result["ok"])
        self.assertTrue(result["detail_probe_stopped"])
        self.assertIn("intervals", result["skipped_endpoints"])
        self.assertEqual(["/sessions", "/laps", "/position"], client._client.paths)

    async def test_openf1_trace_probe_stops_after_first_missing_location(self):
        client = OpenF1Client()
        client._client = _MissingLocationClient()
        drivers = [
            Driver(id="a", number=12, code="ANT", first_name="A", last_name="Antonelli", nationality="ITA", team="Mercedes"),
            Driver(id="b", number=44, code="HAM", first_name="L", last_name="Hamilton", nationality="GBR", team="Ferrari"),
            Driver(id="c", number=63, code="RUS", first_name="G", last_name="Russell", nationality="GBR", team="Mercedes"),
        ]

        result = await client._build_trace(11315, drivers)

        self.assertFalse(result["ok"])
        self.assertEqual(["/drivers", "/location"], client._client.paths)


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


class _MissingEndpointClient:
    def __init__(self):
        self.calls = 0

    async def get(self, path, params=None, **kwargs):
        self.calls += 1
        request = httpx.Request("GET", f"https://api.openf1.org/v1{path}")
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("missing", request=request, response=response)


class _SlowMissingEndpointClient:
    def __init__(self):
        self.calls = 0

    async def get(self, path, params=None, **kwargs):
        self.calls += 1
        await asyncio.sleep(0.01)
        request = httpx.Request("GET", f"https://api.openf1.org/v1{path}")
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("missing", request=request, response=response)


class _SessionYearRecorder:
    def __init__(self):
        self.years = []

    async def get(self, path, params=None, **kwargs):
        if path == "/sessions" and params and "year" in params:
            self.years.append(params["year"])
        return _Response([])


class _SessionFeatureYearRecorder:
    def __init__(self):
        self.years = []

    async def get(self, path, params=None, **kwargs):
        if path == "/sessions" and params and "year" in params:
            self.years.append(params["year"])
            return _Response([{
                "session_key": 11315,
                "meeting_key": 2042,
                "session_name": "Race",
                "session_type": "Race",
                "year": params["year"],
                "country_name": "Spain",
                "location": "Barcelona",
            }])
        request = httpx.Request("GET", f"https://api.openf1.org/v1{path}")
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("missing", request=request, response=response)


class _MissingDetailClient:
    def __init__(self):
        self.paths = []

    async def get(self, path, params=None, **kwargs):
        self.paths.append(path)
        request = httpx.Request("GET", f"https://api.openf1.org/v1{path}")
        if path == "/sessions":
            return _Response([{
                "session_key": 11315,
                "meeting_key": 2042,
                "session_name": "Race",
                "session_type": "Race",
                "year": 2026,
                "country_name": "Spain",
                "location": "Barcelona",
            }])
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("missing", request=request, response=response)


class _MissingLocationClient:
    def __init__(self):
        self.paths = []

    async def get(self, path, params=None, **kwargs):
        self.paths.append(path)
        request = httpx.Request("GET", f"https://api.openf1.org/v1{path}")
        if path == "/drivers":
            return _Response([
                {"driver_number": 12},
                {"driver_number": 44},
                {"driver_number": 63},
            ])
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("missing", request=request, response=response)


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
