"""Historical race loader for F1 backtesting."""

from __future__ import annotations

from typing import Any


class HistoricalRaceLoader:
    def __init__(self, client):
        self._client = client

    async def load_season(self, season: int) -> list[dict[str, Any]]:
        if hasattr(self._client, "get_historical_race_results"):
            races = await self._client.get_historical_race_results(season)
        else:
            races = await self._client._fetch_season_results(season)
        return sorted(
            [race for race in races if str(race.get("round") or "").isdigit()],
            key=lambda race: int(race.get("round") or 0),
        )

    async def load_range(self, start_season: int, end_season: int) -> dict[int, list[dict[str, Any]]]:
        seasons = {}
        for season in range(start_season, end_season + 1):
            seasons[season] = await self.load_season(season)
        return seasons
