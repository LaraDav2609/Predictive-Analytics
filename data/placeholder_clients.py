"""Placeholder clients for sports not yet implemented."""

from data.base_client import SportsDataClient
from models.sport import Sport, Competition


class PlaceholderClient(SportsDataClient):
    """Base placeholder for sports coming soon."""

    def __init__(self, sport: Sport, label: str):
        self._sport = sport
        self._label = label

    def get_sport(self) -> Sport:
        return self._sport

    async def get_competitions(self) -> list[Competition]:
        return []

    async def refresh(self) -> None:
        raise NotImplementedError(f"{self._label} data client coming soon")

    def is_available(self) -> bool:
        return False


class BasketballClient(PlaceholderClient):
    def __init__(self):
        super().__init__(Sport.BASKETBALL, "NBA Basketball")


class BaseballClient(PlaceholderClient):
    def __init__(self):
        super().__init__(Sport.BASEBALL, "MLB Baseball")


class TennisClient(PlaceholderClient):
    def __init__(self):
        super().__init__(Sport.TENNIS, "Tennis (ATP/WTA)")


class NFLClient(PlaceholderClient):
    def __init__(self):
        super().__init__(Sport.AMERICAN_FOOTBALL, "NFL Football")


class CricketClient(PlaceholderClient):
    def __init__(self):
        super().__init__(Sport.CRICKET, "Cricket (ICC)")
