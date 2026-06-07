from abc import ABC, abstractmethod
from common.models.sport import Sport, Competition


class SportsDataClient(ABC):
    """Abstract base class for sport-specific data clients."""

    @abstractmethod
    def get_sport(self) -> Sport:
        ...

    @abstractmethod
    async def get_competitions(self) -> list[Competition]:
        ...

    @abstractmethod
    async def refresh(self) -> None:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...
