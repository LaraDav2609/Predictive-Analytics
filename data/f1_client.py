"""Client for Ergast/Jolpica F1 API (free, no auth)."""

import logging
from datetime import datetime, timezone

import httpx

from data.base_client import SportsDataClient
from models.sport import Sport, Competition
from models.f1 import Driver, Constructor, Race, RaceResult, RacePrediction

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"


class F1Client(SportsDataClient):
    def __init__(self):
        self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)
        self._drivers: list[Driver] = []
        self._constructors: list[Constructor] = []
        self._races: list[Race] = []
        self._loaded = False

    def get_sport(self) -> Sport:
        return Sport.FORMULA_ONE

    async def get_competitions(self) -> list[Competition]:
        return [Competition(
            id="F1-2026",
            name="Formula 1 World Championship 2026",
            sport=Sport.FORMULA_ONE,
            season="2026",
            country="International",
        )]

    def is_available(self) -> bool:
        return True

    async def refresh(self) -> None:
        await self._fetch_driver_standings()
        await self._fetch_constructor_standings()
        await self._fetch_race_calendar()
        self._loaded = True
        logger.info("F1 data refreshed: %d drivers, %d constructors, %d races",
                     len(self._drivers), len(self._constructors), len(self._races))

    async def _fetch_driver_standings(self) -> None:
        try:
            resp = await self._client.get("/2025/driverstandings.json")
            resp.raise_for_status()
            data = resp.json()
            standings_lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if not standings_lists:
                self._drivers = self._get_fallback_drivers()
                return
            self._drivers = []
            for entry in standings_lists[0].get("DriverStandings", []):
                d = entry.get("Driver", {})
                constructors = entry.get("Constructors", [{}])
                team = constructors[0].get("name", "") if constructors else ""
                self._drivers.append(Driver(
                    id=d.get("driverId", ""),
                    number=int(d.get("permanentNumber", 0)) if d.get("permanentNumber") else None,
                    code=d.get("code", ""),
                    first_name=d.get("givenName", ""),
                    last_name=d.get("familyName", ""),
                    nationality=d.get("nationality", ""),
                    team=team,
                    points=float(entry.get("points", 0)),
                    wins=int(entry.get("wins", 0)),
                    position=int(entry.get("position", 0)),
                ))
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 driver standings: %s", e)
            if not self._drivers:
                self._drivers = self._get_fallback_drivers()

    async def _fetch_constructor_standings(self) -> None:
        try:
            resp = await self._client.get("/2025/constructorstandings.json")
            resp.raise_for_status()
            data = resp.json()
            standings_lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if not standings_lists:
                self._constructors = self._get_fallback_constructors()
                return
            self._constructors = []
            for entry in standings_lists[0].get("ConstructorStandings", []):
                c = entry.get("Constructor", {})
                self._constructors.append(Constructor(
                    id=c.get("constructorId", ""),
                    name=c.get("name", ""),
                    nationality=c.get("nationality", ""),
                    points=float(entry.get("points", 0)),
                    wins=int(entry.get("wins", 0)),
                    position=int(entry.get("position", 0)),
                ))
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 constructor standings: %s", e)
            if not self._constructors:
                self._constructors = self._get_fallback_constructors()

    async def _fetch_race_calendar(self) -> None:
        try:
            resp = await self._client.get("/2026.json")
            resp.raise_for_status()
            data = resp.json()
            races_data = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not races_data:
                self._races = self._get_fallback_races()
                return
            self._races = []
            for r in races_data:
                circuit = r.get("Circuit", {})
                location = circuit.get("Location", {})
                date_str = r.get("date", "2026-03-01")
                time_str = r.get("time", "14:00:00Z").rstrip("Z")
                dt = datetime.fromisoformat(f"{date_str}T{time_str}").replace(tzinfo=timezone.utc)
                self._races.append(Race(
                    round=int(r.get("round", 0)),
                    name=r.get("raceName", ""),
                    circuit=circuit.get("circuitName", ""),
                    country=location.get("country", ""),
                    date=dt,
                ))
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 calendar: %s", e)
            if not self._races:
                self._races = self._get_fallback_races()

    def get_drivers(self) -> list[Driver]:
        return self._drivers

    def get_constructors(self) -> list[Constructor]:
        return self._constructors

    def get_races(self) -> list[Race]:
        return self._races

    def get_race_by_round(self, round_num: int) -> Race | None:
        return next((r for r in self._races if r.round == round_num), None)

    @staticmethod
    def _get_fallback_drivers() -> list[Driver]:
        fallback = [
            ("max_verstappen", 1, "VER", "Max", "Verstappen", "Dutch", "Red Bull", 0, 0),
            ("lando_norris", 4, "NOR", "Lando", "Norris", "British", "McLaren", 0, 0),
            ("charles_leclerc", 16, "LEC", "Charles", "Leclerc", "Monegasque", "Ferrari", 0, 0),
            ("lewis_hamilton", 44, "HAM", "Lewis", "Hamilton", "British", "Ferrari", 0, 0),
            ("carlos_sainz", 55, "SAI", "Carlos", "Sainz", "Spanish", "Williams", 0, 0),
            ("oscar_piastri", 81, "PIA", "Oscar", "Piastri", "Australian", "McLaren", 0, 0),
            ("george_russell", 63, "RUS", "George", "Russell", "British", "Mercedes", 0, 0),
            ("fernando_alonso", 14, "ALO", "Fernando", "Alonso", "Spanish", "Aston Martin", 0, 0),
            ("pierre_gasly", 10, "GAS", "Pierre", "Gasly", "French", "Alpine", 0, 0),
            ("yuki_tsunoda", 22, "TSU", "Yuki", "Tsunoda", "Japanese", "Red Bull", 0, 0),
        ]
        return [
            Driver(id=did, number=num, code=code, first_name=fn, last_name=ln,
                   nationality=nat, team=team, points=pts, wins=wins, position=i + 1)
            for i, (did, num, code, fn, ln, nat, team, pts, wins) in enumerate(fallback)
        ]

    @staticmethod
    def _get_fallback_constructors() -> list[Constructor]:
        fallback = [
            ("red_bull", "Red Bull Racing", "Austrian"),
            ("mclaren", "McLaren", "British"),
            ("ferrari", "Ferrari", "Italian"),
            ("mercedes", "Mercedes", "German"),
            ("aston_martin", "Aston Martin", "British"),
            ("alpine", "Alpine F1 Team", "French"),
            ("williams", "Williams", "British"),
            ("haas", "Haas F1 Team", "American"),
            ("sauber", "Kick Sauber", "Swiss"),
            ("rb", "RB", "Italian"),
        ]
        return [
            Constructor(id=cid, name=name, nationality=nat, position=i + 1)
            for i, (cid, name, nat) in enumerate(fallback)
        ]

    @staticmethod
    def _get_fallback_races() -> list[Race]:
        races = [
            (1, "Australian Grand Prix", "Albert Park", "Australia", "2026-03-15"),
            (2, "Bahrain Grand Prix", "Bahrain International Circuit", "Bahrain", "2026-03-29"),
            (3, "Saudi Arabian Grand Prix", "Jeddah Corniche Circuit", "Saudi Arabia", "2026-04-05"),
            (4, "Japanese Grand Prix", "Suzuka Circuit", "Japan", "2026-04-19"),
            (5, "Chinese Grand Prix", "Shanghai International Circuit", "China", "2026-05-03"),
            (6, "Miami Grand Prix", "Miami International Autodrome", "USA", "2026-05-17"),
            (7, "Emilia Romagna Grand Prix", "Autodromo Enzo e Dino Ferrari", "Italy", "2026-05-31"),
            (8, "Monaco Grand Prix", "Circuit de Monaco", "Monaco", "2026-06-07"),
            (9, "Spanish Grand Prix", "Circuit de Barcelona-Catalunya", "Spain", "2026-06-21"),
            (10, "Canadian Grand Prix", "Circuit Gilles Villeneuve", "Canada", "2026-06-28"),
            (11, "Austrian Grand Prix", "Red Bull Ring", "Austria", "2026-07-05"),
            (12, "British Grand Prix", "Silverstone Circuit", "Great Britain", "2026-07-19"),
        ]
        return [
            Race(round=r, name=n, circuit=c, country=co,
                 date=datetime.fromisoformat(f"{d}T14:00:00").replace(tzinfo=timezone.utc))
            for r, n, c, co, d in races
        ]

    async def close(self):
        await self._client.aclose()
