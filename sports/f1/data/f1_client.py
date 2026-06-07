"""Client for Ergast/Jolpica F1 API (free, no auth)."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote, urlparse

import httpx

from common.data.base_client import SportsDataClient
from common.data.weather_client import OpenMeteoClient
from common.models.sport import Sport, Competition
from sports.f1.models.f1 import Driver, Constructor, Race, RaceResult, RacePrediction

logger = logging.getLogger(__name__)

BASE_URL = "https://api.jolpi.ca/ergast/f1"

CONSTRUCTOR_TITLE_FALLBACKS = {
    "ferrari": [1961, 1964, 1975, 1976, 1977, 1979, 1982, 1983, 1999, 2000, 2001, 2002, 2003, 2004, 2007, 2008],
    "mclaren": [1974, 1984, 1985, 1988, 1989, 1990, 1991, 1998, 2024, 2025],
    "mercedes": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    "red_bull": [2010, 2011, 2012, 2013, 2022, 2023],
    "williams": [1980, 1981, 1986, 1987, 1992, 1993, 1994, 1996, 1997],
    "renault": [2005, 2006],
    "alpine": [2005, 2006],
}

CONSTRUCTOR_DRIVER_TITLE_FALLBACKS = {
    "ferrari": [
        (1952, "Alberto Ascari", "ASC"), (1953, "Alberto Ascari", "ASC"), (1956, "Juan Manuel Fangio", "FAN"),
        (1958, "Mike Hawthorn", "HAW"), (1961, "Phil Hill", "HIL"), (1964, "John Surtees", "SUR"),
        (1975, "Niki Lauda", "LAU"), (1977, "Niki Lauda", "LAU"), (1979, "Jody Scheckter", "SCH"),
        (2000, "Michael Schumacher", "MSC"), (2001, "Michael Schumacher", "MSC"), (2002, "Michael Schumacher", "MSC"),
        (2003, "Michael Schumacher", "MSC"), (2004, "Michael Schumacher", "MSC"), (2007, "Kimi Raikkonen", "RAI"),
    ],
    "mclaren": [
        (1974, "Emerson Fittipaldi", "FIT"), (1976, "James Hunt", "HUN"), (1984, "Niki Lauda", "LAU"),
        (1985, "Alain Prost", "PRO"), (1986, "Alain Prost", "PRO"), (1988, "Ayrton Senna", "SEN"),
        (1989, "Alain Prost", "PRO"), (1990, "Ayrton Senna", "SEN"), (1991, "Ayrton Senna", "SEN"),
        (1998, "Mika Hakkinen", "HAK"), (1999, "Mika Hakkinen", "HAK"), (2008, "Lewis Hamilton", "HAM"),
        (2025, "Lando Norris", "NOR"),
    ],
    "mercedes": [
        (1954, "Juan Manuel Fangio", "FAN"), (1955, "Juan Manuel Fangio", "FAN"),
        (2014, "Lewis Hamilton", "HAM"), (2015, "Lewis Hamilton", "HAM"), (2016, "Nico Rosberg", "ROS"),
        (2017, "Lewis Hamilton", "HAM"), (2018, "Lewis Hamilton", "HAM"), (2019, "Lewis Hamilton", "HAM"),
        (2020, "Lewis Hamilton", "HAM"),
    ],
    "red_bull": [
        (2010, "Sebastian Vettel", "VET"), (2011, "Sebastian Vettel", "VET"),
        (2012, "Sebastian Vettel", "VET"), (2013, "Sebastian Vettel", "VET"),
        (2021, "Max Verstappen", "VER"), (2022, "Max Verstappen", "VER"),
        (2023, "Max Verstappen", "VER"), (2024, "Max Verstappen", "VER"),
    ],
    "williams": [
        (1980, "Alan Jones", "JON"), (1982, "Keke Rosberg", "ROS"), (1987, "Nelson Piquet", "PIQ"),
        (1992, "Nigel Mansell", "MAN"), (1993, "Alain Prost", "PRO"), (1996, "Damon Hill", "HIL"),
        (1997, "Jacques Villeneuve", "VIL"),
    ],
    "renault": [(2005, "Fernando Alonso", "ALO"), (2006, "Fernando Alonso", "ALO")],
    "alpine": [(2005, "Fernando Alonso", "ALO"), (2006, "Fernando Alonso", "ALO")],
}

F1_HEADSHOT_FALLBACKS = {
    "ALB": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/A/ALEALB01_Alexander_Albon/alealb01.png.transform/9col/image.png",
    "ALO": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/F/FERALO01_Fernando_Alonso/feralo01.png.transform/9col/image.png",
    "ANT": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/K/ANDANT01_Kimi_Antonelli/andant01.png.transform/9col/image.png",
    "BEA": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/O/OLIBEA01_Oliver_Bearman/olibea01.png.transform/9col/image.png",
    "BOR": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/G/GABBOR01_Gabriel_Bortoleto/gabbor01.png.transform/9col/image.png",
    "BOT": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/V/VALBOT01_Valtteri_Bottas/valbot01.png.transform/9col/image.png",
    "COL": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/F/FRACOL01_Franco_Colapinto/fracol01.png.transform/9col/image.png",
    "GAS": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/P/PIEGAS01_Pierre_Gasly/piegas01.png.transform/9col/image.png",
    "HAD": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/I/ISAHAD01_Isack_Hadjar/isahad01.png.transform/9col/image.png",
    "HAM": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LEWHAM01_Lewis_Hamilton/lewham01.png.transform/9col/image.png",
    "HUL": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/N/NICHUL01_Nico_Hulkenberg/nichul01.png.transform/9col/image.png",
    "LAW": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LIALAW01_Liam_Lawson/lialaw01.png.transform/9col/image.png",
    "LEC": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/C/CHALEC01_Charles_Leclerc/chalec01.png.transform/9col/image.png",
    "LIN": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/A/ARVLIN01_Arvid_Lindblad/arvlin01.png.transform/9col/image.png",
    "NOR": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LANNOR01_Lando_Norris/lannor01.png.transform/9col/image.png",
    "OCO": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/E/ESTOCO01_Esteban_Ocon/estoco01.png.transform/9col/image.png",
    "PER": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/S/SERPER01_Sergio_Perez/serper01.png.transform/9col/image.png",
    "PIA": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/O/OSCPIA01_Oscar_Piastri/oscpia01.png.transform/9col/image.png",
    "RUS": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/G/GEORUS01_George_Russell/georus01.png.transform/9col/image.png",
    "SAI": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/C/CARSAI01_Carlos_Sainz/carsai01.png.transform/9col/image.png",
    "STR": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/L/LANSTR01_Lance_Stroll/lanstr01.png.transform/9col/image.png",
    "VER": "https://media.formula1.com/d_driver_fallback_image.png/content/dam/fom-website/drivers/M/MAXVER01_Max_Verstappen/maxver01.png.transform/9col/image.png",
}


class F1Client(SportsDataClient):
    def __init__(self):
        self._season = datetime.now(timezone.utc).year
        self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)
        self._drivers: list[Driver] = []
        self._constructors: list[Constructor] = []
        self._races: list[Race] = []
        self._loaded = False
        self._refresh_lock = asyncio.Lock()
        self._photo_cache: dict[str, str | None] = {}
        self._driver_seasons_cache: dict[str, list[dict]] = {}
        self._constructor_seasons_cache: dict[str, list[dict]] = {}
        self._constructor_driver_titles_cache: dict[tuple[str, int], dict | None] = {}
        self._season_results_cache: dict[int, list[dict]] = {}
        self._season_qualifying_cache: dict[int, list[dict]] = {}
        self._prediction_features_cache: dict[tuple[int, int], dict] = {}
        self._weather_client = OpenMeteoClient()

    def get_sport(self) -> Sport:
        return Sport.FORMULA_ONE

    async def get_competitions(self) -> list[Competition]:
        return [Competition(
            id=f"F1-{self._season}",
            name=f"Formula 1 World Championship {self._season}",
            sport=Sport.FORMULA_ONE,
            season=str(self._season),
            country="International",
        )]

    def is_available(self) -> bool:
        return True

    @property
    def season(self) -> int:
        return self._season

    async def refresh(self) -> None:
        async with self._refresh_lock:
            self._season_results_cache.pop(self._season, None)
            self._prediction_features_cache.clear()
            await self._fetch_driver_standings()
            await self._fetch_constructor_standings()
            await self._fetch_race_calendar()
            self._loaded = True
            logger.info("F1 data refreshed: %d drivers, %d constructors, %d races",
                         len(self._drivers), len(self._constructors), len(self._races))

    async def _fetch_driver_standings(self) -> None:
        try:
            resp = await self._client.get(f"/{self._season}/driverstandings.json")
            resp.raise_for_status()
            data = resp.json()
            standings_lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if not standings_lists:
                self._drivers = self._get_fallback_drivers()
                return
            drivers: list[Driver] = []
            seen: set[str] = set()
            for entry in standings_lists[0].get("DriverStandings", []):
                d = entry.get("Driver", {})
                constructors = entry.get("Constructors", [{}])
                team = constructors[0].get("name", "") if constructors else ""
                driver_id = d.get("driverId", "")
                driver_key = driver_id or d.get("code", "")
                if not driver_key or driver_key in seen:
                    continue
                seen.add(driver_key)
                drivers.append(Driver(
                    id=driver_id,
                    number=int(d.get("permanentNumber", 0)) if d.get("permanentNumber") else None,
                    code=d.get("code", ""),
                    first_name=d.get("givenName", ""),
                    last_name=d.get("familyName", ""),
                    nationality=d.get("nationality", ""),
                    team=team,
                    profile_url=d.get("url"),
                    photo_url=await self._resolve_driver_photo(
                        driver_id,
                        d.get("url"),
                        int(d.get("permanentNumber", 0)) if d.get("permanentNumber") else None,
                        d.get("code"),
                    ),
                    points=float(entry.get("points", 0)),
                    wins=int(entry.get("wins", 0)),
                    position=int(entry.get("position", 0)),
                ))
            self._drivers = drivers
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 driver standings: %s", e)
            if not self._drivers:
                self._drivers = self._get_fallback_drivers()

    async def _fetch_constructor_standings(self) -> None:
        try:
            resp = await self._client.get(f"/{self._season}/constructorstandings.json")
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
            resp = await self._client.get(f"/{self._season}.json")
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
                date_str = r.get("date", f"{self._season}-03-01")
                time_str = r.get("time", "14:00:00Z").rstrip("Z")
                dt = datetime.fromisoformat(f"{date_str}T{time_str}").replace(tzinfo=timezone.utc)
                sessions = _parse_calendar_sessions(r, dt)
                has_sprint = bool(r.get("Sprint") or r.get("SprintQualifying") or r.get("SprintShootout"))
                self._races.append(Race(
                    round=int(r.get("round", 0)),
                    name=r.get("raceName", ""),
                    circuit=circuit.get("circuitName", ""),
                    country=location.get("country", ""),
                    date=dt,
                    circuit_id=circuit.get("circuitId"),
                    locality=location.get("locality"),
                    latitude=_safe_float(location.get("lat")),
                    longitude=_safe_float(location.get("long")),
                    has_sprint=has_sprint,
                    sessions=sessions,
                    status="COMPLETED" if dt < datetime.now(timezone.utc) else "SCHEDULED",
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

    async def get_prediction_features(self, lookback_races: int = 8) -> dict:
        """Build live, historical, circuit, and weather features for the F1 predictor."""
        cache_key = (self._season, lookback_races)
        cached = self._prediction_features_cache.get(cache_key)
        if cached:
            return cached

        current_races = await self._fetch_season_results(self._season)
        previous_races = []
        for season in range(self._season - 1, max(2022, self._season - 3), -1):
            previous_races.extend(await self._fetch_season_results(season))
        races_data = current_races
        all_races = current_races + previous_races

        driver_results: dict[str, list[dict]] = {}
        constructor_results: dict[str, list[dict]] = {}
        track_results: dict[str, dict[str, list[dict]]] = {}
        teammate_results: dict[tuple[int, int, str], list[dict]] = {}
        for race in all_races:
            round_num = int(race.get("round", 0) or 0)
            season = int(race.get("season", self._season) or self._season)
            track_key = _track_key_from_raw_race(race)
            for item in race.get("Results") or []:
                driver = item.get("Driver") or {}
                constructor = item.get("Constructor") or {}
                driver_id = driver.get("driverId", "")
                constructor_name = (constructor.get("name") or "").lower()
                position_raw = item.get("position")
                grid_raw = item.get("grid")
                result = {
                    "round": round_num,
                    "season": season,
                    "track_key": track_key,
                    "driver_id": driver_id,
                    "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                    "team": constructor.get("name", ""),
                    "position": int(position_raw) if str(position_raw).isdigit() else None,
                    "grid": int(grid_raw) if str(grid_raw).lstrip("-").isdigit() else None,
                    "points": float(item.get("points", 0) or 0),
                    "status": item.get("status") or "Unknown",
                    "fastest_lap_rank": ((item.get("FastestLap") or {}).get("rank")),
                }
                if driver_id:
                    driver_results.setdefault(driver_id, []).append(result)
                    track_results.setdefault(track_key, {}).setdefault(driver_id, []).append(result)
                if constructor_name:
                    constructor_results.setdefault(constructor_name, []).append(result)
                    teammate_results.setdefault((season, round_num, constructor_name), []).append(result)

        teammate_deltas: dict[str, list[float]] = {}
        for entries in teammate_results.values():
            classified = [entry for entry in entries if isinstance(entry.get("position"), int)]
            if len(classified) < 2:
                continue
            for entry in classified:
                teammates = [other for other in classified if other.get("driver_id") != entry.get("driver_id")]
                if not teammates:
                    continue
                best_mate = min(teammates, key=lambda item: item["position"])
                teammate_deltas.setdefault(entry["driver_id"], []).append(float(best_mate["position"] - entry["position"]))

        max_driver_points = max((d.points for d in self._drivers), default=1.0) or 1.0
        max_constructor_points = max((c.points for c in self._constructors), default=1.0) or 1.0

        driver_features = {}
        for driver in self._drivers:
            results = sorted(driver_results.get(driver.id, []), key=lambda item: (item["season"], item["round"]), reverse=True)
            current_results = [item for item in results if item.get("season") == self._season]
            recent = results[:lookback_races]
            classified = [item for item in recent if isinstance(item.get("position"), int)]
            grids = [item["grid"] for item in recent if isinstance(item.get("grid"), int) and item.get("grid", 0) > 0]
            grid_deltas = [
                item["grid"] - item["position"]
                for item in recent
                if isinstance(item.get("grid"), int)
                and item.get("grid", 0) > 0
                and isinstance(item.get("position"), int)
            ]
            avg_finish = (
                sum(item["position"] for item in classified) / len(classified)
                if classified else None
            )
            avg_grid = sum(grids) / len(grids) if grids else None
            avg_grid_delta = sum(grid_deltas) / len(grid_deltas) if grid_deltas else 0.0
            recent_points = sum(item.get("points", 0.0) for item in recent)
            podiums = sum(1 for item in recent if isinstance(item.get("position"), int) and item["position"] <= 3)
            wins = sum(1 for item in recent if item.get("position") == 1)
            dnfs = sum(1 for item in recent if not _is_finished_status(item.get("status")))
            mechanical_dnfs = sum(1 for item in recent if _is_mechanical_status(item.get("status")))
            incident_dnfs = sum(1 for item in recent if _is_incident_status(item.get("status")))
            teammate_delta = _avg(teammate_deltas.get(driver.id) or [])

            standing_score = (driver.points / max_driver_points) if max_driver_points else 0.0
            finish_score = 0.45 if avg_finish is None else max(0.0, min(1.0, (21 - avg_finish) / 20))
            points_score = min(1.0, recent_points / max(1.0, lookback_races * 25.0))
            podium_score = min(1.0, podiums / max(1.0, min(lookback_races, 4)))
            reliability_score = 1.0 - min(0.7, dnfs / max(1, len(recent)) if recent else 0.15)
            qualifying_pace_score = 0.48 if avg_grid is None else max(0.02, min(1.0, (22 - avg_grid) / 21))
            race_pace_score = max(0.02, min(1.0, finish_score + max(-0.12, min(0.12, avg_grid_delta / 50.0))))
            teammate_score = max(0.02, min(1.0, 0.50 + teammate_delta / 12.0))
            trend_score = _trend_score(current_results or recent)
            form_score = (
                0.34 * finish_score
                + 0.22 * points_score
                + 0.15 * podium_score
                + 0.04 * standing_score
                + 0.13 * qualifying_pace_score
                + 0.12 * trend_score
            )

            driver_features[driver.id] = {
                "starts": len(results),
                "current_season_starts": len(current_results),
                "recent_starts": len(recent),
                "recent_points": round(recent_points, 2),
                "recent_avg_finish": round(avg_finish, 2) if avg_finish is not None else None,
                "recent_avg_grid": round(avg_grid, 2) if avg_grid is not None else None,
                "recent_grid_delta": round(avg_grid_delta, 2),
                "recent_wins": wins,
                "recent_podiums": podiums,
                "recent_dnfs": dnfs,
                "mechanical_dnfs": mechanical_dnfs,
                "incident_dnfs": incident_dnfs,
                "form_score": round(form_score, 4),
                "reliability_score": round(reliability_score, 4),
                "qualifying_pace_score": round(qualifying_pace_score, 4),
                "race_pace_score": round(race_pace_score, 4),
                "teammate_score": round(teammate_score, 4),
                "trend_score": round(trend_score, 4),
                "recent_summary": _recent_driver_summary(recent, avg_finish, podiums, dnfs),
            }

        constructor_features = {}
        for constructor in self._constructors:
            key = constructor.name.lower()
            results = sorted(constructor_results.get(key, []), key=lambda item: item["round"], reverse=True)
            recent = results[:lookback_races * 2]
            positions = [item["position"] for item in recent if isinstance(item.get("position"), int)]
            avg_finish = sum(positions) / len(positions) if positions else None
            recent_points = sum(item.get("points", 0.0) for item in recent)
            standings_score = constructor.points / max_constructor_points
            finish_score = 0.45 if avg_finish is None else max(0.0, min(1.0, (21 - avg_finish) / 20))
            points_score = min(1.0, recent_points / max(1.0, lookback_races * 43.0))
            constructor_features[key] = {
                "team_score": round(0.24 * standings_score + 0.34 * points_score + 0.42 * finish_score, 4),
                "recent_points": round(recent_points, 2),
                "recent_avg_finish": round(avg_finish, 2) if avg_finish is not None else None,
                "recent_starts": len(recent),
                "reliability_score": round(1.0 - min(0.55, sum(1 for item in recent if not _is_finished_status(item.get("status"))) / max(1, len(recent)) if recent else 0.10), 4),
            }

        weather_by_round = {}
        weather_by_round_session = {}
        for race in _weather_candidate_races(self._races):
            race_key = str(race.round)
            weather_by_round[race_key] = await self._weather_client.get_race_weather(race.latitude, race.longitude, race.date)
            session_weather = {}
            for session in race.sessions or []:
                session_code = _weather_session_key(session.get("code") or session.get("name"))
                session_dt = _prediction_weather_session_datetime(session, race.date)
                session_weather[session_code] = await self._weather_client.get_session_weather(
                    race.latitude,
                    race.longitude,
                    session_dt,
                    session=session_code,
                )
            if session_weather:
                weather_by_round_session[race_key] = session_weather

        result = {
            "season": self._season,
            "lookback_races": lookback_races,
            "completed_races": len(races_data),
            "total_races": len(self._races),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "drivers": driver_features,
            "constructors": constructor_features,
            "races": {
                str(race.round): {
                    "round": race.round,
                    "name": race.name,
                    "circuit": race.circuit,
                    "circuit_id": race.circuit_id,
                    "country": race.country,
                    "locality": race.locality,
                    "latitude": race.latitude,
                    "longitude": race.longitude,
                    "date": race.date.isoformat(),
                    "has_sprint": race.has_sprint,
                    "sessions": race.sessions,
                }
                for race in self._races
            },
            "track_history": _track_history_features(track_results),
            "weather_by_round": weather_by_round,
            "weather_by_round_session": weather_by_round_session,
            "source_coverage": {
                "current_season_races": len(current_races),
                "historical_races": len(previous_races),
                "weather_races": len(weather_by_round),
                "weather_sessions": sum(len(items) for items in weather_by_round_session.values()),
            },
        }
        self._prediction_features_cache[cache_key] = result
        return result

    async def _fetch_season_results(self, season: int) -> list[dict]:
        if season in self._season_results_cache:
            return self._season_results_cache[season]
        try:
            resp = await self._client.get(f"/{season}/results.json?limit=2000")
            resp.raise_for_status()
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 season results for %s: %s", season, e)
            races = []
        self._season_results_cache[season] = races
        return races

    async def get_historical_race_results(self, season: int) -> list[dict]:
        """Return raw Jolpica race-result rows for a season.

        This is intentionally raw-ish so internal tools like backtesting can
        replay races without reshaping through live-season dashboard models.
        """
        return await self._fetch_season_results(season)

    async def get_historical_qualifying_results(self, season: int) -> list[dict]:
        """Return raw Jolpica qualifying rows for a season, grouped by race."""

        if season in self._season_qualifying_cache:
            return self._season_qualifying_cache[season]
        try:
            resp = await self._client.get(f"/{season}/qualifying.json?limit=2000")
            resp.raise_for_status()
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 season qualifying for %s: %s", season, e)
            races = []
        self._season_qualifying_cache[season] = races
        return races

    def get_race_by_round(self, round_num: int) -> Race | None:
        return next((r for r in self._races if r.round == round_num), None)

    async def get_race_profile(self, round_num: int, predictor=None) -> dict:
        race = self.get_race_by_round(round_num)
        if not race:
            return {"ok": False, "reason": "Race not found"}

        if race.status == "SCHEDULED" and predictor:
            race.prediction = predictor.predict_race(race)

        round_results = await self._fetch_round_results(round_num, "results")
        qualifying = await self._fetch_round_results(round_num, "qualifying")
        sprint = await self._fetch_round_results(round_num, "sprint")
        sessions = await self._fetch_race_sessions(round_num, race)

        return {
            "ok": True,
            "race": race.model_dump(mode="json"),
            "sessions": sessions,
            "results": round_results,
            "qualifying": qualifying,
            "sprint": sprint,
            "driver_standings": [d.model_dump() for d in self._drivers],
            "constructor_standings": [c.model_dump() for c in self._constructors],
            "context": _race_context(race, sessions, round_results, qualifying, sprint),
        }

    async def get_driver_profile(self, driver_id: str, season: int | None = None) -> dict:
        driver = self._resolve_driver(driver_id)
        if not driver:
            return {"ok": False, "reason": "Driver not found"}

        driver_id = driver.id
        seasons = await self._fetch_driver_seasons(driver_id)
        selected_season = season or self._season
        if seasons and not any(item.get("season") == selected_season for item in seasons):
            return {
                "ok": False,
                "reason": f"Driver did not participate in {selected_season}",
                "driver": driver.model_dump(),
                "seasons": seasons,
            }

        results = await self._fetch_driver_results(driver_id, selected_season)
        stats = _summarize_driver_results(results, selected_season)
        career = _summarize_driver_career(seasons, self._season)

        return {
            "ok": True,
            "driver": driver.model_dump(),
            "seasons": seasons,
            "stats": stats,
            "career": career,
            "results": results,
        }

    def _resolve_driver(self, value: str) -> Driver | None:
        normalized = _slug(value)
        for driver in self._drivers:
            aliases = {
                _slug(driver.id),
                _slug(driver.code),
                _slug(driver.last_name),
                _slug(f"{driver.first_name} {driver.last_name}"),
            }
            if normalized in aliases:
                return driver
        return None

    async def get_constructor_profile(self, constructor_id: str, season: int | None = None) -> dict:
        constructor = next((c for c in self._constructors if c.id == constructor_id), None)
        if not constructor:
            return {"ok": False, "reason": "Constructor not found"}

        seasons = await self._fetch_constructor_seasons(constructor_id)
        selected_season = season or self._season
        if seasons and not any(item.get("season") == selected_season for item in seasons):
            return {
                "ok": False,
                "reason": f"Constructor did not participate in {selected_season}",
                "constructor": constructor.model_dump(),
                "seasons": seasons,
            }

        results = await self._fetch_constructor_results(constructor_id, selected_season)
        stats = _summarize_constructor_results(results, selected_season)
        career = _summarize_constructor_career(seasons, constructor_id, self._season)

        return {
            "ok": True,
            "constructor": constructor.model_dump(),
            "seasons": seasons,
            "stats": stats,
            "career": career,
            "results": results,
        }

    async def _fetch_driver_seasons(self, driver_id: str) -> list[dict]:
        if driver_id in self._driver_seasons_cache:
            return self._driver_seasons_cache[driver_id]
        try:
            races_data = await self._fetch_paginated_races(f"/drivers/{driver_id}/results.json")
            by_season: dict[int, list[dict]] = {}
            for race in races_data:
                season_raw = race.get("season")
                if not str(season_raw).isdigit():
                    continue
                season = int(season_raw)
                result = self._parse_driver_result(race)
                if result:
                    by_season.setdefault(season, []).append(result)

            seasons = []
            for season, results in by_season.items():
                summary = _summarize_driver_results(results, season)
                teams = sorted({result.get("team") for result in results if result.get("team")})
                season_position = await self._fetch_driver_standing_position(driver_id, season)
                seasons.append({
                    "season": season,
                    "position": season_position,
                    "starts": summary["starts"],
                    "points": summary["points"],
                    "wins": summary["wins"],
                    "podiums": summary["podiums"],
                    "best_finish": summary["best_finish"],
                    "average_finish": summary["average_finish"],
                    "teams": teams,
                })
            seasons.sort(key=lambda item: item["season"], reverse=True)
            self._driver_seasons_cache[driver_id] = seasons
            return seasons
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 driver seasons for %s: %s", driver_id, e)
            return []

    async def _fetch_driver_standing_position(self, driver_id: str, season: int) -> int | None:
        try:
            resp = await self._client.get(f"/{season}/drivers/{driver_id}/driverstandings.json")
            resp.raise_for_status()
            data = resp.json()
            standings_lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if not standings_lists:
                return None
            standings = standings_lists[0].get("DriverStandings", [])
            if not standings:
                return None
            position = standings[0].get("position")
            return int(position) if str(position).isdigit() else None
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 standing position for %s in %s: %s", driver_id, season, e)
            return None

    async def _fetch_constructor_seasons(self, constructor_id: str) -> list[dict]:
        if constructor_id in self._constructor_seasons_cache:
            return self._constructor_seasons_cache[constructor_id]
        try:
            races_data = await self._fetch_paginated_races(f"/constructors/{constructor_id}/results.json")
            by_season: dict[int, list[dict]] = {}
            for race in races_data:
                season_raw = race.get("season")
                if not str(season_raw).isdigit():
                    continue
                season = int(season_raw)
                result = self._parse_constructor_result(race)
                if result:
                    by_season.setdefault(season, []).append(result)

            seasons = []
            title_years = set(CONSTRUCTOR_TITLE_FALLBACKS.get(constructor_id, []))
            current_constructor = next((c for c in self._constructors if c.id == constructor_id), None)
            for season, results in by_season.items():
                summary = _summarize_constructor_results(results, season)
                season_position = 1 if season in title_years else None
                if season == self._season and current_constructor:
                    season_position = current_constructor.position
                driver_title = _constructor_driver_title_fallback(constructor_id, season)
                drivers = sorted({
                    entry.get("driver_code") or entry.get("driver_name")
                    for result in results
                    for entry in result.get("entries", [])
                    if entry.get("driver_code") or entry.get("driver_name")
                })
                seasons.append({
                    "season": season,
                    "position": season_position,
                    "races": summary["races"],
                    "entries": summary["entries"],
                    "points": summary["points"],
                    "wins": summary["wins"],
                    "podiums": summary["podiums"],
                    "best_finish": summary["best_finish"],
                    "driver_title": driver_title,
                    "drivers": drivers,
                })
            seasons.sort(key=lambda item: item["season"], reverse=True)
            self._constructor_seasons_cache[constructor_id] = seasons
            return seasons
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 constructor seasons for %s: %s", constructor_id, e)
            return []

    async def _fetch_constructor_standing_position(self, constructor_id: str, season: int) -> int | None:
        try:
            resp = await self._client.get(f"/{season}/constructors/{constructor_id}/constructorstandings.json")
            resp.raise_for_status()
            data = resp.json()
            standings_lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if not standings_lists:
                return None
            standings = standings_lists[0].get("ConstructorStandings", [])
            if not standings:
                return None
            position = standings[0].get("position")
            return int(position) if str(position).isdigit() else None
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 constructor standing position for %s in %s: %s", constructor_id, season, e)
            return None

    async def _fetch_constructor_driver_title(self, constructor_id: str, season: int) -> dict | None:
        cache_key = (constructor_id, season)
        if cache_key in self._constructor_driver_titles_cache:
            return self._constructor_driver_titles_cache[cache_key]
        try:
            resp = await self._client.get(f"/{season}/driverstandings/1.json")
            resp.raise_for_status()
            data = resp.json()
            standings_lists = data.get("MRData", {}).get("StandingsTable", {}).get("StandingsLists", [])
            if not standings_lists:
                self._constructor_driver_titles_cache[cache_key] = None
                return None
            standings = standings_lists[0].get("DriverStandings", [])
            if not standings:
                self._constructor_driver_titles_cache[cache_key] = None
                return None
            champion = standings[0]
            constructors = champion.get("Constructors", [])
            if not any((item.get("constructorId") == constructor_id) for item in constructors):
                self._constructor_driver_titles_cache[cache_key] = None
                return None
            driver = champion.get("Driver", {})
            title = {
                "season": season,
                "driver_id": driver.get("driverId", ""),
                "driver_code": driver.get("code", ""),
                "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
            }
            self._constructor_driver_titles_cache[cache_key] = title
            return title
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 driver title for constructor %s in %s: %s", constructor_id, season, e)
            self._constructor_driver_titles_cache[cache_key] = None
            return None

    async def _fetch_paginated_races(self, path: str) -> list[dict]:
        races: list[dict] = []
        offset = 0
        limit = 100
        while True:
            separator = "&" if "?" in path else "?"
            resp = await self._client.get(f"{path}{separator}limit={limit}&offset={offset}")
            resp.raise_for_status()
            data = resp.json()
            mrdata = data.get("MRData", {})
            batch = mrdata.get("RaceTable", {}).get("Races", [])
            races.extend(batch)

            total = int(mrdata.get("total", len(races)) or len(races))
            returned_limit = int(mrdata.get("limit", limit) or limit)
            returned_offset = int(mrdata.get("offset", offset) or offset)
            next_offset = returned_offset + returned_limit
            if next_offset >= total or not batch:
                break
            offset = next_offset
        return races

    async def _fetch_driver_results(self, driver_id: str, season: int | None = None) -> list[dict]:
        selected_season = season or self._season
        try:
            resp = await self._client.get(f"/{selected_season}/drivers/{driver_id}/results.json?limit=1000")
            resp.raise_for_status()
            data = resp.json()
            races_data = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            results = []
            for race in races_data:
                result = self._parse_driver_result(race)
                if result:
                    results.append(result)
            return sorted(results, key=lambda item: item["round"], reverse=True)
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 driver results for %s in %s: %s", driver_id, selected_season, e)
            return []

    async def _fetch_constructor_results(self, constructor_id: str, season: int | None = None) -> list[dict]:
        selected_season = season or self._season
        try:
            resp = await self._client.get(f"/{selected_season}/constructors/{constructor_id}/results.json?limit=1000")
            resp.raise_for_status()
            data = resp.json()
            races_data = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            results = []
            for race in races_data:
                result = self._parse_constructor_result(race)
                if result:
                    results.append(result)
            return sorted(results, key=lambda item: item["round"], reverse=True)
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 constructor results for %s in %s: %s", constructor_id, selected_season, e)
            return []

    async def _fetch_race_sessions(self, round_num: int, race: Race) -> list[dict]:
        try:
            resp = await self._client.get(f"/{self._season}/{round_num}.json")
            resp.raise_for_status()
            data = resp.json()
            races_data = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            raw = races_data[0] if races_data else {}
        except (httpx.HTTPError, KeyError, ValueError):
            raw = {}

        session_specs = [
            ("fp1", "Practice 1", "FirstPractice", -2, "Free practice baseline and install laps"),
            ("fp2", "Practice 2", "SecondPractice", -2, "Long-run pace and setup direction"),
            ("fp3", "Practice 3", "ThirdPractice", -1, "Final race-trim checks"),
            ("sprint_qualifying", "Sprint Qualifying", "SprintQualifying", -1, "Sprint grid shootout"),
            ("sprint", "Sprint", "Sprint", -1, "Short-form points race"),
            ("qualifying", "Qualifying", "Qualifying", -1, "Grand Prix grid order"),
            ("race", "Race", None, 0, "Full-distance Grand Prix"),
        ]
        race_dt = race.date
        sessions = []
        has_sprint = bool(race.has_sprint or raw.get("Sprint") or raw.get("SprintQualifying") or raw.get("SprintShootout"))
        for code, name, api_key, offset_days, note in session_specs:
            if code in {"sprint", "sprint_qualifying"} and not has_sprint:
                continue
            dt = race_dt
            if api_key and raw.get(api_key):
                session_raw = raw.get(api_key) or {}
                date_str = session_raw.get("date") or race_dt.date().isoformat()
                time_str = (session_raw.get("time") or "14:00:00Z").rstrip("Z")
                try:
                    dt = datetime.fromisoformat(f"{date_str}T{time_str}").replace(tzinfo=timezone.utc)
                except ValueError:
                    dt = race_dt
            elif offset_days:
                dt = race_dt.replace(day=race_dt.day) + _safe_day_delta(offset_days)
            status = "completed" if dt < datetime.now(timezone.utc) else "scheduled"
            sessions.append({
                "code": code,
                "name": name,
                "date": dt.isoformat(),
                "status": status,
                "note": note,
            })
        return sessions

    async def _fetch_round_results(self, round_num: int, result_type: str) -> list[dict]:
        try:
            endpoint = {
                "results": "results",
                "qualifying": "qualifying",
                "sprint": "sprint",
            }.get(result_type, "results")
            resp = await self._client.get(f"/{self._season}/{round_num}/{endpoint}.json?limit=1000")
            resp.raise_for_status()
            data = resp.json()
            races_data = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            if not races_data:
                return []
            race = races_data[0]
            if result_type == "qualifying":
                return _parse_qualifying_results(race)
            if result_type == "sprint":
                return _parse_sprint_results(race)
            return _parse_race_results(race)
        except (httpx.HTTPError, KeyError, ValueError) as e:
            logger.warning("Failed to fetch F1 %s for round %s: %s", result_type, round_num, e)
            return []

    def _parse_driver_result(self, race: dict) -> dict | None:
        result = (race.get("Results") or [{}])[0]
        if not result:
            return None
        constructor = result.get("Constructor") or {}
        grid_raw = result.get("grid")
        position_raw = result.get("position")
        return {
            "season": int(race.get("season", self._season)),
            "round": int(race.get("round", 0)),
            "race_name": race.get("raceName", ""),
            "date": race.get("date"),
            "circuit": (race.get("Circuit") or {}).get("circuitName", ""),
            "country": ((race.get("Circuit") or {}).get("Location") or {}).get("country", ""),
            "team": constructor.get("name", ""),
            "grid": int(grid_raw) if str(grid_raw).lstrip("-").isdigit() else None,
            "position": int(position_raw) if str(position_raw).isdigit() else None,
            "points": float(result.get("points", 0)),
            "status": result.get("status") or "Unknown",
            "time": (result.get("Time") or {}).get("time"),
            "fastest_lap_rank": ((result.get("FastestLap") or {}).get("rank")),
        }

    def _parse_constructor_result(self, race: dict) -> dict | None:
        raw_results = race.get("Results") or []
        entries = []
        for result in raw_results:
            driver = result.get("Driver") or {}
            grid_raw = result.get("grid")
            position_raw = result.get("position")
            entries.append({
                "driver_id": driver.get("driverId", ""),
                "driver_code": driver.get("code", ""),
                "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                "grid": int(grid_raw) if str(grid_raw).lstrip("-").isdigit() else None,
                "position": int(position_raw) if str(position_raw).isdigit() else None,
                "points": float(result.get("points", 0)),
                "status": result.get("status") or "Unknown",
            })
        if not entries:
            return None

        positions = [entry["position"] for entry in entries if isinstance(entry.get("position"), int)]
        return {
            "season": int(race.get("season", self._season)),
            "round": int(race.get("round", 0)),
            "race_name": race.get("raceName", ""),
            "date": race.get("date"),
            "circuit": (race.get("Circuit") or {}).get("circuitName", ""),
            "country": ((race.get("Circuit") or {}).get("Location") or {}).get("country", ""),
            "points": round(sum(entry["points"] for entry in entries), 2),
            "best_finish": min(positions) if positions else None,
            "entries": sorted(entries, key=lambda entry: entry.get("position") or 99),
        }

    async def _resolve_driver_photo(self, driver_id: str, profile_url: str | None, number: int | None, code: str | None) -> str | None:
        if driver_id in self._photo_cache:
            return self._photo_cache[driver_id]
        openf1_photo = await self._resolve_openf1_photo(number, code)
        if openf1_photo:
            self._photo_cache[driver_id] = openf1_photo
            return openf1_photo
        fallback_photo = F1_HEADSHOT_FALLBACKS.get((code or "").upper())
        if fallback_photo:
            self._photo_cache[driver_id] = fallback_photo
            return fallback_photo
        if not profile_url:
            self._photo_cache[driver_id] = None
            return None
        try:
            path = unquote(urlparse(profile_url).path)
            title = path.rsplit("/", 1)[-1]
            if not title:
                self._photo_cache[driver_id] = None
                return None
            resp = await self._client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                headers={"User-Agent": "SportsPredictionsDashboard/0.1 (local development)"},
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            photo = ((data.get("thumbnail") or {}).get("source")) or ((data.get("originalimage") or {}).get("source"))
            self._photo_cache[driver_id] = photo
            return photo
        except Exception:
            self._photo_cache[driver_id] = None
            return None

    async def _resolve_openf1_photo(self, number: int | None, code: str | None) -> str | None:
        if not number and not code:
            return None
        try:
            query = f"name_acronym={code}" if code else f"driver_number={number}"
            resp = await self._client.get(
                f"https://api.openf1.org/v1/drivers?{query}",
                timeout=15.0,
            )
            resp.raise_for_status()
            rows = resp.json()
            rows = [row for row in rows if row.get("headshot_url")]
            if not rows and code and number:
                resp = await self._client.get(
                    f"https://api.openf1.org/v1/drivers?driver_number={number}",
                    timeout=15.0,
                )
                resp.raise_for_status()
                rows = [row for row in resp.json() if row.get("headshot_url")]
            if not rows:
                return None
            rows.sort(key=lambda row: (int(row.get("session_key") or 0), int(row.get("meeting_key") or 0)), reverse=True)
            return _high_quality_f1_image(rows[0].get("headshot_url"))
        except Exception:
            return None

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
        await self._weather_client.close()


def _is_finished_status(status: str | None) -> bool:
    if not status:
        return False
    normalized = status.lower()
    return normalized == "finished" or normalized.startswith("+") or "lap" in normalized


def _is_mechanical_status(status: str | None) -> bool:
    normalized = str(status or "").lower()
    mechanical_tokens = [
        "engine", "gearbox", "hydraulics", "power unit", "powerunit", "electrical",
        "transmission", "brakes", "brake", "suspension", "oil", "water", "fuel",
        "wheel", "puncture", "tyre", "tire", "overheating",
    ]
    return any(token in normalized for token in mechanical_tokens)


def _is_incident_status(status: str | None) -> bool:
    normalized = str(status or "").lower()
    return any(token in normalized for token in ["accident", "collision", "spun", "damage", "crash", "withdrew"])


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _trend_score(results: list[dict]) -> float:
    ordered = sorted(results, key=lambda item: (item.get("season", 0), item.get("round", 0)))
    classified = [item for item in ordered if isinstance(item.get("position"), int)]
    if len(classified) < 2:
        return 0.50
    midpoint = max(1, len(classified) // 2)
    early = classified[:midpoint]
    late = classified[midpoint:]
    early_avg = _avg([float(item["position"]) for item in early])
    late_avg = _avg([float(item["position"]) for item in late])
    return max(0.02, min(1.0, 0.50 + (early_avg - late_avg) / 18.0))


def _track_key_from_raw_race(race: dict) -> str:
    circuit = race.get("Circuit") or {}
    location = circuit.get("Location") or {}
    text = f"{race.get('raceName', '')} {circuit.get('circuitName', '')} {location.get('country', '')}"
    normalized = _slug(text)
    for key in [
        "albertpark", "shanghai", "suzuka", "miami", "bahrain", "jeddah", "monaco",
        "barcelona", "redbullring", "silverstone", "spa", "hungaroring", "zandvoort",
        "monza", "baku", "marinabay", "cota", "mexico", "interlagos", "lasvegas",
        "losail", "yasmarina", "imola",
    ]:
        if key in normalized:
            return key
    if "gilles" in normalized or "canadian" in normalized:
        return "gilles"
    return normalized[:32] or "default"


def _track_history_features(track_results: dict[str, dict[str, list[dict]]]) -> dict[str, dict]:
    rows = {}
    for track_key, by_driver in track_results.items():
        driver_rows = {}
        for driver_id, results in by_driver.items():
            classified = [item for item in results if isinstance(item.get("position"), int)]
            if not classified:
                continue
            points = sum(float(item.get("points") or 0.0) for item in results)
            avg_finish = _avg([float(item["position"]) for item in classified])
            wins = sum(1 for item in classified if item.get("position") == 1)
            podiums = sum(1 for item in classified if item.get("position", 99) <= 3)
            driver_rows[driver_id] = {
                "starts": len(results),
                "avg_finish": round(avg_finish, 2),
                "points": round(points, 2),
                "wins": wins,
                "podiums": podiums,
                "track_score": round(max(0.02, min(1.0, (22 - avg_finish) / 21 + min(0.10, podiums * 0.025))), 4),
            }
        rows[track_key] = {"drivers": driver_rows, "source": "jolpica_multi_season_results", "missing_data": not bool(driver_rows)}
    return rows


def _weather_candidate_races(races: list[Race]) -> list[Race]:
    now = datetime.now(timezone.utc)
    candidates = []
    for race in races:
        days = (race.date - now).days
        if -3 <= days <= 16:
            candidates.append(race)
    if not candidates:
        next_race = next((race for race in races if race.date >= now), None)
        if next_race:
            candidates.append(next_race)
    return candidates[:4]


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_session_datetime(raw: dict, fallback: datetime) -> datetime:
    if not raw:
        return fallback
    date_str = raw.get("date") or fallback.date().isoformat()
    time_str = (raw.get("time") or fallback.time().isoformat()).rstrip("Z")
    try:
        return datetime.fromisoformat(f"{date_str}T{time_str}").replace(tzinfo=timezone.utc)
    except ValueError:
        return fallback


def _prediction_weather_session_datetime(session: dict, fallback: datetime) -> datetime:
    value = (session or {}).get("date")
    if not value:
        return fallback
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return fallback


def _weather_session_key(session: str | None) -> str:
    value = str(session or "race").lower().replace("-", "_").replace(" ", "_")
    if value.startswith("qual"):
        return "qualifying"
    if value.startswith("sprint_qual"):
        return "sprint_qualifying"
    if value.startswith("sprint"):
        return "sprint"
    if value.startswith("fp1") or "practice_1" in value:
        return "fp1"
    if value.startswith("fp2") or "practice_2" in value:
        return "fp2"
    if value.startswith("fp3") or "practice_3" in value:
        return "fp3"
    return "race"


def _parse_calendar_sessions(raw_race: dict, race_dt: datetime) -> list[dict]:
    specs = [
        ("fp1", "Practice 1", "FirstPractice", -2),
        ("fp2", "Practice 2", "SecondPractice", -2),
        ("fp3", "Practice 3", "ThirdPractice", -1),
        ("sprint_qualifying", "Sprint Qualifying", "SprintQualifying", -1),
        ("sprint", "Sprint", "Sprint", -1),
        ("qualifying", "Qualifying", "Qualifying", -1),
        ("race", "Race", None, 0),
    ]
    has_sprint = bool(raw_race.get("Sprint") or raw_race.get("SprintQualifying") or raw_race.get("SprintShootout"))
    sessions = []
    now = datetime.now(timezone.utc)
    for code, name, api_key, offset_days in specs:
        if code in {"sprint", "sprint_qualifying"} and not has_sprint:
            continue
        dt = race_dt + _safe_day_delta(offset_days)
        if api_key and raw_race.get(api_key):
            dt = _parse_session_datetime(raw_race.get(api_key) or {}, race_dt)
        sessions.append({
            "code": code,
            "name": name,
            "date": dt.isoformat(),
            "status": "completed" if dt < now else "scheduled",
        })
    return sessions


def _slug(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _recent_driver_summary(recent: list[dict], avg_finish: float | None, podiums: int, dnfs: int) -> str:
    if not recent:
        return "No current-season race starts yet"
    pieces = [f"{len(recent)} recent starts"]
    if avg_finish is not None:
        pieces.append(f"avg finish P{avg_finish:.1f}")
    if podiums:
        pieces.append(f"{podiums} podium{'s' if podiums != 1 else ''}")
    if dnfs:
        pieces.append(f"{dnfs} DNF{'s' if dnfs != 1 else ''}")
    return ", ".join(pieces)


def _summarize_driver_results(results: list[dict], season: int) -> dict:
    classified = [
        result for result in results
        if isinstance(result.get("position"), int) and result.get("position") > 0
    ]
    finishes = [result["position"] for result in classified]
    points = sum(float(result.get("points") or 0) for result in results)
    wins = sum(1 for result in results if result.get("position") == 1)
    podiums = sum(1 for result in results if isinstance(result.get("position"), int) and result["position"] <= 3)
    dnfs = sum(1 for result in results if not _is_finished_status(result.get("status")))

    return {
        "season": season,
        "starts": len(results),
        "classified_finishes": len(classified),
        "wins": wins,
        "podiums": podiums,
        "points": round(points, 2),
        "dnfs": dnfs,
        "best_finish": min(finishes) if finishes else None,
        "average_finish": round(sum(finishes) / len(finishes), 2) if finishes else None,
        "last_result": results[0] if results else None,
    }


def _summarize_driver_career(seasons: list[dict], current_season: int) -> dict:
    title_years = [
        item.get("season")
        for item in seasons
        if item.get("position") == 1 and item.get("season") and item.get("season") < current_season
    ]
    title_years.sort(reverse=True)
    current = next((item for item in seasons if item.get("season") == current_season), None)
    return {
        "titles": len(title_years),
        "title_years": title_years,
        "current_season": current_season,
        "current_position": current.get("position") if current else None,
        "current_points": current.get("points") if current else None,
        "current_is_leader": bool(current and current.get("position") == 1),
        "current_status": "in_progress",
    }


def _summarize_constructor_career(seasons: list[dict], constructor_id: str, current_season: int) -> dict:
    constructor_title_years = sorted(set(CONSTRUCTOR_TITLE_FALLBACKS.get(constructor_id, []) + [
        item.get("season")
        for item in seasons
        if item.get("position") == 1 and item.get("season") and item.get("season") < current_season
    ]), reverse=True)
    driver_titles = [
        item.get("driver_title")
        for item in seasons
        if item.get("driver_title") and item.get("season") and item.get("season") < current_season
    ]
    fallback_driver_titles = [
        {
            "season": season,
            "driver_id": "",
            "driver_code": code,
            "driver_name": name,
        }
        for season, name, code in CONSTRUCTOR_DRIVER_TITLE_FALLBACKS.get(constructor_id, [])
    ]
    by_season = {item.get("season"): item for item in driver_titles if item.get("season")}
    for item in fallback_driver_titles:
        if item.get("season") and item.get("season") < current_season:
            by_season.setdefault(item.get("season"), item)
    driver_titles = list(by_season.values())
    driver_titles.sort(key=lambda item: item.get("season", 0), reverse=True)
    current = next((item for item in seasons if item.get("season") == current_season), None)
    return {
        "constructor_titles": len(constructor_title_years),
        "constructor_title_years": constructor_title_years,
        "driver_titles": len(driver_titles),
        "driver_title_years": [item.get("season") for item in driver_titles if item.get("season")],
        "driver_title_drivers": driver_titles,
        "current_season": current_season,
        "current_position": current.get("position") if current else None,
        "current_points": current.get("points") if current else None,
        "current_is_leader": bool(current and current.get("position") == 1),
        "current_status": "in_progress",
    }


def _constructor_driver_title_fallback(constructor_id: str, season: int) -> dict | None:
    for title_season, name, code in CONSTRUCTOR_DRIVER_TITLE_FALLBACKS.get(constructor_id, []):
        if title_season == season:
            return {
                "season": season,
                "driver_id": "",
                "driver_code": code,
                "driver_name": name,
            }
    return None


def _safe_day_delta(days: int) -> timedelta:
    return timedelta(days=days)


def _parse_race_results(race: dict) -> list[dict]:
    results = []
    for item in race.get("Results") or []:
        driver = item.get("Driver") or {}
        constructor = item.get("Constructor") or {}
        position_raw = item.get("position")
        grid_raw = item.get("grid")
        results.append({
            "position": int(position_raw) if str(position_raw).isdigit() else None,
            "driver_id": driver.get("driverId", ""),
            "driver_code": driver.get("code", ""),
            "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
            "team": constructor.get("name", ""),
            "grid": int(grid_raw) if str(grid_raw).lstrip("-").isdigit() else None,
            "points": float(item.get("points", 0)),
            "status": item.get("status") or "Unknown",
            "time": (item.get("Time") or {}).get("time"),
            "fastest_lap_rank": ((item.get("FastestLap") or {}).get("rank")),
        })
    return results


def _parse_sprint_results(race: dict) -> list[dict]:
    results = []
    for item in race.get("SprintResults") or []:
        driver = item.get("Driver") or {}
        constructor = item.get("Constructor") or {}
        position_raw = item.get("position")
        grid_raw = item.get("grid")
        results.append({
            "position": int(position_raw) if str(position_raw).isdigit() else None,
            "driver_id": driver.get("driverId", ""),
            "driver_code": driver.get("code", ""),
            "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
            "team": constructor.get("name", ""),
            "grid": int(grid_raw) if str(grid_raw).lstrip("-").isdigit() else None,
            "points": float(item.get("points", 0)),
            "status": item.get("status") or "Unknown",
            "time": (item.get("Time") or {}).get("time"),
        })
    return results


def _parse_qualifying_results(race: dict) -> list[dict]:
    results = []
    for item in race.get("QualifyingResults") or []:
        driver = item.get("Driver") or {}
        constructor = item.get("Constructor") or {}
        position_raw = item.get("position")
        results.append({
            "position": int(position_raw) if str(position_raw).isdigit() else None,
            "driver_id": driver.get("driverId", ""),
            "driver_code": driver.get("code", ""),
            "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
            "team": constructor.get("name", ""),
            "q1": item.get("Q1"),
            "q2": item.get("Q2"),
            "q3": item.get("Q3"),
        })
    return results


def _race_context(race: Race, sessions: list[dict], results: list[dict], qualifying: list[dict], sprint: list[dict]) -> dict:
    completed_sessions = sum(1 for item in sessions if item.get("status") == "completed")
    next_session = next((item for item in sessions if item.get("status") != "completed"), None)
    prediction_count = len((race.prediction.driver_predictions if race.prediction else {}) or {})
    return {
        "has_results": bool(results),
        "has_qualifying": bool(qualifying),
        "has_sprint": bool(sprint) or any(item.get("code") == "sprint" for item in sessions),
        "completed_sessions": completed_sessions,
        "total_sessions": len(sessions),
        "next_session": next_session,
        "prediction_count": prediction_count,
    }


def _summarize_constructor_results(results: list[dict], season: int) -> dict:
    entries = [
        entry for result in results
        for entry in result.get("entries", [])
    ]
    finishes = [
        entry["position"] for entry in entries
        if isinstance(entry.get("position"), int) and entry.get("position") > 0
    ]
    points = sum(float(result.get("points") or 0) for result in results)
    wins = sum(1 for result in results if result.get("best_finish") == 1)
    podiums = sum(1 for entry in entries if isinstance(entry.get("position"), int) and entry["position"] <= 3)

    return {
        "season": season,
        "races": len(results),
        "entries": len(entries),
        "wins": wins,
        "podiums": podiums,
        "points": round(points, 2),
        "best_finish": min(finishes) if finishes else None,
        "average_finish": round(sum(finishes) / len(finishes), 2) if finishes else None,
        "last_result": results[0] if results else None,
    }


def _high_quality_f1_image(url: str | None) -> str | None:
    if not url:
        return None
    return (
        url.replace(".transform/1col/", ".transform/9col/")
        .replace(".transform/2col/", ".transform/9col/")
        .replace(".transform/3col/", ".transform/9col/")
        .replace(".transform/4col/", ".transform/9col/")
        .replace(".transform/5col/", ".transform/9col/")
        .replace(".transform/6col/", ".transform/9col/")
    )
