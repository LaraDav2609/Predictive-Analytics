"""First-pitch weather + static park run-environment factors for MLB venues.

Run scoring in baseball is strongly modulated by the *run environment*: altitude
and air density (Coors plays huge, Petco plays small), temperature (warm air carries
the ball), and wind (blowing out vs in relative to the park's orientation). This
module supplies two optional, leak-free signals the game model can use:

  1. ``PARK_FACTORS`` — a static, documented run factor per venue (~0.90 pitcher-
     friendly … ~1.15+ hitter-friendly). Approximate, multi-year public averages;
     not a live feed.
  2. ``WeatherClient.first_pitch_weather(...)`` — temperature / wind speed / wind
     direction at first pitch from **Open-Meteo** (FREE, no API key), joined to a
     static venue lat/long + home-plate→center-field azimuth table so a caller can
     resolve wind into an out-to-CF / in-from-CF component.

Both are OPTIONAL and fail soft: any network / parse error yields ``None`` so the
model degrades gracefully to its park-neutral behaviour. HTTP uses
``common.data.http.make_async_client`` (OS trust store — required on the corporate
MITM machine).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Optional

from common.data.http import make_async_client

logger = logging.getLogger(__name__)

# Open-Meteo: free, no key. Archive (historical, for backtests) + forecast (live).
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# ─────────────────────────── static park tables ───────────────────────────
# Static run factor per venue (park factor): >1 favours hitters/runs, <1 favours
# pitchers. APPROXIMATE multi-year public averages (Statcast / ESPN / Baseball-
# Reference park factors hover in this range); intended as a small bounded prior,
# not a precise live number. Keyed by BOTH the MLBClient full team name (canonical)
# and the venue name so either can look it up.
PARK_FACTORS: dict[str, float] = {
    # Extreme hitter park (altitude).
    "colorado rockies": 1.15, "coors field": 1.15,
    # Hitter-leaning.
    "cincinnati reds": 1.08, "great american ball park": 1.08,
    "boston red sox": 1.06, "fenway park": 1.06,
    "texas rangers": 1.05, "globe life field": 1.05,
    "kansas city royals": 1.04, "kauffman stadium": 1.04,
    "baltimore orioles": 1.03, "oriole park at camden yards": 1.03,
    "arizona diamondbacks": 1.03, "chase field": 1.03,
    "philadelphia phillies": 1.03, "citizens bank park": 1.03,
    "chicago cubs": 1.02, "wrigley field": 1.02,
    "atlanta braves": 1.02, "truist park": 1.02,
    "toronto blue jays": 1.02, "rogers centre": 1.02,
    # Roughly neutral.
    "washington nationals": 1.01, "nationals park": 1.01,
    "minnesota twins": 1.00, "target field": 1.00,
    "houston astros": 1.00, "minute maid park": 1.00, "daikin park": 1.00,
    "milwaukee brewers": 1.00, "american family field": 1.00,
    "chicago white sox": 1.00, "guaranteed rate field": 1.00, "rate field": 1.00,
    "los angeles angels": 0.99, "angel stadium": 0.99,
    "st. louis cardinals": 0.99, "busch stadium": 0.99,
    "pittsburgh pirates": 0.99, "pnc park": 0.99,
    "new york yankees": 0.99, "yankee stadium": 0.99,
    "los angeles dodgers": 0.98, "dodger stadium": 0.98,
    "new york mets": 0.97, "citi field": 0.97,
    "cleveland guardians": 0.97, "progressive field": 0.97,
    "detroit tigers": 0.97, "comerica park": 0.97,
    "tampa bay rays": 0.97, "tropicana field": 0.97,
    "miami marlins": 0.96, "loandepot park": 0.96,
    # Pitcher-leaning.
    "oakland athletics": 0.95, "oakland coliseum": 0.95, "sutter health park": 0.98,
    "seattle mariners": 0.95, "t-mobile park": 0.95,
    "san francisco giants": 0.94, "oracle park": 0.94,
    # Extreme pitcher park.
    "san diego padres": 0.92, "petco park": 0.92,
}

NEUTRAL_PARK_FACTOR = 1.00


# Venue lat/long + park orientation azimuth (home-plate → center-field, degrees from
# true north). APPROXIMATE — coordinates rounded to the ballpark, azimuths are rough
# public figures; good enough to resolve wind into an out-to-CF component but not
# survey-grade. Keyed by canonical team name.
VENUES: dict[str, dict] = {
    "arizona diamondbacks": {"venue": "Chase Field", "lat": 33.4455, "lon": -112.0667, "cf_azimuth": 0},
    "atlanta braves": {"venue": "Truist Park", "lat": 33.8907, "lon": -84.4677, "cf_azimuth": 25},
    "baltimore orioles": {"venue": "Oriole Park at Camden Yards", "lat": 39.2839, "lon": -76.6217, "cf_azimuth": 30},
    "boston red sox": {"venue": "Fenway Park", "lat": 42.3467, "lon": -71.0972, "cf_azimuth": 45},
    "chicago cubs": {"venue": "Wrigley Field", "lat": 41.9484, "lon": -87.6553, "cf_azimuth": 30},
    "chicago white sox": {"venue": "Rate Field", "lat": 41.8300, "lon": -87.6339, "cf_azimuth": 130},
    "cincinnati reds": {"venue": "Great American Ball Park", "lat": 39.0975, "lon": -84.5069, "cf_azimuth": 120},
    "cleveland guardians": {"venue": "Progressive Field", "lat": 41.4962, "lon": -81.6852, "cf_azimuth": 0},
    "colorado rockies": {"venue": "Coors Field", "lat": 39.7559, "lon": -104.9942, "cf_azimuth": 0},
    "detroit tigers": {"venue": "Comerica Park", "lat": 42.3390, "lon": -83.0485, "cf_azimuth": 150},
    "houston astros": {"venue": "Daikin Park", "lat": 29.7573, "lon": -95.3555, "cf_azimuth": 345},
    "kansas city royals": {"venue": "Kauffman Stadium", "lat": 39.0517, "lon": -94.4803, "cf_azimuth": 0},
    "los angeles angels": {"venue": "Angel Stadium", "lat": 33.8003, "lon": -117.8827, "cf_azimuth": 40},
    "los angeles dodgers": {"venue": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400, "cf_azimuth": 25},
    "miami marlins": {"venue": "loanDepot Park", "lat": 25.7781, "lon": -80.2197, "cf_azimuth": 40},
    "milwaukee brewers": {"venue": "American Family Field", "lat": 43.0280, "lon": -87.9712, "cf_azimuth": 0},
    "minnesota twins": {"venue": "Target Field", "lat": 44.9817, "lon": -93.2776, "cf_azimuth": 90},
    "new york mets": {"venue": "Citi Field", "lat": 40.7571, "lon": -73.8458, "cf_azimuth": 25},
    "new york yankees": {"venue": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262, "cf_azimuth": 15},
    "oakland athletics": {"venue": "Sutter Health Park", "lat": 38.5804, "lon": -121.5133, "cf_azimuth": 60},
    "philadelphia phillies": {"venue": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665, "cf_azimuth": 15},
    "pittsburgh pirates": {"venue": "PNC Park", "lat": 40.4469, "lon": -80.0057, "cf_azimuth": 120},
    "san diego padres": {"venue": "Petco Park", "lat": 32.7073, "lon": -117.1566, "cf_azimuth": 0},
    "san francisco giants": {"venue": "Oracle Park", "lat": 37.7786, "lon": -122.3893, "cf_azimuth": 90},
    "seattle mariners": {"venue": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325, "cf_azimuth": 0},
    "st. louis cardinals": {"venue": "Busch Stadium", "lat": 38.6226, "lon": -90.1928, "cf_azimuth": 60},
    "tampa bay rays": {"venue": "Tropicana Field", "lat": 27.7683, "lon": -82.6534, "cf_azimuth": 0},
    "texas rangers": {"venue": "Globe Life Field", "lat": 32.7473, "lon": -97.0838, "cf_azimuth": 0},
    "toronto blue jays": {"venue": "Rogers Centre", "lat": 43.6414, "lon": -79.3894, "cf_azimuth": 0},
    "washington nationals": {"venue": "Nationals Park", "lat": 38.8730, "lon": -77.0074, "cf_azimuth": 30},
}

# Domed / retractable-roof parks where surface wind rarely reaches the field —
# wind-out signal is meaningless; the caller should ignore wind for these.
DOMED_VENUES = {
    "arizona diamondbacks", "houston astros", "miami marlins", "milwaukee brewers",
    "tampa bay rays", "texas rangers", "toronto blue jays", "seattle mariners",
}


# ─────────────────────────── lookups ───────────────────────────
def _canon(key: str) -> Optional[str]:
    """Resolve a team-name / venue-name to a canonical key present in the tables."""
    if not key:
        return None
    from sports.baseball.analytics.odds_ingest import normalize_team
    s = key.strip().lower()
    if s in VENUES:
        return s
    canon = normalize_team(key)
    if canon and canon in VENUES:
        return canon
    return None


def park_factor(team_or_venue: str) -> float:
    """Static run factor for a team OR venue name. Returns ``NEUTRAL_PARK_FACTOR``
    (1.00) for anything unknown — never raises."""
    if not team_or_venue:
        return NEUTRAL_PARK_FACTOR
    s = team_or_venue.strip().lower()
    if s in PARK_FACTORS:
        return PARK_FACTORS[s]
    canon = _canon(team_or_venue)
    if canon and canon in PARK_FACTORS:
        return PARK_FACTORS[canon]
    return NEUTRAL_PARK_FACTOR


def venue_info(team_or_venue: str) -> Optional[dict]:
    """{venue, lat, lon, cf_azimuth, domed} for a team/venue, or ``None`` if unknown."""
    canon = _canon(team_or_venue)
    if not canon:
        return None
    info = dict(VENUES[canon])
    info["domed"] = canon in DOMED_VENUES
    info["park_factor"] = PARK_FACTORS.get(canon, NEUTRAL_PARK_FACTOR)
    return info


class WeatherClient:
    """Fetches first-pitch weather from Open-Meteo (free, no key). Fail-soft: returns
    ``None`` on any error so the model degrades to park-neutral."""

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout

    async def first_pitch_weather(self, team_or_venue: str, when) -> Optional[dict]:
        """Temperature (°F), wind speed (mph) + direction (deg) near first pitch.

        ``when`` is a date/datetime; the hour is used if present (else 19:00 local ~
        a typical first pitch). Uses the archive endpoint for past dates and the
        forecast endpoint otherwise. Returns ``None`` on unknown venue or any
        failure — the model treats missing weather as neutral."""
        info = venue_info(team_or_venue)
        if not info:
            return None
        hour = when.hour if isinstance(when, datetime) and when.hour else 19
        day = when.date() if isinstance(when, datetime) else when
        if not isinstance(day, date):
            return None
        historical = day < date.today()
        url = ARCHIVE_URL if historical else FORECAST_URL
        params = {
            "latitude": info["lat"], "longitude": info["lon"],
            "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "timezone": "auto",
            "start_date": day.isoformat(), "end_date": day.isoformat(),
        }
        try:
            async with make_async_client(timeout=self._timeout) as http:
                resp = await http.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                return None
            idx = min(range(len(times)),
                      key=lambda i: abs(_hour_of(times[i]) - hour))
            return {
                "venue": info["venue"],
                "temperature_f": _at(hourly.get("temperature_2m"), idx),
                "wind_speed_mph": _at(hourly.get("wind_speed_10m"), idx),
                "wind_direction_deg": _at(hourly.get("wind_direction_10m"), idx),
                "cf_azimuth": info["cf_azimuth"],
                "domed": info["domed"],
                "source": "open-meteo",
            }
        except Exception as exc:                       # noqa: BLE001 — fail soft by design
            logger.debug("weather fetch failed for %s: %s", team_or_venue, exc)
            return None


def _hour_of(iso: str) -> int:
    try:
        return int(iso[11:13])
    except (ValueError, IndexError):
        return 0


def _at(seq, idx):
    try:
        return seq[idx]
    except (TypeError, IndexError):
        return None


def wind_out_component(wind_speed_mph, wind_direction_deg, cf_azimuth) -> Optional[float]:
    """Signed wind-toward-center-field speed (mph): + = blowing out (helps hitters),
    - = blowing in. ``wind_direction_deg`` is the direction the wind blows FROM
    (meteorological convention); the vector blows TOWARD ``dir+180``. Returns ``None``
    if any input is missing."""
    import math
    if wind_speed_mph is None or wind_direction_deg is None or cf_azimuth is None:
        return None
    blow_toward = (float(wind_direction_deg) + 180.0) % 360.0
    theta = math.radians(blow_toward - float(cf_azimuth))
    return float(wind_speed_mph) * math.cos(theta)


# ─────────────────── efficient per-venue seasonal weather (backtest) ───────────────────
# Naively calling ``first_pitch_weather`` once per game is ~2600 Open-Meteo requests for a
# full season. Instead we fetch ONE archive request per VENUE covering the whole season's
# hourly temperature + wind, cache the parsed hour grid to disk, then resolve each game's
# first-pitch weather with a pure-Python local lookup. ~30 venue calls total.
#
# The archive is requested in ``timezone=UTC`` so the hour index is unambiguous: MLB game
# datetimes are UTC (statsapi ``gameDate`` ends in Z), and matching UTC-hour to UTC-hour
# avoids any DST / venue-timezone guessing. Everything fails soft — a venue that errors
# simply contributes no weather for its games (the model falls back to park-only).

_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts", "baseball_weather")
)


def _venue_cache_path(canon: str, season: int) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in canon)
    return os.path.join(_CACHE_DIR, f"{safe}_{season}.json")


def _parse_hour_grid(hourly: dict) -> dict[str, dict]:
    """Open-Meteo hourly arrays → {"YYYY-MM-DDTHH": {temperature_f, wind_speed_mph,
    wind_direction_deg}}. Null-safe: missing/short arrays just yield fewer keys."""
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    winds = hourly.get("wind_speed_10m") or []
    dirs = hourly.get("wind_direction_10m") or []
    grid: dict[str, dict] = {}
    for i, t in enumerate(times):
        if not isinstance(t, str) or len(t) < 13:
            continue
        grid[t[:13]] = {                       # key to the hour: "2023-04-01T19"
            "temperature_f": _at(temps, i),
            "wind_speed_mph": _at(winds, i),
            "wind_direction_deg": _at(dirs, i),
        }
    return grid


async def _fetch_venue_season(canon: str, info: dict, season: int, *,
                              timeout: float, use_cache: bool) -> dict[str, dict]:
    """Hour grid for ONE venue across the season (cached to disk). Returns {} on any
    failure so the caller degrades to park-only for that venue's games."""
    path = _venue_cache_path(canon, season)
    if use_cache and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass
    # Regular MLB season spans ~late March → early November.
    start, end = f"{season}-03-01", f"{season}-11-30"
    params = {
        "latitude": info["lat"], "longitude": info["lon"],
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "UTC",
        "start_date": start, "end_date": end,
    }
    try:
        async with make_async_client(timeout=timeout) as http:
            resp = await http.get(ARCHIVE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        grid = _parse_hour_grid(data.get("hourly", {}))
    except Exception as exc:                       # noqa: BLE001 — fail soft by design
        logger.debug("season weather fetch failed for %s %s: %s", canon, season, exc)
        return {}
    if grid:
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(grid, fh)
        except OSError as exc:
            logger.debug("could not cache weather for %s: %s", canon, exc)
    return grid


def _lookup_hour(grid: dict[str, dict], when: datetime) -> Optional[dict]:
    """Nearest-hour weather (within ±3h) for a UTC first-pitch datetime. Returns None
    if the grid has nothing near that hour."""
    if not grid or not isinstance(when, datetime):
        return None
    dt = when.astimezone(timezone.utc) if when.tzinfo else when.replace(tzinfo=timezone.utc)
    key = dt.strftime("%Y-%m-%dT%H")
    if key in grid:
        return grid[key]
    # Fall back to the closest available hour on the same date (handles rare gaps).
    day = dt.strftime("%Y-%m-%d")
    best, best_gap = None, 4
    for h in range(24):
        k = f"{day}T{h:02d}"
        if k in grid:
            gap = abs(h - dt.hour)
            if gap < best_gap:
                best, best_gap = grid[k], gap
    return best


async def season_weather_by_game(games, season: int, *, timeout: float = 30.0,
                                 use_cache: bool = True) -> dict[int, dict]:
    """{game_id: weather_dict} for a full season using ~one Open-Meteo request per
    venue (NOT one per game). Leak-free (historical archive) and disk-cached so re-runs
    are instant. Each weather_dict matches ``first_pitch_weather``'s shape so it drops
    straight into ``replay(..., weather_by_game=...)`` / ``predict_game(weather=...)``.

    A game whose venue is unknown, or whose venue fetch failed, simply gets no entry —
    the model then applies park-only (or fully neutral) run environment for it."""
    # Group games by the home team's canonical venue key (the park they are played in).
    by_canon: dict[str, list] = {}
    for g in games:
        home = getattr(g, "home_team", None)
        canon = _canon(getattr(g, "venue", None) or home) or _canon(home)
        if canon:
            by_canon.setdefault(canon, []).append(g)

    out: dict[int, dict] = {}
    for canon, canon_games in by_canon.items():
        info = venue_info(canon)
        if not info:
            continue
        grid = await _fetch_venue_season(canon, info, season, timeout=timeout, use_cache=use_cache)
        if not grid:
            continue
        for g in canon_games:
            wx = _lookup_hour(grid, getattr(g, "date", None))
            if wx is None:
                continue
            out[g.id] = {
                "venue": info["venue"],
                "temperature_f": wx.get("temperature_f"),
                "wind_speed_mph": wx.get("wind_speed_mph"),
                "wind_direction_deg": wx.get("wind_direction_deg"),
                "cf_azimuth": info["cf_azimuth"],
                "domed": info["domed"],
                "source": "open-meteo-archive",
            }
    return out
