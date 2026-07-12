"""Per-game pitcher-usage ingestion from the MLB Stats API boxscore.

The single biggest un-built baseball signal after the starting pitcher is **bullpen
fatigue**: relievers who threw a lot of pitches over the prior day or three are less
available / less effective tonight. To build a REAL fatigue signal (not a schedule
proxy) we need actual per-game reliever workload, which lives in the boxscore's
per-pitcher ``pitchesThrown`` fields.

This module fetches the finished-game boxscore
(``https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore``), and for each team
extracts, per pitcher, how many pitches were thrown and whether that pitcher was a
RELIEVER (not the game's starter). The starter is the first pitcher listed for the
team AND has ``gamesStarted == 1`` in the game line; everyone else is a reliever.

Design mirrors ``weather_client.season_weather_by_game``:

  * Everything fails soft — a game that errors or is unparseable simply yields no
    usage entry, so downstream fatigue degrades to neutral for that game.
  * Each game's parsed usage is cached to disk under
    ``artifacts/baseball_boxscore/{gamePk}.json`` (gitignored) so the ~2600-game
    cold fetch happens once and every re-run is instant.
  * Fetching is concurrent but BOUNDED (a semaphore) so we don't hammer StatsAPI.

HTTP uses ``common.data.http.make_async_client`` (OS trust store — required on the
corporate MITM machine).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from common.data.http import make_async_client

logger = logging.getLogger(__name__)

BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"

# Bounded concurrency so a full-season backfill (~2600 games) does not overwhelm the
# public StatsAPI or the local connection pool.
DEFAULT_CONCURRENCY = 12

_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts", "baseball_boxscore")
)


def _cache_path(game_pk: int) -> str:
    return os.path.join(_CACHE_DIR, f"{game_pk}.json")


def _as_int(value) -> Optional[int]:
    """Best-effort int (StatsAPI pitch counts are ints but be defensive)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pitches_for(pitching: dict) -> Optional[int]:
    """Pitches thrown for one pitcher's game line, with documented fallbacks.

    Prefer ``pitchesThrown``; fall back to ``numberOfPitches`` (same value in modern
    lines) and finally to ``battersFaced`` (a coarse workload proxy when pitch counts
    are absent, e.g. very old games). ``None`` when nothing usable is present."""
    if not isinstance(pitching, dict):
        return None
    for key in ("pitchesThrown", "numberOfPitches"):
        v = _as_int(pitching.get(key))
        if v is not None:
            return v
    # No pitch count at all — battersFaced is a rough stand-in (~4 pitches/batter, but
    # we keep it in "pitches-ish" units and let the fatigue module treat it uniformly).
    return _as_int(pitching.get("battersFaced"))


def _is_reliever(idx: int, pitching: dict) -> bool:
    """A pitcher is a reliever if they did NOT start this game. The starter is the
    first pitcher listed (``idx == 0``) and carries ``gamesStarted == 1`` in the game
    line; everyone else is a reliever. We require BOTH signals to agree for idx 0 so a
    malformed line doesn't silently misclassify the starter as a reliever."""
    gs = _as_int(pitching.get("gamesStarted"))
    if idx == 0 and gs == 1:
        return False
    # Defensive: if a later pitcher somehow reports gamesStarted==1 (openers / data
    # quirks), still treat only the very first listed pitcher as the starter.
    if idx == 0:
        return False
    return True


def _lineup_for(team: dict) -> list:
    """The 9 STARTER batter player-ids actually in the lineup, from the boxscore's
    ``battingOrder`` (StatsAPI lists exactly the 9 starters in batting-order sequence;
    pinch hitters / substitutions are NOT in this list). Null-safe: returns ``[]`` when
    absent so a partial block still parses. Ids are coerced to int and de-duplicated in
    order (defensive against malformed payloads)."""
    if not isinstance(team, dict):
        return []
    order = team.get("battingOrder") or []
    if not isinstance(order, (list, tuple)):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for pid in order:
        iv = _as_int(pid)
        if iv is None or iv in seen:
            continue
        seen.add(iv)
        out.append(iv)
    return out


def _parse_team(team: dict) -> dict:
    """One team block from the boxscore → ``{"pitches": int, "relievers": int,
    "lineup": [ids]}``. ``pitches``/``relievers`` count only RELIEVER pitches / distinct
    relievers (what ``bullpen_fatigue`` reads); ``lineup`` is the 9 starting batters in
    order (what ``lineup_strength`` reads). Null-safe: missing players / stats are skipped
    rather than raising, so a partial block still yields a usable (smaller) count."""
    if not isinstance(team, dict):
        return {"pitches": 0, "relievers": 0, "lineup": []}
    pitcher_ids = team.get("pitchers") or []
    players = team.get("players") or {}
    total_pitches = 0
    relievers = 0
    for idx, pid in enumerate(pitcher_ids):
        player = players.get(f"ID{pid}") if isinstance(players, dict) else None
        pitching = ((player or {}).get("stats") or {}).get("pitching") or {}
        if not _is_reliever(idx, pitching):
            continue
        pitches = _pitches_for(pitching)
        if pitches is None:
            continue
        total_pitches += pitches
        relievers += 1
    return {"pitches": total_pitches, "relievers": relievers, "lineup": _lineup_for(team)}


def parse_boxscore(data: dict) -> Optional[dict]:
    """Full boxscore JSON → ``{"home": {...}, "away": {...}}`` per-team usage, or ``None``
    if the payload has no usable teams block. Each side carries reliever ``pitches`` /
    ``relievers`` (bullpen-fatigue signal) AND the 9-starter ``lineup`` (lineup-strength
    signal). Pure / null-safe so it is easy to unit-test without the network."""
    teams = (data or {}).get("teams") if isinstance(data, dict) else None
    if not isinstance(teams, dict) or "home" not in teams or "away" not in teams:
        return None
    return {"home": _parse_team(teams.get("home")), "away": _parse_team(teams.get("away"))}


def _has_lineup(parsed: Optional[dict]) -> bool:
    """A parsed entry is CURRENT-SCHEMA iff both sides carry a ``lineup`` key. Older
    cache files (written before lineups were captured) lack it and must be re-fetched so
    lineups actually populate — a missing key, not a stale value, is the version marker."""
    if not isinstance(parsed, dict):
        return False
    for side in ("home", "away"):
        team = parsed.get(side)
        if not isinstance(team, dict) or "lineup" not in team:
            return False
    return True


def _load_cached(game_pk: int) -> Optional[dict]:
    """Return a cached parsed entry ONLY if it is the current schema (includes lineups).
    A pre-lineup cache file returns ``None`` here so the caller re-fetches it once and
    rewrites it in the new shape — bullpen-only fields are preserved by the re-parse."""
    path = _cache_path(game_pk)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            parsed = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not _has_lineup(parsed):
        return None                                    # old shape → force a re-fetch
    return parsed


def _store_cache(game_pk: int, parsed: dict) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_cache_path(game_pk), "w", encoding="utf-8") as fh:
            json.dump(parsed, fh)
    except OSError as exc:                              # pragma: no cover - disk edge
        logger.debug("could not cache boxscore for %s: %s", game_pk, exc)


async def _fetch_one(http, sem: asyncio.Semaphore, game_pk: int, *,
                     use_cache: bool) -> Optional[dict]:
    """Cached-or-fetched parsed usage for one game. Returns ``None`` on any failure so
    the caller simply omits that game (fatigue degrades to neutral)."""
    if use_cache:
        cached = _load_cached(game_pk)
        if cached is not None:
            return cached
    async with sem:
        try:
            resp = await http.get(BOXSCORE_URL.format(game_pk=game_pk))
            resp.raise_for_status()
            parsed = parse_boxscore(resp.json())
        except Exception as exc:                        # noqa: BLE001 — fail soft by design
            logger.debug("boxscore fetch failed for %s: %s", game_pk, exc)
            return None
    if parsed is None:
        return None
    _store_cache(game_pk, parsed)
    return parsed


async def season_bullpen_usage(games, *, concurrency: int = DEFAULT_CONCURRENCY,
                               timeout: float = 30.0, use_cache: bool = True) -> dict[int, dict]:
    """{gamePk: {"home": {pitches, relievers}, "away": {...}}} for finished games.

    Fetches each game's boxscore concurrently (bounded by ``concurrency``) and caches
    every parsed result to disk, so the expensive cold pass (~2600 games) runs once and
    subsequent calls resolve entirely from cache. Only games that look FINISHED (both
    scores present) are fetched; a game whose boxscore errors is simply absent from the
    returned map. Leak-freeness is the fatigue module's job — this only reports raw
    per-game usage."""
    finished = []
    seen: set[int] = set()
    for g in games:
        gid = getattr(g, "id", None)
        if gid is None or gid in seen:
            continue
        if getattr(g, "home_score", None) is None or getattr(g, "away_score", None) is None:
            continue
        seen.add(gid)
        finished.append(gid)

    out: dict[int, dict] = {}
    sem = asyncio.Semaphore(max(1, concurrency))
    async with make_async_client(timeout=timeout) as http:
        tasks = [_fetch_one(http, sem, gid, use_cache=use_cache) for gid in finished]
        for gid, parsed in zip(finished, await asyncio.gather(*tasks)):
            if parsed is not None:
                out[gid] = parsed
    return out
