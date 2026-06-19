"""PandaScore-backed CSGO/CS2 data client.

Implements the :class:`CsgoDataClient` contract against the PandaScore esports
API (https://developers.pandascore.co). One ``refresh()`` pulls real teams plus
upcoming / running / recent matches, and derives a provisional global Elo rating
for every team by replaying recent results — so the baseline predictor produces
real probabilities immediately. Phase 1 refines these into per-map ratings.

Auth: a PandaScore bearer token (``PANDASCORE_TOKEN``). Without it, the factory
falls back to the in-memory stub so the dashboard still boots.

The HTTP client is injectable so tests can drive it with an ``httpx.MockTransport``
and stay off the network.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from games.csgo.analytics.ratings import rate_matches
from games.csgo.data.csgo_client import CsgoDataClient
from games.csgo.identity import build_aliases
from games.csgo.models.csgo import CsgoMatch, CsgoPlayer, CsgoTeam, MapScore

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.pandascore.co"

# PandaScore match.status -> our CsgoMatch.status
_STATUS_MAP = {
    "not_started": "SCHEDULED",
    "running": "LIVE",
    "finished": "FINAL",
    "canceled": "FINAL",
    "postponed": "SCHEDULED",
}


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # PandaScore returns ISO-8601 like "2026-06-08T17:00:00Z".
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _acronym(name: str, acronym: str | None) -> str:
    if acronym:
        return acronym.upper()
    return (name or "")[:4].upper()


_GAME_STATUS_MAP = {
    "not_started": "NOT_STARTED",
    "not_played": "NOT_STARTED",
    "running": "RUNNING",
    "finished": "FINISHED",
}


def _roster_ids(players) -> list[int]:
    out: list[int] = []
    for p in players or []:
        pid = p.get("id") if isinstance(p, dict) else None
        if pid is not None:
            out.append(int(pid))
    return out


def _map_player(raw: dict, team_id: int) -> CsgoPlayer | None:
    pid = raw.get("id")
    if pid is None:
        return None
    real = " ".join(x for x in (raw.get("first_name"), raw.get("last_name")) if x)
    return CsgoPlayer(
        id=int(pid), name=raw.get("name") or "", real_name=real,
        nationality=raw.get("nationality") or "", role=raw.get("role") or "", team_id=team_id,
    )


def _map_games(games, team1_id, team2_id) -> list[MapScore]:
    out: list[MapScore] = []
    for g in games or []:
        if not isinstance(g, dict):
            continue
        winner = g.get("winner") if isinstance(g.get("winner"), dict) else {}
        wid = winner.get("id")
        map_obj = g.get("map") if isinstance(g.get("map"), dict) else {}
        out.append(MapScore(
            order=g.get("position"),
            map_name=map_obj.get("name"),
            winner_id=int(wid) if wid is not None else None,
            status=_GAME_STATUS_MAP.get(str(g.get("status")), "NOT_STARTED"),
        ))
    return out


class PandaScoreCsgoClient(CsgoDataClient):
    """Live CSGO/CS2 data from PandaScore."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        game: str = "csgo",
        client: httpx.AsyncClient | None = None,
        lookback_days: int = 180,
        max_teams: int = 100,
        k_factor: float = 30.0,
        page_size: int = 100,
        max_pages: int = 10,
        timeout: float = 15.0,
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._game = game.strip("/")
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        self._owns_client = client is None
        self._lookback_days = lookback_days
        self._max_teams = max_teams
        self._k = k_factor
        self._page_size = page_size
        self._max_pages = max_pages

        self._teams: list[CsgoTeam] = []
        self._matches: list[CsgoMatch] = []
        self._past: list[CsgoMatch] = []
        self._players: dict[int, CsgoPlayer] = {}
        self._team_players: dict[int, list[int]] = {}
        self._loaded = False

    # ── HTTP plumbing ────────────────────────────────────────────────────────
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}

    async def _get(self, path: str, params: dict | None = None) -> list[dict]:
        resp = await self._client.get(self._base_url + path, params=params or {}, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    async def _paged(self, path: str, params: dict | None = None) -> list[dict]:
        out: list[dict] = []
        base = dict(params or {})
        for page in range(1, self._max_pages + 1):
            chunk = await self._get(path, {**base, "page[size]": self._page_size, "page[number]": page})
            if not chunk:
                break
            out.extend(chunk)
            if len(chunk) < self._page_size or len(out) >= self._max_teams * 50:
                break
        return out

    # ── Mapping ──────────────────────────────────────────────────────────────
    def _map_team(self, raw: dict) -> CsgoTeam | None:
        tid = raw.get("id")
        name = raw.get("name")
        if tid is None or not name:
            return None
        acr = _acronym(name, raw.get("acronym"))
        return CsgoTeam(
            id=int(tid),
            name=name,
            abbreviation=acr,
            region=raw.get("location") or "",
            world_rank=None,
            rating=1500.0,
            aliases=build_aliases(name, acr),
            source_ids={"pandascore": str(tid)},
            roster=_roster_ids(raw.get("players")),
        )

    def _map_match(self, raw: dict) -> CsgoMatch | None:
        opponents = [o.get("opponent") or {} for o in (raw.get("opponents") or [])]
        opponents = [o for o in opponents if o.get("id") is not None]
        if len(opponents) < 2:
            return None  # TBD / single-side entries are not predictable
        o1, o2 = opponents[0], opponents[1]

        results = {r.get("team_id"): r.get("score") for r in (raw.get("results") or [])}
        s1, s2 = results.get(o1.get("id")), results.get(o2.get("id"))

        when = _parse_dt(raw.get("begin_at")) or _parse_dt(raw.get("scheduled_at"))
        if when is None:
            return None

        n_games = raw.get("number_of_games")
        best_of = int(n_games) if isinstance(n_games, int) and n_games > 0 else 3

        serie = raw.get("serie") or {}
        tournament = raw.get("tournament") or {}
        league = raw.get("league") or {}
        event = serie.get("full_name") or tournament.get("name") or league.get("name")
        event_slug = serie.get("slug") or tournament.get("slug") or league.get("slug")

        a1 = _acronym(o1.get("name") or "", o1.get("acronym"))
        a2 = _acronym(o2.get("name") or "", o2.get("acronym"))

        # Winner: explicit winner_id, else infer from the series score.
        winner_id = raw.get("winner_id")
        if winner_id is None and isinstance(s1, int) and isinstance(s2, int) and s1 != s2:
            winner_id = o1.get("id") if s1 > s2 else o2.get("id")
        winner_code = None
        if winner_id == o1.get("id"):
            winner_code = a1
        elif winner_id == o2.get("id"):
            winner_code = a2

        return CsgoMatch(
            id=str(raw.get("id")),
            team1=o1.get("name") or "TBD",
            team2=o2.get("name") or "TBD",
            team1_id=int(o1.get("id")),
            team2_id=int(o2.get("id")),
            team1_abbrev=a1,
            team2_abbrev=a2,
            team1_aliases=build_aliases(o1.get("name"), o1.get("acronym")),
            team2_aliases=build_aliases(o2.get("name"), o2.get("acronym")),
            date=when,
            event=event,
            event_slug=event_slug,
            best_of=best_of,
            status=_STATUS_MAP.get(str(raw.get("status")), "SCHEDULED"),
            team1_score=int(s1) if isinstance(s1, int) else None,
            team2_score=int(s2) if isinstance(s2, int) else None,
            winner_id=int(winner_id) if winner_id is not None else None,
            winner_code=winner_code,
            map_scores=_map_games(raw.get("games"), o1.get("id"), o2.get("id")),
            team1_roster=_roster_ids(o1.get("players")),
            team2_roster=_roster_ids(o2.get("players")),
            source_ids={"pandascore": str(raw.get("id"))},
        )

    # ── CsgoDataClient contract ──────────────────────────────────────────────
    async def refresh(self) -> None:
        try:
            teams_raw = await self._paged(f"/{self._game}/teams", {"sort": "name"})
            upcoming_raw = await self._paged(f"/{self._game}/matches/upcoming", {"sort": "begin_at"})
            running_raw = await self._get(f"/{self._game}/matches/running", {"page[size]": self._page_size})
            past_raw = await self._paged(f"/{self._game}/matches/past", {"sort": "-begin_at"})

            teams = [t for t in (self._map_team(r) for r in teams_raw) if t][: self._max_teams]
            upcoming = [m for m in (self._map_match(r) for r in upcoming_raw) if m]
            running = [m for m in (self._map_match(r) for r in running_raw) if m]
            past = [m for m in (self._map_match(r) for r in past_raw) if m]

            # Fit Glicko-2 ratings from recent results and stamp them onto teams.
            ratings = rate_matches(past)
            by_id = {t.id: t for t in teams}
            for tid, rating in ratings.items():
                if tid in by_id:
                    by_id[tid].rating = round(rating.rating, 1)
                    by_id[tid].rating_deviation = round(rating.rd, 1)

            players: dict[int, CsgoPlayer] = {}
            team_players: dict[int, list[int]] = {}
            for raw in teams_raw:
                tid = raw.get("id")
                if tid is None:
                    continue
                ids: list[int] = []
                for p in raw.get("players") or []:
                    cp = _map_player(p, int(tid))
                    if cp:
                        players[cp.id] = cp
                        ids.append(cp.id)
                if ids:
                    team_players[int(tid)] = ids

            self._teams = teams
            self._matches = running + upcoming  # live first, then scheduled
            self._past = past
            self._players = players
            self._team_players = team_players
            self._loaded = True
            logger.info(
                "PandaScore CSGO: %d teams, %d upcoming, %d live, %d past results",
                len(teams), len(upcoming), len(running), len(past),
            )
        except httpx.HTTPError as exc:
            logger.warning("PandaScore CSGO refresh failed (keeping prior data): %s", exc)

    def get_teams(self) -> list[CsgoTeam]:
        return list(self._teams)

    def get_matches(self) -> list[CsgoMatch]:
        return list(self._matches)

    def get_match(self, match_id: str) -> CsgoMatch | None:
        return next((m for m in (self._matches + self._past) if m.id == match_id), None)

    def get_past_matches(self) -> list[CsgoMatch]:
        """Recent finished matches with scores — training data for Phase 1."""
        return list(self._past)

    def get_players(self, team_id: int) -> list[CsgoPlayer]:
        return [self._players[pid] for pid in self._team_players.get(team_id, []) if pid in self._players]

    def get_player(self, player_id: int) -> CsgoPlayer | None:
        return self._players.get(player_id)

    def is_available(self) -> bool:
        return bool(self._token) and (self._loaded or not self._teams)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
