"""Liquipedia-backed CS2 history (free API).

Pulls finished + upcoming CS2 matches from the Liquipedia v3 API
(https://api.liquipedia.net/api/v3) — a *free* source good for ratings + the
backtest. Requires a free API key (request one from Liquipedia) and, per their
terms, a descriptive User-Agent with contact info and respectful rate limiting
(default one request / 2s). Disabled by default; the factory falls back to the
stub when no key is set.

⚠️ Terms: Liquipedia data is CC-BY-SA and their API is intended for non-commercial
use — fine for research/backtesting, but for a commercial/betting product get their
explicit permission (or use a licensed feed). The field mapping below follows the
documented v3 `match` shape; if a field name differs on the live API, adjust
``_map_match`` — it's the single mapping site.
"""

from __future__ import annotations

import asyncio
import logging
import zlib
from datetime import datetime, timedelta, timezone

import httpx

from games.csgo.data.csgo_client import CsgoDataClient
from games.csgo.identity import build_aliases, normalize
from games.csgo.models.csgo import CsgoMatch, CsgoTeam, MapScore

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.liquipedia.net/api/v3"


def _stable_id(name: str) -> int:
    return (zlib.crc32(normalize(name).encode("utf-8")) % 2_000_000_000) + 1


def _parse_dt(value) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    dt = None
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(str(value).strip(), fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class LiquipediaCsgoClient(CsgoDataClient):
    def __init__(
        self,
        api_key: str,
        *,
        user_agent: str,
        base_url: str = DEFAULT_BASE_URL,
        wiki: str = "counterstrike",
        client: httpx.AsyncClient | None = None,
        lookback_days: int = 180,
        rate_limit_s: float = 2.0,
        page_size: int = 100,
        max_pages: int = 20,
        timeout: float = 20.0,
    ) -> None:
        self._key = api_key
        self._ua = user_agent
        self._base_url = base_url.rstrip("/")
        self._wiki = wiki
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))
        self._owns_client = client is None
        self._lookback_days = lookback_days
        self._rate_limit_s = rate_limit_s
        self._page_size = page_size
        self._max_pages = max_pages

        self._teams: list[CsgoTeam] = []
        self._matches: list[CsgoMatch] = []
        self._past: list[CsgoMatch] = []
        self._loaded = False

    # ── HTTP (rate-limited, ToS-compliant headers) ───────────────────────────
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Apikey {self._key}", "User-Agent": self._ua, "Accept": "application/json"}

    async def _get(self, endpoint: str, conditions: str) -> list[dict]:
        out: list[dict] = []
        for page in range(self._max_pages):
            if self._rate_limit_s > 0 and (page > 0 or out):
                await asyncio.sleep(self._rate_limit_s)
            params = {
                "wiki": self._wiki, "limit": self._page_size, "offset": page * self._page_size,
                "conditions": conditions, "order": "date DESC",
            }
            resp = await self._client.get(f"{self._base_url}/{endpoint}", params=params, headers=self._headers())
            resp.raise_for_status()
            chunk = resp.json().get("result", [])
            if not chunk:
                break
            out.extend(chunk)
            if len(chunk) < self._page_size:
                break
        return out

    # ── mapping ──────────────────────────────────────────────────────────────
    @staticmethod
    def _opponents(raw: dict) -> list[dict]:
        opps = raw.get("match2opponents") or raw.get("opponents") or []
        return [o for o in opps if (o.get("name") or o.get("template"))]

    @staticmethod
    def _games(raw: dict, t1_id: int, t2_id: int, name_to_id: dict) -> list[MapScore]:
        out: list[MapScore] = []
        for i, g in enumerate(raw.get("match2games") or raw.get("games") or [], start=1):
            if not isinstance(g, dict):
                continue
            winner = g.get("winner")
            wid = None
            if winner in (1, "1"):
                wid = t1_id
            elif winner in (2, "2"):
                wid = t2_id
            out.append(MapScore(order=i, map_name=g.get("map") or None,
                                winner_id=wid, status="FINISHED" if winner else "NOT_STARTED"))
        return out

    def _map_match(self, raw: dict) -> CsgoMatch | None:
        opps = self._opponents(raw)
        if len(opps) < 2:
            return None
        n1 = opps[0].get("name") or opps[0].get("template") or ""
        n2 = opps[1].get("name") or opps[1].get("template") or ""
        if not n1 or not n2:
            return None
        t1id, t2id = _stable_id(n1), _stable_id(n2)

        def _score(o):
            try:
                return int(o.get("score"))
            except (TypeError, ValueError):
                return None
        s1, s2 = _score(opps[0]), _score(opps[1])

        when = _parse_dt(raw.get("date"))
        if when is None:
            return None
        bestof = raw.get("bestof")
        best_of = int(bestof) if isinstance(bestof, (int, str)) and str(bestof).isdigit() and int(bestof) > 0 else 3

        winner = raw.get("winner")
        winner_id = t1id if winner in (1, "1") else t2id if winner in (2, "2") else None
        if winner_id is None and isinstance(s1, int) and isinstance(s2, int) and s1 != s2:
            winner_id = t1id if s1 > s2 else t2id
        finished = str(raw.get("finished")) in ("1", "true", "True") or winner_id is not None

        return CsgoMatch(
            id=str(raw.get("match2id") or raw.get("objectname") or f"{t1id}-{t2id}-{when.date()}"),
            team1=n1, team2=n2, team1_id=t1id, team2_id=t2id,
            team1_aliases=build_aliases(n1, None), team2_aliases=build_aliases(n2, None),
            date=when, event=raw.get("tournament") or raw.get("pagename") or None,
            best_of=best_of, status="FINAL" if finished else "SCHEDULED",
            team1_score=s1, team2_score=s2, winner_id=winner_id,
            map_scores=self._games(raw, t1id, t2id, {}),
            source_ids={"liquipedia": str(raw.get("match2id") or "")},
        )

    # ── CsgoDataClient contract ──────────────────────────────────────────────
    async def refresh(self) -> None:
        try:
            since = (datetime.now(timezone.utc) - timedelta(days=self._lookback_days)).strftime("%Y-%m-%d")
            past_raw = await self._get("match", f"[[date::>{since}]] AND [[finished::true]]")
            upcoming_raw = await self._get("match", "[[finished::false]]")

            past = [m for m in (self._map_match(r) for r in past_raw) if m]
            upcoming = [m for m in (self._map_match(r) for r in upcoming_raw) if m]

            teams: dict[int, CsgoTeam] = {}
            for m in past + upcoming:
                for tid, name in ((m.team1_id, m.team1), (m.team2_id, m.team2)):
                    if tid not in teams:
                        teams[tid] = CsgoTeam(id=tid, name=name, abbreviation=name[:4].upper(),
                                              aliases=build_aliases(name, None), source_ids={"liquipedia": str(tid)})

            self._past = sorted(past, key=lambda x: x.date)
            self._matches = upcoming
            self._teams = list(teams.values())
            self._loaded = True
            logger.info("Liquipedia CSGO: %d past, %d upcoming, %d teams", len(past), len(upcoming), len(teams))
        except httpx.HTTPError as exc:
            logger.warning("Liquipedia refresh failed (keeping prior data): %s", exc)

    def get_teams(self) -> list[CsgoTeam]:
        return list(self._teams)

    def get_matches(self) -> list[CsgoMatch]:
        return list(self._matches)

    def get_match(self, match_id: str) -> CsgoMatch | None:
        return next((m for m in (self._matches + self._past) if m.id == match_id), None)

    def get_past_matches(self) -> list[CsgoMatch]:
        return list(self._past)

    def is_available(self) -> bool:
        return bool(self._key) and (self._loaded or not self._past)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
