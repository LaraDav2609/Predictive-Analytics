"""Local-file CS2 history backfill — read results from a user-supplied export.

A zero-cost, zero-scraping way to run real ratings + backtests before paying for a
data API: you export CS2 match results from HLTV / Bo3.gg / a Kaggle dump / a hand
CSV, drop the file in, and this adapter turns it into past matches + teams. Our code
never scrapes — it only reads the file you provide.

Enable with ``CSGO_DATA_PROVIDER=localfile`` + ``CSGO_HISTORY_FILE=/path/to/file``.

CSV schema (header row, case-insensitive; extra columns ignored):
    date, event, team1, team2, best_of, team1_score, team2_score[, winner, team1_id, team2_id, id]
  - date: YYYY-MM-DD or ISO-8601
  - team1_score/team2_score: maps won (series score); winner inferred if absent
  - team ids: optional — derived stably from the team name (crc32) when missing

JSON schema: a list of match objects (or {"matches": [...]}) matching the CsgoMatch
model — use this when you also have per-map results (map_scores) or rosters.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import zlib
from datetime import datetime, timezone

from games.csgo.data.csgo_client import CsgoDataClient
from games.csgo.identity import build_aliases, normalize
from games.csgo.models.csgo import CsgoMatch, CsgoTeam

logger = logging.getLogger(__name__)


def _stable_team_id(raw_id, name: str) -> int:
    if raw_id not in (None, ""):
        try:
            return int(raw_id)
        except (TypeError, ValueError):
            pass
    return (zlib.crc32(normalize(name).encode("utf-8")) % 2_000_000_000) + 1


def _int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_dt(value) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return None


class LocalHistoryCsgoClient(CsgoDataClient):
    def __init__(self, path: str) -> None:
        self._path = path
        self._teams: list[CsgoTeam] = []
        self._matches: list[CsgoMatch] = []   # upcoming/scheduled (usually empty for a backfill)
        self._past: list[CsgoMatch] = []      # finished, with winners
        self._loaded = False
        self._load()

    # ── loading ──────────────────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            logger.warning("CSGO history file not readable (%s): %s", self._path, exc)
            return
        try:
            matches = self._parse_json(text) if self._path.lower().endswith(".json") else self._parse_csv(text)
        except Exception as exc:  # malformed file — degrade, don't crash startup
            logger.warning("Failed to parse CSGO history (%s): %s", self._path, exc)
            return
        self._ingest(matches)
        self._loaded = bool(self._past or self._matches)
        logger.info("CSGO local history: %d past matches, %d teams from %s",
                    len(self._past), len(self._teams), self._path)

    def _parse_json(self, text: str) -> list[CsgoMatch]:
        data = json.loads(text)
        raw = data.get("matches") if isinstance(data, dict) else data
        out: list[CsgoMatch] = []
        for d in raw or []:
            try:
                out.append(CsgoMatch.model_validate(d))
            except Exception:
                continue
        return out

    def _parse_csv(self, text: str) -> list[CsgoMatch]:
        out: list[CsgoMatch] = []
        reader = csv.DictReader(io.StringIO(text))
        for idx, row in enumerate(reader):
            r = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            t1 = r.get("team1") or r.get("team_1") or r.get("home")
            t2 = r.get("team2") or r.get("team_2") or r.get("away")
            if not t1 or not t2:
                continue
            t1id = _stable_team_id(r.get("team1_id"), t1)
            t2id = _stable_team_id(r.get("team2_id"), t2)
            s1 = _int(r.get("team1_score") or r.get("score1") or r.get("maps1"))
            s2 = _int(r.get("team2_score") or r.get("score2") or r.get("maps2"))
            best_of = _int(r.get("best_of") or r.get("bo")) or 3
            date = _parse_dt(r.get("date") or r.get("datetime")) or datetime(2020, 1, 1, tzinfo=timezone.utc)

            winner_id = self._resolve_winner(r.get("winner"), t1, t2, t1id, t2id, s1, s2)
            status = "FINAL" if winner_id is not None else "SCHEDULED"

            out.append(CsgoMatch(
                id=r.get("id") or f"hist-{idx}-{t1id}-{t2id}",
                team1=t1, team2=t2, team1_id=t1id, team2_id=t2id,
                team1_aliases=build_aliases(t1, None), team2_aliases=build_aliases(t2, None),
                date=date, event=r.get("event") or None, best_of=best_of, status=status,
                team1_score=s1, team2_score=s2, winner_id=winner_id,
                source_ids={"localfile": r.get("id") or f"hist-{idx}"},
            ))
        return out

    @staticmethod
    def _resolve_winner(winner, t1, t2, t1id, t2id, s1, s2) -> int | None:
        if winner:
            w = str(winner).strip()
            if w in ("1", t1) or normalize(w) == normalize(t1):
                return t1id
            if w in ("2", t2) or normalize(w) == normalize(t2):
                return t2id
        if s1 is not None and s2 is not None and s1 != s2:
            return t1id if s1 > s2 else t2id
        return None

    def _ingest(self, matches: list[CsgoMatch]) -> None:
        teams: dict[int, CsgoTeam] = {}
        past: list[CsgoMatch] = []
        upcoming: list[CsgoMatch] = []
        for m in matches:
            for tid, name, abbr in ((m.team1_id, m.team1, m.team1_abbrev), (m.team2_id, m.team2, m.team2_abbrev)):
                if tid not in teams:
                    teams[tid] = CsgoTeam(
                        id=tid, name=name, abbreviation=abbr or name[:4].upper(),
                        aliases=build_aliases(name, abbr or None), source_ids={"localfile": str(tid)},
                    )
            if m.winner_id is not None or (isinstance(m.team1_score, int) and isinstance(m.team2_score, int)
                                           and m.team1_score != m.team2_score):
                past.append(m)
            else:
                upcoming.append(m)
        self._teams = list(teams.values())
        self._matches = upcoming
        self._past = sorted(past, key=lambda x: x.date)

    # ── CsgoDataClient contract ──────────────────────────────────────────────
    async def refresh(self) -> None:
        self._load()   # re-read the file (lets you update it and POST /refresh)

    def get_teams(self) -> list[CsgoTeam]:
        return list(self._teams)

    def get_matches(self) -> list[CsgoMatch]:
        return list(self._matches)

    def get_match(self, match_id: str) -> CsgoMatch | None:
        return next((m for m in (self._matches + self._past) if m.id == match_id), None)

    def get_past_matches(self) -> list[CsgoMatch]:
        return list(self._past)

    def is_available(self) -> bool:
        return self._loaded
