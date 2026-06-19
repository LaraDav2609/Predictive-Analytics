"""CSGO data client — interface + a stub implementation with sample data.

The real implementation (HLTV / PandaScore-backed) plugs in behind
`CsgoDataClient` later; the stub lets the whole predict -> edge -> bridge ->
dashboard loop run today without an external API dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

from games.csgo.identity import build_aliases
from games.csgo.models.csgo import CsgoMatch, CsgoTeam, MapScore


class CsgoDataClient(ABC):
    """Contract a CSGO data source must implement (HLTV, PandaScore, ...)."""

    @abstractmethod
    def get_teams(self) -> list[CsgoTeam]: ...

    @abstractmethod
    def get_matches(self) -> list[CsgoMatch]: ...

    @abstractmethod
    def get_match(self, match_id: str) -> CsgoMatch | None: ...

    @abstractmethod
    async def refresh(self) -> None: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    def get_past_matches(self) -> list[CsgoMatch]:
        """Finished matches with results, for backtesting. Default: none."""
        return []


_SAMPLE_TEAMS = [
    CsgoTeam(id=1, name="Natus Vincere", abbreviation="NAVI", region="EU", world_rank=1, rating=1920.0),
    CsgoTeam(id=2, name="FaZe Clan", abbreviation="FAZE", region="EU", world_rank=2, rating=1875.0),
    CsgoTeam(id=3, name="Team Vitality", abbreviation="VIT", region="EU", world_rank=3, rating=1890.0),
    CsgoTeam(id=4, name="G2 Esports", abbreviation="G2", region="EU", world_rank=4, rating=1840.0),
    CsgoTeam(id=5, name="Team Spirit", abbreviation="SPIRIT", region="EU", world_rank=5, rating=1855.0),
    CsgoTeam(id=6, name="MOUZ", abbreviation="MOUZ", region="EU", world_rank=6, rating=1820.0),
]


class StubCsgoClient(CsgoDataClient):
    """In-memory sample data so the architecture runs end-to-end without a feed."""

    def __init__(self) -> None:
        self._teams = [
            t.model_copy(update={
                "aliases": build_aliases(t.name, t.abbreviation),
                "source_ids": {"stub": str(t.id)},
            })
            for t in _SAMPLE_TEAMS
        ]
        self._matches = self._build_matches()
        self._history = self._build_history()

    def _build_matches(self) -> list[CsgoMatch]:
        by_id = {t.id: t for t in self._teams}
        base = datetime(2026, 6, 8, 17, 0, tzinfo=timezone.utc)
        pairs = [(1, 2), (3, 4), (5, 6), (1, 3)]
        matches: list[CsgoMatch] = []
        for i, (a, b) in enumerate(pairs):
            ta, tb = by_id[a], by_id[b]
            matches.append(CsgoMatch(
                id=f"2026-iem-katowice-{ta.abbreviation}-{tb.abbreviation}".lower(),
                team1=ta.name, team2=tb.name,
                team1_id=ta.id, team2_id=tb.id,
                team1_abbrev=ta.abbreviation, team2_abbrev=tb.abbreviation,
                team1_aliases=build_aliases(ta.name, ta.abbreviation),
                team2_aliases=build_aliases(tb.name, tb.abbreviation),
                date=base + timedelta(hours=3 * i),
                event="IEM Katowice 2026", event_slug="iem-katowice-2026", best_of=3,
                source_ids={"stub": f"{ta.id}-{tb.id}"},
            ))
        return matches

    def _build_history(self) -> list[CsgoMatch]:
        """Synthetic finished-match history so the backtest endpoint returns real
        (deterministic) metrics without an external feed. Higher-rated teams win
        ~2/3 of the time; periodic upsets keep it non-trivial."""
        teams = list(self._teams)
        pool = ["Mirage", "Inferno", "Nuke", "Ancient", "Anubis"]
        base = datetime(2026, 2, 1, 17, 0, tzinfo=timezone.utc)
        out: list[CsgoMatch] = []
        idx = 0
        for rnd in range(6):
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    ta, tb = teams[i], teams[j]
                    higher, lower = (ta, tb) if ta.rating >= tb.rating else (tb, ta)
                    upset = (rnd + ta.id + tb.id) % 3 == 0
                    winner, loser = (lower, higher) if upset else (higher, lower)
                    ws, ls = 2, idx % 2  # 2-0 or 2-1
                    maps: list[MapScore] = []
                    order = 1
                    for _ in range(ws):
                        maps.append(MapScore(order=order, map_name=pool[order % len(pool)], winner_id=winner.id)); order += 1
                    for _ in range(ls):
                        maps.append(MapScore(order=order, map_name=pool[order % len(pool)], winner_id=loser.id)); order += 1
                    out.append(CsgoMatch(
                        id=f"hist-{idx}", team1=ta.name, team2=tb.name,
                        team1_id=ta.id, team2_id=tb.id,
                        team1_abbrev=ta.abbreviation, team2_abbrev=tb.abbreviation,
                        team1_aliases=build_aliases(ta.name, ta.abbreviation),
                        team2_aliases=build_aliases(tb.name, tb.abbreviation),
                        date=base + timedelta(days=idx),
                        event="Stub League", event_slug="stub-league", best_of=3, status="FINAL",
                        team1_score=ws if winner.id == ta.id else ls,
                        team2_score=ws if winner.id == tb.id else ls,
                        winner_id=winner.id, winner_code=winner.abbreviation,
                        map_scores=maps, source_ids={"stub": f"hist-{idx}"},
                    ))
                    idx += 1
        return out

    def get_past_matches(self) -> list[CsgoMatch]:
        return list(self._history)

    def get_teams(self) -> list[CsgoTeam]:
        return list(self._teams)

    def get_matches(self) -> list[CsgoMatch]:
        return list(self._matches)

    def get_match(self, match_id: str) -> CsgoMatch | None:
        return next((m for m in self._matches if m.id == match_id), None)

    async def refresh(self) -> None:
        # Stub data is static; a live client would re-fetch here.
        return None

    def is_available(self) -> bool:
        return True
