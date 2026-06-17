"""Historical race loader for F1 backtesting."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class HistoricalRaceLoader:
    def __init__(
        self,
        client,
        cache_dir: str | Path | None = None,
        use_disk_cache: bool = True,
        evidence_cache_dir: str | Path | None = None,
        use_evidence_cache: bool = True,
    ):
        self._client = client
        self._use_disk_cache = use_disk_cache and os.getenv("F1_BACKTEST_CACHE_DISABLE", "").lower() not in {"1", "true", "yes"}
        self._use_evidence_cache = use_evidence_cache and os.getenv("F1_BACKTEST_EVIDENCE_CACHE_DISABLE", "").lower() not in {"1", "true", "yes"}
        self._cache_dir = Path(cache_dir) if cache_dir is not None else Path.cwd() / "artifacts" / "f1_backtest_cache"
        self._evidence_cache_dir = (
            Path(evidence_cache_dir)
            if evidence_cache_dir is not None
            else Path.cwd() / "artifacts" / "f1_weekend_evidence_cache"
        )
        self._cache_stats = {
            "enabled": self._use_disk_cache,
            "cache_dir": str(self._cache_dir),
            "policy": "read_through",
            "evidence_cache": {
                "enabled": self._use_evidence_cache,
                "cache_dir": str(self._evidence_cache_dir),
                "files_found": 0,
                "races_enriched": 0,
                "practice_rows": 0,
                "qualifying_rows": 0,
                "grid_rows": 0,
                "errors": 0,
            },
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "refreshes": 0,
            "bypassed": 0,
            "errors": 0,
        }

    @property
    def cache_stats(self) -> dict[str, Any]:
        return dict(self._cache_stats)

    async def load_season(self, season: int, cache_policy: str = "read_through") -> list[dict[str, Any]]:
        season = int(season)
        cache_policy = _cache_policy(cache_policy)
        self._cache_stats["policy"] = cache_policy
        can_cache = self._can_use_disk_cache(season)
        if can_cache and cache_policy != "refresh":
            cached = self._read_cache(season)
            if cached is not None:
                # Keep the durable season cache immutable-ish, but always overlay
                # local weekend evidence files. This lets cache-only deep
                # backtests pick up newly captured FP/quali/grid summaries
                # without another Jolpica/OpenF1 fetch.
                return self._merge_evidence_cache(season, cached)
            if cache_policy == "cache_only":
                raise ValueError(f"historical cache miss for season {season}")
        elif can_cache and cache_policy == "refresh":
            self._cache_stats["refreshes"] += 1
        elif cache_policy == "cache_only":
            raise ValueError(f"historical cache unavailable for season {season}")

        if hasattr(self._client, "get_historical_race_results"):
            races = await self._client.get_historical_race_results(season)
        else:
            races = await self._client._fetch_season_results(season)
        races = await self._merge_qualifying(season, races)
        races = self._merge_evidence_cache(season, races)
        rows = sorted(
            [race for race in races if str(race.get("round") or "").isdigit()],
            key=lambda race: int(race.get("round") or 0),
        )
        if can_cache:
            self._write_cache(season, rows)
        return rows

    async def load_range(self, start_season: int, end_season: int, cache_policy: str = "read_through") -> dict[int, list[dict[str, Any]]]:
        seasons = {}
        for season in range(start_season, end_season + 1):
            seasons[season] = await self.load_season(season, cache_policy=cache_policy)
        return seasons

    async def _merge_qualifying(self, season: int, races: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not hasattr(self._client, "get_historical_qualifying_results"):
            return races
        try:
            qualifying_races = await self._client.get_historical_qualifying_results(season)
        except Exception:
            return races
        by_round = {
            int(race.get("round") or 0): race.get("QualifyingResults") or []
            for race in qualifying_races
            if str(race.get("round") or "").isdigit()
        }
        if not by_round:
            return races
        merged = []
        for race in races:
            round_num = int(race.get("round") or 0) if str(race.get("round") or "").isdigit() else 0
            if round_num in by_round and not race.get("QualifyingResults"):
                race = {**race, "QualifyingResults": by_round[round_num]}
            merged.append(race)
        return merged

    def _merge_evidence_cache(self, season: int, races: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._use_evidence_cache:
            return races
        merged = []
        for race in races:
            if not str(race.get("round") or "").isdigit():
                merged.append(race)
                continue
            payload = self._read_evidence_payload(season, int(race.get("round") or 0))
            if not payload:
                merged.append(race)
                continue
            enriched = _merge_race_evidence(race, payload)
            if enriched is not race:
                self._cache_stats["evidence_cache"]["races_enriched"] += 1
                self._cache_stats["evidence_cache"]["practice_rows"] += len(_practice_rows_from_evidence(enriched))
                self._cache_stats["evidence_cache"]["qualifying_rows"] += len(enriched.get("QualifyingResults") or [])
                self._cache_stats["evidence_cache"]["grid_rows"] += len(enriched.get("GridResults") or enriched.get("Grid") or [])
            merged.append(enriched)
        return merged

    def _read_evidence_payload(self, season: int, round_num: int) -> dict[str, Any] | None:
        for path in self._evidence_paths(season, round_num):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self._cache_stats["evidence_cache"]["files_found"] += 1
                    return payload
            except Exception:
                self._cache_stats["evidence_cache"]["errors"] += 1
        return None

    def _evidence_paths(self, season: int, round_num: int) -> list[Path]:
        return [
            self._evidence_cache_dir / f"season_{int(season)}_round_{int(round_num):02d}.json",
            self._evidence_cache_dir / f"season_{int(season)}_round_{int(round_num)}.json",
            self._evidence_cache_dir / f"{int(season)}_{int(round_num):02d}.json",
            self._evidence_cache_dir / f"{int(season)}_{int(round_num)}.json",
            self._evidence_cache_dir / str(int(season)) / f"round_{int(round_num):02d}.json",
            self._evidence_cache_dir / str(int(season)) / f"round_{int(round_num)}.json",
        ]

    def _can_use_disk_cache(self, season: int) -> bool:
        if not self._use_disk_cache:
            self._cache_stats["bypassed"] += 1
            return False
        current_season = int(getattr(self._client, "season", datetime.now(timezone.utc).year))
        # Completed seasons are immutable enough for durable local backtest cache.
        # Current/in-progress seasons should be fetched fresh unless a caller later
        # adds an explicit short-TTL policy.
        if season >= current_season:
            self._cache_stats["bypassed"] += 1
            return False
        return True

    def _cache_path(self, season: int) -> Path:
        return self._cache_dir / f"season_{int(season)}_merged.json"

    def _read_cache(self, season: int) -> list[dict[str, Any]] | None:
        path = self._cache_path(season)
        if not path.exists():
            self._cache_stats["misses"] += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            races = payload.get("races")
            if not isinstance(races, list):
                raise ValueError("cache payload missing races list")
            if not races:
                raise ValueError("cache payload has no race rows")
            self._cache_stats["hits"] += 1
            return sorted(
                [race for race in races if str(race.get("round") or "").isdigit()],
                key=lambda race: int(race.get("round") or 0),
            )
        except Exception:
            self._cache_stats["errors"] += 1
            self._cache_stats["misses"] += 1
            return None

    def _write_cache(self, season: int, races: list[dict[str, Any]]) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "season": int(season),
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "source": "jolpica_results_plus_qualifying",
                "races": races,
            }
            self._cache_path(season).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._cache_stats["writes"] += 1
        except Exception:
            self._cache_stats["errors"] += 1


def _cache_policy(value: str | None) -> str:
    policy = str(value or "read_through").strip().lower().replace("-", "_").replace(" ", "_")
    if policy in {"cache", "cacheonly", "offline"}:
        return "cache_only"
    if policy in {"force_refresh", "reload"}:
        return "refresh"
    if policy not in {"read_through", "cache_only", "refresh"}:
        return "read_through"
    return policy


def _merge_race_evidence(race: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(race)
    changed = False

    practice_rows = _payload_rows(payload, "PracticeResults", "practice_results", "practice_laps")
    practice_sessions = _payload_rows(payload, "PracticeSessions", "practice_sessions", "practice")
    if practice_rows and not merged.get("PracticeResults"):
        merged["PracticeResults"] = practice_rows
        changed = True
    elif practice_sessions and not merged.get("PracticeSessions"):
        merged["PracticeSessions"] = practice_sessions
        changed = True

    qualifying_rows = _payload_rows(payload, "QualifyingResults", "qualifying_results", "qualifying")
    if qualifying_rows and not merged.get("QualifyingResults"):
        merged["QualifyingResults"] = qualifying_rows
        changed = True

    grid_rows = _payload_rows(payload, "GridResults", "grid_results", "grid")
    if grid_rows and not merged.get("GridResults"):
        merged["GridResults"] = grid_rows
        changed = True

    sprint_rows = _payload_rows(payload, "SprintResults", "sprint_results", "sprint")
    if sprint_rows and not merged.get("SprintResults"):
        merged["SprintResults"] = sprint_rows
        changed = True

    race_inputs = _payload_dict(payload, "RaceInputs", "race_inputs", "live_inputs")
    if race_inputs and not merged.get("RaceInputs"):
        merged["RaceInputs"] = race_inputs
        changed = True

    weekend_evidence = _payload_dict(payload, "WeekendEvidence", "weekend_evidence")
    if weekend_evidence and not merged.get("WeekendEvidence"):
        merged["WeekendEvidence"] = weekend_evidence
        changed = True

    source = _payload_dict(payload, "source", "metadata") or {}
    merged.setdefault("EvidenceSources", [])
    if changed:
        evidence_source = {
            "source": source.get("source") or payload.get("source") or "local_weekend_evidence_cache",
            "captured_at": source.get("captured_at") or payload.get("captured_at"),
            "coverage": source.get("coverage") or payload.get("coverage") or {},
        }
        merged["EvidenceSources"] = [*(merged.get("EvidenceSources") or []), evidence_source]
        return merged
    return race


def _payload_rows(payload: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            rows = value.get("rows") or value.get("results") or value.get("drivers")
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
            if isinstance(rows, dict):
                return [row for row in rows.values() if isinstance(row, dict)]
    return []


def _payload_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _practice_rows_from_evidence(race: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(race.get("PracticeResults"), list):
        return list(race.get("PracticeResults") or [])
    rows = []
    for session in race.get("PracticeSessions") or race.get("Practice") or []:
        if isinstance(session, dict):
            rows.extend(session.get("Results") or session.get("results") or [])
    return [row for row in rows if isinstance(row, dict)]
