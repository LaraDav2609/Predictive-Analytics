"""pybaseball wrapper for FanGraphs/Baseball-Reference advanced metrics.

pybaseball scrapes FanGraphs and Baseball-Reference, so the first call
for a given year is slow (~5-30s) and downloads a CSV. We cache the
resulting DataFrames in memory, keyed by year and stat group. Player
lookups filter the cached DataFrames.

Graceful degradation: if pybaseball isn't installed, is_available() is
False and all stat methods return empty lists.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import pybaseball  # type: ignore
    PYBASEBALL_AVAILABLE = True
except ImportError:
    pybaseball = None  # type: ignore
    PYBASEBALL_AVAILABLE = False


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        # filter NaN
        if f != f:
            return None
        return f
    except (TypeError, ValueError):
        return None


class PybaseballClient:
    """Thin wrapper around pybaseball with per-year DataFrame caches."""

    # FanGraphs column → response key + label.
    BATTING_COLS = {
        "WAR":   "war",
        "wRC+":  "wrc_plus",
        "wOBA":  "woba",
        "ISO":   "iso",
        "BABIP": "babip",
        "K%":    "k_pct",
        "BB%":   "bb_pct",
        "Off":   "off_runs",
        "Def":   "def_runs",
    }
    PITCHING_COLS = {
        "WAR":   "war",
        "FIP":   "fip",
        "xFIP":  "xfip",
        "K/9":   "k_per_9",
        "BB/9":  "bb_per_9",
        "K%":    "k_pct",
        "BB%":   "bb_pct",
        "BABIP": "babip",
        "LOB%":  "lob_pct",
        "ERA-":  "era_minus",
        "FIP-":  "fip_minus",
    }

    def __init__(self) -> None:
        self._batting_cache: dict[int, Any] = {}
        self._pitching_cache: dict[int, Any] = {}
        self._id_cache: dict[int, Optional[int]] = {}  # MLBAM ID → FanGraphs ID

    def is_available(self) -> bool:
        return PYBASEBALL_AVAILABLE

    def _lookup_fangraphs_id(self, mlb_id: int) -> Optional[int]:
        if mlb_id in self._id_cache:
            return self._id_cache[mlb_id]
        if not PYBASEBALL_AVAILABLE:
            return None
        try:
            df = pybaseball.playerid_reverse_lookup([mlb_id], key_type="mlbam")
            if df.empty:
                self._id_cache[mlb_id] = None
                return None
            fg_id_raw = df.iloc[0].get("key_fangraphs")
            fg_id = int(fg_id_raw) if fg_id_raw and fg_id_raw == fg_id_raw else None  # NaN check
            self._id_cache[mlb_id] = fg_id
            return fg_id
        except Exception as e:
            logger.warning("FanGraphs ID lookup failed for MLB %s: %s", mlb_id, e)
            self._id_cache[mlb_id] = None
            return None

    def _get_batting_year(self, year: int):
        if year in self._batting_cache:
            return self._batting_cache[year]
        if not PYBASEBALL_AVAILABLE:
            return None
        try:
            df = pybaseball.batting_stats(year, qual=1)
            self._batting_cache[year] = df
            return df
        except Exception as e:
            logger.warning("batting_stats(%d) failed: %s", year, e)
            self._batting_cache[year] = None
            return None

    def _get_pitching_year(self, year: int):
        if year in self._pitching_cache:
            return self._pitching_cache[year]
        if not PYBASEBALL_AVAILABLE:
            return None
        try:
            df = pybaseball.pitching_stats(year, qual=1)
            self._pitching_cache[year] = df
            return df
        except Exception as e:
            logger.warning("pitching_stats(%d) failed: %s", year, e)
            self._pitching_cache[year] = None
            return None

    @staticmethod
    def _row_to_dict(row, col_map: dict[str, str]) -> dict[str, Optional[float]]:
        out: dict[str, Optional[float]] = {}
        for src_col, key in col_map.items():
            if src_col in row.index:
                out[key] = _to_float(row[src_col])
            else:
                out[key] = None
        return out

    def get_advanced_batting(self, mlb_id: int, years: list[int]) -> list[dict[str, Any]]:
        """Return [{season, team, war, wrc_plus, woba, iso, babip, k_pct, bb_pct, ...}, ...]."""
        if not PYBASEBALL_AVAILABLE:
            return []
        fg_id = self._lookup_fangraphs_id(mlb_id)
        if fg_id is None:
            return []
        out: list[dict[str, Any]] = []
        for year in years:
            df = self._get_batting_year(year)
            if df is None:
                continue
            try:
                match = df[df["IDfg"] == fg_id]
            except KeyError:
                continue
            if match.empty:
                continue
            row = match.iloc[0]
            stats = self._row_to_dict(row, self.BATTING_COLS)
            stats["season"] = year
            stats["team"] = str(row.get("Team", "")) if "Team" in row.index else ""
            out.append(stats)
        return out

    def get_advanced_pitching(self, mlb_id: int, years: list[int]) -> list[dict[str, Any]]:
        """Return [{season, team, war, fip, xfip, k_per_9, bb_per_9, ...}, ...]."""
        if not PYBASEBALL_AVAILABLE:
            return []
        fg_id = self._lookup_fangraphs_id(mlb_id)
        if fg_id is None:
            return []
        out: list[dict[str, Any]] = []
        for year in years:
            df = self._get_pitching_year(year)
            if df is None:
                continue
            try:
                match = df[df["IDfg"] == fg_id]
            except KeyError:
                continue
            if match.empty:
                continue
            row = match.iloc[0]
            stats = self._row_to_dict(row, self.PITCHING_COLS)
            stats["season"] = year
            stats["team"] = str(row.get("Team", "")) if "Team" in row.index else ""
            out.append(stats)
        return out
