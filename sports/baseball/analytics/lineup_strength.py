"""Leak-free "confirmed lineups" signal for MLB games, from the real boxscore lineup.

The starting pitcher is the biggest single lever; the next is WHO ELSE is actually in
the batter's box. Tonight's confirmed lineup captures stars resting, injuries, call-ups
and September roster churn that a season-long team-strength number misses. This module
turns each team's 9 STARTERS (the boxscore ``battingOrder``) into one aggregate hitting
quality by anchoring every batter to their **PRIOR-SEASON wOBA** (a wOBA-scale proxy from
OBP/SLG when the feed has no true wOBA — see ``baseball_routes._woba_from_hitting``).

Leak-freeness comes from the anchor: prior-season wOBA is fully known before this season
starts, so scoring tonight's lineup with it never peeks at tonight's outcome. The lineups
themselves come from the FINISHED-game boxscore, which is exactly how the model would see
a *confirmed* lineup at first pitch — the same information a bettor has pre-game.

Design mirrors ``bullpen_fatigue`` / ``schedule_fatigue``:

  * ``lineup_quality`` reduces one 9-man batting order to a mean prior-season wOBA, with a
    league-average fallback for any starter missing an anchor (rookies / call-ups / sub-
    threshold PA), so a lineup never collapses to a tiny sample of known bats.
  * ``lineup_by_game`` produces the per-game home/away qualities in one pass; a game whose
    boxscore lineup is absent simply yields ``None`` for that side (the model flags it
    absent rather than assuming league-average).

Everything is a small bounded prior. The honest expectation is a TINY effect — v1 uses
LAST season's batter quality, which is stale for exactly the players (breakouts, call-ups)
where a lineup signal should matter most, and the market prices confirmed lineups quickly.
The goal is to build and MEASURE a clean lineup signal, not manufacture a positive result.
A within-season, heavier batter rating is the v2 that might actually move the needle.
"""
from __future__ import annotations

from typing import Optional

# League-average wOBA (on the OBP/SLG-derived proxy scale used by ``_woba_from_hitting``,
# whose qualified-hitter mean is ~.329). Used as the fallback quality for any starter with
# no prior-season anchor so a lineup with a rookie or two isn't scored off only its veterans.
LEAGUE_AVG_WOBA = 0.320


def lineup_quality(batting_order, woba_map: dict, league_avg: float = LEAGUE_AVG_WOBA) -> Optional[float]:
    """Mean prior-season wOBA of the STARTERS in ``batting_order`` (a list of batter ids),
    using ``league_avg`` for any id missing from ``woba_map``. Returns ``None`` for an
    empty / unusable order (the caller then flags the lineup absent). Null-safe: a
    non-list order or a ``None`` map yields ``None`` rather than raising."""
    if not batting_order or not isinstance(batting_order, (list, tuple)):
        return None
    lookup = woba_map or {}
    total = 0.0
    count = 0
    for pid in batting_order:
        try:
            key = int(pid)
        except (TypeError, ValueError):
            continue
        total += float(lookup.get(key, league_avg))
        count += 1
    if count == 0:
        return None
    return total / count


def _side_lineup(usage_entry: Optional[dict], side: str) -> Optional[list]:
    """The ``lineup`` id-list for one side of a game's boxscore usage entry, or ``None``
    when the entry / side / lineup is missing (backward-compatible with older usage
    entries that predate the lineup field)."""
    if not isinstance(usage_entry, dict):
        return None
    team = usage_entry.get(side)
    if not isinstance(team, dict):
        return None
    lineup = team.get("lineup")
    if not isinstance(lineup, (list, tuple)) or not lineup:
        return None
    return list(lineup)


def lineup_by_game(games, usage: dict[int, dict], woba_map: dict,
                   league_avg: float = LEAGUE_AVG_WOBA) -> dict[int, dict]:
    """One pass over the season → ``{game_id: {home_lineup_quality, away_lineup_quality}}``
    where each quality is the mean prior-season wOBA of that team's 9 confirmed starters
    (``None`` when that side's lineup is absent from the boxscore usage).

    ``usage`` is ``{game_id: {"home": {..., "lineup": [ids]}, "away": {...}}}`` as returned
    by ``season_bullpen_usage``; ``woba_map`` is ``{player_id: prior_season_woba}`` from
    ``_prior_season_woba_map``. Leak-free: the lineup is the confirmed first-pitch roster
    and the quality anchor is the fully-known prior season — no lookahead. A game missing
    from ``usage`` (or with a missing lineup) simply gets ``None`` qualities. Only games
    with an id participate; anything else is skipped so callers get a partial map rather
    than an error."""
    out: dict[int, dict] = {}
    lookup = woba_map or {}
    for g in games:
        gid = getattr(g, "id", None)
        if gid is None:
            continue
        entry = usage.get(gid) if usage else None
        h_line = _side_lineup(entry, "home")
        a_line = _side_lineup(entry, "away")
        out[gid] = {
            "home_lineup_quality": lineup_quality(h_line, lookup, league_avg) if h_line else None,
            "away_lineup_quality": lineup_quality(a_line, lookup, league_avg) if a_line else None,
            "home_lineup_size": len(h_line) if h_line else 0,
            "away_lineup_size": len(a_line) if a_line else 0,
            "league_avg_woba": league_avg,
        }
    return out
