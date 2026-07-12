"""Leak-free bullpen-fatigue signal for MLB games, from real boxscore usage.

``schedule_fatigue`` is a *proxy* (games played, days rest) built from the schedule
alone. This is the REAL thing: how hard each team's BULLPEN was actually worked over
the prior 1-3 calendar days, computed from per-game reliever ``pitchesThrown`` in the
boxscore (see ``data/boxscore_client.season_bullpen_usage``).

For each game, per team, we sum reliever pitches thrown over that team's games in the
trailing ``LOOKBACK_DAYS`` window, counting ONLY games strictly BEFORE this game's
calendar date — so it is trivially leak-free (tonight's bullpen usage never informs
tonight's prediction). We also count the distinct relievers used in that window (a
depth-depletion proxy). The game-level signal the model consumes is the *difference*
between the two teams (a more-taxed bullpen nudges that team's win prob DOWN),
symmetric with how ``schedule_fatigue`` uses a rest/density gap.

Usage mirrors ``fatigue_by_game`` / ``season_weather_by_game``: build
``bullpen_by_game(games, usage)`` once (a single leak-free pass over the season) →
``{game_id: {home_bullpen_load, away_bullpen_load, ...}}`` and pass it to
``replay(..., bullpen_by_game=...)`` / ``predict_game(bullpen=...)``.

Everything is a small bounded prior. The honest expectation is a TINY effect —
bullpen fatigue is real but second-order over a full season, and much of it is
already priced by the market. The goal is to build and MEASURE it, not manufacture a
positive result.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

# Trailing window over which prior reliever workload accumulates into "fatigue". Three
# calendar days captures the usual back-to-back(-to-back) bullpen grind without letting
# a week-old outing count.
LOOKBACK_DAYS = 3
# Typical single-game bullpen load is ~120-160 pitches; over a 3-day window a heavily
# used pen can approach ~300-400. We normalize the raw pitch total by this scale so the
# game-level gap is O(1) and the model weight stays interpretable and bounded.
LOAD_SCALE = 150.0


def _as_date(d) -> Optional[date]:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return None


def _team_usage(usage_entry: Optional[dict], side: str) -> Optional[dict]:
    """One team's {"pitches", "relievers"} out of a game's usage entry, or ``None``."""
    if not isinstance(usage_entry, dict):
        return None
    team = usage_entry.get(side)
    if not isinstance(team, dict):
        return None
    return team


def bullpen_by_game(games, usage: dict[int, dict]) -> dict[int, dict]:
    """One leak-free pass over the season → ``{game_id: {...}}`` with, for both the home
    and away side, ``bullpen_load`` (normalized reliever pitches thrown over the trailing
    ``LOOKBACK_DAYS``, strictly before this game's date) and ``relievers_used`` (distinct
    reliever appearances in that window). ``None`` loads for a team with no prior game in
    the window (so the model can flag it absent rather than assume a fresh pen).

    ``usage`` is ``{game_id: {"home": {pitches, relievers}, "away": {...}}}`` as returned
    by ``season_bullpen_usage``. A game missing from ``usage`` simply contributes no prior
    load (it is skipped when accumulating), and a game whose OWN usage is missing still
    gets a leak-free lookback over whatever prior games ARE present.

    Only games with home/away team ids, an id and a parseable date participate; anything
    else is skipped so callers get a partial map rather than an error."""
    dated = []
    for g in games:
        d = _as_date(getattr(g, "date", None))
        h = getattr(g, "home_team_id", None)
        a = getattr(g, "away_team_id", None)
        if d is None or h is None or a is None or getattr(g, "id", None) is None:
            continue
        dated.append((d, g))
    dated.sort(key=lambda t: t[0])

    # Each team's ordered history of (date, reliever_pitches) it has ALREADY played.
    history: dict[int, list[tuple[date, int]]] = {}
    out: dict[int, dict] = {}
    for d, g in dated:
        h, a = g.home_team_id, g.away_team_id
        h_load, h_cnt = _window_load(history.get(h), d)
        a_load, a_cnt = _window_load(history.get(a), d)
        out[g.id] = {
            "home_bullpen_load": h_load,
            "away_bullpen_load": a_load,
            "home_relievers_used": h_cnt,
            "away_relievers_used": a_cnt,
            "lookback_days": LOOKBACK_DAYS,
        }
        # Fold THIS game's actual reliever usage into each team's history for future games.
        entry = usage.get(g.id) if usage else None
        h_team = _team_usage(entry, "home")
        a_team = _team_usage(entry, "away")
        if h_team is not None:
            history.setdefault(h, []).append((d, int(h_team.get("pitches") or 0)))
        if a_team is not None:
            history.setdefault(a, []).append((d, int(a_team.get("pitches") or 0)))
    return out


def _window_load(prior: Optional[list[tuple[date, int]]], current: date):
    """(normalized reliever-pitch load, distinct-game count) over the trailing
    ``LOOKBACK_DAYS`` strictly before ``current``. Returns ``(None, 0)`` when the team
    has no prior game with usage in the window — the model then flags bullpen absent for
    that side rather than assuming a rested pen."""
    if not prior:
        return None, 0
    lo = current - timedelta(days=LOOKBACK_DAYS)
    total = 0
    games_in_window = 0
    for d, pitches in prior:
        if lo <= d < current:
            total += pitches
            games_in_window += 1
    if games_in_window == 0:
        return None, 0
    return total / LOAD_SCALE, games_in_window
