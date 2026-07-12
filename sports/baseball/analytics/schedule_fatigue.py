"""Leak-free schedule fatigue / rest signals for MLB games.

Computed from the SCHEDULE ALONE — no extra data feed — so it is free and trivially
leak-free: for each game we look only at that team's games played strictly BEFORE the
game's calendar date. Two raw signals per team:

  * ``days_rest`` — calendar days since the team's previous game (capped). A team coming
    off a day (or more) off is fresher than one on the tail of a long stretch; the very
    first game of the window has no prior game (``None``).
  * ``games_last_n`` — how many games the team played in the trailing ``N`` days (a
    density / fatigue proxy — e.g. the back end of a stretch of many games in few days).

The GAME-LEVEL signal the model consumes is the *difference* between the two teams
(home minus away) for each, so a fresher / less-taxed home side nudges the home win
probability up a hair, symmetric with how ``recent_form`` uses a gap. Everything is a
small bounded prior — the goal is a decomposable, learnable lever, measured honestly,
not a hand-tuned edge (MLB schedules are balanced, so the expected effect is tiny).

Usage mirrors ``weather_by_game``: build ``fatigue_by_game(games)`` once (a single pass
over the season schedule) → ``{game_id: {home_days_rest, away_days_rest, ...}}`` and pass
it to ``replay(..., fatigue_by_game=...)`` / ``predict_game(fatigue=...)``.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

# A team hasn't meaningfully "rested" more the longer the gap; cap so an all-star break
# or a season-opening gap doesn't blow up the signal.
MAX_REST_DAYS = 6
# Trailing window for the games-played density proxy (a week captures the usual
# 6-or-7-games-in-7-days grind vs a lighter stretch).
FATIGUE_WINDOW_DAYS = 7


def _as_date(d) -> Optional[date]:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return None


def fatigue_by_game(games) -> dict[int, dict]:
    """One leak-free pass over the season schedule → ``{game_id: {...}}`` with, for both
    the home and away side, ``days_rest`` (calendar days since that team's previous game,
    capped at ``MAX_REST_DAYS``; ``None`` for its first game in the window) and
    ``games_last_n`` (games that team played in the trailing ``FATIGUE_WINDOW_DAYS`` days,
    strictly before the current date).

    Only games with a home/away team id and a parseable date participate; anything else is
    skipped so callers get a partial map rather than an error."""
    dated = []
    for g in games:
        d = _as_date(getattr(g, "date", None))
        h = getattr(g, "home_team_id", None)
        a = getattr(g, "away_team_id", None)
        if d is None or h is None or a is None or getattr(g, "id", None) is None:
            continue
        dated.append((d, g))
    dated.sort(key=lambda t: t[0])

    # Each team's ordered list of dates it has ALREADY played (strictly before current).
    played: dict[int, list[date]] = {}
    out: dict[int, dict] = {}
    for d, g in dated:
        h, a = g.home_team_id, g.away_team_id
        out[g.id] = {
            "home_days_rest": _days_rest(played.get(h), d),
            "away_days_rest": _days_rest(played.get(a), d),
            "home_games_last_n": _games_last_n(played.get(h), d),
            "away_games_last_n": _games_last_n(played.get(a), d),
            "window_days": FATIGUE_WINDOW_DAYS,
        }
        played.setdefault(h, []).append(d)
        played.setdefault(a, []).append(d)
    return out


def _days_rest(prior_dates: Optional[list[date]], current: date) -> Optional[int]:
    """Calendar days since the most recent prior game (capped). ``None`` when the team has
    no game before ``current`` in the window."""
    if not prior_dates:
        return None
    gap = (current - prior_dates[-1]).days
    if gap < 0:
        return None
    return min(gap, MAX_REST_DAYS)


def _games_last_n(prior_dates: Optional[list[date]], current: date) -> int:
    """How many prior games fell within the trailing ``FATIGUE_WINDOW_DAYS`` days
    (strictly before ``current``)."""
    if not prior_dates:
        return 0
    lo = current - _timedelta_days(FATIGUE_WINDOW_DAYS)
    return sum(1 for d in prior_dates if lo <= d < current)


def _timedelta_days(n: int):
    from datetime import timedelta
    return timedelta(days=n)
