"""Within-season, as-of-date starting-pitcher form (FIP).

Prior-season ERA (P4) barely moved the backtest — last year's ERA is a poor read
on this year's talent. This computes each starter's form from THIS season's game
log up to (but not including) the game being predicted — leak-free — as **FIP**
(Fielding Independent Pitching: HR/BB/HBP/K only, so it strips out defense and
batted-ball luck and is far more stable than ERA). The within-season number is
regressed toward a prior anchor (prior-season ERA, else league average) by a fixed
number of innings, so a pitcher with three good starts isn't over-rated.

Reused by the backtest (all starters, cached per season) and the single-game
analysis (just the two starters).
"""
from __future__ import annotations

from typing import Any

FIP_CONST = 3.10       # puts FIP on the ERA scale (league FIP ≈ league ERA)
LEAGUE_FIP = 4.10      # fallback anchor when there is no prior-season number
REG_IP = 45.0          # innings of the prior anchor blended in (regression to the mean)


def parse_ip(text: Any) -> float:
    """Baseball innings notation → true innings: '5.2' == 5 + 2/3."""
    try:
        whole, _, frac = str(text).partition(".")
        outs = int(frac) if frac else 0
        return int(whole or 0) + outs / 3.0
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


async def fetch_pitcher_logs(client, pitcher_ids, season: int) -> dict[int, list[dict]]:
    """Per-start game logs for the given pitcher ids in a season, keyed by id and
    sorted by date. One StatsAPI call per pitcher (cache the whole map per season)."""
    logs: dict[int, list[dict]] = {}
    for pid in pitcher_ids:
        if pid is None:
            continue
        try:
            resp = await client._client.get(
                f"/people/{int(pid)}/stats",
                params={"stats": "gameLog", "group": "pitching", "season": season},
            )
            resp.raise_for_status()
            splits = (resp.json().get("stats") or [{}])[0].get("splits") or []
        except Exception:
            continue
        starts: list[dict] = []
        for s in splits:
            st = s.get("stat") or {}
            if not st.get("gamesStarted"):          # count starts only
                continue
            ip = parse_ip(st.get("inningsPitched"))
            if ip <= 0:
                continue
            starts.append({
                "date": str(s.get("date") or "")[:10],
                "ip": ip,
                "hr": _int(st.get("homeRuns")), "bb": _int(st.get("baseOnBalls")),
                "hbp": _int(st.get("hitByPitch")), "k": _int(st.get("strikeOuts")),
            })
        starts.sort(key=lambda x: x["date"])
        logs[int(pid)] = starts
    return logs


def rating_asof(starts: list[dict] | None, before_date: Any, prior: float | None = None) -> float | None:
    """Shrunk within-season FIP (ERA-scale, lower = better) from the pitcher's starts
    strictly before ``before_date``. ``prior`` (prior-season ERA) anchors the blend;
    returns the anchor when there is no in-season history yet, or None if there is no
    signal at all."""
    bd = str(before_date)[:10]
    anchor = prior if prior is not None else LEAGUE_FIP
    if not starts:
        return anchor
    ip = hr = bb = hbp = k = 0.0
    for s in starts:
        if s["date"] >= bd:
            break
        ip += s["ip"]; hr += s["hr"]; bb += s["bb"]; hbp += s["hbp"]; k += s["k"]
    if ip <= 0:
        return anchor
    within_fip = (13.0 * hr + 3.0 * (bb + hbp) - 2.0 * k) / ip + FIP_CONST
    return round((ip * within_fip + REG_IP * anchor) / (ip + REG_IP), 3)


def starter_ids(games) -> set[int]:
    """Unique probable-starter ids across a set of games."""
    ids: set[int] = set()
    for g in games:
        if getattr(g, "home_pitcher_id", None):
            ids.add(int(g.home_pitcher_id))
        if getattr(g, "away_pitcher_id", None):
            ids.add(int(g.away_pitcher_id))
    return ids
