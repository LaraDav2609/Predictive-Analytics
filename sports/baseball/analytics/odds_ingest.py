"""Join raw historical closing-odds files to the season schedule by gamePk.

The CLV gate (``clv.run_clv_backtest``) needs closing moneylines keyed by the same
``gamePk`` the leak-free backtest scores on. But public odds archives are keyed by
DATE + TEAM NAMES, not gamePk. This module bridges that gap: it normalises team
names to a canonical key, parses the two common file shapes, and matches each odds
row to a scheduled game on ``(date, home_team, away_team)`` with a +/-1 day
tolerance (odds files are frequently timezone-shifted by a day).

Two input shapes are supported and auto-detected:

  (a) a *generic* CSV with columns like ``date, home_team, away_team, home_ml,
      away_ml`` (one row per game); and
  (b) the well-known **sportsbookreviewsonline** MLB format — two rows per game
      (visitor then home), with columns ``Date, VH, Team, Close`` where ``Close``
      is the American closing moneyline for that team.

Pure-python, no third-party deps. Returns {gamePk: odds_row} where each row has
``home_ml`` / ``away_ml`` so ``clv.market_fair_home`` works unchanged. Returns {}
on a missing file and skips any row it cannot parse.
"""
from __future__ import annotations

import csv
import os
import re
from datetime import date, datetime, timedelta
from typing import Optional


# ─────────────────────────── team-name normalisation ───────────────────────────
# Canonical key = lowercased full team name from MLBClient. Every variant a public
# odds file is likely to use (nickname, city, abbreviation, historical alias) maps
# to that canonical key so a join can be made regardless of the source's spelling.
_CANONICAL = {
    "arizona diamondbacks": {"arizona", "diamondbacks", "dbacks", "d-backs", "ari", "az", "arizona d'backs"},
    "atlanta braves": {"atlanta", "braves", "atl"},
    "baltimore orioles": {"baltimore", "orioles", "bal", "os"},
    "boston red sox": {"boston", "red sox", "redsox", "bos"},
    "chicago cubs": {"chi cubs", "chicago cubs", "cubs", "chc", "chn"},
    "chicago white sox": {"chi white sox", "chicago white sox", "white sox", "whitesox", "cws", "chw", "cha"},
    "cincinnati reds": {"cincinnati", "reds", "cin"},
    "cleveland guardians": {"cleveland", "guardians", "indians", "cleveland indians", "cle"},
    "colorado rockies": {"colorado", "rockies", "col"},
    "detroit tigers": {"detroit", "tigers", "det"},
    "houston astros": {"houston", "astros", "hou"},
    "kansas city royals": {"kansas city", "royals", "kc", "kan", "kcr"},
    "los angeles angels": {"la angels", "los angeles angels", "anaheim", "angels",
                            "laa", "ana", "los angeles angels of anaheim"},
    "los angeles dodgers": {"la dodgers", "los angeles dodgers", "dodgers", "lad", "lan"},
    "miami marlins": {"miami", "marlins", "florida", "florida marlins", "mia", "fla"},
    "milwaukee brewers": {"milwaukee", "brewers", "mil"},
    "minnesota twins": {"minnesota", "twins", "min"},
    "new york mets": {"ny mets", "new york mets", "mets", "nym", "nyn"},
    "new york yankees": {"ny yankees", "new york yankees", "yankees", "nyy", "nya"},
    "oakland athletics": {"oakland", "athletics", "a's", "as", "oak", "ath", "oakland a's"},
    "philadelphia phillies": {"philadelphia", "phillies", "phi"},
    "pittsburgh pirates": {"pittsburgh", "pirates", "pit"},
    "san diego padres": {"san diego", "padres", "sd", "sdp"},
    "san francisco giants": {"san francisco", "giants", "sf", "sfg"},
    "seattle mariners": {"seattle", "mariners", "sea"},
    "st. louis cardinals": {"st louis", "st. louis", "saint louis", "cardinals", "stl", "sln"},
    "tampa bay rays": {"tampa bay", "rays", "devil rays", "tampa bay devil rays", "tb", "tbr", "tba"},
    "texas rangers": {"texas", "rangers", "tex"},
    "toronto blue jays": {"toronto", "blue jays", "bluejays", "jays", "tor"},
    "washington nationals": {"washington", "nationals", "nats", "wsh", "was", "wsn"},
}

# Flatten to variant -> canonical (built once).
_VARIANT_TO_CANON: dict[str, str] = {}
for _canon, _aliases in _CANONICAL.items():
    _VARIANT_TO_CANON[_canon] = _canon
    for _a in _aliases:
        _VARIANT_TO_CANON[_a] = _canon


def _squash(name: str) -> str:
    """Lowercase, strip punctuation runs, collapse whitespace."""
    s = (name or "").strip().lower()
    s = s.replace(".", ".")          # keep the dot for "st. louis" pre-lookup
    s = re.sub(r"[^a-z0-9'\-\. ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_team(name: str) -> Optional[str]:
    """Map any common team-name / city / nickname / abbreviation variant to the
    canonical full-name key (e.g. ``"NYY"`` / ``"Yankees"`` / ``"NY Yankees"`` ->
    ``"new york yankees"``). Returns ``None`` for an unrecognised name."""
    if not name:
        return None
    s = _squash(name)
    if s in _VARIANT_TO_CANON:
        return _VARIANT_TO_CANON[s]
    # Try without a trailing dot on abbreviations / "st" spelling variants.
    s2 = s.replace(".", "").strip()
    if s2 in _VARIANT_TO_CANON:
        return _VARIANT_TO_CANON[s2]
    s2 = re.sub(r"\s+", " ", s.replace(".", " ")).strip()
    if s2 in _VARIANT_TO_CANON:
        return _VARIANT_TO_CANON[s2]
    # Substring fallback: a source might carry extra words ("New York Yankees (AL)").
    for variant, canon in _VARIANT_TO_CANON.items():
        if len(variant) >= 4 and variant in s:
            return canon
    return None


# ─────────────────────────── date parsing ───────────────────────────
_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y",
    "%Y%m%d", "%d/%m/%Y",
)


def _parse_date(raw, season: Optional[int] = None) -> Optional[date]:
    """Parse a date cell. Handles ISO / US formats and the sportsbookreviewsonline
    ``MMDD`` / ``Mdd`` integer form (needs the season to supply the year)."""
    if raw in (None, ""):
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    s = str(raw).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    # sportsbookreviewsonline: "Date" like 402 (Apr 2) or 1015 (Oct 15) — MMDD/Mdd.
    if season and s.isdigit() and 3 <= len(s) <= 4:
        digits = s.zfill(4)
        try:
            m, d = int(digits[:2]), int(digits[2:])
            if 1 <= m <= 12 and 1 <= d <= 31:
                return date(int(season), m, d)
        except ValueError:
            pass
    return None


def _to_ml(raw) -> Optional[float]:
    """Parse an American moneyline cell; ``None`` for blanks / pick'em placeholders."""
    if raw in (None, ""):
        return None
    s = str(raw).strip().replace("+", "")
    if s.upper() in ("NL", "PK", "PICK", "N/A", "NA", "-"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v != 0 else None


# ─────────────────────────── schedule index ───────────────────────────
def _game_date(g) -> Optional[date]:
    d = getattr(g, "date", None)
    if isinstance(d, datetime):
        return d.date()
    return d if isinstance(d, date) else _parse_date(d)


def _schedule_index(schedule_games) -> dict[tuple, list]:
    """``{(date, home_canon, away_canon): [game, ...]}`` — finished or not, all
    scheduled games. Only games whose both team names normalise are indexed.

    The value is a **list** so a doubleheader (two games with the same
    ``(date, home, away)``) keeps BOTH ``gamePk``s instead of the second
    silently overwriting the first. ``_find_game`` then hands out one unused
    ``gamePk`` per odds row.
    """
    idx: dict[tuple, list] = {}
    for g in schedule_games or []:
        gd = _game_date(g)
        h = normalize_team(getattr(g, "home_team", None))
        a = normalize_team(getattr(g, "away_team", None))
        if gd is None or h is None or a is None:
            continue
        idx.setdefault((gd, h, a), []).append(g)
    return idx


def _matchup_schedule(schedule_games) -> dict[tuple, list]:
    """``{(home_canon, away_canon): [(date, game), ...]}`` sorted by date.

    Grouping by matchup (not by exact date) is what lets the matcher align a
    series/doubleheader sequence to its odds rows even when the schedule's UTC
    ``gameDate`` is a systematic calendar day ahead of the game-local odds date
    (night games at US venues roll into the next UTC day)."""
    ms: dict[tuple, list] = {}
    for g in schedule_games or []:
        gd = _game_date(g)
        h = normalize_team(getattr(g, "home_team", None))
        a = normalize_team(getattr(g, "away_team", None))
        if gd is None or h is None or a is None:
            continue
        ms.setdefault((h, a), []).append((gd, g))
    for lst in ms.values():
        lst.sort(key=lambda t: (t[0], getattr(t[1], "id", 0)))
    return ms


def _find_game(idx: dict[tuple, list], gd: date, home_c: str, away_c: str,
               tol_days: int = 1, used: Optional[set] = None):
    """Return one scheduled game for ``(date, home, away)`` allowing a
    +/-``tol_days`` day shift.

    Dates are tried nearest-first (exact, then -1, +1, -2, +2, ...) so a game on
    the true calendar day always wins over a tolerance neighbour. When ``used`` is
    supplied, already-assigned ``gamePk``s are skipped — this lets the two games of
    a doubleheader each claim a distinct odds row instead of both collapsing onto
    the first ``gamePk``. Retained as a single-lookup helper; bulk matching uses
    ``_match_parsed`` (sequence alignment per matchup)."""
    order = [0]
    for d in range(1, tol_days + 1):
        order += [-d, d]
    for delta in order:
        for game in idx.get((gd + timedelta(days=delta), home_c, away_c), ()):
            if used is None or getattr(game, "id") not in used:
                return game
    return None


# ─────────────────────────── format detection + parsing ───────────────────────────
def _detect_format(fieldnames) -> str:
    """'generic' (one row per game, home_/away_ columns) or 'sbr'
    (sportsbookreviewsonline — two rows per game with a VH + Close column)."""
    lower = {(f or "").strip().lower() for f in (fieldnames or [])}
    has_pairwise = any(c in lower for c in ("home_team", "home", "hometeam")) and \
        any(c in lower for c in ("away_team", "away", "awayteam", "visitor"))
    has_home_ml = any(c in lower for c in ("home_ml", "home_moneyline", "home_ml_close",
                                           "home_dec", "home_implied"))
    if has_pairwise and has_home_ml:
        return "generic"
    # SBR: per-team rows keyed by VH (V/H visitor/home) + a Close moneyline column.
    if "team" in lower and (("vh" in lower) or ("v/h" in lower)) and \
            any(c in lower for c in ("close", "closeml", "closing", "ml", "moneyline")):
        return "sbr"
    if has_pairwise and any("ml" in c or "moneyline" in c or "dec" in c for c in lower):
        return "generic"
    return "sbr" if "team" in lower and "vh" in lower else "generic"


def _get(row: dict, *keys) -> Optional[str]:
    """First non-empty value among ``keys`` (case/space-insensitive on the header)."""
    norm = {(k or "").strip().lower(): v for k, v in row.items()}
    for k in keys:
        v = norm.get(k)
        if v not in (None, ""):
            return v
    return None


# A parsed odds row, normalised and format-agnostic, ready for schedule matching.
# ``date``/``home``/``away`` are canonical; ``odds`` is the {home_ml, away_ml, ...} row.
# ``bad_team`` flags a row whose team names would not normalise (a team-side miss).
class _ParsedGame:
    __slots__ = ("date", "home", "away", "odds", "bad_team")

    def __init__(self, gd, home, away, odds, bad_team=False):
        self.date = gd
        self.home = home
        self.away = away
        self.odds = odds
        self.bad_team = bad_team


def _parse_generic_rows(reader, season) -> list:
    """One ``_ParsedGame`` per generic (home_/away_) odds row."""
    parsed: list = []
    for row in reader:
        try:
            gd = _parse_date(_get(row, "date", "gamedate", "game_date"), season)
            home = normalize_team(_get(row, "home_team", "home", "hometeam"))
            away = normalize_team(_get(row, "away_team", "away", "awayteam", "visitor"))
            if gd is None or home is None or away is None:
                parsed.append(_ParsedGame(gd, home, away, None, bad_team=True))
                continue
            hml = _to_ml(_get(row, "home_ml", "home_moneyline", "home_ml_close", "homeml"))
            aml = _to_ml(_get(row, "away_ml", "away_moneyline", "away_ml_close", "awayml"))
            odds = {"home_ml": hml, "away_ml": aml}
            for k in ("home_dec", "away_dec", "home_implied", "away_implied"):
                v = _get(row, k)
                if v not in (None, ""):
                    odds[k] = v
            parsed.append(_ParsedGame(gd, home, away, odds))
        except Exception:
            parsed.append(_ParsedGame(None, None, None, None, bad_team=True))
    return parsed


def _parse_sbr_rows(reader, season) -> list:
    """One ``_ParsedGame`` per sportsbookreviewsonline (visitor, home) row pair."""
    rows = list(reader)
    parsed: list = []
    i = 0
    while i < len(rows) - 1:
        v_row, h_row = rows[i], rows[i + 1]
        vh1 = (_get(v_row, "vh", "v/h") or "").strip().upper()
        vh2 = (_get(h_row, "vh", "v/h") or "").strip().upper()
        if vh1 not in ("V", "N") or vh2 != "H":     # 'N' = neutral-site visitor in some files
            i += 1
            continue
        try:
            gd = _parse_date(_get(v_row, "date"), season) or _parse_date(_get(h_row, "date"), season)
            away = normalize_team(_get(v_row, "team"))
            home = normalize_team(_get(h_row, "team"))
            if gd is None or away is None or home is None:
                parsed.append(_ParsedGame(gd, home, away, None, bad_team=True))
            else:
                aml = _to_ml(_get(v_row, "close", "closeml", "closing", "ml", "moneyline"))
                hml = _to_ml(_get(h_row, "close", "closeml", "closing", "ml", "moneyline"))
                parsed.append(_ParsedGame(gd, home, away, {"home_ml": hml, "away_ml": aml}))
        except Exception:
            parsed.append(_ParsedGame(None, None, None, None, bad_team=True))
        i += 2
    return parsed


def _has_price(odds: dict) -> bool:
    if odds is None:
        return False
    if odds.get("home_ml") is not None or odds.get("away_ml") is not None:
        return True
    return any(k in odds for k in ("home_dec", "away_dec", "home_implied", "away_implied"))


def _match_parsed(parsed: list, matchup_sched: dict, md: dict, tol_days: int, diag) -> dict:
    """Assign parsed odds rows to schedule ``gamePk``s by per-matchup sequence
    alignment.

    For each ``(home, away)`` matchup, both the odds rows and the scheduled games
    are sorted by date, then each odds row claims the earliest still-unused
    scheduled game within +/-``tol_days`` (scanning in date order). This is robust
    to the systematic UTC-vs-local one-day shift (a whole series slides by a day
    together, so aligning the ordered sequences recovers every game) and to
    doubleheaders (two same-day games become two consecutive slots that the two
    odds rows fill in turn). Diagnostics are tallied once at the end.
    """
    from collections import defaultdict

    by_matchup: dict[tuple, list] = defaultdict(list)
    for p in parsed:
        if not p.bad_team and _has_price(p.odds):
            by_matchup[(p.home, p.away)].append(p)

    out: dict = {}
    unmatched: list = []
    for key, olist in by_matchup.items():
        olist.sort(key=lambda p: p.date)
        slots = matchup_sched.get(key, [])          # [(date, game)] sorted
        used_idx: set = set()
        for p in olist:
            pick = None
            for j, (sd, game) in enumerate(slots):
                if j in used_idx:
                    continue
                if abs((sd - p.date).days) <= tol_days:
                    pick = (j, game)
                    break
            if pick is None:
                unmatched.append(p)
                continue
            used_idx.add(pick[0])
            out[getattr(pick[1], "id")] = p.odds

    if diag is not None:
        diag["matched"] = len(out)
        diag["unmatched_by_team"] = sum(1 for p in parsed if p.bad_team)
        diag["unmatched_no_price"] = sum(
            1 for p in parsed if not p.bad_team and not _has_price(p.odds))
        by_date = no_matchup = 0
        for p in unmatched:
            if (p.home, p.away) in md or (p.away, p.home) in md:
                by_date += 1
            else:
                no_matchup += 1
        diag["unmatched_by_date"] = by_date
        diag["unmatched_no_matchup"] = no_matchup
    return out


def load_and_match(path: str, schedule_games, *, season: Optional[int] = None,
                   tol_days: int = 1, diagnostics: bool = False):
    """Load a raw historical-odds file and join it to ``schedule_games`` by gamePk.

    Returns ``{gamePk: {"home_ml": .., "away_ml": ..}}`` ready for
    ``clv.run_clv_backtest`` (via ``market_fair_home``). Auto-detects the generic vs
    sportsbookreviewsonline format. Returns ``{}`` if the file is absent. ``season``
    is used only to resolve year-less (MMDD) dates in the SBR format; if omitted it
    is inferred from the schedule.

    Matching is timezone- and doubleheader-aware:

      * Each odds row is joined on ``(date, home, away)`` with a +/-``tol_days`` day
        tolerance, tried nearest-first (odds archives are frequently one calendar day
        off because the schedule's ``gameDate`` is UTC while the odds date is
        game-local — a night game rolls to the next UTC day).
      * A doubleheader (two ``gamePk``s sharing one ``(date, home, away)``) hands one
        distinct ``gamePk`` to each of its odds rows instead of collapsing both onto
        the first.
      * To keep the tolerance from stealing a neighbour day's game, matching runs in
        two passes: every odds row that has an EXACT-date game claims it first, then a
        second pass resolves the remainder within +/-``tol_days``.

    When ``diagnostics=True`` returns ``(mapping, diag_dict)`` where ``diag_dict`` has
    counts of ``matched``, ``unmatched_by_team``, ``unmatched_by_date``,
    ``unmatched_no_matchup``, ``unmatched_no_price`` and ``errors``.
    """
    diag = {"matched": 0, "unmatched_by_team": 0, "unmatched_by_date": 0,
            "unmatched_no_matchup": 0, "unmatched_no_price": 0, "errors": 0,
            "total_rows": 0} if diagnostics else None

    def _finish(mapping):
        if diagnostics:
            return mapping, diag
        return mapping

    if not path or not os.path.exists(path):
        return _finish({})
    if season is None:
        for g in schedule_games or []:
            gd = _game_date(g)
            if gd is not None:
                season = gd.year
                break
    matchup_sched = _matchup_schedule(schedule_games)
    if not matchup_sched:
        return _finish({})
    md = {k: [d for d, _g in v] for k, v in matchup_sched.items()}
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            fmt = _detect_format(reader.fieldnames)
            rows = list(reader)
    except (OSError, UnicodeDecodeError):
        return _finish({})

    parse = _parse_sbr_rows if fmt == "sbr" else _parse_generic_rows
    parsed = parse(iter(rows), season)
    if diagnostics:
        diag["total_rows"] = len(parsed)
    out = _match_parsed(parsed, matchup_sched, md, tol_days, diag)
    return _finish(out)
