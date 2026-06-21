"""Match a CS2 match to a prediction market, with confidence + reason codes.

Markets are noisy: outright/futures ("NAVI to win the Major"), wrong week's
rematch, alias soup ("Na'Vi" vs "NAVI"). The matcher scores a (match, market)
pair on team aliases, event name, start-time window and best-of, and returns an
explicit confidence + reason codes so the dashboard/backtest can gate on it.

Key precision rule: a real match-winner market must mention BOTH teams. One team
alone is treated as an outright/futures and rejected — the main source of false
positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from games.csgo.identity import alias_keys, normalize
from games.csgo.models.csgo import CsgoMatch

MATCH_THRESHOLD = 0.6
TIME_NEAR_SECONDS = 24 * 3600
TIME_FAR_SECONDS = 72 * 3600


@dataclass
class MarketRef:
    venue: str = ""
    title: str = ""
    event: str = ""
    series: str = ""
    ticker: str = ""               # kalshi ticker or polymarket slug
    start_time: datetime | None = None


@dataclass
class MatchResult:
    is_match: bool
    confidence: float              # 0..1
    reasons: list[str] = field(default_factory=list)


def _haystack(market: MarketRef) -> tuple[str, set[str]]:
    raw = " ".join([market.title, market.event, market.series, market.ticker])
    norm = normalize(raw)
    return norm, set(norm.split())


def _team_hit(aliases: list[str], haystack: str, tokens: set[str]) -> bool:
    """A single-word alias must be a standalone token (precision); a multi-word
    alias must appear as a substring."""
    for key in alias_keys(aliases):
        if " " in key:
            if key and key in haystack:
                return True
        elif key in tokens:
            return True
    return False


def _aliases_for(match: CsgoMatch, side: int) -> list[str]:
    if side == 1:
        return match.team1_aliases or [a for a in (match.team1, match.team1_abbrev) if a]
    return match.team2_aliases or [a for a in (match.team2, match.team2_abbrev) if a]


def score_match(match: CsgoMatch, market: MarketRef) -> MatchResult:
    haystack, tokens = _haystack(market)
    reasons: list[str] = []

    t1 = _team_hit(_aliases_for(match, 1), haystack, tokens)
    t2 = _team_hit(_aliases_for(match, 2), haystack, tokens)

    if not t1 and not t2:
        return MatchResult(False, 0.0, ["NO_TEAMS"])
    if not (t1 and t2):
        # Outright / futures / wrong market — reject to avoid false positives.
        return MatchResult(False, 0.2, ["ONE_TEAM_ONLY"])

    confidence = 0.6
    reasons.append("BOTH_TEAMS")

    if match.event:
        ev = normalize(match.event)
        ev_tokens = {t for t in ev.split() if len(t) > 3}  # ignore tiny tokens/years
        if ev_tokens and ev_tokens & tokens:
            confidence += 0.2
            reasons.append("EVENT_MATCH")

    if market.start_time is not None and match.date is not None:
        delta = abs((market.start_time - match.date).total_seconds())
        if delta <= TIME_NEAR_SECONDS:
            confidence += 0.15
            reasons.append("TIME_WINDOW")
        elif delta > TIME_FAR_SECONDS:
            confidence -= 0.3
            reasons.append("TIME_MISMATCH")

    bo = f"bo{match.best_of}"
    if bo in tokens or f"best of {match.best_of}" in haystack:
        confidence += 0.05
        reasons.append("BEST_OF")

    confidence = max(0.0, min(1.0, confidence))
    return MatchResult(confidence >= MATCH_THRESHOLD, confidence, reasons)


def best_market(match: CsgoMatch, markets: list[MarketRef]) -> tuple[MarketRef | None, MatchResult]:
    """Return the highest-confidence matching market (or None if nothing clears)."""
    best: tuple[MarketRef | None, MatchResult] = (None, MatchResult(False, 0.0, ["NO_MARKETS"]))
    for m in markets:
        res = score_match(match, m)
        if res.confidence > best[1].confidence:
            best = (m, res)
    if best[0] is not None and not best[1].is_match:
        return (None, best[1])
    return best
