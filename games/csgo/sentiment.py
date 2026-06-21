"""Bounded CS2 news/sentiment modifier.

Structured news items (stand-in, roster change, illness, visa issue, bootcamp,
patch) nudge a pre-game probability by a *tightly capped* amount — sentiment is a
modifier, never the core model. Each item carries a kind + affected team + severity
and maps to a small signed delta; the total shift is clamped to ±MAX_TOTAL_SHIFT and
every applied item is reported as a reason code.

Mirrors the Predictive-Sentiment idea (structured signals → bounded score) without
coupling to that service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel

# Signed prob impact (toward/away from the *affected* team) at severity 1.0.
# Negative = bad news for the affected team (lowers their win prob).
_KIND_IMPACT: dict[str, float] = {
    "stand_in": -0.040,
    "roster_change": -0.030,
    "illness": -0.025,
    "visa_issue": -0.030,
    "travel_issue": -0.020,
    "bootcamp": +0.015,
    "coach_change": -0.015,
    "patch": 0.0,           # meta shift affects both sides; neutral here
}

MAX_TOTAL_SHIFT = 0.05      # sentiment can move a price at most 5 points


class NewsItem(BaseModel):
    kind: str
    team: Optional[int] = None     # 1 | 2 | None (e.g. a patch)
    severity: float = 0.5          # 0..1
    summary: str = ""


@dataclass
class SentimentAdjustment:
    delta: float                   # applied to team1 prob (+ favors team1)
    adjusted_prob: float           # team1 prob after the (clamped) shift
    reasons: list[str] = field(default_factory=list)


def apply_sentiment(pregame_team1_prob: float, items: list[NewsItem]) -> SentimentAdjustment:
    delta = 0.0
    reasons: list[str] = []
    for it in items or []:
        impact = _KIND_IMPACT.get(it.kind, 0.0)
        if impact == 0.0:
            continue
        signed = impact * max(0.0, min(1.0, it.severity))
        if it.team == 1:
            delta += signed                       # affects team1 directly
            reasons.append(f"{it.kind}:team1")
        elif it.team == 2:
            delta -= signed                       # bad news for team2 helps team1
            reasons.append(f"{it.kind}:team2")

    delta = max(-MAX_TOTAL_SHIFT, min(MAX_TOTAL_SHIFT, delta))
    adjusted = min(0.99, max(0.01, pregame_team1_prob + delta))
    return SentimentAdjustment(round(delta, 4), round(adjusted, 4), reasons)
