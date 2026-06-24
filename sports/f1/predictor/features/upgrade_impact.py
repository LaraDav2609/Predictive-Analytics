"""Causal upgrade-impact tracking via a manual upgrade calendar + difference-in-differences.

There is NO structured upstream feed for car upgrades, so the treatment indicator is a
*curated* calendar of ``{constructor_id, round, component}`` entries (populate
``DEFAULT_UPGRADE_CALENDAR`` or point ``F1_UPGRADE_CALENDAR`` at a JSON file). Impact is
estimated by diff-in-diff on field/teammate-relative pace scores (which cancel global
pace shifts): ``mean(post) - mean(pre)`` strictly after the declared round, requiring
``>= min_post`` post-upgrade races, bounded to ``±cap`` with confidence from sample size
and full source/explanation metadata.

Honest by construction: with no calendar configured it produces nothing — it does NOT
fabricate an "upgrade impact" from the weak sentiment keyword signal.
"""
from __future__ import annotations

import json
import os
from statistics import mean
from typing import Any

DEFAULT_UPGRADE_CALENDAR: list[dict[str, Any]] = []
UPGRADE_CALENDAR_ENV = "F1_UPGRADE_CALENDAR"
DEFAULT_CAP = 0.06


def load_upgrade_calendar() -> list[dict[str, Any]]:
    """Load the curated upgrade calendar from F1_UPGRADE_CALENDAR (JSON list), else the
    in-module default (empty until populated)."""
    path = os.environ.get(UPGRADE_CALENDAR_ENV)
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return list(DEFAULT_UPGRADE_CALENDAR)


def estimate_upgrade_impact(*, pre_scores, post_scores, cap: float = DEFAULT_CAP, min_post: int = 2) -> dict[str, Any]:
    """Diff-in-diff on field-relative pace scores (higher = faster vs the field).
    Requires >= min_post post-upgrade races; bounds the estimate to ±cap."""
    pre = [float(x) for x in (pre_scores or []) if x is not None]
    post = [float(x) for x in (post_scores or []) if x is not None]
    if len(post) < min_post or not pre:
        return {"applied": False, "reason": "insufficient_post_races", "impact": 0.0,
                "modifier": 1.0, "confidence": 0.0, "pre_races": len(pre), "post_races": len(post)}
    impact = mean(post) - mean(pre)
    bounded = max(-cap, min(cap, impact))
    confidence = round(min(0.7, 0.2 + 0.06 * min(len(pre), 5) + 0.06 * min(len(post), 5)), 3)
    return {
        "applied": True,
        "impact": round(impact, 4),
        "bounded_impact": round(bounded, 4),
        "modifier": round(1.0 + bounded, 4),
        "confidence": confidence,
        "pre_mean": round(mean(pre), 4),
        "post_mean": round(mean(post), 4),
        "pre_races": len(pre),
        "post_races": len(post),
        "cap": cap,
        "source": "upgrade_calendar_diff_in_diff",
    }


def build_upgrade_impacts(
    calendar: list[dict[str, Any]],
    pace_history: dict[str, dict[Any, float]],
    *,
    current_round: int | None = None,
    cap: float = DEFAULT_CAP,
    min_post: int = 2,
) -> list[dict[str, Any]]:
    """For each calendar upgrade, diff-in-diff the constructor's relative pace before vs
    after the declared round. ``pace_history``: ``{constructor_id: {round: rel_pace}}``."""
    impacts = []
    for up in calendar or []:
        cid = str(up.get("constructor_id"))
        declared = int(up.get("round") or 0)
        hist = (pace_history or {}).get(cid) or {}
        pre, post = [], []
        for r, v in hist.items():
            rn = int(r)
            if rn < declared:
                pre.append(v)
            elif current_round is None or rn <= int(current_round):
                post.append(v)
        est = estimate_upgrade_impact(pre_scores=pre, post_scores=post, cap=cap, min_post=min_post)
        impacts.append({"constructor_id": cid, "round": declared, "component": up.get("component"), **est})
    return impacts
