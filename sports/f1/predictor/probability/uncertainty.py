"""Per-driver uncertainty intervals and trade-conviction flags.

Derived entirely from the Monte-Carlo finish-position distribution the simulator
already produces (no simulator change) plus the audit's confidence/governance
signals. Each driver gets:
  - a confidence-adjusted win-probability band (model + MC sampling uncertainty),
  - optimistic / median / pessimistic finishing positions (P10 / P50 / P90),
  - upside (podium) and downside (DNF) scenarios,
  - a conviction label + ``do_not_trade`` flag for low-confidence / wide-outcome cases.

The band is explicitly a *confidence-adjusted* band, not a frequentist CI: model
uncertainty (heuristic inputs) dominates MC sampling error, so the width scales
with (1 - confidence). This keeps the bands honest about the model's real limits.
"""
from __future__ import annotations

import math
from typing import Any

# Conviction thresholds (documented, tune-able).
_LOW_CONFIDENCE = 0.42
_FLOOR_CONFIDENCE = 0.30
_WIDE_BAND = 0.18
_WIDE_OUTCOME = 8


def _position_quantiles(positions: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    """Return (P10, P50, P90) finishing positions from a {position: prob} map.
    Lower position = better finish, so P10 is the optimistic end."""
    pairs: list[tuple[int, float]] = []
    for pos, prob in positions.items():
        try:
            k = int(pos)
            p = float(prob or 0.0)
        except (TypeError, ValueError):
            continue
        if p > 0:
            pairs.append((k, p))
    if not pairs:
        return None, None, None
    pairs.sort(key=lambda kp: kp[0])
    total = sum(p for _, p in pairs) or 1.0
    cum = 0.0
    q: dict[float, int | None] = {0.10: None, 0.50: None, 0.90: None}
    for k, p in pairs:
        cum += p / total
        for target in q:
            if q[target] is None and cum >= target:
                q[target] = k
    last = pairs[-1][0]
    return (q[0.10] if q[0.10] is not None else pairs[0][0],
            q[0.50] if q[0.50] is not None else last,
            q[0.90] if q[0.90] is not None else last)


def driver_uncertainty(
    distribution: dict[str, Any],
    win_prob: float,
    confidence: float,
    iterations: int = 0,
) -> dict[str, Any]:
    p = max(0.0, min(1.0, float(win_prob or 0.0)))
    conf = max(0.0, min(1.0, float(confidence or 0.0)))
    positions = distribution.get("positions") or {}

    var = p * (1.0 - p)
    mc_se = math.sqrt(var / iterations) if iterations and iterations > 0 else 0.0
    # Model uncertainty dominates MC sampling error: widen as confidence falls.
    model_width = (1.0 - conf) * 0.5 * math.sqrt(var)
    half = 1.645 * mc_se + model_width
    low = max(0.0, p - half)
    high = min(1.0, p + half)
    p10, p50, p90 = _position_quantiles(positions)

    return {
        "win_prob": round(p, 4),
        "win_prob_low": round(low, 4),
        "win_prob_high": round(high, 4),
        "band_width": round(high - low, 4),
        "mc_standard_error": round(mc_se, 5),
        "finish_best": p10,        # optimistic (P10)
        "finish_median": p50,
        "finish_worst": p90,       # pessimistic (P90)
        "upside_podium": round(float(distribution.get("podium") or 0.0), 4),
        "downside_dnf": round(float(distribution.get("dnf") or 0.0), 4),
        "basis": "confidence-adjusted band (model + MC sampling); P10/P50/P90 from finish distribution",
    }


def assess_conviction(
    *,
    win_prob: float,
    confidence: float,
    uncertainty: dict[str, Any],
    governance_capped: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    conf = float(confidence or 0.0)
    band = float(uncertainty.get("band_width") or 0.0)
    best = uncertainty.get("finish_best")
    worst = uncertainty.get("finish_worst")
    spread = (worst - best) if (best is not None and worst is not None) else None

    if conf < _LOW_CONFIDENCE:
        reasons.append("low_confidence")
    if band >= _WIDE_BAND:
        reasons.append("wide_probability_band")
    if spread is not None and spread >= _WIDE_OUTCOME:
        reasons.append("wide_outcome_range")
    if governance_capped:
        reasons.append("evidence_capped")

    if conf >= 0.62 and band < 0.10 and not governance_capped and (spread is None or spread < 5):
        conviction = "high"
    elif conf < _LOW_CONFIDENCE or band >= _WIDE_BAND or (spread is not None and spread >= _WIDE_OUTCOME):
        conviction = "low"
    else:
        conviction = "medium"

    do_not_trade = (
        conviction == "low"
        or conf < _FLOOR_CONFIDENCE
        or ("wide_outcome_range" in reasons and "low_confidence" in reasons)
    )
    return {
        "conviction": conviction,
        "do_not_trade": bool(do_not_trade),
        "reasons": reasons,
    }
