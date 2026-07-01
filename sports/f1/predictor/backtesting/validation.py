"""Walk-forward, out-of-sample validation harness for F1 win-probability models.

This is the *foundation* the trading roadmap (task A4/A7) depends on: a
model-agnostic way to (1) select which races form the training vs. held-out
evaluation set over time, and (2) judge a model — or a calibration fit — on data
it was **not** fit on, against an honest baseline.

Why it exists
-------------
The existing backtest (:mod:`service`) prevents *per-race* leakage via
``RaceReplayBuilder`` (the target race's own result is never a feature), but it
still reports metrics on the same races used to tune weights, and it has no
train/validation split. The ``/backtest/calibration/fit`` route added a single
"fit on all-but-last-season, eval on last" split; this module generalises that to
proper **expanding-window walk-forward** folds and adds a **pass/fail gate**
comparing the model to a uniform baseline.

Design
------
- **Model-agnostic.** Consumes backtest ``race_rows`` — dicts carrying
  ``season``, ``round``, ``probability_distribution`` (``[{driver_id,
  win_probability}]``) and ``actual_winner``. Works for the current heuristic and
  any future trained model without change.
- **Honest, time-ordered.** Folds train only on races *before* the validation
  block, so every evaluated race is genuinely out-of-sample. Validation blocks
  are disjoint, so pooling them evaluates each held-out race exactly once, each
  with a calibrator fit on its own past.
- **Reuses** :mod:`empirical_calibration` (``fit_winner_calibrator``,
  ``winner_pairs_from_runs``, ``brier_score``) so numbers are directly comparable
  to the calibration-fit endpoint.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from sports.f1.predictor.probability.empirical_calibration import (
    MIN_POSITIVES,
    MIN_SAMPLES,
    EmpiricalCalibrator,
    brier_score,
    fit_winner_calibrator,
    winner_pairs_from_runs,
)

# Tolerance (in Brier units) within which calibration is treated as "not worse".
_CAL_TOLERANCE = 1e-4
# A predicted probability is clamped away from 0/1 before taking a log.
_EPS = 1e-6


# --------------------------------------------------------------------------
# Row helpers
# --------------------------------------------------------------------------
def _race_key(row: dict[str, Any]) -> tuple[int, int]:
    """Time-ordering key for a race row: (season, round)."""
    try:
        season = int(row.get("season") or 0)
    except (TypeError, ValueError):
        season = 0
    try:
        rnd = int(row.get("round") or row.get("round_num") or 0)
    except (TypeError, ValueError):
        rnd = 0
    return (season, rnd)


def sort_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows sorted chronologically by (season, round)."""
    return sorted((r for r in rows if isinstance(r, dict)), key=_race_key)


def _actual_winner(row: dict[str, Any]) -> Any:
    actual = row.get("actual_winner")
    if isinstance(actual, dict):
        return actual.get("driver_id") or actual.get("driver")
    return actual


def _field(row: dict[str, Any]) -> dict[str, float]:
    """{driver_id: win_probability} for a race row (raw model distribution)."""
    out: dict[str, float] = {}
    for item in row.get("probability_distribution") or row.get("distribution") or []:
        driver_id = item.get("driver_id") or item.get("driver")
        prob = item.get("win_probability")
        if driver_id is None or prob is None:
            continue
        try:
            out[str(driver_id)] = float(prob)
        except (TypeError, ValueError):
            continue
    return out


def _uniform_pairs(rows: Sequence[dict[str, Any]]) -> list[tuple[float, float]]:
    """Baseline: every driver in a race gets 1/field_size."""
    pairs: list[tuple[float, float]] = []
    for row in rows:
        field = _field(row)
        n = len(field)
        if not n:
            continue
        share = 1.0 / n
        actual = _actual_winner(row)
        for driver_id in field:
            pairs.append((share, 1.0 if driver_id == actual else 0.0))
    return pairs


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _log_loss(pairs: Sequence[tuple[float, float]]) -> float:
    """Mean binary log loss over (probability, outcome) pairs."""
    if not pairs:
        return 0.0
    total = 0.0
    for prob, outcome in pairs:
        p = min(1.0 - _EPS, max(_EPS, float(prob)))
        y = 1.0 if float(outcome) >= 0.5 else 0.0
        total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
    return total / len(pairs)


def _expected_calibration_error(pairs: Sequence[tuple[float, float]], bins: int = 10) -> float:
    """Weighted mean gap between predicted probability and observed frequency."""
    if not pairs:
        return 0.0
    buckets: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(bins)]  # [pred_sum, outcome_sum, count]
    for prob, outcome in pairs:
        p = min(1.0, max(0.0, float(prob)))
        idx = min(bins - 1, int(p * bins))
        buckets[idx][0] += p
        buckets[idx][1] += 1.0 if float(outcome) >= 0.5 else 0.0
        buckets[idx][2] += 1.0
    n = len(pairs)
    ece = 0.0
    for pred_sum, outcome_sum, count in buckets:
        if count <= 0:
            continue
        ece += (count / n) * abs(pred_sum / count - outcome_sum / count)
    return ece


def _winner_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    """Fraction of races whose top-probability driver actually won.

    Monotonic calibration never changes the arg-max, so this is reported once and
    applies to both raw and calibrated probabilities.
    """
    graded = 0
    correct = 0
    for row in rows:
        field = _field(row)
        if not field:
            continue
        graded += 1
        top = max(field.items(), key=lambda kv: kv[1])[0]
        if top == _actual_winner(row):
            correct += 1
    return round(correct / graded, 4) if graded else 0.0


def _metric_block(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    return {
        "brier": round(brier_score([p for p, _ in pairs], [y for _, y in pairs]), 6),
        "log_loss": round(_log_loss(pairs), 6),
        "ece": round(_expected_calibration_error(pairs), 6),
        "samples": len(pairs),
        "positives": sum(1 for _, y in pairs if y >= 0.5),
    }


# --------------------------------------------------------------------------
# Fold construction (race selection over time)
# --------------------------------------------------------------------------
def build_walk_forward_folds(
    rows: Sequence[dict[str, Any]],
    *,
    min_train_races: int = 20,
    val_block: int = 5,
    step: int | None = None,
    max_folds: int | None = None,
) -> list[dict[str, Any]]:
    """Expanding-window folds over chronologically ordered races.

    Fold ``k`` trains on ``ordered[:cut]`` and validates on the next ``val_block``
    races ``ordered[cut:cut+val_block]``; ``cut`` starts at ``min_train_races`` and
    advances by ``step`` (default ``val_block``, giving disjoint validation blocks
    that together cover every race after the initial training window).
    """
    ordered = sort_rows(rows)
    n = len(ordered)
    step = step or val_block
    folds: list[dict[str, Any]] = []
    cut = max(1, int(min_train_races))
    while cut < n:
        val = ordered[cut : cut + val_block]
        if not val:
            break
        train = ordered[:cut]
        folds.append(
            {
                "index": len(folds),
                "train": train,
                "val": val,
                "train_size": len(train),
                "val_size": len(val),
                "train_span": [_race_key(train[0]), _race_key(train[-1])],
                "val_span": [_race_key(val[0]), _race_key(val[-1])],
            }
        )
        if max_folds and len(folds) >= max_folds:
            break
        cut += step
    return folds


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_split(
    train_rows: Sequence[dict[str, Any]],
    val_rows: Sequence[dict[str, Any]],
    *,
    method: str = "isotonic",
) -> dict[str, Any]:
    """Fit a calibrator on ``train_rows``; score raw / calibrated / baseline on
    ``val_rows``. Returned metrics are all out-of-sample."""
    calibrator = fit_winner_calibrator(train_rows, method=method)
    raw_pairs = winner_pairs_from_runs(val_rows)
    cal_pairs = [(calibrator.apply(p), y) for p, y in raw_pairs]
    base_pairs = _uniform_pairs(val_rows)
    return {
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "calibrator": {
            "method": calibrator.method,
            "is_identity": calibrator.is_identity(),
            "sample_count": calibrator.sample_count,
            "positive_count": calibrator.positive_count,
        },
        "winner_accuracy": _winner_accuracy(val_rows),
        "raw": _metric_block(raw_pairs),
        "calibrated": _metric_block(cal_pairs),
        "baseline_uniform": _metric_block(base_pairs),
    }


def _gate(
    raw: dict[str, Any],
    calibrated: dict[str, Any],
    baseline: dict[str, Any],
    *,
    val_races: int,
    calibrator_active: bool,
) -> dict[str, Any]:
    """A7 go/no-go gate. The model must be more informative than a uniform
    baseline before its edges can be trusted for trading."""
    reasons: list[str] = []

    sufficient = raw["samples"] >= MIN_SAMPLES and raw["positives"] >= MIN_POSITIVES
    if not sufficient:
        reasons.append(
            f"insufficient out-of-sample data ({raw['samples']} samples / "
            f"{raw['positives']} winners; need {MIN_SAMPLES}/{MIN_POSITIVES})"
        )

    beats_brier = raw["brier"] < baseline["brier"]
    beats_logloss = raw["log_loss"] < baseline["log_loss"]
    beats_baseline = beats_brier and beats_logloss
    if beats_baseline:
        reasons.append("model beats the uniform baseline on Brier and log loss out-of-sample")
    else:
        if not beats_brier:
            reasons.append(f"model Brier {raw['brier']} does not beat baseline {baseline['brier']}")
        if not beats_logloss:
            reasons.append(f"model log loss {raw['log_loss']} does not beat baseline {baseline['log_loss']}")

    calibration_helps = calibrator_active and calibrated["brier"] <= raw["brier"] + _CAL_TOLERANCE
    if calibrator_active and not calibration_helps:
        reasons.append(
            f"calibration does not help out-of-sample (Brier {raw['brier']} → {calibrated['brier']}); keep it off"
        )
    elif calibration_helps:
        reasons.append("calibration improves (or holds) out-of-sample Brier; safe to enable")

    passed = bool(sufficient and beats_baseline)
    return {
        "passed": passed,
        "sufficient_data": sufficient,
        "beats_baseline": beats_baseline,
        "recommend_enable_calibration": bool(calibration_helps and passed),
        "val_races": val_races,
        "reasons": reasons,
        "verdict": (
            "PASS — model is informative out-of-sample; trustworthy enough to proceed to paper edges"
            if passed
            else "FAIL — do not trade on these probabilities yet"
        ),
    }


def run_walk_forward_validation(
    rows: Sequence[dict[str, Any]],
    *,
    method: str = "isotonic",
    min_train_races: int = 20,
    val_block: int = 5,
    step: int | None = None,
) -> dict[str, Any]:
    """Full walk-forward validation over ``rows``.

    Trains a calibrator on each fold's past, evaluates its disjoint validation
    block, pools the held-out predictions across folds, and applies the A7 gate.
    """
    ordered = sort_rows(rows)
    folds = build_walk_forward_folds(
        ordered, min_train_races=min_train_races, val_block=val_block, step=step
    )
    if not folds:
        return {
            "ok": False,
            "reason": "not_enough_races_for_walk_forward",
            "race_count": len(ordered),
            "min_required": int(min_train_races) + 1,
            "config": {"method": method, "min_train_races": min_train_races, "val_block": val_block},
        }

    pooled_raw: list[tuple[float, float]] = []
    pooled_cal: list[tuple[float, float]] = []
    pooled_base: list[tuple[float, float]] = []
    pooled_val_rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    calibrator_active = False

    for fold in folds:
        calibrator = fit_winner_calibrator(fold["train"], method=method)
        calibrator_active = calibrator_active or not calibrator.is_identity()
        raw_pairs = winner_pairs_from_runs(fold["val"])
        cal_pairs = [(calibrator.apply(p), y) for p, y in raw_pairs]
        base_pairs = _uniform_pairs(fold["val"])
        pooled_raw.extend(raw_pairs)
        pooled_cal.extend(cal_pairs)
        pooled_base.extend(base_pairs)
        pooled_val_rows.extend(fold["val"])
        fold_reports.append(
            {
                "index": fold["index"],
                "train_size": fold["train_size"],
                "val_size": fold["val_size"],
                "train_span": fold["train_span"],
                "val_span": fold["val_span"],
                "calibrator": {
                    "method": calibrator.method,
                    "is_identity": calibrator.is_identity(),
                    "sample_count": calibrator.sample_count,
                },
                "winner_accuracy": _winner_accuracy(fold["val"]),
                "raw": _metric_block(raw_pairs),
                "calibrated": _metric_block(cal_pairs),
                "baseline_uniform": _metric_block(base_pairs),
            }
        )

    raw = _metric_block(pooled_raw)
    calibrated = _metric_block(pooled_cal)
    baseline = _metric_block(pooled_base)
    aggregate = {
        "winner_accuracy": _winner_accuracy(pooled_val_rows),
        "raw": raw,
        "calibrated": calibrated,
        "baseline_uniform": baseline,
        "improvement": {
            "calibration_brier_delta": round(calibrated["brier"] - raw["brier"], 6),
            "vs_baseline_brier_delta": round(raw["brier"] - baseline["brier"], 6),
            "vs_baseline_log_loss_delta": round(raw["log_loss"] - baseline["log_loss"], 6),
        },
    }
    gate = _gate(
        raw,
        calibrated,
        baseline,
        val_races=len(pooled_val_rows),
        calibrator_active=calibrator_active,
    )
    return {
        "ok": True,
        "config": {
            "method": method,
            "min_train_races": int(min_train_races),
            "val_block": int(val_block),
            "step": int(step or val_block),
        },
        "race_count": len(ordered),
        "fold_count": len(folds),
        "aggregate": aggregate,
        "gate": gate,
        "last_fold": fold_reports[-1] if fold_reports else None,
        "folds": fold_reports,
    }
