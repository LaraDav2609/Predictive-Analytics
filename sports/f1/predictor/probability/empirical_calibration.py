"""Empirical probability calibration (isotonic / Platt) fit from backtest outcomes.

The stage-aware :mod:`calibration` module only does hand-tuned *temperature*
scaling — a sharpen/flatten knob, not a data-fit calibration. This module adds a
real calibrator that learns a monotonic map from predicted probability to the
*observed* outcome frequency, fit on (predicted, actual) pairs collected from the
backtester, and applied at serve time.

Design goals:
- **Safe by default.** With no fitted artifact the calibrator is the identity, so
  serving behaviour is unchanged until an artifact is explicitly provided
  (``F1_CALIBRATION_ARTIFACT``). Mirrors the ``F1_ML_ARTIFACT_PATH`` pattern.
- **No heavy deps.** Pure-Python isotonic (pool-adjacent-violators) and Platt
  (logistic) fits; the fitted map is stored as interpolation breakpoints, so
  ``apply`` is identical regardless of method and the artifact is plain JSON.
- **Honest.** Refuses to fit on too little data (returns identity) so it never
  manufactures a calibration map from noise.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# Below this many (predicted, outcome) pairs, or this few positive outcomes, we
# decline to fit and fall back to identity — a calibration map from a handful of
# races is worse than none.
MIN_SAMPLES = 60
MIN_POSITIVES = 8

CALIBRATION_ARTIFACT_ENV = "F1_CALIBRATION_ARTIFACT"


@dataclass
class EmpiricalCalibrator:
    """A monotonic predicted->calibrated probability map stored as breakpoints."""

    breakpoints: list[tuple[float, float]] = field(default_factory=list)
    method: str = "identity"
    source: str | None = None
    sample_count: int = 0
    positive_count: int = 0

    # ----- application ---------------------------------------------------
    def is_identity(self) -> bool:
        return not self.breakpoints

    def apply(self, probability: float) -> float:
        """Map a single probability through the calibrator (clamped to [0, 1])."""
        try:
            p = float(probability)
        except (TypeError, ValueError):
            return 0.0
        p = min(1.0, max(0.0, p))
        pts = self.breakpoints
        if not pts:
            return p
        if p <= pts[0][0]:
            return pts[0][1]
        if p >= pts[-1][0]:
            return pts[-1][1]
        # linear interpolation between bracketing breakpoints
        lo = 0
        hi = len(pts) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if pts[mid][0] <= p:
                lo = mid
            else:
                hi = mid
        x0, y0 = pts[lo]
        x1, y1 = pts[hi]
        if x1 == x0:
            return min(1.0, max(0.0, y1))
        t = (p - x0) / (x1 - x0)
        return min(1.0, max(0.0, y0 + t * (y1 - y0)))

    def calibrate_field(
        self,
        field_probs: dict[str, float],
        *,
        normalize_to: float | None = 1.0,
    ) -> dict[str, float]:
        """Calibrate a whole market field (e.g. per-driver win probs).

        ``normalize_to`` renormalises the calibrated values to a target sum —
        1.0 for a winner market (exactly one winner), 3.0 for podium, etc. Pass
        ``None`` to leave the per-outcome calibrated values unnormalised.
        """
        mapped = {key: self.apply(value) for key, value in field_probs.items()}
        if normalize_to is None:
            return {key: round(value, 6) for key, value in mapped.items()}
        total = sum(mapped.values())
        if total <= 0:
            count = max(len(mapped), 1)
            share = normalize_to / count
            return {key: round(share, 6) for key in mapped}
        scale = normalize_to / total
        return {key: round(value * scale, 6) for key, value in mapped.items()}

    # ----- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "source": self.source,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "breakpoints": [[round(x, 6), round(y, 6)] for x, y in self.breakpoints],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmpiricalCalibrator":
        raw = data.get("breakpoints") or []
        points = []
        for item in raw:
            try:
                points.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError, IndexError):
                continue
        points.sort(key=lambda p: p[0])
        return cls(
            breakpoints=points,
            method=str(data.get("method") or ("isotonic" if points else "identity")),
            source=data.get("source"),
            sample_count=int(data.get("sample_count") or 0),
            positive_count=int(data.get("positive_count") or 0),
        )

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "EmpiricalCalibrator":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def identity(cls) -> "EmpiricalCalibrator":
        return cls()


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def _clean_pairs(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for prob, outcome in zip(probabilities, outcomes):
        try:
            p = min(1.0, max(0.0, float(prob)))
            y = 1.0 if float(outcome) >= 0.5 else 0.0
        except (TypeError, ValueError):
            continue
        pairs.append((p, y))
    return pairs


def _isotonic_fit(values: list[float], weights: list[float]) -> list[float]:
    """Weighted pool-adjacent-violators. Returns one non-decreasing fitted value
    per input (inputs are distinct, pre-aggregated x positions)."""
    blocks: list[list[float]] = []  # [weighted_sum, weight, input_count]
    for value, weight in zip(values, weights):
        blocks.append([value * weight, weight, 1])
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) >= (blocks[-1][0] / blocks[-1][1]):
            s2, w2, c2 = blocks.pop()
            s1, w1, c1 = blocks.pop()
            blocks.append([s1 + s2, w1 + w2, c1 + c2])
    fitted: list[float] = []
    for weighted_sum, weight, count in blocks:
        mean = weighted_sum / weight
        fitted.extend([mean] * int(count))
    return fitted


def _breakpoints_from_fit(xs: list[float], fitted: list[float]) -> list[tuple[float, float]]:
    # Collapse to one breakpoint per distinct x (PAV makes ``fitted`` constant
    # within a pooled block, so the last value for a given x is representative).
    points: list[tuple[float, float]] = []
    for x, y in zip(xs, fitted):
        if points and abs(points[-1][0] - x) < 1e-9:
            points[-1] = (x, y)
        else:
            points.append((x, y))
    # Anchor the ends so extrapolation is well-defined.
    if points and points[0][0] > 0.0:
        points.insert(0, (0.0, points[0][1]))
    if points and points[-1][0] < 1.0:
        points.append((1.0, points[-1][1]))
    return points


def fit_isotonic(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    *,
    source: str | None = None,
) -> EmpiricalCalibrator:
    """Fit an isotonic (monotonic) calibration map. Returns identity if too sparse."""
    pairs = _clean_pairs(probabilities, outcomes)
    positives = sum(1 for _, y in pairs if y >= 0.5)
    if len(pairs) < MIN_SAMPLES or positives < MIN_POSITIVES:
        return EmpiricalCalibrator.identity()
    # Aggregate outcomes by distinct predicted probability so tied x-values get a
    # single pooled estimate (correct isotonic-with-ties handling).
    groups: dict[float, list[float]] = {}
    for prob, outcome in pairs:
        bucket = groups.setdefault(prob, [0.0, 0.0])
        bucket[0] += outcome
        bucket[1] += 1.0
    xs = sorted(groups)
    means = [groups[x][0] / groups[x][1] for x in xs]
    weights = [groups[x][1] for x in xs]
    fitted = _isotonic_fit(means, weights)
    breakpoints = _breakpoints_from_fit(xs, fitted)
    return EmpiricalCalibrator(
        breakpoints=breakpoints,
        method="isotonic",
        source=source,
        sample_count=len(pairs),
        positive_count=positives,
    )


def fit_platt(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    *,
    source: str | None = None,
    iterations: int = 200,
    learning_rate: float = 0.5,
) -> EmpiricalCalibrator:
    """Fit a Platt (logistic) calibration map sampled into breakpoints.

    Logistic on the *logit* of the raw probability: calibrated = sigmoid(A*z + B)
    where z = logit(p). Fit A, B by gradient descent on log loss.
    """
    pairs = _clean_pairs(probabilities, outcomes)
    positives = sum(1 for _, y in pairs if y >= 0.5)
    if len(pairs) < MIN_SAMPLES or positives < MIN_POSITIVES:
        return EmpiricalCalibrator.identity()

    def logit(p: float) -> float:
        p = min(1 - 1e-6, max(1e-6, p))
        return math.log(p / (1 - p))

    def sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    zs = [logit(p) for p, _ in pairs]
    ys = [y for _, y in pairs]
    a, b = 1.0, 0.0
    n = len(pairs)
    for _ in range(max(1, iterations)):
        ga = gb = 0.0
        for z, y in zip(zs, ys):
            pred = sigmoid(a * z + b)
            err = pred - y
            ga += err * z
            gb += err
        a -= learning_rate * ga / n
        b -= learning_rate * gb / n

    # Sample the fitted logistic on a grid of raw probabilities -> breakpoints.
    grid = [i / 50.0 for i in range(51)]
    breakpoints = [(p, sigmoid(a * logit(p) + b)) for p in grid]
    return EmpiricalCalibrator(
        breakpoints=breakpoints,
        method="platt",
        source=source,
        sample_count=n,
        positive_count=positives,
    )


# --------------------------------------------------------------------------
# Helpers: build pairs from backtest output, score, load default
# --------------------------------------------------------------------------
def winner_pairs_from_runs(runs: Iterable[dict[str, Any]]) -> list[tuple[float, float]]:
    """Extract (win_probability, is_winner) pairs from backtest race runs.

    Each run is expected to carry ``probability_distribution`` (rows with
    ``driver_id``/``win_probability``) and an ``actual_winner`` driver id.
    """
    pairs: list[tuple[float, float]] = []
    for run in runs or []:
        actual = run.get("actual_winner")
        if isinstance(actual, dict):
            actual = actual.get("driver_id") or actual.get("driver")
        rows = run.get("probability_distribution") or run.get("distribution") or []
        for row in rows:
            driver_id = row.get("driver_id") or row.get("driver")
            prob = row.get("win_probability")
            if driver_id is None or prob is None:
                continue
            pairs.append((float(prob), 1.0 if driver_id == actual else 0.0))
    return pairs


def fit_winner_calibrator(
    runs: Iterable[dict[str, Any]],
    *,
    method: str = "isotonic",
    source: str | None = None,
) -> EmpiricalCalibrator:
    """Fit a winner-market calibrator directly from backtest ``runs``.

    Operator flow: run the deep backtest (``GET /api/f1/backtest`` / the
    ``F1BacktestService``), pass its ``runs`` here, then ``save()`` the result to
    the path named by ``F1_CALIBRATION_ARTIFACT`` to switch on serve-time
    calibration.
    """
    pairs = winner_pairs_from_runs(runs)
    if not pairs:
        return EmpiricalCalibrator.identity()
    probabilities = [p for p, _ in pairs]
    outcomes = [y for _, y in pairs]
    if method == "platt":
        return fit_platt(probabilities, outcomes, source=source)
    return fit_isotonic(probabilities, outcomes, source=source)


def brier_score(probabilities: Sequence[float], outcomes: Sequence[float]) -> float:
    pairs = _clean_pairs(probabilities, outcomes)
    if not pairs:
        return 0.0
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def load_default() -> EmpiricalCalibrator:
    """Load the calibrator from ``F1_CALIBRATION_ARTIFACT``; identity if unset/bad."""
    path = os.environ.get(CALIBRATION_ARTIFACT_ENV)
    if not path:
        return EmpiricalCalibrator.identity()
    try:
        calibrator = EmpiricalCalibrator.load(path)
        calibrator.source = calibrator.source or path
        return calibrator
    except Exception:
        return EmpiricalCalibrator.identity()
