"""Trained win-probability model (A1) — learns feature weights from history.

The served model is a hand-tuned heuristic Monte-Carlo; its probabilities are
only mildly better than uniform (see the A7 validation harness). This module
learns a model that maps the *same* leakage-safe per-driver features (the backtest
``component_scores``) to a win probability, fit on past races and evaluated on
held-out races via the same walk-forward protocol.

It is deliberately conservative: a regularised **logistic** model is the default
because, on the small F1 sample (dozens of races/season), tree ensembles overfit —
the walk-forward numbers, not intuition, pick the winner. The output is a per-race
softmax-normalised field (a winner market sums to 1).

This module is the *training + evaluation* half. Wiring a fitted artifact into the
serve path (so live predictions use it) is a separate step (roadmap A5); until then
this proves — with an honest out-of-sample number — whether a trained model beats
the heuristic before anything depends on it.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from sports.f1.predictor.backtesting.validation import sort_rows

WIN_MODEL_ARTIFACT_ENV = "F1_WIN_MODEL_ARTIFACT"

# Leakage-safe pre-race features exposed per driver in backtest component_scores.
# NB: 'performance' is excluded — it is the heuristic's own output (would leak).
FEATURES: list[str] = [
    "qualifying_pace", "race_pace", "form", "car_performance", "driver_skill",
    "grid_position", "grid_penalty", "reliability", "dnf_probability",
    "track_fit", "tire_strategy", "weather_risk", "team", "sentiment",
]

_EPS = 1e-6


def _num(value: Any) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def build_field(row: dict[str, Any]) -> tuple[list[str], list[list[float]], list[float]]:
    """From one full backtest race row → (driver_ids, feature_matrix, is_winner)."""
    component_scores = row.get("component_scores") or {}
    actual = row.get("actual_winner")
    if isinstance(actual, dict):
        actual = actual.get("driver_id") or actual.get("driver")
    driver_ids: list[str] = []
    matrix: list[list[float]] = []
    labels: list[float] = []
    for driver_id, feats in component_scores.items():
        driver_ids.append(str(driver_id))
        matrix.append([_num((feats or {}).get(f)) for f in FEATURES])
        labels.append(1.0 if driver_id == actual else 0.0)
    return driver_ids, matrix, labels


def _softmax_normalise(scores: Sequence[float]) -> list[float]:
    clipped = [max(s, 1e-9) for s in scores]
    total = sum(clipped)
    if total <= 0:
        n = max(len(clipped), 1)
        return [1.0 / n] * len(clipped)
    return [s / total for s in clipped]


def _make_estimator(kind: str):
    """Lazily construct an sklearn estimator (import deferred to keep module light)."""
    if kind == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, class_weight="balanced")


class TrainedWinModel:
    """Fit on backtest rows; predict a per-race win-probability field."""

    def __init__(self, kind: str = "logistic"):
        self.kind = kind if kind in {"logistic", "gbm"} else "logistic"
        self._est = None
        self._fitted = False

    def fit(self, rows: Sequence[dict[str, Any]]) -> "TrainedWinModel":
        X: list[list[float]] = []
        y: list[float] = []
        for row in rows:
            _, matrix, labels = build_field(row)
            X.extend(matrix)
            y.extend(labels)
        # need at least two classes to fit a classifier
        if len(X) < 2 or len(set(y)) < 2:
            self._fitted = False
            return self
        self._est = _make_estimator(self.kind)
        self._est.fit(X, y)
        self._fitted = True
        return self

    def is_fitted(self) -> bool:
        return self._fitted

    def predict_field(self, row: dict[str, Any]) -> dict[str, float]:
        driver_ids, matrix, _ = build_field(row)
        if not driver_ids:
            return {}
        if not self._fitted:
            share = 1.0 / len(driver_ids)
            return {d: share for d in driver_ids}
        raw = [float(p[1]) for p in self._est.predict_proba(matrix)]
        normed = _softmax_normalise(raw)
        return {driver_ids[i]: normed[i] for i in range(len(driver_ids))}

    def predicted_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """A race row whose probability_distribution comes from this model —
        shaped so the A7 validation metrics can score it directly."""
        field = self.predict_field(row)
        return {
            "season": row.get("season"),
            "round": row.get("round"),
            "actual_winner": row.get("actual_winner"),
            "probability_distribution": [
                {"driver_id": d, "win_probability": p} for d, p in field.items()
            ],
        }

    def linear_artifact(self, source: str | None = None) -> dict[str, Any] | None:
        """Serialise a fitted **logistic** model as plain linear coefficients.

        The logistic model is just intercept + weights on the features, so it can
        be applied at serve time with a pure-Python sigmoid — no sklearn, no
        pickle, versionless JSON (mirrors the empirical calibrator artifact).
        Returns None for unfitted or non-logistic (e.g. gbm) models.
        """
        if self.kind != "logistic" or not self._fitted:
            return None
        return {
            "method": "logistic",
            "features": list(FEATURES),
            "weights": [float(w) for w in self._est.coef_[0]],
            "intercept": float(self._est.intercept_[0]),
            "source": source,
        }


@dataclass
class ServeWinModel:
    """Pure-Python serve-time applier for the trained win model.

    Applies the linear (logistic) model to each driver's features and normalises
    the field to sum to 1 (winner market) — identical to
    ``TrainedWinModel.predict_field`` but with no sklearn dependency at serve time.
    Safe by default: an empty/identity model leaves probabilities unchanged.
    """

    features: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    intercept: float = 0.0
    method: str = "identity"
    source: str | None = None

    def is_identity(self) -> bool:
        return not self.weights

    def _win_score(self, feats: dict[str, Any]) -> float:
        z = self.intercept
        for name, weight in zip(self.features, self.weights):
            z += weight * _num((feats or {}).get(name))
        z = max(-60.0, min(60.0, z))
        return 1.0 / (1.0 + math.exp(-z))  # sigmoid

    def apply_field(self, components_by_driver: dict[str, dict[str, Any]]) -> dict[str, float]:
        """{driver_id: components} -> {driver_id: normalised win probability}."""
        if self.is_identity() or not components_by_driver:
            n = max(len(components_by_driver), 1)
            return {d: 1.0 / n for d in components_by_driver}
        scores = {d: self._win_score(f) for d, f in components_by_driver.items()}
        total = sum(scores.values())
        if total <= 0:
            n = max(len(scores), 1)
            return {d: 1.0 / n for d in scores}
        return {d: v / total for d, v in scores.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "features": list(self.features),
            "weights": [round(float(w), 8) for w in self.weights],
            "intercept": round(float(self.intercept), 8),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServeWinModel":
        weights = [float(w) for w in (data.get("weights") or [])]
        return cls(
            features=[str(f) for f in (data.get("features") or [])],
            weights=weights,
            intercept=float(data.get("intercept") or 0.0),
            method=str(data.get("method") or ("logistic" if weights else "identity")),
            source=data.get("source"),
        )

    @classmethod
    def identity(cls) -> "ServeWinModel":
        return cls()

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "ServeWinModel":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def fit_serve_win_model(rows: Sequence[dict[str, Any]], *, source: str | None = None) -> ServeWinModel:
    """Fit a logistic win model on backtest rows and return a serve applier."""
    artifact = TrainedWinModel("logistic").fit(rows).linear_artifact(source=source)
    if not artifact:
        return ServeWinModel.identity()
    return ServeWinModel.from_dict(artifact)


_default_win_model: ServeWinModel | None = None


def load_default_win_model() -> ServeWinModel:
    """Load the serve win model from ``F1_WIN_MODEL_ARTIFACT`` (identity if unset).

    Cached; call :func:`reset_default_win_model` after (re)fitting to reload.
    """
    global _default_win_model
    if _default_win_model is not None:
        return _default_win_model
    path = os.environ.get(WIN_MODEL_ARTIFACT_ENV)
    if not path:
        _default_win_model = ServeWinModel.identity()
        return _default_win_model
    try:
        model = ServeWinModel.load(path)
        model.source = model.source or path
        _default_win_model = model
    except Exception:
        _default_win_model = ServeWinModel.identity()
    return _default_win_model


def reset_default_win_model() -> None:
    global _default_win_model
    _default_win_model = None


# --------------------------------------------------------------------------
# Metrics (self-contained so this module does not reach into validation internals)
# --------------------------------------------------------------------------
def _winner_pairs(rows: Sequence[dict[str, Any]]) -> list[tuple[float, float]]:
    pairs = []
    for row in rows:
        actual = row.get("actual_winner")
        for item in row.get("probability_distribution") or []:
            did = item.get("driver_id")
            prob = item.get("win_probability")
            if did is None or prob is None:
                continue
            pairs.append((float(prob), 1.0 if did == actual else 0.0))
    return pairs


def _uniform_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        field = row.get("probability_distribution") or []
        n = len(field)
        share = (1.0 / n) if n else 0.0
        out.append({
            "actual_winner": row.get("actual_winner"),
            "probability_distribution": [
                {"driver_id": d.get("driver_id"), "win_probability": share} for d in field
            ],
        })
    return out


def _brier(pairs: Sequence[tuple[float, float]]) -> float:
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else 0.0


def _log_loss_winner(rows: Sequence[dict[str, Any]]) -> float:
    losses = []
    for row in rows:
        actual = row.get("actual_winner")
        p = next((float(i.get("win_probability") or 0.0) for i in row.get("probability_distribution") or [] if i.get("driver_id") == actual), None)
        if p is None:
            continue
        losses.append(-math.log(min(1 - _EPS, max(_EPS, p))))
    return sum(losses) / len(losses) if losses else 0.0


def _winner_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    correct = graded = 0
    for row in rows:
        field = row.get("probability_distribution") or []
        if not field:
            continue
        graded += 1
        top = max(field, key=lambda i: float(i.get("win_probability") or 0.0))
        if top.get("driver_id") == row.get("actual_winner"):
            correct += 1
    return round(correct / graded, 4) if graded else 0.0


def _block(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pairs = _winner_pairs(rows)
    return {
        "brier": round(_brier(pairs), 6),
        "log_loss": round(_log_loss_winner(rows), 6),
        "winner_accuracy": _winner_accuracy(rows),
        "samples": len(pairs),
    }


def walk_forward_train_eval(
    full_rows: Sequence[dict[str, Any]],
    *,
    model_kind: str = "logistic",
    min_train_races: int = 20,
    val_block: int = 5,
) -> dict[str, Any]:
    """Walk-forward retrain: each fold trains a ``TrainedWinModel`` on prior races
    and predicts the held-out block. Returns pooled out-of-sample metrics for the
    trained model vs the heuristic (the row's own distribution) vs a uniform
    baseline, plus a gate: does the trained model beat the heuristic OOS?"""
    rows = [r for r in sort_rows(full_rows) if r.get("component_scores")]
    n = len(rows)
    if n < int(min_train_races) + 1:
        return {
            "ok": False,
            "reason": "not_enough_races_for_walk_forward",
            "race_count": n,
            "min_required": int(min_train_races) + 1,
        }

    trained_val: list[dict[str, Any]] = []
    heuristic_val: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    cut = int(min_train_races)
    while cut < n:
        val = rows[cut : cut + int(val_block)]
        if not val:
            break
        model = TrainedWinModel(model_kind).fit(rows[:cut])
        trained_block = [model.predicted_row(r) for r in val]
        trained_val.extend(trained_block)
        heuristic_val.extend(val)
        fold_reports.append({
            "index": len(fold_reports),
            "train_size": cut,
            "val_size": len(val),
            "fitted": model.is_fitted(),
            "trained": _block(trained_block),
            "heuristic": _block(val),
        })
        cut += int(val_block)

    trained = _block(trained_val)
    heuristic = _block(heuristic_val)
    baseline = _block(_uniform_rows(heuristic_val))

    beats_heuristic = trained["brier"] < heuristic["brier"] and trained["log_loss"] < heuristic["log_loss"]
    beats_baseline = trained["brier"] < baseline["brier"] and trained["log_loss"] < baseline["log_loss"]
    reasons = []
    if beats_heuristic:
        reasons.append("trained model beats the heuristic on Brier and log loss out-of-sample")
    else:
        reasons.append(f"trained model does not clearly beat the heuristic (Brier {trained['brier']} vs {heuristic['brier']}, log loss {trained['log_loss']} vs {heuristic['log_loss']})")
    if not beats_baseline:
        reasons.append("trained model does not beat the uniform baseline")

    return {
        "ok": True,
        "config": {"model_kind": model_kind, "min_train_races": int(min_train_races), "val_block": int(val_block), "features": FEATURES},
        "race_count": n,
        "fold_count": len(fold_reports),
        "aggregate": {"trained": trained, "heuristic": heuristic, "baseline_uniform": baseline},
        "improvement": {
            "trained_vs_heuristic_brier_delta": round(trained["brier"] - heuristic["brier"], 6),
            "trained_vs_heuristic_log_loss_delta": round(trained["log_loss"] - heuristic["log_loss"], 6),
        },
        "gate": {
            "beats_heuristic": beats_heuristic,
            "beats_baseline": beats_baseline,
            "verdict": (
                "TRAINED MODEL WINS — beats the heuristic out-of-sample; candidate to wire into serving (A5)"
                if beats_heuristic else
                "NO IMPROVEMENT YET — trained model does not beat the heuristic out-of-sample on these features"
            ),
            "reasons": reasons,
        },
        "folds": fold_reports,
    }
