"""CS2 calibrated ensemble model.

A transparent weighted-logistic layer on top of the Glicko baseline: it starts
from the rating-implied per-map probability (in logit space) and adds bounded
contributions from engineered features (form, map advantage, roster/stand-ins,
head-to-head). The result is lifted to the series via best-of math and can be
post-hoc calibrated (Platt/Isotonic from common.ml.calibration).

`feature_vector()` exposes the same inputs as a flat dict so a LightGBM/XGBoost
model can be trained and dropped in later behind the same interface — the linear
weights here are the explainable v1 baseline.

Outputs: winner (series), map1_winner (per-map), and over_2.5_maps (bo3 distance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from games.csgo.analytics.ratings import confidence_from_rd, win_probability
from games.csgo.features import MatchFeatures, _series_win_prob


@dataclass
class EnsembleWeights:
    """Contributions in logit units (modest, hand-set v1; tune/fit later)."""
    form: float = 0.8
    map_adv: float = 1.0
    map_glicko: float = 1.2
    roster: float = 1.2
    h2h: float = 0.3


@dataclass
class ModelOutput:
    winner_prob: float                       # team1 series win prob (calibrated)
    map1_team1_prob: float                   # per-map (map 1) win prob
    over_2_5_maps_prob: Optional[float]      # bo3: series goes the distance; else None
    confidence: float
    model_version: str
    provenance: dict[str, str] = field(default_factory=dict)


def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _avg_map_adv(f: MatchFeatures) -> float:
    """Mean (team1 - team2) map win-rate over the likely map pool, in [-1, 1]."""
    maps = f.likely_maps or []
    if not maps:
        return 0.0
    diffs = [f.team1.map_strength.get(m, 0.5) - f.team2.map_strength.get(m, 0.5) for m in maps]
    return sum(diffs) / len(diffs)


class CsgoEnsembleModel:
    VERSION = "csgo-ensemble-v1"

    def __init__(self, weights: EnsembleWeights | None = None, calibrator=None) -> None:
        self.weights = weights or EnsembleWeights()
        self.calibrator = calibrator  # optional object with .transform(np.ndarray)

    def feature_vector(self, f: MatchFeatures) -> dict[str, float]:
        """Flat numeric features — training surface for a future GBM."""
        base = win_probability(f.team1.rating, f.team1.rating_deviation,
                               f.team2.rating, f.team2.rating_deviation)
        return {
            "base_per_map": base,
            "rating_diff": f.rating_diff,
            "form_diff": f.form_diff,
            "map_adv": _avg_map_adv(f),
            "map_edge": f.map_edge,
            "roster_diff": f.team1.roster_stability - f.team2.roster_stability,
            "stand_ins": float(f.team1.stand_in_count + f.team2.stand_in_count),
            "h2h_centered": (0.0 if f.h2h_team1_winrate is None else (f.h2h_team1_winrate - 0.5)),
            "h2h_shrink": (0.0 if not f.h2h_sample else f.h2h_sample / (f.h2h_sample + 5.0)),
            "best_of": float(f.best_of),
            "event_tier_weight": f.event_tier_weight,
        }

    def _per_map_prob(self, f: MatchFeatures) -> float:
        v = self.feature_vector(f)
        w = self.weights
        logit = _logit(v["base_per_map"])
        logit += w.form * v["form_diff"]
        logit += w.map_adv * v["map_adv"]
        logit += w.map_glicko * v["map_edge"]
        logit += w.roster * v["roster_diff"]
        logit += w.h2h * v["h2h_centered"] * v["h2h_shrink"]
        p = _sigmoid(logit)
        if self.calibrator is not None:
            import numpy as np
            p = float(self.calibrator.transform(np.array([p]))[0])
        return min(1 - 1e-6, max(1e-6, p))

    def predict(self, f: MatchFeatures) -> ModelOutput:
        p_map = self._per_map_prob(f)
        winner = _series_win_prob(p_map, f.best_of)

        over_2_5 = None
        if f.best_of == 3:
            over_2_5 = round(2.0 * p_map * (1.0 - p_map), 4)   # P(series reaches map 3)

        # Confidence: rating certainty, reduced when a stand-in is present.
        conf = confidence_from_rd(f.team1.rating_deviation, f.team2.rating_deviation)
        if f.team1.stand_in_count or f.team2.stand_in_count:
            conf = round(conf * 0.85, 3)

        return ModelOutput(
            winner_prob=round(winner, 4),
            map1_team1_prob=round(p_map, 4),
            over_2_5_maps_prob=over_2_5,
            confidence=conf,
            model_version=self.VERSION,
            provenance={**f.provenance, "model": self.VERSION,
                        "calibrated": "yes" if self.calibrator is not None else "no"},
        )
