"""End-to-end pre-game pipeline: history → ratings → features → ensemble.

Fits global + per-map Glicko ratings from finished matches, builds features per
upcoming match, and runs the calibrated ensemble — so the live /matches predictions
use the full model (winner / map1 / over-2.5 + confidence + provenance) instead of
the bare Glicko baseline. Duck-types the predictor interface (`predict_matches`,
`load_teams`) the API routes expect, plus `fit(past, teams)`.
"""

from __future__ import annotations

import logging

from games.csgo.analytics.model import CsgoEnsembleModel
from games.csgo.analytics.ratings import rate_maps, rate_matches
from games.csgo.features import FeatureExtractor
from games.csgo.models.csgo import CsgoMatch, CsgoPrediction, CsgoTeam

logger = logging.getLogger(__name__)


class CsgoModelPipeline:
    def __init__(self, model: CsgoEnsembleModel | None = None) -> None:
        self.model = model or CsgoEnsembleModel()
        self._past: list[CsgoMatch] = []
        self._ratings: dict = {}
        self._map_ratings: dict = {}
        self._teams_by_id: dict[int, CsgoTeam] = {}

    def fit(self, past_matches: list[CsgoMatch], teams: list[CsgoTeam]) -> None:
        self._past = list(past_matches or [])
        self._ratings = rate_matches(self._past)
        self._map_ratings = rate_maps(self._past)
        self._teams_by_id = {t.id: t for t in (teams or [])}

        # Fit a Platt scaler on the model's walk-forward predictions over history and apply
        # it to live predictions (None when too little history / no sklearn → stays raw).
        from games.csgo.analytics.backtest import fit_calibrator
        self.model.calibrator = fit_calibrator(self._past, self._teams_by_id)

        logger.info("CSGO pipeline fit: %d past matches, %d team ratings, %d map ratings, calibrated=%s",
                    len(self._past), len(self._ratings), len(self._map_ratings),
                    self.model.calibrator is not None)

    def load_teams(self, teams: list[CsgoTeam]) -> None:
        """Startup compatibility — teams only, before history is available."""
        self._teams_by_id = {t.id: t for t in (teams or [])}

    def predict(self, match: CsgoMatch) -> CsgoPrediction:
        feats = FeatureExtractor(self._past, self._ratings, self._teams_by_id,
                                 map_ratings=self._map_ratings).extract(match)
        out = self.model.predict(feats)
        return CsgoPrediction(
            team1_win_prob=out.winner_prob,
            team2_win_prob=round(1.0 - out.winner_prob, 4),
            confidence=out.confidence,
            model_version=out.model_version,
            map1_team1_win_prob=out.map1_team1_prob,
            over_2_5_maps_prob=out.over_2_5_maps_prob,
            feature_provenance=out.provenance,
        )

    def predict_matches(self, matches: list[CsgoMatch]) -> list[CsgoMatch]:
        for m in matches:
            if m.status == "SCHEDULED":
                m.prediction = self.predict(m)
        return matches
