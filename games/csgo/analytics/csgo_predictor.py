"""CSGO match predictor — baseline Elo win probability.

This is the reuse proof for the shared core: the model's probabilities feed
`common.ml.markets.edge` / `common.ml.markets.kelly` for edge + sizing, and
serialize to `common.ml.types.OutcomeProbability` for the shared Redis bridge
(`common.ml.bridge.outcome_publisher`, domain="csgo"). No betting/ML code is
re-implemented here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from math import comb

from common.ml.markets.edge import EdgeOpportunity, MarketQuote, compute_edge
from common.ml.markets.kelly import KellySize, size_position
from common.ml.types import OutcomeProbability
from games.csgo.models.csgo import CsgoMatch, CsgoPrediction, CsgoTeam

logger = logging.getLogger(__name__)

DOMAIN = "csgo"


class CsgoPredictor:
    """Predicts CS2/CSGO match outcomes from team Elo ratings."""

    def __init__(self) -> None:
        self._teams: dict[int, CsgoTeam] = {}
        self._version = "csgo-elo-v1"

    def load_teams(self, teams: list[CsgoTeam]) -> None:
        self._teams = {t.id: t for t in teams}
        logger.info("CSGO predictor loaded %d team ratings", len(self._teams))

    def _rating(self, team_id: int) -> float:
        team = self._teams.get(team_id)
        return team.rating if team else 1500.0

    def predict(self, match: CsgoMatch) -> CsgoPrediction:
        r1 = self._rating(match.team1_id)
        r2 = self._rating(match.team2_id)
        # Per-map Elo expected score, then lift to the series via best-of math.
        p_map = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
        p1 = self._best_of_win_prob(p_map, match.best_of)
        gap = abs(r1 - r2)
        confidence = round(0.4 + min(0.4, gap / 1000.0), 3)
        return CsgoPrediction(
            team1_win_prob=round(p1, 4),
            team2_win_prob=round(1.0 - p1, 4),
            confidence=confidence,
            model_version=self._version,
        )

    def predict_matches(self, matches: list[CsgoMatch]) -> list[CsgoMatch]:
        for m in matches:
            if m.status == "SCHEDULED":
                m.prediction = self.predict(m)
        return matches

    @staticmethod
    def _best_of_win_prob(p_map: float, best_of: int) -> float:
        """P(win a best-of-N) given per-map win prob — needs ceil(N/2) map wins."""
        if best_of <= 1:
            return p_map
        need = best_of // 2 + 1
        return sum(
            comb(best_of, k) * (p_map ** k) * ((1.0 - p_map) ** (best_of - k))
            for k in range(need, best_of + 1)
        )

    def to_outcome_probabilities(self, match: CsgoMatch) -> list[OutcomeProbability]:
        """Serialize the match prediction into the shared OutcomeProbability contract."""
        pred = match.prediction or self.predict(match)
        knowable = datetime.now(timezone.utc)
        code1 = match.team1_abbrev or match.team1
        code2 = match.team2_abbrev or match.team2
        return [
            OutcomeProbability(domain=DOMAIN, entity_id=match.id, entity_code=code1,
                               market="winner", probability=pred.team1_win_prob,
                               knowable_as_of=knowable, model_version=self._version),
            OutcomeProbability(domain=DOMAIN, entity_id=match.id, entity_code=code2,
                               market="winner", probability=pred.team2_win_prob,
                               knowable_as_of=knowable, model_version=self._version),
        ]

    def publish(self, matches: list[CsgoMatch], publisher) -> int:
        """Push every match's outcome probabilities over the shared Redis bridge.

        `publisher` is a common.ml.bridge.outcome_publisher.OutcomePublisher (or the
        InMemoryOutcomePublisher for tests/dry-runs)."""
        probs: list[OutcomeProbability] = []
        for m in matches:
            probs.extend(self.to_outcome_probabilities(m))
        publisher.publish_batch(probs)
        return len(probs)

    @staticmethod
    def edge_and_stake(
        model_prob: float,
        market_price: float,
        bankroll_usd: float = 1000.0,
        fee_bps: float = 200.0,
    ) -> tuple[EdgeOpportunity | None, KellySize]:
        """Reuse the shared betting core: edge vs. a market price + fractional-Kelly stake."""
        quote = MarketQuote(venue="kalshi", market_id="demo",
                            best_bid=market_price, best_ask=market_price, fee_bps=fee_bps)
        return compute_edge(model_prob, quote), size_position(model_prob, market_price, bankroll_usd)
