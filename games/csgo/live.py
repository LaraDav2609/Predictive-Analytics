"""Live in-game state + a transparent live win-probability updater for CS2.

Takes a pre-game probability and the current match state (series score, map round
score, side, man-count, bomb, economy, opening kill) and returns an updated series
win probability — plus the current-map and current-round probabilities, and the
drivers that moved it (for explainability). Pure and deterministic, so it's tested
without Redis. The result serializes to the shared OutcomeProbability contract and
publishes over the csgo:prob:* bridge when a publisher is supplied.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import comb
from typing import Optional

from pydantic import BaseModel

from common.ml.types import OutcomeProbability

MODEL_VERSION = "csgo-live-v1"
DOMAIN = "csgo"


class LiveMatchState(BaseModel):
    best_of: int = 3
    maps_won_team1: int = 0
    maps_won_team2: int = 0

    current_map: Optional[str] = None
    map_in_progress: bool = True
    team1_rounds: int = 0
    team2_rounds: int = 0
    rounds_to_win: int = 13               # CS2 MR12 → first to 13

    team1_side: Optional[str] = None      # "CT" | "T"
    players_alive_team1: Optional[int] = None   # current round, 0..5
    players_alive_team2: Optional[int] = None
    bomb_planted: bool = False
    bomb_planter_team: Optional[int] = None     # 1 | 2
    opening_kill_team: Optional[int] = None     # 1 | 2
    team1_equipment_value: Optional[int] = None
    team2_equipment_value: Optional[int] = None
    note: Optional[str] = None


class LiveProbability(BaseModel):
    team1_win_prob: float                 # series
    team2_win_prob: float
    current_map_team1_prob: Optional[float] = None
    round_team1_prob: Optional[float] = None
    model_version: str = MODEL_VERSION
    drivers: dict = {}


def _race(a: int, b: int, p: float) -> float:
    """P(A reaches `a` wins before B reaches `b`), per-event win prob p for A."""
    if a <= 0:
        return 1.0
    if b <= 0:
        return 0.0
    p = min(1 - 1e-9, max(1e-9, p))
    # A wins iff it gets its a-th success before B gets its b-th:
    #   sum_{k=0}^{b-1} C(a-1+k, k) p^a (1-p)^k
    return sum(comb(a - 1 + k, k) * p ** a * (1 - p) ** k for k in range(b))


def _q_base(per_map_prob: float, state: LiveMatchState) -> float:
    """Steady per-round win prob for team1: a damped version of the map edge, plus side."""
    q = 0.5 + (per_map_prob - 0.5) * 0.4
    if state.team1_side == "CT":
        q += 0.02
    elif state.team1_side == "T":
        q -= 0.02
    return min(0.9, max(0.1, q))


def _round_prob(state: LiveMatchState, q_base: float) -> tuple[float, dict]:
    p = q_base
    drivers: dict = {}
    a1, a2 = state.players_alive_team1, state.players_alive_team2
    if a1 is not None and a2 is not None:
        diff = a1 - a2
        p += 0.09 * diff                    # ~9% per man advantage
        drivers["man_advantage"] = diff
    if state.bomb_planted and state.bomb_planter_team in (1, 2):
        p += 0.15 if state.bomb_planter_team == 1 else -0.15
        drivers["bomb_planted_by"] = state.bomb_planter_team
    if state.opening_kill_team in (1, 2):
        p += 0.06 if state.opening_kill_team == 1 else -0.06
        drivers["opening_kill"] = state.opening_kill_team
    return min(0.98, max(0.02, p)), drivers


def _map_prob(state: LiveMatchState, q: float) -> tuple[float, Optional[float], dict]:
    a1 = state.rounds_to_win - state.team1_rounds
    a2 = state.rounds_to_win - state.team2_rounds
    if a1 <= 0:
        return 1.0, None, {}
    if a2 <= 0:
        return 0.0, None, {}

    has_round = (state.players_alive_team1 is not None or state.bomb_planted
                 or state.opening_kill_team is not None)
    if has_round:
        p_round, drivers = _round_prob(state, q)
        # Current round is live: fold it in, then race the remainder at q.
        prob = p_round * _race(a1 - 1, a2, q) + (1 - p_round) * _race(a1, a2 - 1, q)
        return prob, p_round, drivers
    return _race(a1, a2, q), None, {}


def update_live_probability(
    pregame_team1_prob: float,
    state: LiveMatchState,
    pregame_per_map_prob: Optional[float] = None,
) -> LiveProbability:
    per_map = pregame_per_map_prob if pregame_per_map_prob is not None else pregame_team1_prob
    q = _q_base(per_map, state)
    drivers: dict = {"q_base": round(q, 3)}

    maps_needed = state.best_of // 2 + 1
    m1 = maps_needed - state.maps_won_team1
    m2 = maps_needed - state.maps_won_team2
    if m1 <= 0:
        return LiveProbability(team1_win_prob=1.0, team2_win_prob=0.0,
                               drivers={"series": "team1_clinched"})
    if m2 <= 0:
        return LiveProbability(team1_win_prob=0.0, team2_win_prob=1.0,
                               drivers={"series": "team2_clinched"})

    cur_map_prob: Optional[float] = None
    round_prob: Optional[float] = None
    if state.map_in_progress and state.current_map is not None:
        cur_map_prob, round_prob, rd = _map_prob(state, q)
        drivers.update(rd)
        drivers["current_map_prob"] = round(cur_map_prob, 3)
        series = cur_map_prob * _race(m1 - 1, m2, per_map) + (1 - cur_map_prob) * _race(m1, m2 - 1, per_map)
    else:
        series = _race(m1, m2, per_map)

    series = min(1 - 1e-6, max(1e-6, series))
    return LiveProbability(
        team1_win_prob=round(series, 4),
        team2_win_prob=round(1 - series, 4),
        current_map_team1_prob=round(cur_map_prob, 4) if cur_map_prob is not None else None,
        round_team1_prob=round(round_prob, 4) if round_prob is not None else None,
        drivers=drivers,
    )


def live_outcome_probabilities(match_id: str, code1: str, code2: str,
                               live: LiveProbability) -> list[OutcomeProbability]:
    """Serialize a live update into the shared OutcomeProbability contract (market='winner')."""
    now = datetime.now(timezone.utc)
    return [
        OutcomeProbability(domain=DOMAIN, entity_id=match_id, entity_code=code1, market="winner",
                           probability=live.team1_win_prob, knowable_as_of=now, model_version=MODEL_VERSION),
        OutcomeProbability(domain=DOMAIN, entity_id=match_id, entity_code=code2, market="winner",
                           probability=live.team2_win_prob, knowable_as_of=now, model_version=MODEL_VERSION),
    ]


def publish_live(match_id: str, code1: str, code2: str, live: LiveProbability, publisher) -> int:
    """Push a live update over the shared Redis bridge (csgo:prob:*). `publisher` is any
    common.ml.bridge OutcomePublisher (or InMemoryOutcomePublisher in tests)."""
    probs = live_outcome_probabilities(match_id, code1, code2, live)
    publisher.publish_batch(probs)
    return len(probs)
