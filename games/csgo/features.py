"""Pre-game feature extraction for CS2 matches.

Turns identity + history (ratings, past matches, rosters) into a structured
`MatchFeatures` object the calibrated model consumes. Every feature degrades to a
neutral default when its data isn't available, so the same extractor works on stub
data and on a full PandaScore/HLTV history. Composable and side-effect free.

Priorities covered: team strength (P3 ratings), recent form adjusted for opponent
(P6), map-specific strength + likely veto (P4), roster stability / stand-ins (P5),
series-format effects (P7), event tier/context (P8), and low-weight head-to-head.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import comb
from typing import Optional

from games.csgo.analytics.ratings import Rating
from games.csgo.models.csgo import CsgoMatch, CsgoTeam

ACTIVE_DUTY_MAPS = ["Mirage", "Inferno", "Nuke", "Ancient", "Anubis", "Dust2", "Vertigo"]

_S_TIER = ("major", "katowice", "cologne", "world final", "blast premier world")
_A_TIER = ("iem", "esl pro league", "blast premier", "esl one", "pgl", "epl", "intel extreme")


@dataclass
class TeamFeatures:
    team_id: int
    rating: float = 1500.0
    rating_deviation: float = 350.0
    recent_form: float = 0.5                 # opponent-adjusted, 0..1
    map_strength: dict[str, float] = field(default_factory=dict)
    roster_stability: float = 1.0            # 1.0 = full core roster
    stand_in_count: int = 0


@dataclass
class MatchFeatures:
    match_id: str
    team1: TeamFeatures
    team2: TeamFeatures
    rating_diff: float = 0.0                 # team1 - team2
    form_diff: float = 0.0
    best_of: int = 3
    format_amplification: float = 0.6        # how a 60%/map team scores in this format
    likely_maps: list[str] = field(default_factory=list)
    event_tier: str = "B"
    event_tier_weight: float = 0.65
    h2h_team1_winrate: Optional[float] = None
    h2h_sample: int = 0
    provenance: dict[str, str] = field(default_factory=dict)


def classify_event_tier(name: Optional[str]) -> tuple[str, float]:
    n = (name or "").lower()
    if any(k in n for k in _S_TIER):
        return "S", 1.0
    if any(k in n for k in _A_TIER):
        return "A", 0.85
    return "B", 0.65


def _winner(m: CsgoMatch) -> Optional[int]:
    if m.winner_id is not None:
        return m.winner_id
    if isinstance(m.team1_score, int) and isinstance(m.team2_score, int) and m.team1_score != m.team2_score:
        return m.team1_id if m.team1_score > m.team2_score else m.team2_id
    return None


def _series_win_prob(p_map: float, best_of: int) -> float:
    if best_of <= 1:
        return p_map
    need = best_of // 2 + 1
    return sum(comb(best_of, k) * p_map ** k * (1 - p_map) ** (best_of - k)
               for k in range(need, best_of + 1))


class FeatureExtractor:
    def __init__(
        self,
        past_matches: list[CsgoMatch],
        ratings: dict[int, Rating] | None = None,
        teams_by_id: dict[int, CsgoTeam] | None = None,
    ) -> None:
        self._past = past_matches or []
        self._ratings = ratings or {}
        self._teams = teams_by_id or {}

    # ── per-team features ────────────────────────────────────────────────────
    def _recent_form(self, team_id: int, n: int = 10) -> float:
        games = sorted(
            [m for m in self._past if team_id in (m.team1_id, m.team2_id) and _winner(m) is not None],
            key=lambda x: x.date, reverse=True,
        )[:n]
        if not games:
            return 0.5
        num = den = 0.0
        for i, m in enumerate(games):
            opp = m.team2_id if m.team1_id == team_id else m.team1_id
            opp_r = self._ratings[opp].rating if opp in self._ratings else 1500.0
            weight = (opp_r / 1500.0) * (0.9 ** i)        # opponent quality × recency decay
            num += weight * (1.0 if _winner(m) == team_id else 0.0)
            den += weight
        return round(num / den, 4) if den else 0.5

    def _map_strength(self, team_id: int) -> dict[str, float]:
        wins: dict[str, int] = defaultdict(int)
        total: dict[str, int] = defaultdict(int)
        for m in self._past:
            if team_id not in (m.team1_id, m.team2_id):
                continue
            for ms in m.map_scores:
                if ms.winner_id is None or not ms.map_name:
                    continue
                total[ms.map_name] += 1
                if ms.winner_id == team_id:
                    wins[ms.map_name] += 1
        return {mp: round(wins[mp] / total[mp], 3) for mp in total if total[mp] > 0}

    def _roster(self, team_id: int, match_roster: list[int]) -> tuple[float, int]:
        core = set(self._teams[team_id].roster) if team_id in self._teams else set()
        if not match_roster or not core:
            return 1.0, 0
        stand_ins = [p for p in match_roster if p not in core]
        return round(max(0.0, 1.0 - len(stand_ins) / 5.0), 3), len(stand_ins)

    def _team_features(self, team_id: int, match_roster: list[int]) -> TeamFeatures:
        r = self._ratings.get(team_id)
        team = self._teams.get(team_id)
        rating = r.rating if r else (team.rating if team else 1500.0)
        rd = r.rd if r else (team.rating_deviation if team else 350.0)
        stability, stand_ins = self._roster(team_id, match_roster)
        return TeamFeatures(
            team_id=team_id,
            rating=round(rating, 1),
            rating_deviation=round(rd, 1),
            recent_form=self._recent_form(team_id),
            map_strength=self._map_strength(team_id),
            roster_stability=stability,
            stand_in_count=stand_ins,
        )

    # ── head-to-head (low weight / contextual) ───────────────────────────────
    def _h2h(self, t1: int, t2: int) -> tuple[Optional[float], int]:
        games = [m for m in self._past if {m.team1_id, m.team2_id} == {t1, t2} and _winner(m) is not None]
        if not games:
            return None, 0
        t1_wins = sum(1 for m in games if _winner(m) == t1)
        return round(t1_wins / len(games), 3), len(games)

    # ── likely veto / map path (heuristic until real veto data) ──────────────
    def _likely_maps(self, f1: TeamFeatures, f2: TeamFeatures, best_of: int) -> list[str]:
        # Maps both teams have played, ranked by combined comfort; cap at best_of.
        shared = set(f1.map_strength) | set(f2.map_strength)
        pool = [mp for mp in shared if mp in ACTIVE_DUTY_MAPS] or list(shared)
        if not pool:
            return ACTIVE_DUTY_MAPS[:max(1, best_of)]
        ranked = sorted(
            pool,
            key=lambda mp: f1.map_strength.get(mp, 0.5) + f2.map_strength.get(mp, 0.5),
            reverse=True,
        )
        return ranked[:max(1, best_of)]

    # ── public ───────────────────────────────────────────────────────────────
    def extract(self, match: CsgoMatch) -> MatchFeatures:
        f1 = self._team_features(match.team1_id, match.team1_roster)
        f2 = self._team_features(match.team2_id, match.team2_roster)
        tier, tier_w = classify_event_tier(match.event)
        h2h_wr, h2h_n = self._h2h(match.team1_id, match.team2_id)
        return MatchFeatures(
            match_id=match.id,
            team1=f1,
            team2=f2,
            rating_diff=round(f1.rating - f2.rating, 1),
            form_diff=round(f1.recent_form - f2.recent_form, 4),
            best_of=match.best_of,
            format_amplification=round(_series_win_prob(0.6, match.best_of), 4),
            likely_maps=self._likely_maps(f1, f2, match.best_of),
            event_tier=tier,
            event_tier_weight=tier_w,
            h2h_team1_winrate=h2h_wr,
            h2h_sample=h2h_n,
            provenance={
                "ratings": "glicko2" if self._ratings else "none",
                "history_matches": str(len(self._past)),
            },
        )
