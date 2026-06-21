"""Glicko-2 team strength ratings for CS2.

Glicko-2 improves on Elo by tracking each team's rating *uncertainty* (RD) and
*volatility*, so win probabilities are opponent-adjusted AND confidence reflects
how well-established a rating is (a 1-2 result by an unrated team moves less than
the same result by a tracked one). Ratings are fit by replaying finished matches
in chronological order; the per-match win probability + confidence feed the
baseline predictor.

Reference: Glickman, "Example of the Glicko-2 system" (2013).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_SCALE = 173.7178          # Glicko-2 internal scale
_BASE = 1500.0
_DEFAULT_RD = 350.0
_DEFAULT_VOL = 0.06


@dataclass
class Rating:
    rating: float = _BASE
    rd: float = _DEFAULT_RD
    vol: float = _DEFAULT_VOL


class Glicko2:
    def __init__(self, tau: float = 0.5) -> None:
        self.tau = tau

    @staticmethod
    def _g(phi: float) -> float:
        return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))

    @staticmethod
    def _expect(mu: float, mu_j: float, phi_j: float) -> float:
        return 1.0 / (1.0 + math.exp(-Glicko2._g(phi_j) * (mu - mu_j)))

    def update(self, r: Rating, opp: Rating, score: float) -> Rating:
        """Return r's new Rating after one game vs opp (score: 1 win / 0 loss / 0.5 draw)."""
        mu = (r.rating - _BASE) / _SCALE
        phi = r.rd / _SCALE
        mu_j = (opp.rating - _BASE) / _SCALE
        phi_j = opp.rd / _SCALE

        g = self._g(phi_j)
        e = self._expect(mu, mu_j, phi_j)
        v = 1.0 / (g * g * e * (1.0 - e))
        delta = v * g * (score - e)

        new_vol = self._new_volatility(phi, v, delta, r.vol)
        phi_star = math.sqrt(phi * phi + new_vol * new_vol)
        phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
        mu_new = mu + phi_new * phi_new * g * (score - e)

        return Rating(rating=_SCALE * mu_new + _BASE, rd=_SCALE * phi_new, vol=new_vol)

    def _new_volatility(self, phi: float, v: float, delta: float, sigma: float) -> float:
        a = math.log(sigma * sigma)
        tau2 = self.tau * self.tau
        phi2 = phi * phi

        def f(x: float) -> float:
            ex = math.exp(x)
            num = ex * (delta * delta - phi2 - v - ex)
            den = 2.0 * (phi2 + v + ex) ** 2
            return num / den - (x - a) / tau2

        A = a
        if delta * delta > phi2 + v:
            B = math.log(delta * delta - phi2 - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau

        fa, fb = f(A), f(B)
        for _ in range(100):
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            if fc * fb <= 0:
                A, fa = B, fb
            else:
                fa = fa / 2.0
            B, fb = C, fc
            if abs(B - A) <= 1e-6:
                break
        return math.exp(A / 2.0)


def rate_matches(matches, k_decay_days: float | None = None) -> dict[int, Rating]:
    """Fit Glicko-2 ratings from a chronological list of finished matches.

    Each match needs team1_id, team2_id, a winner (winner_id, else series score),
    and a date. Unknown teams start unrated (1500 / RD 350). Sequential per-match
    updates approximate rating periods of size 1 — fine for a continuously-played
    circuit like CS2.
    """
    engine = Glicko2()
    ratings: dict[int, Rating] = {}

    def winner_of(m) -> int | None:
        if getattr(m, "winner_id", None) is not None:
            return m.winner_id
        s1, s2 = getattr(m, "team1_score", None), getattr(m, "team2_score", None)
        if isinstance(s1, int) and isinstance(s2, int) and s1 != s2:
            return m.team1_id if s1 > s2 else m.team2_id
        return None

    for m in sorted(matches, key=lambda x: x.date):
        w = winner_of(m)
        if w is None:
            continue
        r1 = ratings.get(m.team1_id, Rating())
        r2 = ratings.get(m.team2_id, Rating())
        s1 = 1.0 if w == m.team1_id else 0.0
        # Update both off each other's PRE-match rating (standard).
        ratings[m.team1_id] = engine.update(r1, r2, s1)
        ratings[m.team2_id] = engine.update(r2, r1, 1.0 - s1)
    return ratings


def win_probability(r1: float, rd1: float, r2: float, rd2: float) -> float:
    """Symmetric per-map win probability of team1, dampened by rating uncertainty.

    Uses an RMS-combined RD so swapping the teams gives the complement (p1+p2=1).
    """
    rd_eff = math.sqrt((rd1 * rd1 + rd2 * rd2) / 2.0)
    g = Glicko2._g(rd_eff / _SCALE)
    return 1.0 / (1.0 + 10.0 ** (-g * (r1 - r2) / 400.0))


def confidence_from_rd(rd1: float, rd2: float, rd_max: float = _DEFAULT_RD) -> float:
    """0.5 (fully uncertain) .. ~0.95 (well-established). Pure estimate certainty."""
    mean_rd = (rd1 + rd2) / 2.0
    certainty = 1.0 - max(0.0, min(1.0, mean_rd / rd_max))
    return round(0.5 + 0.45 * certainty, 3)


def rate_maps(matches) -> dict[tuple[int, str], Rating]:
    """Per-(team, map) Glicko ratings from per-map results, in chronological order.

    CS2 strength is highly map-dependent, so a team carries a separate rating for
    each active-duty map. Replays every finished map (from each match's map_scores).
    """
    engine = Glicko2()
    ratings: dict[tuple[int, str], Rating] = {}
    events: list[tuple] = []
    for m in matches:
        for ms in getattr(m, "map_scores", None) or []:
            if ms.winner_id is None or not ms.map_name:
                continue
            events.append((m.date, m.team1_id, m.team2_id, ms.map_name, ms.winner_id))
    events.sort(key=lambda e: e[0])

    for _, t1, t2, map_name, winner in events:
        k1, k2 = (t1, map_name), (t2, map_name)
        r1 = ratings.get(k1, Rating())
        r2 = ratings.get(k2, Rating())
        s1 = 1.0 if winner == t1 else 0.0
        ratings[k1] = engine.update(r1, r2, s1)
        ratings[k2] = engine.update(r2, r1, 1.0 - s1)
    return ratings


def get_map_rating(map_ratings, team_id: int, map_name: str, fallback: float) -> tuple[float, float]:
    """A team's (rating, rd) on a map; falls back to its global rating with high RD."""
    r = map_ratings.get((team_id, map_name)) if map_ratings else None
    return (r.rating, r.rd) if r else (fallback, _DEFAULT_RD)
