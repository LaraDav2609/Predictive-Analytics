"""Decomposable MLB game win-probability model.

Produces a home win probability as an **additive breakdown of named analysis
components** — a 0.500 coin-flip baseline plus a contribution from each signal
(team strength, recent form, home-field, starting pitcher) — so the dashboard can
show every signal's contribution next to the combined number, the way the F1
"why this pick" decomposition does.

Everything is computed from a light ``TeamState`` (a running win/loss + runs
for/against + last-10 record), so the SAME code serves live predictions (state
from current standings) and the leak-free backtest (state rebuilt from only prior
games). Starting pitcher / bullpen / park are structured slots that stay neutral
until wired to richer (still free) data — surfaced honestly rather than hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# MLB home teams win ~54% across modern seasons — the standing home-field edge.
HOME_FIELD_EDGE = 0.035
PYTHAG_EXP = 1.83          # Pythagenpat-ish exponent for run-based win expectancy
FORM_WEIGHT = 0.08         # how much a full last-10 gap swings the probability
FORM_CAP = 0.05
PITCHER_WEIGHT = 0.020     # per run of prior-season ERA gap between the starters
PITCHER_CAP = 0.06
PROB_FLOOR, PROB_CEIL = 0.05, 0.95


@dataclass
class TeamState:
    """A team's record as of some point in time (used live and in backtest)."""
    runs_for: float = 0.0
    runs_against: float = 0.0
    wins: int = 0
    losses: int = 0
    last10: list[int] = field(default_factory=list)   # most-recent-last, 1=win 0=loss

    @property
    def games(self) -> int:
        return self.wins + self.losses

    def record_game(self, runs_for: int, runs_against: int) -> None:
        self.runs_for += runs_for
        self.runs_against += runs_against
        won = 1 if runs_for > runs_against else 0
        self.wins += won
        self.losses += 1 - won
        self.last10.append(won)
        if len(self.last10) > 10:
            self.last10.pop(0)

    def pythag_wpct(self) -> float:
        rf, ra = self.runs_for, self.runs_against
        if rf <= 0 and ra <= 0:
            return 0.5
        return rf ** PYTHAG_EXP / (rf ** PYTHAG_EXP + ra ** PYTHAG_EXP)

    def last10_pct(self) -> float:
        return sum(self.last10) / len(self.last10) if self.last10 else 0.5


def log5(p_home: float, p_away: float) -> float:
    """Bill James log5: P(home beats away) from each team's win pct vs a .500 field."""
    denom = p_home * (1 - p_away) + (1 - p_home) * p_away
    return 0.5 if denom <= 0 else (p_home * (1 - p_away)) / denom


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def predict_game(
    home: TeamState,
    away: TeamState,
    *,
    home_pitcher: str | None = None,
    away_pitcher: str | None = None,
    home_pitcher_era: float | None = None,
    away_pitcher_era: float | None = None,
    min_games: int = 10,
) -> dict[str, Any]:
    """Home win probability + its additive component breakdown.

    Contributions sum (with the 0.500 baseline) to the final probability, so the UI
    can render a clean waterfall: 0.500 -> +strength -> +home field -> +form ->
    +pitcher -> final. Returns confidence 0 when neither team has ``min_games`` of
    history (the model refuses to pretend it has a read)."""
    h_pyth, a_pyth = home.pythag_wpct(), away.pythag_wpct()
    strength = log5(h_pyth, a_pyth) - 0.5                       # matchup lean vs coin flip
    form = _clamp((home.last10_pct() - away.last10_pct()) * FORM_WEIGHT, -FORM_CAP, FORM_CAP)
    home_field = HOME_FIELD_EDGE
    # Starting pitcher: prior-season ERA gap (lower ERA is better → favours that team).
    pitcher = 0.0
    pitcher_modeled = False
    if home_pitcher_era is not None and away_pitcher_era is not None:
        pitcher = _clamp((float(away_pitcher_era) - float(home_pitcher_era)) * PITCHER_WEIGHT, -PITCHER_CAP, PITCHER_CAP)
        pitcher_modeled = True

    raw = 0.5 + strength + home_field + form + pitcher
    home_win_prob = _clamp(raw, PROB_FLOOR, PROB_CEIL)

    enough = home.games >= min_games and away.games >= min_games
    confidence = _clamp(min(home.games, away.games) / 60.0, 0.0, 0.9) if enough else 0.0

    components = {
        "team_strength": {
            "contribution": round(strength, 4),
            "home_pyth_wpct": round(h_pyth, 3),
            "away_pyth_wpct": round(a_pyth, 3),
            "home_record": f"{home.wins}-{home.losses}",
            "away_record": f"{away.wins}-{away.losses}",
            "detail": "Pythagorean win expectancy (runs for/against) combined via log5",
        },
        "home_field": {
            "contribution": round(home_field, 4),
            "detail": "MLB home teams win ~54%",
        },
        "recent_form": {
            "contribution": round(form, 4),
            "home_last10": f"{sum(home.last10)}-{len(home.last10) - sum(home.last10)}",
            "away_last10": f"{sum(away.last10)}-{len(away.last10) - sum(away.last10)}",
            "detail": "Last-10 win-rate gap, bounded",
        },
        "starting_pitcher": {
            "contribution": round(pitcher, 4),
            "modeled": pitcher_modeled,
            "home": home_pitcher,
            "away": away_pitcher,
            "home_rating": round(float(home_pitcher_era), 2) if home_pitcher_era is not None else None,
            "away_rating": round(float(away_pitcher_era), 2) if away_pitcher_era is not None else None,
            "detail": ("Starter form (FIP-scale, lower is better) — the better starter favours that team" if pitcher_modeled
                       else "No reliable starter form for one side — neutral"),
        },
    }
    return {
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(1.0 - home_win_prob, 4),
        "confidence": round(confidence, 4),
        "leak_free": enough,
        "baseline": 0.5,
        "components": components,
        "model_version": "mlb-decomp-v1",
    }
