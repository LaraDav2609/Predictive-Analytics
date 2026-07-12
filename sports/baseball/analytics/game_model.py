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
# Run environment (park + weather): a park/wind that inflates run scoring adds
# variance, which very slightly regresses the favourite toward the coin flip. Kept
# tiny and bounded — the goal is a decomposable, learnable signal, not a hand-tuned
# edge. Sign: a hotter run environment shrinks |strength lean| a touch.
ENV_WEIGHT = 0.015         # per unit of (run-environment index - 1.0)
ENV_CAP = 0.02
# Schedule fatigue / rest (from the schedule alone, leak-free): a fresher / less-taxed
# home side (more days rest, fewer games in the trailing week than the away side) gets a
# tiny nudge up. Bounded and 0 (flagged not-modeled) when no fatigue data is supplied so
# default serve behaviour is unchanged. Effect expected small — MLB schedules are balanced.
REST_WEIGHT = 0.004        # per net day of rest advantage (home - away), capped
DENSITY_WEIGHT = 0.003     # per net fewer games in the trailing window (home - away), capped
FATIGUE_CAP = 0.015
# Bullpen fatigue (from REAL boxscore reliever workload, leak-free): a team whose
# bullpen threw a lot of pitches over the prior 1-3 days is a touch more vulnerable
# tonight. The game-level signal is the load GAP (home minus away, in normalized
# ~150-pitch units); a MORE-taxed home pen nudges the home win probability DOWN.
# Bounded and 0 (flagged not-modeled) when no bullpen usage is supplied so default
# serve behaviour is unchanged. Effect expected small — much of it is priced already.
BULLPEN_WEIGHT = 0.010     # per unit of normalized reliever-pitch load gap (home - away)
BULLPEN_CAP = 0.015
# Confirmed lineups (from the REAL boxscore batting order, leak-free): each team's mean
# PRIOR-SEASON wOBA over its 9 confirmed starters. The game-level signal is the quality
# GAP (home minus away, in wOBA units ~±0.05 at the extremes); a better-hitting home
# lineup nudges the home win probability UP. Bounded and 0 (flagged not-modeled) when no
# lineup dict is supplied so default serve behaviour is unchanged. Effect expected small
# — v1's anchor is LAST season's quality, stale for the call-ups where it should matter,
# and the market prices confirmed lineups fast.
LINEUP_WEIGHT = 0.60       # per wOBA point of lineup-quality gap (home - away)
LINEUP_CAP = 0.03
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
    park_factor: float | None = None,
    weather: dict | None = None,
    fatigue: dict | None = None,
    bullpen: dict | None = None,
    lineup: dict | None = None,
    min_games: int = 10,
) -> dict[str, Any]:
    """Home win probability + its additive component breakdown.

    Contributions sum (with the 0.500 baseline) to the final probability, so the UI
    can render a clean waterfall: 0.500 -> +strength -> +home field -> +form ->
    +pitcher -> final. Returns confidence 0 when neither team has ``min_games`` of
    history (the model refuses to pretend it has a read)."""
    h_pyth, a_pyth = home.pythag_wpct(), away.pythag_wpct()
    strength = log5(h_pyth, a_pyth) - 0.5                       # matchup lean vs coin flip
    form_gap = home.last10_pct() - away.last10_pct()            # raw last-10 gap (pre-weight)
    form = _clamp(form_gap * FORM_WEIGHT, -FORM_CAP, FORM_CAP)
    home_field = HOME_FIELD_EDGE
    # Starting pitcher: prior-season ERA gap (lower ERA is better → favours that team).
    pitcher = 0.0
    pitcher_modeled = False
    pitcher_gap = 0.0
    if home_pitcher_era is not None and away_pitcher_era is not None:
        pitcher_gap = float(away_pitcher_era) - float(home_pitcher_era)   # raw ERA/FIP gap
        pitcher = _clamp(pitcher_gap * PITCHER_WEIGHT, -PITCHER_CAP, PITCHER_CAP)
        pitcher_modeled = True

    # Run environment (OPTIONAL): park run factor + (if present) wind blowing out to
    # center. A hotter run environment adds scoring variance, nudging the favourite a
    # hair back toward the coin flip. Decomposable: its own bounded additive term that
    # is exactly 0 (and flagged not-modeled) when no park/weather data is supplied, so
    # default serve behaviour and the existing components are unchanged.
    env = 0.0
    env_modeled = False
    env_index = 1.0            # 1.0 == neutral run environment
    wind_out = None
    if park_factor is not None or weather:
        pf = float(park_factor) if park_factor is not None else 1.0
        env_index = pf
        if weather and not weather.get("domed"):
            from sports.baseball.data.weather_client import wind_out_component
            wind_out = wind_out_component(
                weather.get("wind_speed_mph"), weather.get("wind_direction_deg"),
                weather.get("cf_azimuth"))
            if wind_out is not None:
                env_index += 0.010 * wind_out       # ~+0.10 run-env per 10mph out to CF
            temp = weather.get("temperature_f")
            if temp is not None:
                env_index += 0.003 * (float(temp) - 70.0)   # warm air carries
        # Direction: shrink the strength lean toward 0.5 in a hot run environment.
        env = _clamp(-strength * ENV_WEIGHT * (env_index - 1.0) / 0.01
                     if strength else 0.0, -ENV_CAP, ENV_CAP)
        env_modeled = True

    # Schedule fatigue / rest (OPTIONAL, leak-free from the schedule alone). A fresher
    # home side (more days rest, fewer games in the trailing window than the away side)
    # nudges the home win probability up a hair. Decomposable: its own bounded additive
    # term, exactly 0 (and flagged not-modeled) when no fatigue dict is supplied so the
    # existing components and default serve behaviour are unchanged.
    fatigue_contrib = 0.0
    fatigue_modeled = False
    rest_gap = 0.0          # home_days_rest - away_days_rest (raw, capped upstream)
    density_gap = 0.0       # away_games_last_n - home_games_last_n (positive = home fresher)
    if fatigue:
        h_rest = fatigue.get("home_days_rest")
        a_rest = fatigue.get("away_days_rest")
        h_dens = fatigue.get("home_games_last_n")
        a_dens = fatigue.get("away_games_last_n")
        if h_rest is not None and a_rest is not None:
            rest_gap = float(h_rest) - float(a_rest)
        if h_dens is not None and a_dens is not None:
            density_gap = float(a_dens) - float(h_dens)
        fatigue_contrib = _clamp(rest_gap * REST_WEIGHT + density_gap * DENSITY_WEIGHT,
                                 -FATIGUE_CAP, FATIGUE_CAP)
        fatigue_modeled = True

    # Bullpen fatigue (OPTIONAL, leak-free from REAL boxscore reliever workload). A team
    # whose pen threw more pitches over the prior 1-3 days is a touch more vulnerable; the
    # signal is the load gap (home minus away) so a MORE-taxed home pen nudges the home win
    # probability DOWN. Decomposable: its own bounded additive term, exactly 0 (and flagged
    # not-modeled) when no bullpen dict is supplied so the existing components and default
    # serve behaviour are unchanged. Needs BOTH sides' loads to be present to fire.
    bullpen_contrib = 0.0
    bullpen_modeled = False
    bullpen_gap = 0.0      # home_bullpen_load - away_bullpen_load (positive = home more taxed)
    if bullpen:
        h_load = bullpen.get("home_bullpen_load")
        a_load = bullpen.get("away_bullpen_load")
        if h_load is not None and a_load is not None:
            bullpen_gap = float(h_load) - float(a_load)
            # More home fatigue (positive gap) → home DOWN, hence the negative sign.
            bullpen_contrib = _clamp(-bullpen_gap * BULLPEN_WEIGHT, -BULLPEN_CAP, BULLPEN_CAP)
            bullpen_modeled = True

    # Confirmed lineups (OPTIONAL, leak-free from the REAL boxscore batting order). Each
    # team's mean prior-season wOBA over its 9 confirmed starters; the signal is the
    # quality gap (home minus away) so a better-hitting home lineup nudges the home win
    # probability UP. Decomposable: its own bounded additive term, exactly 0 (and flagged
    # not-modeled) when no lineup dict is supplied so the existing components and default
    # serve behaviour are unchanged. Needs BOTH sides' qualities present to fire.
    lineup_contrib = 0.0
    lineup_modeled = False
    lineup_gap = 0.0       # home_lineup_quality - away_lineup_quality (positive = home better)
    if lineup:
        h_qual = lineup.get("home_lineup_quality")
        a_qual = lineup.get("away_lineup_quality")
        if h_qual is not None and a_qual is not None:
            lineup_gap = float(h_qual) - float(a_qual)
            lineup_contrib = _clamp(lineup_gap * LINEUP_WEIGHT, -LINEUP_CAP, LINEUP_CAP)
            lineup_modeled = True

    raw = (0.5 + strength + home_field + form + pitcher + env + fatigue_contrib
           + bullpen_contrib + lineup_contrib)
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
        "run_environment": {
            "contribution": round(env, 4),
            "modeled": env_modeled,
            "park_factor": round(float(park_factor), 3) if park_factor is not None else None,
            "run_env_index": round(env_index, 3) if env_modeled else None,
            "wind_out_mph": round(wind_out, 1) if wind_out is not None else None,
            "temperature_f": (weather or {}).get("temperature_f") if weather else None,
            "detail": ("Park run factor + wind/temperature — a hotter run environment adds "
                       "variance and regresses the favourite toward the coin flip"
                       if env_modeled else "No park/weather data — neutral"),
        },
        "schedule_fatigue": {
            "contribution": round(fatigue_contrib, 4),
            "modeled": fatigue_modeled,
            "home_days_rest": (fatigue or {}).get("home_days_rest") if fatigue else None,
            "away_days_rest": (fatigue or {}).get("away_days_rest") if fatigue else None,
            "home_games_last_n": (fatigue or {}).get("home_games_last_n") if fatigue else None,
            "away_games_last_n": (fatigue or {}).get("away_games_last_n") if fatigue else None,
            "detail": ("Days of rest + trailing-week game density (schedule-only, leak-free) "
                       "— a fresher home side gets a small nudge"
                       if fatigue_modeled else "No schedule fatigue data — neutral"),
        },
        "bullpen_fatigue": {
            "contribution": round(bullpen_contrib, 4),
            "modeled": bullpen_modeled,
            "home_bullpen_load": (bullpen or {}).get("home_bullpen_load") if bullpen else None,
            "away_bullpen_load": (bullpen or {}).get("away_bullpen_load") if bullpen else None,
            "home_relievers_used": (bullpen or {}).get("home_relievers_used") if bullpen else None,
            "away_relievers_used": (bullpen or {}).get("away_relievers_used") if bullpen else None,
            "detail": ("Reliever pitches thrown over the prior 1-3 days (real boxscore usage, "
                       "leak-free) — a more-taxed bullpen nudges that team down"
                       if bullpen_modeled else "No bullpen usage data — neutral"),
        },
        "confirmed_lineups": {
            "contribution": round(lineup_contrib, 4),
            "modeled": lineup_modeled,
            "home_lineup_quality": round(float((lineup or {}).get("home_lineup_quality")), 3)
                if lineup and (lineup or {}).get("home_lineup_quality") is not None else None,
            "away_lineup_quality": round(float((lineup or {}).get("away_lineup_quality")), 3)
                if lineup and (lineup or {}).get("away_lineup_quality") is not None else None,
            "detail": ("Mean prior-season wOBA of tonight's 9 confirmed starters (real "
                       "boxscore lineup, leak-free) — a better-hitting lineup nudges that "
                       "team up (captures stars resting / call-ups)"
                       if lineup_modeled else "No confirmed lineup data — neutral"),
        },
    }
    return {
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(1.0 - home_win_prob, 4),
        "confidence": round(confidence, 4),
        "leak_free": enough,
        "baseline": 0.5,
        "components": components,
        # Raw, pre-weight signals — the feature vector a trained blend learns over.
        "features": {
            "strength": round(strength, 6),
            "form_gap": round(form_gap, 6),
            "pitcher_gap": round(pitcher_gap, 6),
            "pitcher_present": 1.0 if pitcher_modeled else 0.0,
            # Raw run-environment signals for the blend to learn over (0 / neutral when
            # absent, so they don't perturb the existing feature vector's defaults).
            "run_env_index": round(env_index - 1.0, 6) if env_modeled else 0.0,
            "wind_out_mph": round(wind_out, 4) if wind_out is not None else 0.0,
            "env_present": 1.0 if env_modeled else 0.0,
            # Raw schedule-fatigue signals (0 / neutral when absent, so they don't perturb
            # the existing feature vector's defaults).
            "rest_gap": round(rest_gap, 6) if fatigue_modeled else 0.0,
            "density_gap": round(density_gap, 6) if fatigue_modeled else 0.0,
            "fatigue_present": 1.0 if fatigue_modeled else 0.0,
            # Raw bullpen-fatigue signals (0 / neutral when absent, so they don't perturb
            # the existing feature vector's defaults).
            "bullpen_gap": round(bullpen_gap, 6) if bullpen_modeled else 0.0,
            "bullpen_present": 1.0 if bullpen_modeled else 0.0,
            # Raw confirmed-lineup signals (0 / neutral when absent, so they don't perturb
            # the existing feature vector's defaults).
            "lineup_gap": round(lineup_gap, 6) if lineup_modeled else 0.0,
            "lineup_present": 1.0 if lineup_modeled else 0.0,
        },
        "model_version": "mlb-decomp-v1",
    }
