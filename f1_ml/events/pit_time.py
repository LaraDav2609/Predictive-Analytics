"""Pit stop time distribution per team.

Standard stop is ~2.0-2.5 s stationary, ~22-25 s pit lane delta total. Variance
is the modeling target — Red Bull / McLaren consistently sub-2.5 s; smaller teams
sit near 3 s with heavier left tail (fumbled stops at 5-10+ s).

Model: Gaussian core + heavy-tail mixture (lognormal or shifted-exponential) for
botched stops. Per-team fit. Two-step EM-style assignment: classify as 'botched'
above μ_clean + 3σ_clean, refit each component.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PitTimeDistribution:
    team_code: str
    mean_clean_s: float
    sigma_clean_s: float
    botch_probability: float    # P(botched stop)
    botch_mean_extra_s: float   # additional time when botched (over clean mean)


def fit(
    historical_pit_stops: pd.DataFrame,
    team_col: str = "team_code",
    duration_col: str = "stationary_time_s",
    botch_threshold_sigmas: float = 3.0,
    min_observations: int = 5,
) -> dict[str, PitTimeDistribution]:
    """Per-team Gaussian-core + heavy-tail mixture fit.

    Two-step procedure (light-EM):
      1. Robust initial estimate: median + MAD-based σ.
      2. Anything above (median + k * MAD-σ) labelled 'botch'; refit the
         clean Gaussian on the rest, model the botch component as the
         extra time over the clean mean.
    """
    if historical_pit_stops.empty:
        return {}

    out: dict[str, PitTimeDistribution] = {}
    for team, grp in historical_pit_stops.groupby(team_col):
        times = np.asarray(grp[duration_col].dropna().values, dtype=float)
        if len(times) < min_observations:
            # Fall back to global default for low-sample teams.
            out[str(team)] = PitTimeDistribution(
                team_code=str(team),
                mean_clean_s=2.5,
                sigma_clean_s=0.3,
                botch_probability=0.05,
                botch_mean_extra_s=2.0,
            )
            continue

        median = float(np.median(times))
        mad = float(np.median(np.abs(times - median)))
        sigma_robust = max(mad * 1.4826, 1e-3)  # MAD → σ for normal

        cutoff = median + botch_threshold_sigmas * sigma_robust
        clean_mask = times <= cutoff
        clean = times[clean_mask]
        botch = times[~clean_mask]

        if len(clean) == 0:
            clean = times  # all flagged — keep them as clean to avoid empty fit
            botch = np.array([])

        mean_clean = float(np.mean(clean))
        sigma_clean = float(max(np.std(clean, ddof=1) if len(clean) > 1 else 0.1, 1e-3))
        p_botch = float(len(botch) / len(times)) if len(times) > 0 else 0.0
        botch_extra = float(np.mean(botch) - mean_clean) if len(botch) > 0 else 2.0

        out[str(team)] = PitTimeDistribution(
            team_code=str(team),
            mean_clean_s=mean_clean,
            sigma_clean_s=sigma_clean,
            botch_probability=p_botch,
            botch_mean_extra_s=max(botch_extra, 0.5),
        )
    return out


def sample(
    dist: PitTimeDistribution,
    rng: np.random.Generator | None = None,
) -> float:
    """One MC draw of the pit-lane delta for this team's next stop."""
    rng = rng or np.random.default_rng()
    if rng.uniform() < dist.botch_probability:
        # Botch component: clean mean + exponential extra time.
        return float(
            rng.normal(dist.mean_clean_s, dist.sigma_clean_s)
            + rng.exponential(max(dist.botch_mean_extra_s, 0.5))
        )
    return float(rng.normal(dist.mean_clean_s, dist.sigma_clean_s))


def expected_loss(dist: PitTimeDistribution) -> float:
    """E[stop time] under the mixture — useful for strategy DP."""
    return float(
        dist.mean_clean_s
        + dist.botch_probability * dist.botch_mean_extra_s
    )
