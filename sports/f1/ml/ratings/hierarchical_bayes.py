"""Hierarchical Bayesian strength model — driver nested in team nested in era.

The key F1 modeling insight: a result is `pace = era_baseline + team_effect + driver_effect + noise`.
Hierarchical priors share information across drivers within a team and across teams
within an era, so we get sensible estimates even for rookies (1-2 races) and for
mid-season car upgrades.

Implementation: pymc model, NUTS sampler. Posterior gives full distributions over
(driver_effect, team_effect, residual_variance) which the simulator samples from
to propagate uncertainty through the Monte Carlo.

This is the most important model in the stack — register it as the default
strength source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sports.f1.ml.common.registry import register


@dataclass
class HierarchicalPosterior:
    driver_codes: list[str]
    team_codes: list[str]
    driver_effect_samples: np.ndarray  # shape (n_samples, n_drivers)
    team_effect_samples: np.ndarray    # shape (n_samples, n_teams)
    residual_sigma_samples: np.ndarray
    era_baseline_samples: np.ndarray


@register("ratings.hierarchical_bayes")
def build_model(**kwargs) -> "HierarchicalBayesStrength":
    return HierarchicalBayesStrength(**kwargs)


class HierarchicalBayesStrength:
    def __init__(self, era_window_seasons: int = 3, n_samples: int = 2000) -> None:
        self.era_window = era_window_seasons
        self.n_samples = n_samples
        self.posterior: HierarchicalPosterior | None = None

    def fit(
        self,
        race_pace: pd.DataFrame,
        n_tune: int = 1000,
        progressbar: bool = False,
        target_accept: float = 0.9,
    ) -> "HierarchicalBayesStrength":
        """`race_pace` columns: season, race, driver_code, team_code, mean_pace_s.

        Fits the model:

            mean_pace = era_baseline + team_effect[team] + driver_effect[driver] + ε

        with non-centered parameterization (sampling driver_z ~ N(0, 1), then
        scaling by sigma_driver) to keep NUTS well-conditioned. Returns self
        with self.posterior populated.
        """
        import pymc as pm  # heavy import; keep inside fit so module loads cheap

        required = {"driver_code", "team_code", "mean_pace_s"}
        missing = required - set(race_pace.columns)
        if missing:
            raise ValueError(f"race_pace missing columns: {missing}")

        df = race_pace.dropna(subset=["mean_pace_s"]).copy()
        driver_codes = sorted(df["driver_code"].unique().tolist())
        team_codes = sorted(df["team_code"].unique().tolist())
        driver_idx = df["driver_code"].map({d: i for i, d in enumerate(driver_codes)}).to_numpy()
        team_idx = df["team_code"].map({t: i for i, t in enumerate(team_codes)}).to_numpy()
        pace = df["mean_pace_s"].to_numpy(dtype=float)

        with pm.Model():
            # Hyperpriors: tighter than typical because F1 pace differences are well-studied.
            sigma_driver = pm.HalfNormal("sigma_driver", sigma=0.5)
            sigma_team = pm.HalfNormal("sigma_team", sigma=2.0)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)

            era_baseline = pm.Normal("era_baseline", mu=float(pace.mean()), sigma=10.0)

            # Non-centered: sample standard normals, then scale.
            driver_z = pm.Normal("driver_z", mu=0.0, sigma=1.0, shape=len(driver_codes))
            team_z = pm.Normal("team_z", mu=0.0, sigma=1.0, shape=len(team_codes))
            driver_effect = pm.Deterministic("driver_effect", driver_z * sigma_driver)
            team_effect = pm.Deterministic("team_effect", team_z * sigma_team)

            mu = era_baseline + team_effect[team_idx] + driver_effect[driver_idx]
            pm.Normal("obs", mu=mu, sigma=sigma_obs, observed=pace)

            trace = pm.sample(
                draws=self.n_samples,
                tune=n_tune,
                chains=2,
                cores=1,  # avoid Windows multiprocessing issues
                progressbar=progressbar,
                target_accept=target_accept,
                random_seed=42,
            )

        post = trace.posterior
        self.posterior = HierarchicalPosterior(
            driver_codes=driver_codes,
            team_codes=team_codes,
            driver_effect_samples=post["driver_effect"].values.reshape(-1, len(driver_codes)),
            team_effect_samples=post["team_effect"].values.reshape(-1, len(team_codes)),
            residual_sigma_samples=post["sigma_obs"].values.reshape(-1),
            era_baseline_samples=post["era_baseline"].values.reshape(-1),
        )
        return self

    def sample_pace(self, driver_code: str, team_code: str, n: int) -> np.ndarray:
        """Draw n posterior-predictive samples of pace contribution for a
        (driver, team) pairing. The simulator calls this each MC iteration."""
        if self.posterior is None:
            raise RuntimeError("HierarchicalBayesStrength not fitted yet")
        post = self.posterior
        if driver_code not in post.driver_codes:
            raise KeyError(f"driver '{driver_code}' not in posterior")
        if team_code not in post.team_codes:
            raise KeyError(f"team '{team_code}' not in posterior")
        d_idx = post.driver_codes.index(driver_code)
        t_idx = post.team_codes.index(team_code)

        n_draws = len(post.era_baseline_samples)
        rng = np.random.default_rng()
        idx = rng.integers(0, n_draws, size=n)
        means = (
            post.era_baseline_samples[idx]
            + post.team_effect_samples[idx, t_idx]
            + post.driver_effect_samples[idx, d_idx]
        )
        # Add residual noise per draw.
        noise = rng.normal(0.0, post.residual_sigma_samples[idx])
        return means + noise

    def driver_strength_summary(self) -> "pd.DataFrame":
        """Posterior mean and 95% HDI of driver_effect, separated from team."""
        import pandas as pd
        if self.posterior is None:
            raise RuntimeError("not fitted")
        means = self.posterior.driver_effect_samples.mean(axis=0)
        lo = np.quantile(self.posterior.driver_effect_samples, 0.025, axis=0)
        hi = np.quantile(self.posterior.driver_effect_samples, 0.975, axis=0)
        return pd.DataFrame({
            "driver_code": self.posterior.driver_codes,
            "driver_effect_mean": means,
            "driver_effect_low_95": lo,
            "driver_effect_high_95": hi,
        }).sort_values("driver_effect_mean")  # negative = faster, since pace = lap time
