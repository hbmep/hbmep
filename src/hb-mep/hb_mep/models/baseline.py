import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import hpdi

from hb_mep.config import HBMepConfig
from hb_mep.models.utils import Site as site
from hb_mep.utils import timing
from hb_mep.utils.constants import (
    REPORTS_DIR,
    INTENSITY,
    RESPONSE,
    PARTICIPANT,
    FEATURES
)

logger = logging.getLogger(__name__)


class Baseline():
    def __init__(self, config: HBMepConfig):
        self.config = config
        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.reports_path = Path(os.path.join(self.current_path, REPORTS_DIR))

        self.name = "Baseline"
        self.link = jax.nn.relu

        self.random_state = 0
        self.columns = [PARTICIPANT] + FEATURES
        self.x = np.linspace(0, 450, 1000)

    def _model(self, intensity, participant, feature0, response_obs=None):
        n_participant = np.unique(participant).shape[0]
        n_feature0 = np.unique(feature0).shape[0]

        with numpyro.plate("n_feature0", n_feature0, dim=-1):
            # Hyperpriors
            a_mean = numpyro.sample(
                "a_mean",
                dist.TruncatedDistribution(dist.Normal(150, 50), low=0)
            )

            a_scale = numpyro.sample(site.a_scale, dist.HalfNormal(20))
            b_scale = numpyro.sample(site.b_scale, dist.HalfNormal(0.1))
            lo_scale = numpyro.sample(site.lo_scale, dist.HalfNormal(0.2))

            noise_offset_scale = numpyro.sample(
                site.noise_offset_scale,
                dist.HalfCauchy(0.2)
            )
            noise_slope_scale = numpyro.sample(
                site.noise_slope_scale,
                dist.HalfCauchy(0.2)
            )

            with numpyro.plate("n_participant", n_participant, dim=-2):
                # Priors
                a = numpyro.sample(
                    site.a,
                    dist.TruncatedDistribution(dist.Normal(a_mean, a_scale), low=0)
                )
                b = numpyro.sample(site.b, dist.HalfNormal(b_scale))

                lo = numpyro.sample(site.lo, dist.HalfNormal(lo_scale))

                noise_offset = numpyro.sample(
                    site.noise_offset,
                    dist.HalfCauchy(noise_offset_scale)
                )
                noise_slope = numpyro.sample(
                    site.noise_slope,
                    dist.HalfCauchy(noise_slope_scale)
                )

        # Model
        mean = \
            lo[participant, feature0] + \
            self.link(
                b[participant, feature0] * (intensity - a[participant, feature0])
            )

        noise = noise_offset[participant, feature0] + noise_slope[participant, feature0] * mean

        with numpyro.plate("data", len(intensity)):
            return numpyro.sample("obs", dist.TruncatedNormal(mean, noise, low=0), obs=response_obs)

    @timing
    def run_inference(self, df: pd.DataFrame) -> tuple[numpyro.infer.mcmc.MCMC, dict]:
        """
        Run MCMC inference
        """
        response = df[RESPONSE].to_numpy().reshape(-1,)
        participant = df[PARTICIPANT].to_numpy().reshape(-1,)
        feature0 = df[FEATURES[0]].to_numpy().reshape(-1,)
        intensity = df[INTENSITY].to_numpy().reshape(-1,)

        # MCMC
        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(nuts_kernel, **self.config.MCMC_PARAMS)
        rng_key = jax.random.PRNGKey(self.random_state)
        logger.info(f"Running inference with {self.name} ...")
        mcmc.run(rng_key, intensity, participant, feature0, response)
        posterior_samples = mcmc.get_samples()

        return mcmc, posterior_samples

    def _get_combinations(self, df: pd.DataFrame):
        combinations = \
            df \
            .groupby(by=self.columns) \
            .size() \
            .to_frame("counts") \
            .reset_index().copy()
        combinations = combinations[self.columns].apply(tuple, axis=1).tolist()
        return combinations

    def _get_estimates(
        self,
        posterior_samples: dict,
        posterior_means: dict,
        c: tuple
    ):
        a = posterior_means[site.a][c]
        b = posterior_means[site.b][c]
        lo = posterior_means[site.lo][c]
        y = lo + self.link(b * (self.x - a))

        threshold_samples = posterior_samples[site.a][:, c[0], c[1]]
        hpdi_interval = hpdi(threshold_samples, prob=0.95)

        return y, threshold_samples, hpdi_interval

    def plot(self, df: pd.DataFrame, posterior_samples: dict):
        combinations = self._get_combinations(df)
        n_combinations = len(combinations)

        posterior_means = {
            p:posterior_samples[p].mean(axis=0) for p in posterior_samples
        }

        fig, ax = plt.subplots(
            n_combinations,
            3,
            figsize=(12, n_combinations * 3),
            constrained_layout=True
        )

        for i, c in enumerate(combinations):
            idx = df[self.columns].apply(tuple, axis=1).isin([c])
            temp_df = df[idx].reset_index(drop=True).copy()

            sns.scatterplot(data=temp_df, x=INTENSITY, y=RESPONSE, ax=ax[i, 0])
            sns.scatterplot(data=temp_df, x=INTENSITY, y=RESPONSE, ax=ax[i, 1], alpha=.4)

            y, threshold_samples, hpdi_interval = self._get_estimates(
                posterior_samples, posterior_means, c
            )

            sns.kdeplot(x=threshold_samples, ax=ax[i, 1], color="blue")
            sns.lineplot(
                x=self.x,
                y=y,
                ax=ax[i, 1],
                color="red",
                alpha=0.4,
                label=f"Mean Posterior"
            )
            sns.kdeplot(x=threshold_samples, color="blue", ax=ax[i, 2])

            ax[i, 2].axvline(hpdi_interval[0], linestyle="--", color="green", label="95% HPDI Interval")
            ax[i, 2].axvline(hpdi_interval[1], linestyle="--", color="green")

            ax[i, 1].set_xlim(right=temp_df[INTENSITY].max() + 10)

            ax[i, 0].set_title(f"{tuple(self.columns)} - {c}")
            ax[i, 1].set_title(f"Model Fit")
            ax[i, 2].set_title(f"Threshold Estimate")
            ax[i, 1].legend(loc="upper right")
            ax[i, 2].legend(loc="upper right")

        return fig
