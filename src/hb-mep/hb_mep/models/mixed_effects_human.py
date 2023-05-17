import logging

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import hpdi

from hb_mep.config import HBMepConfig
from hb_mep.models.baseline import Baseline
from hb_mep.models.utils import Site as site
from hb_mep.utils import timing
from hb_mep.utils.constants import (
    INTENSITY,
    RESPONSE,
    PARTICIPANT,
    FEATURES
)

logger = logging.getLogger(__name__)


class MixedEffectsHuman(Baseline):
    def __init__(self, config: HBMepConfig):
        super(MixedEffectsHuman, self).__init__(config=config)
        self.name = "Mixed_Effects_Human"

        self.columns = ["participant", "method"]
        self.x = np.linspace(0, 15, 100)

    def _model(self, intensity, participant, feature1, response_obs=None):
        n_participant = np.unique(participant).shape[0]
        n_feature1 = np.unique(feature1).shape[0]

        """ Hyperpriors """
        delta_mean = numpyro.sample(site.delta_mean, dist.Normal(0, 10))
        delta_scale = numpyro.sample(site.delta_scale, dist.HalfNormal(2.0))

        # Baseline threshold
        baseline_mean_global_mean = numpyro.sample(
            "baseline_mean_global_mean",
            dist.HalfNormal(2.0)
        )
        baseline_scale_global_scale = numpyro.sample(
            site.baseline_scale_global_scale,
            dist.HalfNormal(2.0)
        )

        baseline_scale = numpyro.sample(
            site.baseline_scale,
            dist.HalfNormal(baseline_scale_global_scale)
        )
        baseline_mean = numpyro.sample(
            site.baseline_mean,
            dist.TruncatedDistribution(dist.Normal(baseline_mean_global_mean, baseline_scale), low=0)
        )

        # Slope
        b_mean_global_scale = numpyro.sample(site.b_mean_global_scale, dist.HalfNormal(5.0))
        b_scale_global_scale = numpyro.sample(site.b_scale_global_scale, dist.HalfNormal(2.0))

        b_mean = numpyro.sample(site.b_mean, dist.HalfNormal(b_mean_global_scale))
        b_scale = numpyro.sample(site.b_scale, dist.HalfNormal(b_scale_global_scale))

        # MEP at rest
        lo_scale_global_scale = numpyro.sample(site.lo_scale_global_scale, dist.HalfNormal(2.0))
        lo_scale = numpyro.sample(site.lo_scale, dist.HalfNormal(lo_scale_global_scale))

        # # Saturation
        # g_shape = numpyro.sample(site.g_shape, dist.HalfNormal(20.0))

        with numpyro.plate("n_participant", n_participant, dim=-1):
            """ Priors """
            baseline = numpyro.sample(
                site.baseline,
                dist.TruncatedDistribution(dist.Normal(baseline_mean, baseline_scale), low=0)
            )
            delta = numpyro.sample(site.delta, dist.Normal(delta_mean, delta_scale))

            with numpyro.plate("n_feature1", n_feature1, dim=-2):
                # Threshold
                a = numpyro.deterministic(
                    site.a,
                    jnp.array([baseline, baseline + delta])
                )

                # Slope
                b = numpyro.sample(
                    site.b,
                    dist.TruncatedDistribution(dist.Normal(b_mean, b_scale), low=0)
                )

                # MEP at rest
                lo = numpyro.sample(site.lo, dist.HalfNormal(lo_scale))

                # # Saturation
                # g = numpyro.sample(site.g, dist.Beta(1, g_shape))

                # Noise
                noise = numpyro.sample(
                    site.noise,
                    dist.HalfCauchy(0.5)
                )

        # Model
        mean = numpyro.deterministic(
            site.mean,
            lo[feature1, participant] + \
            jax.nn.relu(b[feature1, participant] * (intensity - a[feature1, participant]))
        )

        sigma = numpyro.deterministic("sigma", noise[feature1, participant])

        penalty = 5 * (jnp.fabs(baseline + delta) - (baseline + delta))
        numpyro.factor(site.penalty, -penalty)

        with numpyro.plate("data", len(intensity)):
            return numpyro.sample("obs", dist.TruncatedNormal(mean, sigma, low=0), obs=response_obs)

    @timing
    def run_inference(self, df: pd.DataFrame) -> tuple[numpyro.infer.mcmc.MCMC, dict]:
        """
        Run MCMC inference
        """
        response = df[RESPONSE].to_numpy().reshape(-1,)
        participant = df[PARTICIPANT].to_numpy().reshape(-1,)
        feature1 = df[FEATURES[1]].to_numpy().reshape(-1,)
        intensity = df[INTENSITY].to_numpy().reshape(-1,)

        # MCMC
        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(nuts_kernel, **self.config.MCMC_PARAMS)
        rng_key = jax.random.PRNGKey(self.random_state)
        logger.info(f"Running inference with {self.name} ...")
        mcmc.run(rng_key, intensity, participant, feature1, response)
        posterior_samples = mcmc.get_samples()

        return mcmc, posterior_samples

    def _get_estimates(
        self,
        posterior_samples: dict,
        posterior_means: dict,
        c: tuple
    ):
        a = posterior_means[site.a][c[::-1]]
        b = posterior_means[site.b][c[::-1]]
        lo = posterior_means[site.lo][c[::-1]]
        # g = posterior_means[site.g][c[::-1]]
        # y = lo - jnp.log(jnp.maximum(g, jnp.exp(-jnp.maximum(0, b * (x - a)))))
        y = lo + jax.nn.relu(b * (self.xx - a))

        threshold_samples = posterior_samples[site.a][:, c[1], c[0]]
        hpdi_interval = hpdi(threshold_samples, prob=0.95)

        return y, threshold_samples, hpdi_interval

    # def _get_estimates(
    #     self,
    #     posterior_samples: dict,
    #     posterior_means: dict,
    #     c: tuple
    # ):
    #     a = posterior_means[site.a][c[::-1]]
    #     b = posterior_means[site.b][c[::-1]]
    #     lo = posterior_means[site.lo][c[::-1]]
    #     g = posterior_means[site.g][c[::-1]]
    #     y = lo - jnp.log(jnp.maximum(g, jnp.exp(-jnp.maximum(0, b * (self.x - a)))))
    #     # y = lo + self.link(b * (self.xx - a))

    #     threshold_samples = posterior_samples[site.a][:, c[1], c[0]]
    #     hpdi_interval = hpdi(threshold_samples, prob=0.95)

    #     return y, threshold_samples, hpdi_interval

    def _get_combinations(self, df: pd.DataFrame):
        combinations = \
            df \
            .groupby(by=self.columns) \
            .size() \
            .to_frame("counts") \
            .reset_index().copy()
        combinations = combinations[self.columns].apply(tuple, axis=1).tolist()
        return combinations

    def plot(self, df: pd.DataFrame, posterior_samples: dict, encoder_dict: dict = None):
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

            ax[i, 1].set_xlim(right=temp_df[INTENSITY].max() + 3)

            if encoder_dict is None:
                title = f"{tuple(self.columns)} - {c}"
            else:
                c0 = encoder_dict["participant"].inverse_transform(np.array([c[0]]))[0]
                c1 = temp_df["sc_level"].unique()[0]
                c2 = encoder_dict["method"].inverse_transform(np.array([c[1]]))[0]

                title = f"({c0}, {c1}, {c2})"

            ax[i, 0].set_title(title)

            ax[i, 1].set_title(f"Model Fit")
            ax[i, 2].set_title(f"Threshold Estimate")
            ax[i, 1].legend(loc="upper right")
            ax[i, 2].legend(loc="upper right")

        return fig
