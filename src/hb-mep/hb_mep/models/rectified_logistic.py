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


class RectifiedLogistic(Baseline):
    def __init__(self, config: HBMepConfig):
        super(RectifiedLogistic, self).__init__(config=config)
        self.name = "Rectified_Logistic"

        self.columns = ["participant", "method"]
        self.x = np.linspace(0, 15, 100)

    def _model(self, intensity, participant, feature1, response_obs=None):
        n_participant = np.unique(participant).shape[0]
        n_feature1 = np.unique(feature1).shape[0]

        """ Hyperpriors """
        a_mean_global_mean = numpyro.sample(
            "a_mean_global_mean",
            dist.HalfNormal(10.0)
        )
        a_scale_global_scale = numpyro.sample(
            "a_scale_global_scale",
            dist.HalfNormal(2.0)
        )

        a_scale = numpyro.sample(
            site.a_scale,
            dist.HalfNormal(a_scale_global_scale)
        )
        a_mean = numpyro.sample(
            site.a_mean,
            dist.TruncatedDistribution(dist.Normal(a_mean_global_mean, a_scale), low=0)
        )

        b_mean_global_scale = numpyro.sample(site.b_mean_global_scale, dist.HalfNormal(10.0))
        b_scale_global_scale = numpyro.sample(site.b_scale_global_scale, dist.HalfNormal(2.0))

        b_mean = numpyro.sample(site.b_mean, dist.HalfNormal(b_mean_global_scale))
        b_scale = numpyro.sample(site.b_scale, dist.HalfNormal(b_scale_global_scale))

        h_scale_global_scale = numpyro.sample("h_scale_global_scale", dist.HalfNormal(10.0))
        h_scale = numpyro.sample("h_scale", dist.HalfNormal(h_scale_global_scale))

        v_scale_global_scale = numpyro.sample("v_scale_global_scale", dist.HalfNormal(10.0))
        v_scale = numpyro.sample("v_scale", dist.HalfNormal(v_scale_global_scale))

        lo_scale_global_scale = numpyro.sample(site.lo_scale_global_scale, dist.HalfNormal(2.0))
        lo_scale = numpyro.sample(site.lo_scale, dist.HalfNormal(lo_scale_global_scale))

        with numpyro.plate("n_participant", n_participant, dim=-1):
            with numpyro.plate("n_feature1", n_feature1, dim=-2):
                """ Priors """
                a = numpyro.sample(
                    site.a,
                    dist.TruncatedDistribution(dist.Normal(a_mean, a_scale), low=0)
                )

                b = numpyro.sample(
                    site.b,
                    dist.TruncatedDistribution(dist.Normal(b_mean, b_scale), low=0)
                )

                h = numpyro.sample("h", dist.HalfNormal(h_scale))
                v = numpyro.sample("v", dist.HalfNormal(v_scale))

                lo = numpyro.sample(site.lo, dist.HalfNormal(lo_scale))

                noise_offset = numpyro.sample(
                    site.noise_offset,
                    dist.HalfCauchy(0.5)
                )
                noise_slope = numpyro.sample(
                    site.noise_slope,
                    dist.HalfCauchy(0.5)
                )

        """ Model """
        mean = numpyro.deterministic(
            site.mean,
            lo[feature1, participant] + \
            jnp.maximum(
                0,
                -1 + \
                (h[feature1, participant] + 1) / \
                jnp.power(
                    1 + \
                    (jnp.power(1 + h[feature1, participant], v[feature1, participant]) - 1) * \
                    jnp.exp(-b[feature1, participant] * (intensity - a[feature1, participant])),
                    1 / v[feature1, participant]
                )
            )
        )

        sigma = numpyro.deterministic(
            "sigma",
            noise_offset[feature1, participant] + \
            noise_slope[feature1, participant] * \
            mean
        )

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
        h = posterior_means["h"][c[::-1]]
        v = posterior_means["v"][c[::-1]]
        lo = posterior_means[site.lo][c[::-1]]

        y = lo + jnp.maximum(
            0,
            -1 + (h + 1) / \
            jnp.power(1 + (jnp.power(1 + h, v) - 1) * jnp.exp(-b * (self.x - a)), 1 / v)
        )

        threshold_samples = posterior_samples[site.a][:, c[1], c[0]]
        hpdi_interval = hpdi(threshold_samples, prob=0.95)

        return y, threshold_samples, hpdi_interval

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
            ax[i, 1].legend(loc="upper left")
            ax[i, 2].legend(loc="upper left")

        return fig
