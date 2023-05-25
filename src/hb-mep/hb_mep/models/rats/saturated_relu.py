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


class SaturatedReLU(Baseline):
    def __init__(self, config: HBMepConfig):
        super(SaturatedReLU, self).__init__(config=config)
        self.name = "Saturated_ReLU"

        self.columns = [PARTICIPANT] + FEATURES
        self.x = np.linspace(0, 450, 1000)

    def _model(self, intensity, participant, feature0, feature1, response_obs=None):
        n_participant = np.unique(participant).shape[0]
        n_feature0 = np.unique(feature0).shape[0]
        n_feature1 = np.unique(feature1).shape[0]

        with numpyro.plate("n_participant", n_participant, dim=-1):
            # Hyperriors
            a_mean = numpyro.sample(
                site.a_mean,
                dist.TruncatedDistribution(dist.Normal(150, 50), low=0)
            )
            a_scale = numpyro.sample(site.a_scale, dist.HalfNormal(20))

            b_scale = numpyro.sample(site.b_scale, dist.HalfNormal(0.1))

            g_shape = numpyro.sample(site.g_shape, dist.HalfNormal(5.0))
            lo_scale = numpyro.sample(site.lo_scale, dist.HalfNormal(0.2))

            noise_offset_scale = numpyro.sample(
                site.noise_offset_scale,
                dist.HalfCauchy(0.2)
            )
            noise_slope_scale = numpyro.sample(
                site.noise_slope_scale,
                dist.HalfCauchy(0.2)
            )

            with numpyro.plate("n_feature0", n_feature0, dim=-2):
                with numpyro.plate("n_feature1", n_feature1, dim=-3):
                    # Priors
                    a = numpyro.sample(
                        site.a,
                        dist.TruncatedDistribution(dist.Normal(a_mean, a_scale), low=0)
                    )
                    b = numpyro.sample(site.b, dist.HalfNormal(b_scale))

                    g = numpyro.sample(site.g, dist.Beta(1, g_shape))
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
            lo[feature1, feature0, participant] - \
            jnp.log(jnp.maximum(
                g[feature1, feature0, participant],
                jnp.exp(-jax.nn.relu(
                    b[feature1, feature0, participant] * (intensity - a[feature1, feature0, participant])
                ))
            ))

        noise = \
            noise_offset[feature1, feature0, participant] + \
            noise_slope[feature1, feature0, participant] * mean

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
        feature1 = df[FEATURES[1]].to_numpy().reshape(-1,)
        intensity = df[INTENSITY].to_numpy().reshape(-1,)

        # MCMC
        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(nuts_kernel, **self.config.MCMC_PARAMS)
        rng_key = jax.random.PRNGKey(self.random_state)
        logger.info(f"Running inference with {self.name} ...")
        mcmc.run(rng_key, intensity, participant, feature0, feature1, response)
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
        g = posterior_means[site.g][c[::-1]]
        y = lo - jnp.log(jnp.maximum(g, jnp.exp(-jnp.maximum(0, b * (self.x - a)))))

        threshold_samples = posterior_samples[site.a][:, c[2], c[1], c[0]]
        hpdi_interval = hpdi(threshold_samples, prob=0.95)

        return y, threshold_samples, hpdi_interval

    def plot(
        self,
        df: pd.DataFrame,
        posterior_samples: dict,
        encoder_dict: dict = None,
        pred: pd.DataFrame = None,
        mat: np.ndarray = None,
        time: np.ndarray = None
    ):
        if pred is not None:
            assert encoder_dict is not None

        if mat is not None:
            assert time is not None

        combinations = self._get_combinations(df)
        n_combinations = len(combinations)

        posterior_means = {
            p:posterior_samples[p].mean(axis=0) for p in posterior_samples
        }

        n_columns = 3 if mat is None else 4

        fig, axes = plt.subplots(
            n_combinations,
            n_columns,
            figsize=(n_columns * 6, n_combinations * 3),
            constrained_layout=True
        )

        for i, c in enumerate(combinations):
            idx = df[self.columns].apply(tuple, axis=1).isin([c])
            temp_df = df[idx].reset_index(drop=True).copy()

            sns.scatterplot(data=temp_df, x=INTENSITY, y=RESPONSE, ax=axes[i, 0])

            if encoder_dict is None:
                title = f"{self.columns} - {c}"
            else:
                c0 = encoder_dict[self.columns[0]].inverse_transform(np.array([c[0]]))[0]
                c1 = encoder_dict[self.columns[1]].inverse_transform(np.array([c[1]]))[0]
                c2 = encoder_dict[self.columns[2]].inverse_transform(np.array([c[2]]))[0]

                title = f"{tuple(self.columns)} - {(c0, c1, c2)}"

            axes[i, 0].set_title(title)

            sns.scatterplot(data=temp_df, x=INTENSITY, y=RESPONSE, alpha=.4, ax=axes[i, 1])

            y, threshold_samples, hpdi_interval = self._get_estimates(
                posterior_samples, posterior_means, c
            )

            sns.kdeplot(x=threshold_samples, color="blue", ax=axes[i, 1])
            sns.lineplot(
                x=self.x,
                y=y,
                color="red",
                alpha=0.4,
                label=f"Mean Posterior",
                ax=axes[i, 1]
            )
            sns.kdeplot(x=threshold_samples, color="blue", ax=axes[i, 2])

            axes[i, 2].axvline(hpdi_interval[0], linestyle="--", color="green", label="95% HPDI Interval")
            axes[i, 2].axvline(hpdi_interval[1], linestyle="--", color="green")

            axes[i, 1].set_xlim(right=temp_df[INTENSITY].max() + 10)

            if pred is not None:
                temp_pred = pred[pred[self.columns].apply(tuple, axis=1).isin([(c0, c1, c2)])]
                prediction = temp_pred[RESPONSE].values
                assert len(prediction) == 1
                posterior_mean = posterior_means[site.a][c[::-1]]
                axes[i, 0].axvline(
                    x=posterior_mean,
                    color="purple",
                    linestyle='--',
                    alpha=.4,
                    label=f"Mean Posterior: {posterior_mean:.1f}"
                )
                axes[i, 0].axvline(
                    x=prediction[0],
                    color="red",
                    linestyle='--',
                    alpha=.4,
                    label=f"Ahmet's pk-pk: {prediction[0]:.1f}"
                )
                axes[i, 2].axvline(
                    x=posterior_mean,
                    color="purple",
                    linestyle='--',
                    alpha=.4,
                    label=f"Mean Posterior: {posterior_mean:.1f}"
                )
                axes[i, 2].axvline(
                    x=prediction[0],
                    color="red",
                    linestyle='--',
                    alpha=.4,
                    label=f"Ahmet's pk-pk: {prediction[0]:.1f}"
                )
                axes[i, 0].legend(loc="upper left")

            axes[i, 1].set_title(f"Model Fit")
            axes[i, 2].set_title(f"Threshold Estimate")

            axes[i, 1].legend(loc="upper left")
            axes[i, 2].legend(loc="upper left")

            if mat is not None:
                ax = axes[i][3]
                temp_mat = mat[idx, :]

                for j in range(temp_mat.shape[0]):
                    x = temp_mat[j, :]/60 + temp_df[INTENSITY].values[j]
                    ax.plot(x, time, color="green", alpha=.4)

                ax.axhline(
                    y=0.003, color="red", linestyle='--', alpha=.4, label="AUC Window"
                )
                ax.axhline(
                    y=0.015, color="red", linestyle='--', alpha=.4
                )

                ax.set_ylim(bottom=-0.001, top=0.02)

                ax.set_xlabel(f"{INTENSITY}")
                ax.set_ylabel(f"Time")

                ax.legend(loc="upper right")
                ax.set_title(f"Motor Evoked Potential")

        return fig
