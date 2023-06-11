import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
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
        self.random_state = 0
        self.rng_key = None

        self.columns = [PARTICIPANT] + FEATURES
        self.x = np.linspace(0, 450, 1000)
        self.xpad = 10

        self._set_rng_key()

    def _set_rng_key(self):
        self.rng_key = jax.random.PRNGKey(self.random_state)

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
            b_scale = numpyro.sample(site.b_scale, dist.HalfNormal(100))
            lo_scale = numpyro.sample(site.lo_scale, dist.HalfNormal(5))

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
                    dist.HalfCauchy(0.01)
                )
                noise_slope = numpyro.sample(
                    site.noise_slope,
                    dist.HalfCauchy(0.05)
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
        logger.info(f"Running inference with {self.name} ...")
        mcmc.run(self.rng_key, intensity, participant, feature0, response)
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

    def _get_threshold_estimates(
        self,
        combination: tuple,
        posterior_samples: dict,
        prob: float = .95
    ):
        threshold_posterior = posterior_samples[site.a][
            :, combination[2], combination[1], combination[0]
        ]
        threshold = threshold_posterior.mean()
        hpdi_interval = hpdi(threshold_posterior, prob=prob)
        return threshold, threshold_posterior, hpdi_interval

    def predict(
        self,
        intensity: np.ndarray,
        combination: tuple,
        posterior_samples: Optional[dict] = None,
        num_samples: int = 100
    ):
        predictive = Predictive(model=self._model, num_samples=num_samples)
        if posterior_samples is not None:
            predictive = Predictive(model=self._model, posterior_samples=posterior_samples)

        participant = np.repeat([combination[0]], intensity.shape[0])
        feature0 = np.repeat([combination[1]], intensity.shape[0])
        feature1 = np.repeat([combination[2]], intensity.shape[0])

        predictions = predictive(
            self.rng_key,
            intensity=intensity,
            participant=participant,
            feature0=feature0,
            feature1=feature1
        )
        return predictions

    def predictive_check(
        self,
        df: pd.DataFrame,
        posterior_samples: Optional[dict] = None,
        n_ppdm: int = 1000,
        n_ppdo: int = 100
    ):
        assert n_ppdm >= n_ppdo

        (post_check, check) = (False, "Prior") if posterior_samples is None else (True, "Posterior")

        combinations = self._get_combinations(df)
        n_combinations = len(combinations)
        intensity = self.x

        n_columns = 3
        predictions = None

        fig, axes = plt.subplots(
            n_combinations,
            n_columns,
            figsize=(n_columns * 6, n_combinations * 3),
            constrained_layout=True
        )

        for i, c in enumerate(combinations):
            idx = df[self.columns].apply(tuple, axis=1).isin([c])
            temp_df = df[idx].reset_index(drop=True).copy()

            if post_check or (not post_check and predictions is None):
                predictions = self.predict(
                    intensity=intensity,
                    combination=c,
                    posterior_samples=posterior_samples,
                    num_samples=n_ppdm
                )
                obs = predictions["obs"]
                mean = predictions["mean"]

            """ Plots """
            sns.scatterplot(data=temp_df, x=INTENSITY, y=RESPONSE, alpha=.4, ax=axes[i, 0])
            sns.lineplot(
                x=intensity,
                y=mean.mean(axis=0),
                label=f"Mean {check}",
                color="r",
                alpha=0.4,
                ax=axes[i, 0]
            )

            choices = jax.random.choice(
                self.rng_key,
                a=np.array(range(obs.shape[0])),
                shape=(n_ppdo,),
                replace=False
            )
            y_obs = obs[choices, :].T
            x = np.tile(intensity, (choices.shape[0], 1)).T

            axes[i, 1].scatter(x, y_obs, s=11)
            sns.scatterplot(
                data=temp_df, x=INTENSITY, y=RESPONSE, color="y", edgecolor="k", ax=axes[i, 1]
            )
            sns.lineplot(
                x=intensity,
                y=mean.mean(axis=0),
                label=f"Mean {check}",
                color="k",
                alpha=.4,
                ax=axes[i, 1]
            )

            # choices = jax.random.choice(
            #     self.rng_key,
            #     a=np.array(range(mean.shape[0])),
            #     shape=(n_ppdm,),
            #     replace=False
            # )
            # y_mean = mean[choices, :].T
            # x = np.tile(intensity, (choices.shape[0], 1)).T

            y_mean = mean.mean(axis=0)
            hpdi_mean = hpdi(mean, prob=.95)
            axes[i, 2].plot(intensity, y_mean, color="k")
            axes[i, 2].fill_between(intensity, hpdi_mean[0, :], hpdi_mean[1, :], color="paleturquoise")

            # axes[i, 2].plot(x, y_mean, color="paleturquoise")
            sns.scatterplot(
                data=temp_df, x=INTENSITY, y=RESPONSE, color="y", edgecolor="k", ax=axes[i, 2]
            )
            # sns.lineplot(
            #     x=intensity,
            #     y=mean.mean(axis=0),
            #     label=f"Mean {check}",
            #     color="k",
            #     alpha=.4,
            #     ax=axes[i, 2]
            # )

            """ Labels """
            axes[i, 0].set_title(f"{self.columns} - {c}")
            axes[i, 1].set_title(f"{check} Predictive")
            axes[i, 2].set_title(f"{check} Predictive Mean")

            """ Legends """
            axes[i, 0].legend(loc="upper left")
            axes[i, 1].legend(loc="upper left")
            axes[i, 2].legend(loc="upper left")

        return fig

    def plot(
        self,
        df: pd.DataFrame,
        posterior_samples: dict,
        encoder_dict: Optional[dict] = None,
        mat: Optional[np.ndarray] = None,
        time: Optional[np.ndarray] = None
    ):
        if mat is not None:
            assert time is not None

        combinations = self._get_combinations(df)
        n_combinations = len(combinations)
        intensity = self.x

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

            predictions = self.predict(
                intensity=intensity, combination=c, posterior_samples=posterior_samples
            )
            mean = predictions["mean"]

            threshold, threshold_posterior, hpdi_interval = self._get_threshold_estimates(
                c, posterior_samples
            )

            """ Plots """
            sns.scatterplot(data=temp_df, x=INTENSITY, y=RESPONSE, ax=axes[i, 0])
            sns.scatterplot(data=temp_df, x=INTENSITY, y=RESPONSE, alpha=.4, ax=axes[i, 1])

            sns.kdeplot(x=threshold_posterior, color="b", ax=axes[i, 1])
            sns.lineplot(
                x=intensity,
                y=mean.mean(axis=0),
                label="Mean Posterior",
                color="r",
                alpha=0.4,
                ax=axes[i, 1]
            )

            sns.kdeplot(x=threshold_posterior, color="b", ax=axes[i, 2])
            axes[i, 2].axvline(
                threshold,
                linestyle="--",
                color="r",
                label=f"Mean Posterior"
            )
            axes[i, 2].axvline(
                hpdi_interval[0],
                linestyle="--",
                color="g",
                label=f"95% HPDI Interval"
            )
            axes[i, 2].axvline(hpdi_interval[1], linestyle="--", color="g")

            """ Labels """
            axes[i, 0].set_title(f"{self.columns} - {c}")

            if encoder_dict is not None:
                c_inverse = []
                for column, value in zip(self.columns, c):
                    c_inverse.append(
                        encoder_dict[column].inverse_transform(np.array([value]))[0]
                    )
                title = f"{tuple(self.columns)} - {tuple(c_inverse)}"
            else:
                title = "Model Fit"

            axes[i, 1].set_title(title)

            skew = stats.skew(a=threshold_posterior)
            kurt = stats.kurtosis(a=threshold_posterior)

            title = f"TH: {threshold:.2f}"
            title += f", CI: ({hpdi_interval[0]:.1f}, {hpdi_interval[1]:.1f})"
            title += f", LEN: {hpdi_interval[1] - hpdi_interval[0]:.1f}"
            title += r', $\overline{\mu_3}$'
            title += f": {skew:.1f}"
            title += f", K: {kurt:.1f}"
            axes[i, 2].set_title(title)

            """ Limits """
            axes[i, 1].set_xlim(
                left=temp_df[INTENSITY].min() - self.xpad, right=temp_df[INTENSITY].max() + self.xpad
            )
            axes[i, 1].set_ylim(
                bottom=0, top=temp_df[RESPONSE].max() + .05
            )

            """ Legends """
            axes[i, 1].legend(loc="upper left")
            axes[i, 2].legend(loc="upper right")

            """ EEG Data """
            if mat is not None:
                ax = axes[i, 3]
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
