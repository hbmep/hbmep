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
        self.n_response = None

        self.columns = [PARTICIPANT] + FEATURES
        self.x = np.linspace(0, 800, 2000)
        self.xpad = 10

        self._set_rng_key()

    def _set_rng_key(self):
        self.rng_key = jax.random.PRNGKey(self.random_state)

    def _model(self, intensity, participant, feature0, response_obs=None):
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        n_data = intensity.shape[0]
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

        with numpyro.plate("data", n_data):
            return numpyro.sample("obs", dist.TruncatedNormal(mean, noise, low=0), obs=response_obs)

    @timing
    def run_inference(self, df: pd.DataFrame) -> tuple[numpyro.infer.mcmc.MCMC, dict]:
        """
        Run MCMC inference
        """
        response = df[RESPONSE].to_numpy()
        self.n_response = response.shape[-1]

        intensity = df[INTENSITY].to_numpy().reshape(-1,)
        participant = df[PARTICIPANT].to_numpy().reshape(-1,)
        feature0 = df[FEATURES[0]].to_numpy().reshape(-1,)

        # MCMC
        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(nuts_kernel, **self.config.MCMC_PARAMS)
        rng_key = jax.random.PRNGKey(self.random_state)
        # logger.info(f"Running inference with {self.name} ...")
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

    def _get_threshold_estimates(
        self,
        combination: tuple,
        posterior_samples: dict,
        prob: float = .95
    ):
        threshold_posterior = posterior_samples[site.a][
            :, combination[1], combination[0], :
        ]
        threshold = threshold_posterior.mean(axis=0)
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

        predictions = predictive(
            self.rng_key,
            intensity=intensity,
            participant=participant,
            feature0=feature0
        )
        return predictions

    @timing
    def predictive_check(
        self,
        df: pd.DataFrame,
        posterior_samples: Optional[dict] = None
    ):
        posterior_check = False if posterior_samples is None else True
        check = "Posterior" if posterior_check else "Prior"

        combinations = self._get_combinations(df)
        n_combinations = len(combinations)

        intensity = self.x

        n_columns = 3 * self.n_response
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

            if posterior_check or predictions is None:
                predictions = self.predict(
                    intensity=intensity,
                    combination=c,
                    posterior_samples=posterior_samples
                )
                obs = predictions["obs"]
                mean = predictions["mean"]

                hpdi_obs = hpdi(obs, prob=.95)
                hpdi_mean = hpdi(mean, prob=.95)

                """ Additional """
                hpdi_obs_90 = hpdi(obs, prob=.90)
                hpdi_obs_80 = hpdi(obs, prob=.80)
                hpdi_obs_65 = hpdi(obs, prob=.65)


            j = 0
            for (r, response) in enumerate(RESPONSE):
                """ Plots """
                sns.scatterplot(data=temp_df, x=INTENSITY, y=response, alpha=.4, ax=axes[i, j])
                sns.lineplot(
                    x=intensity,
                    y=mean.mean(axis=0)[:, r],
                    label=f"Mean {check}",
                    color="r",
                    alpha=0.4,
                    ax=axes[i, j]
                )

                axes[i, j + 1].plot(
                    intensity,
                    obs.mean(axis=0)[:, r],
                    color="k",
                    label="Mean Prediction"
                )
                axes[i, j + 1].fill_between(
                    intensity,
                    hpdi_obs[0, :, r],
                    hpdi_obs[1, :, r],
                    color="paleturquoise",
                    label="95% HPDI"
                )

                """ Additional """
                axes[i, j + 1].fill_between(
                    intensity,
                    hpdi_obs_90[0, :, r],
                    hpdi_obs_90[1, :, r],
                    color="C1",
                    label="90% HPDI"
                )
                axes[i, j + 1].fill_between(
                    intensity,
                    hpdi_obs_80[0, :, r],
                    hpdi_obs_80[1, :, r],
                    color="C2",
                    label="80% HPDI"
                )
                axes[i, j + 1].fill_between(
                    intensity,
                    hpdi_obs_65[0, :, r],
                    hpdi_obs_65[1, :, r],
                    color="C3",
                    label="65% HPDI"
                )

                sns.scatterplot(
                    data=temp_df, x=INTENSITY, y=response, color="y", edgecolor="k", ax=axes[i, j + 1]
                )

                axes[i, j + 2].plot(
                    intensity,
                    mean.mean(axis=0)[:, r],
                    color="k",
                    label=f"Mean {check}"
                )
                axes[i, j + 2].fill_between(
                    intensity, hpdi_mean[0, :, r], hpdi_mean[1, :, r], color="paleturquoise", label="95% HPDI"
                )
                sns.scatterplot(
                    data=temp_df, x=INTENSITY, y=response, color="y", edgecolor="k", ax=axes[i, j + 2]
                )

                """ Labels """
                axes[i, j].set_title(f"{response} - {tuple(self.columns)} - {c}")
                axes[i, j + 1].set_title(f"{check} Predictive")
                axes[i, j + 2].set_title(f"{check} Predictive Mean")

                """ Limits """
                axes[i, j].set_xlim(
                    left=max(0, temp_df[INTENSITY].min() - 5 * self.xpad),
                    right=temp_df[INTENSITY].max() + 5 * self.xpad
                )
                axes[i, j].set_xlim(
                    left=max(0, temp_df[INTENSITY].min() - 5 * self.xpad),
                    right=temp_df[INTENSITY].max() + 5 * self.xpad
                )
                axes[i, j + 1].set_xlim(
                    left=max(0, temp_df[INTENSITY].min() - 5 * self.xpad),
                    right=temp_df[INTENSITY].max() + 5 * self.xpad
                )
                axes[i, j + 2].set_xlim(
                    left=max(0, temp_df[INTENSITY].min() - 5 * self.xpad),
                    right=temp_df[INTENSITY].max() + 5 * self.xpad
                )
                """ Legends """
                axes[i, j].legend(loc="upper left")
                axes[i, j + 1].legend(loc="upper left")
                axes[i, j + 2].legend(loc="upper left")

                j += 3

        return fig

    @timing
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

        n_columns = 3 * self.n_response
        if mat is not None: n_columns += self.n_response

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

            j = 0
            for (r, response) in enumerate(RESPONSE):
                """ Plots """
                sns.scatterplot(data=temp_df, x=INTENSITY, y=response, ax=axes[i, j])
                sns.scatterplot(data=temp_df, x=INTENSITY, y=response, alpha=.4, ax=axes[i, j + 1])

                sns.kdeplot(x=threshold_posterior[:, r], color="b", ax=axes[i, j + 1])
                sns.lineplot(
                    x=intensity,
                    y=mean.mean(axis=0)[:, r],
                    label="Mean Posterior",
                    color="r",
                    alpha=0.4,
                    ax=axes[i, j + 1]
                )

                sns.kdeplot(x=threshold_posterior[:, r], color="b", ax=axes[i, j + 2])
                axes[i, j + 2].axvline(
                    threshold[r],
                    linestyle="--",
                    color="r",
                    label=f"Mean Posterior"
                )
                axes[i, j + 2].axvline(
                    hpdi_interval[:, r][0],
                    linestyle="--",
                    color="g",
                    label="95% HPDI"
                )
                axes[i, j + 2].axvline(hpdi_interval[:, r][1], linestyle="--", color="g")

                """ Labels """
                axes[i, j].set_title(f"{response} - {tuple(self.columns)} - {c}")

                if encoder_dict is not None:
                    c_inverse = []
                    for column, value in zip(self.columns, c):
                        c_inverse.append(
                            encoder_dict[column].inverse_transform(np.array([value]))[0]
                        )
                    title = f"{response} - {tuple(c_inverse)}"
                else:
                    title = f"{response} - Model Fit"

                axes[i, j + 1].set_title(title)

                skew = stats.skew(a=threshold_posterior[:, r])
                kurt = stats.kurtosis(a=threshold_posterior[:, r])

                title = f"{response} - TH: {threshold[r]:.2f}"
                title += f", CI: ({hpdi_interval[:, r][0]:.1f}, {hpdi_interval[:, r][1]:.1f})"
                title += f", LEN: {hpdi_interval[:, r][1] - hpdi_interval[:, r][0]:.1f}"
                title += r', $\overline{\mu_3}$'
                title += f": {skew:.1f}"
                title += f", K: {kurt:.1f}"
                axes[i, j + 2].set_title(title)

                """ Limits """
                axes[i, j + 1].set_xlim(
                    left=max(0, temp_df[INTENSITY].min() - 2 * self.xpad),
                    right=temp_df[INTENSITY].max() + self.xpad
                )
                axes[i, j + 1].set_ylim(
                    bottom=0, top=temp_df[response].max() + .05
                )

                """ Legends """
                axes[i, j + 1].legend(loc="upper left")
                axes[i, j + 2].legend(loc="upper right")

                j += 3

                """ EEG Data """
                if mat is not None:
                    ax = axes[i, j]
                    temp_mat = mat[idx, :, r]

                    for k in range(temp_mat.shape[0]):
                        x = temp_mat[k, :]/60 + temp_df[INTENSITY].values[k]
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

                    j += 1

        return fig
