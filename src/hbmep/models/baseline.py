import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import jax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

from hbmep.config import MepConfig
from hbmep.dataset import MepDataset
from hbmep.models.utils import Site as site
from hbmep.utils import (
    timing,
    floor,
    ceil,
    evaluate_posterior_mean,
    evaluate_hpdi_interval
)
from hbmep.utils.constants import (
    BASELINE,
    RECRUITMENT_CURVES,
    PRIOR_PREDICTIVE,
    POSTERIOR_PREDICTIVE
)

logger = logging.getLogger(__name__)


class Baseline(MepDataset):
    def __init__(self, config: MepConfig):
        super(Baseline, self).__init__(config=config)
        self.model_name = BASELINE
        self.random_state = 0
        self.rng_key = jax.random.PRNGKey(self.random_state)

        self.mcmc_params = config.MCMC_PARAMS
        self.recruitment_curves_path = os.path.join(self.run_dir, RECRUITMENT_CURVES)
        self.prior_predictive_path = os.path.join(self.run_dir, PRIOR_PREDICTIVE)
        self.posterior_predictive = os.path.join(self.run_dir, POSTERIOR_PREDICTIVE)

    def _model(self, subject, features, intensity, response_obs=None):
        intensity = intensity.reshape(-1, 1)
        intensity = np.tile(intensity, (1, self.n_response))

        feature0 = features[0].reshape(-1,)

        n_data = intensity.shape[0]
        n_subject = np.unique(subject).shape[0]
        n_feature0 = np.unique(feature0).shape[0]

        with numpyro.plate("n_feature0", n_feature0, dim=-1):
            """ Hyperpriors """
            a_mean = numpyro.sample(
                "a_mean",
                dist.TruncatedDistribution(dist.Normal(150, 50), low=0)
            )

            a_scale = numpyro.sample(site.a_scale, dist.HalfNormal(20))
            b_scale = numpyro.sample(site.b_scale, dist.HalfNormal(100))
            lo_scale = numpyro.sample(site.lo_scale, dist.HalfNormal(5))

            with numpyro.plate("n_subject", n_subject, dim=-2):
                """ Priors """
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

        """ Model """
        mean = \
            lo[subject, feature0] + \
            self.link(
                b[subject, feature0] * (intensity - a[subject, feature0])
            )
        noise = noise_offset[subject, feature0] + noise_slope[subject, feature0] * mean

        with numpyro.plate("data", n_data):
            return numpyro.sample("obs", dist.TruncatedNormal(mean, noise, low=0), obs=response_obs)

    @timing
    def run_inference(self, df: pd.DataFrame) -> tuple[numpyro.infer.mcmc.MCMC, dict]:
        """ Prepare dataset """
        subject = df[self.subject].to_numpy().reshape(-1,)
        features = df[self.features].to_numpy().T
        intensity = df[self.intensity].to_numpy().reshape(-1,)
        response = df[self.response].to_numpy()

        """ MCMC """
        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(nuts_kernel, **self.mcmc_params)
        rng_key = jax.random.PRNGKey(self.random_state)

        logger.info(f"Running inference with {self.model_name} ...")
        mcmc.run(rng_key, subject, features, intensity, response)
        posterior_samples = mcmc.get_samples()
        return mcmc, posterior_samples

    def _estimate_threshold(
        self,
        combination: tuple[int],
        posterior_samples: dict,
        prob: float = .95
    ):
        """ Set index """
        ind = [slice(None)] + list(combination) + [slice(None)]
        ind = ind[::-1]

        """ Posterior mean """
        posterior_samples = posterior_samples[site.a][tuple(ind)]
        threshold = evaluate_posterior_mean(
            posterior_samples=posterior_samples,
            prob=prob
        )

        """ HPDI Interval """
        hpdi_interval = evaluate_hpdi_interval(
            posterior_samples=posterior_samples,
            prob=prob
        )
        return threshold, posterior_samples, hpdi_interval

    def _predict(
        self,
        intensity: np.ndarray,
        combination: tuple[int],
        posterior_samples: Optional[dict] = None,
        num_samples: int = 100
    ):
        if posterior_samples is None:   # Prior predictive
            predictive = Predictive(
                model=self._model, num_samples=num_samples
            )
        else:   # Posterior predictive
            predictive = Predictive(
                model=self._model, posterior_samples=posterior_samples
            )

        """ Prepare dataset """
        combination = np.array(list(combination))
        combination = np.tile(combination, (intensity.shape[0], 1)).T
        subject = combination[0]
        features = combination[1:]

        """ Predictions """
        predictions = predictive(
            self.rng_key,
            subject=subject,
            features=features,
            intensity=intensity
        )
        return predictions

    @timing
    def render_recruitment_curves(
        self,
        df: pd.DataFrame,
        posterior_samples: dict,
        encoder_dict: Optional[dict] = None,
        mat: Optional[np.ndarray] = None,
        time: Optional[np.ndarray] = None,
        auc_window: Optional[list[float]] = None
    ):
        if mat is not None:
            assert time is not None
            assert auc_window is not None

        """ Setup pdf layout """
        combinations = self._make_combinations(df=df, columns=self.columns)
        n_combinations = len(combinations)

        n_columns_per_response = 3
        if mat is not None: n_columns_per_response += 1

        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * self.n_response

        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1

        """ Iterate over pdf pages """
        pdf = PdfPages(self.recruitment_curves_path)
        combination_counter = 0

        for page in range(n_pdf_pages):
            n_rows_current_page = min(
                n_fig_rows,
                n_combinations - page * n_fig_rows
            )

            fig, axes = plt.subplots(
                n_rows_current_page,
                n_fig_columns,
                figsize=(n_fig_columns * 5, n_rows_current_page * 3),
                constrained_layout=True,
                squeeze=False
            )

            """ Iterate over combinations """
            for i in range(n_rows_current_page):
                combination = combinations[combination_counter]

                """ Filter dataframe """
                ind = df[self.columns].apply(tuple, axis=1).isin([combination])
                temp_df = df[ind].reset_index(drop=True).copy()

                """ Tickmarks """
                min_intensity = temp_df[self.intensity].min()
                min_intensity = floor(min_intensity, base=self.base)
                max_intensity = temp_df[self.intensity].max()
                max_intensity = ceil(max_intensity, base=self.base)

                n_points = min(2000, ceil((max_intensity - min_intensity) / 5, base=100))
                x_space = np.linspace(min_intensity, max_intensity, n_points)
                x_ticks = np.arange(min_intensity, max_intensity, self.base)

                """ Predictions """
                predictions = self._predict(
                    intensity=x_space,
                    combination=combination,
                    posterior_samples=posterior_samples
                )
                mu = predictions[site.mu]
                mu_posterior_mean = evaluate_posterior_mean(mu)

                """ Threshold estimate """
                threshold, threshold_posterior, hpdi_interval = \
                    self._estimate_threshold(combination, posterior_samples)

                """ Iterate over responses """
                for (r, response) in enumerate(self.response):
                    j = n_columns_per_response * r

                    """ EEG Data """
                    if mat is not None:
                        ax = axes[i, j]
                        temp_mat = mat[ind, :, r]

                        for k in range(temp_mat.shape[0]):
                            x = temp_mat[k, :]/60 + temp_df[self.intensity].values[k]
                            ax.plot(x, time, color="green", alpha=.4)

                        ax.axhline(
                            y=auc_window[0],
                            color="red",
                            linestyle='--',
                            alpha=.4,
                            label=f"AUC Window {auc_window}"
                        )
                        ax.axhline(
                            y=auc_window[1],
                            color="red",
                            linestyle='--',
                            alpha=.4
                        )

                        ax.set_xticks(ticks=x_ticks)
                        ax.tick_params(axis="x", rotation=90)
                        ax.set_xlim(left=min_intensity, right=max_intensity)
                        ax.set_ylim(bottom=-0.001, top=auc_window[1] + .005)

                        ax.set_xlabel(f"{self.intensity}")
                        ax.set_ylabel(f"Time")
                        ax.legend(loc="upper right")
                        ax.set_title(f"Motor Evoked Potential")

                        j += 1

                    """ Plots """
                    sns.scatterplot(
                        data=temp_df,
                        x=self.intensity,
                        y=response,
                        ax=axes[i, j]
                    )
                    sns.scatterplot(
                        data=temp_df,
                        x=self.intensity,
                        y=response,
                        alpha=.4,
                        ax=axes[i, j + 1]
                    )

                    sns.kdeplot(
                        x=threshold_posterior[:, r],
                        color="b",
                        ax=axes[i, j + 1],
                        alpha=.4
                    )
                    sns.lineplot(
                        x=x_space,
                        y=mu_posterior_mean[:, r],
                        label="Mean Posterior",
                        color="r",
                        alpha=0.4,
                        ax=axes[i, j + 1]
                    )

                    sns.kdeplot(
                        x=threshold_posterior[:, r],
                        color="b",
                        ax=axes[i, j + 2]
                    )
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
                    axes[i, j + 2].axvline(
                        hpdi_interval[:, r][1],
                        linestyle="--",
                        color="g"
                    )

                    """ Labels """
                    title = f"{response} - {tuple(self.columns)} - {combination}"
                    axes[i, j].set_title(title)

                    """ Inverted labels """
                    if encoder_dict is not None:
                        combination_inverse = self._invert_combination(
                            combination=combination,
                            colums=self.columns,
                            encoder_dict=encoder_dict
                        )
                        title = f"{response} - {tuple(combination_inverse)}"
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

                    """ Ticks """
                    for k in [j, j + 1]:
                        ax = axes[i, k]
                        ax.set_xticks(ticks=x_ticks)
                        ax.tick_params(axis="x", rotation=90)
                        ax.set_xlim(left=min_intensity, right=max_intensity)

                    """ Legends """
                    axes[i, j + 1].legend(loc="upper left")
                    axes[i, j + 2].legend(loc="upper right")

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {self.recruitment_curves_path}")
        return

    @timing
    def render_predictive_check(
        self,
        df: pd.DataFrame,
        posterior_samples: Optional[dict] = None
    ):
        """ Posterior / Prior Predictive Check """
        is_posterior_check = True
        dest_path = self.posterior_predictive
        if posterior_samples is None: is_posterior_check = False
        if posterior_samples is None: dest_path = self.prior_predictive_path
        check_type = "Posterior" if is_posterior_check else "Prior"

        """ Setup pdf layout """
        combinations = self._make_combinations(df=df, columns=self.columns)
        n_combinations = len(combinations)

        n_columns_per_response = 3
        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * self.n_response

        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1

        """ Iterate over pdf pages """
        pdf = PdfPages(dest_path)
        combination_counter = 0
        predictions = None

        for page in range(n_pdf_pages):
            n_rows_current_page = min(
                n_fig_rows,
                n_combinations - page * n_fig_rows
            )

            fig, axes = plt.subplots(
                n_rows_current_page,
                n_fig_columns,
                figsize=(n_fig_columns * 5, n_rows_current_page * 3),
                constrained_layout=True,
                squeeze=False
            )

            """ Iterate over combinations """
            for i in range(n_rows_current_page):
                combination = combinations[combination_counter]

                """ Filter dataframe """
                ind = df[self.columns].apply(tuple, axis=1).isin([combination])
                temp_df = df[ind].reset_index(drop=True).copy()

                """ Tickmarks """
                min_intensity = temp_df[self.intensity].min()
                min_intensity = floor(min_intensity, base=self.base)
                max_intensity = temp_df[self.intensity].max()
                max_intensity = ceil(max_intensity, base=self.base)

                n_points = min(2000, ceil((max_intensity - min_intensity) / 5, base=100))
                x_space = np.linspace(min_intensity, max_intensity, n_points)
                x_ticks = np.arange(min_intensity, max_intensity, self.base)

                if is_posterior_check or predictions is None:
                    """ Predictions """
                    predictions = self._predict(
                        intensity=x_space,
                        combination=combination,
                        posterior_samples=posterior_samples
                    )
                    obs = predictions[site.obs]
                    mu = predictions[site.mu]

                    """ Posterior mean """
                    obs_posterior_mean = evaluate_posterior_mean(obs)
                    mu_posterior_mean = evaluate_posterior_mean(mu)

                    """ HPDI Intervals """
                    hpdi_obs_95 = evaluate_hpdi_interval(obs, prob=.95)
                    hpdi_obs_85 = evaluate_hpdi_interval(obs, prob=.85)
                    hpdi_obs_65 = evaluate_hpdi_interval(obs, prob=.65)

                    hpdi_mu_95 = evaluate_hpdi_interval(mu, prob=.95)

                """ Iterate over responses """
                for (r, response) in enumerate(self.response):
                    j = n_columns_per_response * r

                    """ Plots """
                    sns.scatterplot(
                        data=temp_df,
                        x=self.intensity,
                        y=response,
                        alpha=.4,
                        ax=axes[i, j]
                    )
                    sns.lineplot(
                        x=x_space,
                        y=mu_posterior_mean[:, r],
                        label=f"Mean {check_type}",
                        color="r",
                        alpha=0.4,
                        ax=axes[i, j]
                    )

                    axes[i, j + 1].plot(
                        x_space,
                        obs_posterior_mean[:, r],
                        color="k",
                        label="Mean Prediction"
                    )
                    axes[i, j + 1].fill_between(
                        x_space,
                        hpdi_obs_95[0, :, r],
                        hpdi_obs_95[1, :, r],
                        color="C1",
                        label="95% HPDI"
                    )
                    axes[i, j + 1].fill_between(
                        x_space,
                        hpdi_obs_85[0, :, r],
                        hpdi_obs_85[1, :, r],
                        color="C2",
                        label="85% HPDI"
                    )
                    axes[i, j + 1].fill_between(
                        x_space,
                        hpdi_obs_65[0, :, r],
                        hpdi_obs_65[1, :, r],
                        color="C3",
                        label="65% HPDI"
                    )

                    sns.scatterplot(
                        data=temp_df,
                        x=self.intensity,
                        y=response,
                        color="y",
                        edgecolor="k",
                        ax=axes[i, j + 1]
                    )

                    axes[i, j + 2].plot(
                        x_space,
                        mu_posterior_mean[:, r],
                        color="k",
                        label=f"Mean {check_type}"
                    )
                    axes[i, j + 2].fill_between(
                        self.x_space,
                        hpdi_mu_95[0, :, r],
                        hpdi_mu_95[1, :, r],
                        color="paleturquoise",
                        label="95% HPDI"
                    )
                    sns.scatterplot(
                        data=temp_df,
                        x=self.intensity,
                        y=response,
                        color="y",
                        edgecolor="k",
                        ax=axes[i, j + 2]
                    )

                    """ Labels """
                    title = f"{response} - {tuple(self.columns)} - {combination}"
                    axes[i, j].set_title(title)
                    axes[i, j + 1].set_title(f"{check_type} Predictive")
                    axes[i, j + 2].set_title(f"{check_type} Predictive Mean")

                    """ Ticks """
                    for k in [j, j + 1, j + 2]:
                        ax = axes[i, k]
                        ax.set_xticks(ticks=x_ticks)
                        ax.tick_params(axis="x", rotation=90)
                        ax.set_xlim(left=min_intensity, right=max_intensity)

                    """ Legends """
                    axes[i, j].legend(loc="upper left")
                    axes[i, j + 1].legend(loc="upper left")
                    axes[i, j + 2].legend(loc="upper left")

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {dest_path}")
        return
