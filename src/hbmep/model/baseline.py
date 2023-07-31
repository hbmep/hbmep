import os
import itertools
import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder

import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive

from hbmep.config import Config
from hbmep.dataset import Dataset
from hbmep.model.utils import Site as site
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
    POSTERIOR_PREDICTIVE,
    MCMC_NC,
    DIAGNOSTICS_CSV,
    LOO_CSV,
    WAIC_CSV,
    RESPONSE
)

logger = logging.getLogger(__name__)


class Baseline(Dataset):
    LINK = BASELINE

    def __init__(self, config: Config):
        super(Baseline, self).__init__(config=config)

        self.random_state = 0
        self.rng_key = jax.random.PRNGKey(self.random_state)
        self.base = config.BASE
        self.mcmc_params = config.MCMC_PARAMS

        self.recruitment_curves_path = os.path.join(self.build_dir, RECRUITMENT_CURVES)
        self.prior_predictive_path = os.path.join(self.build_dir, PRIOR_PREDICTIVE)
        self.posterior_predictive_path = os.path.join(self.build_dir, POSTERIOR_PREDICTIVE)
        self.mcmc_path = os.path.join(self.build_dir, MCMC_NC)
        self.diagnostics_path = os.path.join(self.build_dir, DIAGNOSTICS_CSV)
        self.loo_path = os.path.join(self.build_dir, LOO_CSV)
        self.waic_path = os.path.join(self.build_dir, WAIC_CSV)

        logger.info(f"Initialized model with {self.LINK} link")

    def _model(self, subject, features, intensity, response_obs=None):
        pass

    @timing
    def simulate(self):
        x_space = np.arange(0, 360, 4)

        n_subject = 3
        n_features = [n_subject]
        n_features += jax.random.choice(self.rng_key, jnp.array([2, 3, 4]), shape=(self.n_features,)).tolist()
        n_features[-1] = 2

        combinations = itertools.product(*[range(i) for i in n_features])
        combinations = list(combinations)

        n_combinations = min(10, len(combinations))
        ind = jax.random.choice(self.rng_key, len(combinations), shape=(n_combinations,), replace=False)
        combinations = [combinations[i] for i in ind]
        combinations = sorted(combinations)

        obs = None
        num_samples = 100
        ind = jax.random.choice(self.rng_key, num_samples, shape=(n_combinations,))

        logger.info("Simulating data ...")
        for i, combination in enumerate(combinations):
            pred = self._predict(intensity=x_space, combination=combination, num_samples=num_samples)
            pred = pred[site.obs][ind[i], ...]
            obs = pred if obs is None else jnp.concatenate([obs, pred], axis=0)

        df = pd.DataFrame(combinations, columns=self.combination_columns) \
            .merge(pd.DataFrame(x_space, columns=[self.intensity]), how="cross") \
            .sort_values(by=self.regressors) \
            .reset_index(drop=True) \
            .copy()

        df[self.response] = obs
        return df

    @timing
    def trace(self, df: pd.DataFrame):
        with numpyro.handlers.seed(rng_seed=self.random_state):
            trace = numpyro.handlers.trace(self._model).get_trace(
                *self._make_exogenous_data(df=df),
                *self._make_endogenous_data(df=df)
            )
        return trace

    @timing
    def _collect_regressors(self, df: pd.DataFrame):
        subject = df[self.subject].to_numpy().reshape(-1,)
        features = df[self.features].to_numpy().T
        intensity = df[self.intensity].to_numpy().reshape(-1,)
        return subject, features, intensity,

    @timing
    def _collect_response(self, df: pd.DataFrame):
        response = df[self.response].to_numpy()
        return response,

    @timing
    def run_inference(self, df: pd.DataFrame) -> tuple[numpyro.infer.mcmc.MCMC, dict]:
        """ Set up NUTS sampler """
        nuts_kernel = NUTS(self._model)
        mcmc = MCMC(nuts_kernel, **self.mcmc_params)
        rng_key = jax.random.PRNGKey(self.random_state)

        """ MCMC inference """
        logger.info(f"Running inference with {self.LINK} ...")
        mcmc.run(rng_key, *self._collect_regressors(df=df), *self._collect_response(df=df))
        posterior_samples = mcmc.get_samples()
        return mcmc, posterior_samples

    @timing
    def run_trace(self, df: pd.DataFrame):
        with numpyro.handlers.seed(rng_seed=self.random_state):
            trace = numpyro.handlers.trace(self._model).get_trace(
                *self._collect_regressors(df=df), *self._collect_response(df=df)
            )
        return trace

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
        encoder_dict: dict[str, LabelEncoder],
        posterior_samples: dict
    ):
        if self.mep_matrix_path is not None:
            mep_matrix = np.load(self.mep_matrix_path)
            a, b = self.mep_window
            time = np.linspace(a, b, mep_matrix.shape[1])

        """ Setup pdf layout """
        combinations = self._make_combinations(df=df, columns=self.combination_columns)
        n_combinations = len(combinations)

        n_columns_per_response = 3
        if self.mep_matrix_path is not None: n_columns_per_response += 1

        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * self.n_response

        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1
        logger.info("Rendering recruitment curves ...")

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
                figsize=(
                    n_fig_columns * self.subplot_cell_width,
                    n_rows_current_page * self.subplot_cell_height
                ),
                constrained_layout=True,
                squeeze=False
            )

            """ Iterate over combinations """
            for i in range(n_rows_current_page):
                combination = combinations[combination_counter]

                """ Filter dataframe """
                ind = df[self.combination_columns].apply(tuple, axis=1).isin([combination])
                temp_df = df[ind].reset_index(drop=True).copy()

                """ Tickmarks """
                min_intensity, max_intensity_ = temp_df[self.intensity].agg([min, max])
                min_intensity = floor(min_intensity, base=self.base)
                max_intensity = ceil(max_intensity_, base=self.base)
                if max_intensity == max_intensity_:
                    max_intensity += self.base
                x_ticks = np.arange(min_intensity, max_intensity, self.base)

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

                    """ MEP data """
                    if self.mep_matrix_path is not None:
                        ax = axes[i, j]
                        temp_mep_matrix = mep_matrix[ind, :, r]

                        for k in range(temp_mep_matrix.shape[0]):
                            x = temp_mep_matrix[k, :] / 60 + temp_df[self.intensity].values[k]
                            ax.plot(x, time, color="g", alpha=.4)

                        if self.mep_size_window is not None:
                            ax.axhline(
                                y=self.mep_size_window[0], color="r", linestyle="--", alpha=.4, label="MEP Size Window"
                            )
                            ax.axhline(
                                y=self.mep_size_window[1], color="r", linestyle="--", alpha=.4
                            )

                        ax.set_xticks(ticks=x_ticks)
                        ax.tick_params(axis="x", rotation=90)
                        ax.set_xlim(left=min_intensity, right=max_intensity)
                        ax.set_ylim(bottom=-0.001, top=self.mep_size_window[1] + .005)
                        ax.set_xlabel(f"{self.intensity}")
                        ax.set_ylabel(f"Time")
                        ax.legend(loc="upper right")
                        ax.set_title(f"{response} - MEP")

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

                    """ Threshold KDE """
                    sns.kdeplot(
                        x=threshold_posterior[:, r],
                        color="b",
                        ax=axes[i, j + 1],
                        alpha=.4
                    )
                    sns.kdeplot(
                        x=threshold_posterior[:, r],
                        color="b",
                        ax=axes[i, j + 2]
                    )

                    """ Plots: Recruitment curve """
                    sns.lineplot(
                        x=x_space,
                        y=mu_posterior_mean[:, r],
                        label="Mean Recruitment Curve",
                        color="r",
                        alpha=0.4,
                        ax=axes[i, j + 1]
                    )

                    """ Plots: Threshold estimate """
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
                    title = f"{tuple(list(self.combination_columns)[::-1] + [RESPONSE])}"
                    title += f": {tuple(list(combination)[::-1] + [r])}"
                    combination_inverse = self._invert_combination(
                        combination=combination,
                        columns=self.combination_columns,
                        encoder_dict=encoder_dict
                    )
                    title += f"\ndecoded: {tuple(list(combination_inverse)[::-1] + [response])}"
                    axes[i, j].set_title(title)
                    axes[i, j + 1].set_title("Model Fit")

                    skew = stats.skew(a=threshold_posterior[:, r])
                    kurt = stats.kurtosis(a=threshold_posterior[:, r])

                    title = f"TH: {threshold[r]:.2f}"
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
                    for k in [j + 1, j + 2]:
                        ax = axes[i, k]
                        ax.legend(loc="upper left")

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
        encoder_dict: dict[str, LabelEncoder],
        posterior_samples: Optional[dict] = None
    ):
        """ Posterior / Prior Predictive Check """
        is_posterior_check = True
        dest_path = self.posterior_predictive_path
        if posterior_samples is None: is_posterior_check = False
        if posterior_samples is None: dest_path = self.prior_predictive_path
        check_type = "Posterior" if is_posterior_check else "Prior"

        """ Setup pdf layout """
        combinations = self._make_combinations(df=df, columns=self.combination_columns)
        n_combinations = len(combinations)

        n_columns_per_response = 3
        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * self.n_response

        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1
        logger.info(f"Rendering {check_type} Predictive Check ...")

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
                figsize=(
                    n_fig_columns * self.subplot_cell_width,
                    n_rows_current_page * self.subplot_cell_height
                ),
                constrained_layout=True,
                squeeze=False
            )

            """ Iterate over combinations """
            for i in range(n_rows_current_page):
                combination = combinations[combination_counter]

                """ Filter dataframe """
                ind = df[self.combination_columns].apply(tuple, axis=1).isin([combination])
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

                    """ HPDI intervals """
                    hpdi_obs_95 = evaluate_hpdi_interval(obs, prob=.95)
                    hpdi_obs_85 = evaluate_hpdi_interval(obs, prob=.85)
                    hpdi_obs_65 = evaluate_hpdi_interval(obs, prob=.65)

                    hpdi_mu_95 = evaluate_hpdi_interval(mu, prob=.95)
                    if not is_posterior_check:
                        hpdi_mu_85 = evaluate_hpdi_interval(mu, prob=.85)
                        hpdi_mu_65 = evaluate_hpdi_interval(mu, prob=.65)

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
                        label=f"Mean Recruitment Curve",
                        color="r",
                        alpha=0.4,
                        ax=axes[i, j]
                    )

                    """ Plots: Predictions """
                    sns.lineplot(
                        x=x_space,
                        y=obs_posterior_mean[:, r],
                        color="k",
                        label=f"Mean Prediction",
                        ax=axes[i, j + 1]
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

                    """ Plots: Recruitment curves """
                    sns.lineplot(
                        x=x_space,
                        y=mu_posterior_mean[:, r],
                        color="k",
                        label=f"Mean Recruitment Curve",
                        ax=axes[i, j + 2]
                    )
                    axes[i, j + 2].fill_between(
                        x_space,
                        hpdi_mu_95[0, :, r],
                        hpdi_mu_95[1, :, r],
                        color="C1",
                        label="95% HPDI"
                    )
                    if not is_posterior_check:
                        axes[i, j + 2].fill_between(
                            x_space,
                            hpdi_mu_85[0, :, r],
                            hpdi_mu_85[1, :, r],
                            color="C2",
                            label="85% HPDI"
                        )
                        axes[i, j + 2].fill_between(
                            x_space,
                            hpdi_mu_65[0, :, r],
                            hpdi_mu_65[1, :, r],
                            color="C3",
                            label="65% HPDI"
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
                    title = f"{tuple(list(self.combination_columns)[::-1] + [RESPONSE])}"
                    title += f": {tuple(list(combination)[::-1] + [r])}"
                    combination_inverse = self._invert_combination(
                        combination=combination,
                        columns=self.combination_columns,
                        encoder_dict=encoder_dict
                    )
                    title += f"\ndecoded: {tuple(list(combination_inverse)[::-1] + [response])}"
                    axes[i, j].set_title(title)
                    axes[i, j + 1].set_title(f"{check_type} Predictive")
                    axes[i, j + 2].set_title(f"{check_type} Predictive Recruitment Curves")

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

    @timing
    def save(self, mcmc: numpyro.infer.mcmc.MCMC):
        """ Save inference data """
        logger.info("Saving inference data ...")
        numpyro_data = az.from_numpyro(mcmc)
        numpyro_data.to_netcdf(self.mcmc_path)
        logger.info(f"Saved to {self.mcmc_path}")

        """ Save convergence diagnostics """
        logger.info("Rendering convergence diagnostics ...")
        az.summary(data=numpyro_data, hdi_prob=.95).to_csv(self.diagnostics_path)
        logger.info(f"Saved to {self.diagnostics_path}")

        """ Model evaluation """
        logger.info("Evaluating model ...")
        score = az.loo(numpyro_data)
        logger.info(f"ELPD LOO (Log): {score.elpd_loo:.2f}")
        score.to_csv(self.loo_path)

        score = az.waic(numpyro_data)
        logger.info(f"ELPD WAIC (Log): {score.elpd_waic:.2f}")
        score.to_csv(self.waic_path)
