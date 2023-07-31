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
from numpyro.diagnostics import hpdi

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

    def _collect_regressors(self, df: pd.DataFrame):
        subject = df[self.subject].to_numpy().reshape(-1,)
        features = df[self.features].to_numpy().T
        intensity = df[self.intensity].to_numpy().reshape(-1,)
        return subject, features, intensity,

    def _collect_response(self, df: pd.DataFrame):
        response = df[self.response].to_numpy()
        return response,

    def _make_index_from_combination(self, combination: tuple[int]):
        ind = [slice(None)] + list(combination) + [slice(None)]
        ind = ind[::-1]
        return tuple(ind)

    def _collect_samples_at_combination(self, combination: tuple[int], samples: np.ndarray):
        return samples[*self._make_index_from_combination(combination=combination)]

    def _make_prediction_dataset(self, df: pd.DataFrame):
        pred_df = df \
            .groupby(by=self.combination_columns) \
            .agg({self.intensity: [min, max]}) \
            .copy()

        pred_df.columns = pred_df.columns.map(lambda x: x[1])
        pred_df = pred_df.reset_index().copy()

        pred_df["min"] = pred_df["min"].apply(lambda x: floor(x, base=self.base))
        pred_df["max"] = pred_df["max"] \
            .apply(lambda x: (x, ceil(x, base=self.base))) \
            .apply(lambda x: x[0] + self.base if x[0] == x[1] else x[1])

        pred_df[self.intensity] = pred_df[["min", "max"]] \
            .apply(lambda x: (x[0], x[1], min(2000, ceil((x[1] - x[0]) / 5, base=100))), axis=1) \
            .apply(lambda x: np.linspace(x[0], x[1], x[2]))
        pred_df = pred_df.explode(column=self.intensity)[self.regressors].copy()
        pred_df[self.intensity] = pred_df[self.intensity].astype(float)

        pred_df.reset_index(drop=True, inplace=True)
        return pred_df

    @timing
    def run_trace(self, df: pd.DataFrame):
        with numpyro.handlers.seed(rng_seed=self.random_state):
            trace = numpyro.handlers.trace(self._model).get_trace(
                *self._collect_regressors(df=df), *self._collect_response(df=df)
            )
        return trace

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
    def predict(
        self,
        df: pd.DataFrame,
        num_samples: int = 100,
        posterior_samples: Optional[dict] = None
    ):
        if posterior_samples is None:   # Prior predictive
            predictive = Predictive(
                model=self._model, num_samples=num_samples
            )
        else:   # Posterior predictive
            predictive = Predictive(
                model=self._model, posterior_samples=posterior_samples
            )

        """ Generate predictions """
        predictions = predictive(self.rng_key, *self._collect_regressors(df=df))
        return predictions

    @timing
    def simulate(self):
        n_subject = 3
        n_features = [n_subject]
        n_features += jax.random.choice(self.rng_key, jnp.array([2, 3, 4]), shape=(self.n_features,)).tolist()
        n_features[-1] = 2

        combinations = itertools.product(*[range(i) for i in n_features])
        combinations = list(combinations)
        combinations = sorted(combinations)

        logger.info("Simulating data ...")
        x_space = np.arange(0, 360, 4)
        df = pd.DataFrame(combinations, columns=self.combination_columns)
        df[self.intensity] = df.apply(lambda _: x_space, axis=1)
        df = df.explode(column=self.intensity).reset_index(drop=True).copy()
        df[self.intensity] = df[self.intensity].astype(float)

        pred = self.predict(df=df)
        obs = pred[site.obs]

        df[self.response] = obs[0, ...]
        return df

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

        """ Generate predictions """
        logger.info("Generating predictions ...")
        pred_df = self._make_prediction_dataset(df=df)
        mu_posterior = self.predict(df=pred_df, posterior_samples=posterior_samples)[site.mu]
        mu_posterior = np.array(mu_posterior)

        """ Setup pdf layout """
        combinations = self._make_combinations(df=df, columns=self.combination_columns)
        n_combinations = len(combinations)

        n_columns_per_response = 3
        if self.mep_matrix_path is not None: n_columns_per_response += 1

        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * self.n_response

        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1

        """ Recruitment curves """
        logger.info("Rendering recruitment curves ...")
        pdf = PdfPages(self.recruitment_curves_path)
        combination_counter = 0

        """ Iterate over pdf pages """
        for page in range(n_pdf_pages):
            """ No. of rows for current page """
            n_rows_current_page = min(
                n_fig_rows,
                n_combinations - page * n_fig_rows
            )

            """ Figure for current page """
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

                """ Filter prediction dataframe based on current combination """
                ind = pred_df[self.combination_columns].apply(tuple, axis=1).isin([combination])
                temp_pred_df = pred_df[ind].reset_index(drop=True).copy()

                """ Predictions for current combination """
                curr_mu_posterior = mu_posterior[:, ind, :]
                curr_mu_posterior_mean = curr_mu_posterior.mean(axis=0)

                """ Filter dataframe based on current combination """
                ind = df[self.combination_columns].apply(tuple, axis=1).isin([combination])
                temp_df = df[ind].reset_index(drop=True).copy()

                """ Tickmarks """
                min_intensity, max_intensity_ = temp_df[self.intensity].agg([min, max])
                min_intensity = floor(min_intensity, base=self.base)
                max_intensity = ceil(max_intensity_, base=self.base)
                if max_intensity == max_intensity_:
                    max_intensity += self.base
                x_ticks = np.arange(min_intensity, max_intensity, self.base)

                """ Estimate threshold """
                threshold_posterior = self._collect_samples_at_combination(
                    combination=combination, samples=posterior_samples[site.a]
                )
                threshold = threshold_posterior.mean(axis=0)
                hpdi_interval = hpdi(threshold_posterior, prob=0.95)

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
                        x=temp_pred_df[self.intensity],
                        y=curr_mu_posterior_mean[:, r],
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

                        left, right = ax.get_xlim()
                        left = max(left, min_intensity - self.base)
                        right = min(right, max_intensity + self.base)
                        ax.set_xlim(left=left, right=right)

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

        """ Generate predictions """
        logger.info("Generating predictions ...")
        pred_df = self._make_prediction_dataset(df=df)
        obs_posterior = self.predict(df=pred_df, posterior_samples=posterior_samples)
        mu_posterior = np.array(obs_posterior[site.mu])
        obs_posterior = np.array(obs_posterior[site.obs])

        """ Setup pdf layout """
        combinations = self._make_combinations(df=df, columns=self.combination_columns)
        n_combinations = len(combinations)

        n_columns_per_response = 3
        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * self.n_response

        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1

        """ Predictive check """
        logger.info(f"Rendering {check_type} Predictive Check ...")
        pdf = PdfPages(dest_path)
        combination_counter = 0

        """ Iterate over pdf pages """
        for page in range(n_pdf_pages):
            """ No. of rows for current page """
            n_rows_current_page = min(
                n_fig_rows,
                n_combinations - page * n_fig_rows
            )

            """ Figure for current page """
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

                """ Filter prediction dataframe based on current combination """
                ind = pred_df[self.combination_columns].apply(tuple, axis=1).isin([combination])
                temp_pred_df = pred_df[ind].reset_index(drop=True).copy()

                """ Predictions for current combination """
                curr_obs_posterior = obs_posterior[:, ind, :]
                curr_mu_posterior = mu_posterior[:, ind, :]

                """ Filter dataframe based on current combination """
                ind = df[self.combination_columns].apply(tuple, axis=1).isin([combination])
                temp_df = df[ind].reset_index(drop=True).copy()

                """ Tickmarks """
                min_intensity, max_intensity_ = temp_df[self.intensity].agg([min, max])
                min_intensity = floor(min_intensity, base=self.base)
                max_intensity = ceil(max_intensity_, base=self.base)
                if max_intensity == max_intensity_:
                    max_intensity += self.base
                x_ticks = np.arange(min_intensity, max_intensity, self.base)

                """ Posterior mean """
                curr_obs_posterior_mean = curr_obs_posterior.mean(axis=0)
                curr_mu_posterior_mean = curr_mu_posterior.mean(axis=0)

                """ HPDI intervals """
                hpdi_obs_95 = hpdi(curr_obs_posterior, prob=.95)
                hpdi_obs_85 = hpdi(curr_obs_posterior, prob=.85)
                hpdi_obs_65 = hpdi(curr_obs_posterior, prob=.65)

                hpdi_mu_95 = hpdi(curr_mu_posterior, prob=.95)
                if not is_posterior_check:
                    hpdi_mu_85 = hpdi(curr_mu_posterior, prob=.85)
                    hpdi_mu_65 = hpdi(curr_mu_posterior, prob=.65)

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
                        x=temp_pred_df[self.intensity],
                        y=curr_mu_posterior_mean[:, r],
                        label=f"Mean Recruitment Curve",
                        color="r",
                        alpha=0.4,
                        ax=axes[i, j]
                    )

                    """ Plots: Predictions """
                    sns.lineplot(
                        x=temp_pred_df[self.intensity],
                        y=curr_obs_posterior_mean[:, r],
                        color="k",
                        label=f"Mean Prediction",
                        ax=axes[i, j + 1]
                    )
                    axes[i, j + 1].fill_between(
                        temp_pred_df[self.intensity],
                        hpdi_obs_95[0, :, r],
                        hpdi_obs_95[1, :, r],
                        color="C1",
                        label="95% HPDI"
                    )
                    axes[i, j + 1].fill_between(
                        temp_pred_df[self.intensity],
                        hpdi_obs_85[0, :, r],
                        hpdi_obs_85[1, :, r],
                        color="C2",
                        label="85% HPDI"
                    )
                    axes[i, j + 1].fill_between(
                        temp_pred_df[self.intensity],
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
                        x=temp_pred_df[self.intensity],
                        y=curr_mu_posterior_mean[:, r],
                        color="k",
                        label=f"Mean Recruitment Curve",
                        ax=axes[i, j + 2]
                    )
                    axes[i, j + 2].fill_between(
                        temp_pred_df[self.intensity],
                        hpdi_mu_95[0, :, r],
                        hpdi_mu_95[1, :, r],
                        color="C1",
                        label="95% HPDI"
                    )
                    if not is_posterior_check:
                        axes[i, j + 2].fill_between(
                            temp_pred_df[self.intensity],
                            hpdi_mu_85[0, :, r],
                            hpdi_mu_85[1, :, r],
                            color="C2",
                            label="85% HPDI"
                        )
                        axes[i, j + 2].fill_between(
                            temp_pred_df[self.intensity],
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

                        left, right = ax.get_xlim()
                        left = max(left, left - self.base)
                        right = min(right, right + self.base)
                        ax.set_xlim(left=left, right=right)

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
