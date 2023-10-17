import os
import itertools
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.diagnostics import hpdi

from hbmep.config import Config
from hbmep.dataset import Dataset
from hbmep.model.utils import Site as site
from hbmep.utils import (
    timing,
    floor,
    ceil
)
from hbmep.utils.constants import (
    BASELINE,
    DATASET_PLOT,
    RECRUITMENT_CURVES,
    PRIOR_PREDICTIVE,
    POSTERIOR_PREDICTIVE,
    MCMC_NC,
    DIAGNOSTICS_CSV,
    LOO_CSV,
    WAIC_CSV
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

        self.dataset_plot_path = os.path.join(self.build_dir, DATASET_PLOT)
        self.recruitment_curves_path = os.path.join(self.build_dir, RECRUITMENT_CURVES)
        self.prior_predictive_path = os.path.join(self.build_dir, PRIOR_PREDICTIVE)
        self.posterior_predictive_path = os.path.join(self.build_dir, POSTERIOR_PREDICTIVE)
        self.mcmc_path = os.path.join(self.build_dir, MCMC_NC)
        self.diagnostics_path = os.path.join(self.build_dir, DIAGNOSTICS_CSV)
        self.loo_path = os.path.join(self.build_dir, LOO_CSV)
        self.waic_path = os.path.join(self.build_dir, WAIC_CSV)

        self.base = config.BASE
        self.subplot_cell_width = 5
        self.subplot_cell_height = 3
        self.response_colors = plt.cm.rainbow(np.linspace(0, 1, self.n_response))
        self.recruitment_curve_props = {
            "label": "Recruitment Curve", "color": "black", "alpha": 0.4
        }
        self.threshold_posterior_props = {"color": "green", "alpha": 0.4}

        logger.info(f"Initialized model with {self.LINK} link")

    def _model(self, subject, features, intensity, response_obs=None):
        raise NotImplementedError

    def _collect_regressors(self, df: pd.DataFrame):
        subject = df[self.subject].to_numpy().reshape(-1,)
        n_subject = df[self.subject].nunique()

        features = df[self.features].to_numpy().T
        n_features = df[self.features].nunique().tolist()

        intensity = df[self.intensity].to_numpy().reshape(-1,)
        n_data = intensity.shape[0]

        return (subject, n_subject), (features, n_features), (intensity, n_data),

    def _collect_response(self, df: pd.DataFrame):
        response = jnp.array(df[self.response].to_numpy())
        return response,

    def _make_index_from_combination(self, combination: tuple[int]):
        ind = [slice(None)] + list(combination) + [slice(None)]
        ind = ind[::-1]
        return tuple(ind)

    def _collect_samples_at_combination(self, combination: tuple[int], samples: np.ndarray):
        return samples[*self._make_index_from_combination(combination=combination)]


    def _plot_staging(
        self,
        destination_path: str,
        df: pd.DataFrame,
        combination_columns: list[str],
        response_columns: list[str],
        posterior_samples: Optional[dict] = None,
        prediction_df: Optional[pd.DataFrame] = None,
        posterior_predictive: Optional[dict] = None,
        encoder_dict: Optional[dict[str, LabelEncoder]] = None
    ):
        if self.mep_matrix_path is not None:
            mep_matrix = np.load(self.mep_matrix_path)
            a, b = self.mep_window
            time = np.linspace(a, b, mep_matrix.shape[1])
            within_mep_size_window = (time > self.mep_size_window[0]) & (time < self.mep_size_window[1])

        if posterior_samples is not None:
            assert (prediction_df is not None) and (posterior_predictive is not None)
            mu_posterior_predictive = np.array(posterior_predictive[site.mu])

        """ Setup pdf layout """
        combinations = self._make_combinations(df=df, columns=combination_columns)
        n_combinations = len(combinations)

        n_response = len(response_columns)
        response_colors = plt.cm.rainbow(np.linspace(0, 1, n_response))
        n_columns_per_response = 1
        if self.mep_matrix_path is not None: n_columns_per_response += 1
        if posterior_samples is not None: n_columns_per_response += 2

        n_fig_rows = 10
        n_fig_columns = n_response * n_columns_per_response
        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1

        """ Iterate over pdf pages """
        logger.info("Rendering ...")
        pdf = PdfPages(destination_path)
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
                curr_combination = combinations[combination_counter]
                curr_combination_inverse = ""

                if encoder_dict is not None:
                    curr_combination_inverse = self._invert_combination(
                        combination=curr_combination,
                        columns=self.combination_columns,
                        encoder_dict=encoder_dict
                    )
                    curr_combination_inverse = ", ".join(map(str, curr_combination_inverse))
                    curr_combination_inverse += "\n"

                """ Filter dataframe based on current combination """
                df_ind = df[self.combination_columns].apply(tuple, axis=1).isin([curr_combination])
                curr_df = df[df_ind].reset_index(drop=True).copy()

                if posterior_samples is not None:
                    """ Filter prediction dataframe based on current combination """
                    prediction_df_ind = prediction_df[self.combination_columns].apply(tuple, axis=1).isin([curr_combination])
                    curr_prediction_df = prediction_df[prediction_df_ind].reset_index(drop=True).copy()

                    """ Predictions for current combination """
                    curr_mu_posterior_predictive = mu_posterior_predictive[:, prediction_df_ind, :]
                    curr_mu_posterior_predictive_map = curr_mu_posterior_predictive.mean(axis=0)

                    """ Threshold estimate for current combination """
                    curr_threshold_posterior = self._collect_samples_at_combination(
                        combination=curr_combination, samples=posterior_samples[site.a]
                    )
                    curr_threshold = curr_threshold_posterior.mean(axis=0)
                    curr_hpdi_interval = hpdi(curr_threshold_posterior, prob=0.95)

                """ Tickmarks """
                min_intensity, max_intensity_ = curr_df[self.intensity].agg([min, max])
                min_intensity = floor(min_intensity, base=self.base)
                max_intensity = ceil(max_intensity_, base=self.base)
                if max_intensity == max_intensity_:
                    max_intensity += self.base
                curr_x_ticks = np.arange(min_intensity, max_intensity, self.base)

                axes[i, 0].set_xlabel(self.intensity)
                axes[i, 0].set_xticks(ticks=curr_x_ticks)
                axes[i, 0].set_xlim(left=min_intensity - (self.base // 2), right=max_intensity + (self.base // 2))
                axes[i, 0].tick_params(axis="x", rotation=90)

                """ Iterate over responses """
                j = 0
                for r, response in enumerate(response_columns):
                    """ Labels """
                    prefix = f"{tuple(list(curr_combination)[::-1] + [r])}: {response} - MEP"

                    if not j:
                        prefix = curr_combination_inverse + prefix

                    """ MEP data """
                    if self.mep_matrix_path is not None:
                        postfix = " - MEP"
                        ax = axes[i, j]
                        mep_response_ind = [i for i, _response in enumerate(self.mep_response) if _response == response][0]
                        curr_mep_matrix = mep_matrix[df_ind, :, mep_response_ind]
                        max_amplitude = curr_mep_matrix[..., within_mep_size_window].max()

                        for k in range(curr_mep_matrix.shape[0]):
                            x = (curr_mep_matrix[k, :] / max_amplitude) * (self.base // 2)
                            x += curr_df[self.intensity].values[k]
                            ax.plot(x, time, color=response_colors[r], alpha=.4)

                        if self.mep_size_window is not None:
                            ax.axhline(
                                y=self.mep_size_window[0], color="r", linestyle="--", alpha=.4, label="MEP Size Window"
                            )
                            ax.axhline(
                                y=self.mep_size_window[1], color="r", linestyle="--", alpha=.4
                            )
                            ax.set_ylim(bottom=-0.001, top=self.mep_size_window[1] + (self.mep_size_window[0] - (-0.001)))

                        ax.set_ylabel("Time")
                        ax.set_title(prefix + postfix)
                        ax.legend(loc="upper right")
                        ax.sharex(axes[i, 0])

                        if j // n_columns_per_response:
                            ax.get_legend().remove()

                        j += 1

                    """ Plots: Scatter Plot """
                    postfix = " - MEP Size"
                    ax = axes[i, j]
                    sns.scatterplot(data=curr_df, x=self.intensity, y=response, color=response_colors[r], ax=ax)

                    ax.set_ylabel(response)
                    ax.set_title(prefix + postfix)
                    ax.sharex(axes[i, 0])

                    j += 1

                    if posterior_samples is not None:
                        """ Plots: Scatter Plot and Recruitment curve """
                        postfix = "Recruitment Curve Fit"
                        ax = axes[i, j]
                        sns.scatterplot(data=curr_df, x=self.intensity, y=response, color=response_colors[r], ax=ax)
                        sns.lineplot(
                            x=curr_prediction_df[self.intensity],
                            y=curr_mu_posterior_predictive_map[:, r],
                            ax=ax,
                            **self.recruitment_curve_props,
                        )
                        sns.kdeplot(
                            x=curr_threshold_posterior[:, r],
                            ax=ax,
                            **self.threshold_posterior_props
                        )
                        ax.set_title(postfix)
                        ax.legend(loc="upper right")
                        ax.sharex(axes[i, 0])
                        ax.sharey(axes[i, j - 1])

                        j += 1

                        """ Plots: Threshold estimate """
                        ax = axes[i, j]
                        postfix = "Threshold Estimate"
                        sns.kdeplot(
                            x=curr_threshold_posterior[:, r],
                            ax=ax,
                            **self.threshold_posterior_props
                        )
                        ax.axvline(
                            curr_threshold[r],
                            linestyle="--",
                            color=response_colors[r],
                            label="Threshold"
                        )
                        ax.axvline(
                            curr_hpdi_interval[:, r][0],
                            linestyle="--",
                            color="black",
                            alpha=.4,
                            label="95% HPDI"
                        )
                        ax.axvline(
                            curr_hpdi_interval[:, r][1],
                            linestyle="--",
                            color="black",
                            alpha=.4
                        )

                        ax.set_xlabel(self.intensity)
                        ax.set_title(postfix)
                        ax.legend(loc="upper right")

                        if j // n_columns_per_response:
                            ax.get_legend().remove()
                            axes[i, j - 1].get_legend().remove()

                        j += 1

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {destination_path}")
        return


    def _plot(
        self,
        df: pd.DataFrame,
        destination_path: str,
        posterior_samples: Optional[dict] = None,
        prediction_df: Optional[pd.DataFrame] = None,
        posterior_predictive: Optional[dict] = None,
        encoder_dict: Optional[dict[str, LabelEncoder]] = None
    ):
        if self.mep_matrix_path is not None:
            mep_matrix = np.load(self.mep_matrix_path)
            a, b = self.mep_window
            time = np.linspace(a, b, mep_matrix.shape[1])
            within_mep_size_window = (time > self.mep_size_window[0]) & (time < self.mep_size_window[1])

        if posterior_samples is not None:
            assert (prediction_df is not None) and (posterior_predictive is not None)
            mu_posterior_predictive = np.array(posterior_predictive[site.mu])

        """ Setup pdf layout """
        combinations = self._make_combinations(df=df, columns=self.combination_columns)
        n_combinations = len(combinations)

        n_columns_per_response = 1
        if self.mep_matrix_path is not None: n_columns_per_response += 1
        if posterior_samples is not None: n_columns_per_response += 2

        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * self.n_response

        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1
        logger.info("Rendering ...")

        """ Iterate over pdf pages """
        pdf = PdfPages(destination_path)
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
                curr_combination = combinations[combination_counter]
                curr_combination_inverse = ""

                if encoder_dict is not None:
                    curr_combination_inverse = self._invert_combination(
                        combination=curr_combination,
                        columns=self.combination_columns,
                        encoder_dict=encoder_dict
                    )
                    curr_combination_inverse = ", ".join(map(str, curr_combination_inverse))
                    curr_combination_inverse += "\n"

                """ Filter dataframe based on current combination """
                df_ind = df[self.combination_columns].apply(tuple, axis=1).isin([curr_combination])
                curr_df = df[df_ind].reset_index(drop=True).copy()

                if posterior_samples is not None:
                    """ Filter prediction dataframe based on current combination """
                    prediction_df_ind = prediction_df[self.combination_columns].apply(tuple, axis=1).isin([curr_combination])
                    curr_prediction_df = prediction_df[prediction_df_ind].reset_index(drop=True).copy()

                    """ Predictions for current combination """
                    curr_mu_posterior_predictive = mu_posterior_predictive[:, prediction_df_ind, :]
                    curr_mu_posterior_predictive_map = curr_mu_posterior_predictive.mean(axis=0)

                    """ Threshold estimate for current combination """
                    curr_threshold_posterior = self._collect_samples_at_combination(
                        combination=curr_combination, samples=posterior_samples[site.a]
                    )
                    curr_threshold = curr_threshold_posterior.mean(axis=0)
                    curr_hpdi_interval = hpdi(curr_threshold_posterior, prob=0.95)

                """ Tickmarks """
                min_intensity, max_intensity_ = curr_df[self.intensity].agg([min, max])
                min_intensity = floor(min_intensity, base=self.base)
                max_intensity = ceil(max_intensity_, base=self.base)
                if max_intensity == max_intensity_:
                    max_intensity += self.base
                curr_x_ticks = np.arange(min_intensity, max_intensity, self.base)

                axes[i, 0].set_xlabel(self.intensity)
                axes[i, 0].set_xticks(ticks=curr_x_ticks)
                axes[i, 0].set_xlim(left=min_intensity - (self.base // 2), right=max_intensity + (self.base // 2))
                axes[i, 0].tick_params(axis="x", rotation=90)

                """ Iterate over responses """
                j = 0
                for r, response in enumerate(self.response):
                    """ Labels """
                    prefix = f"{tuple(list(curr_combination)[::-1] + [r])}: {response} - MEP"

                    if not j:
                        prefix = curr_combination_inverse + prefix

                    """ MEP data """
                    if self.mep_matrix_path is not None:
                        postfix = " - MEP"
                        ax = axes[i, j]
                        mep_response_ind = [i for i, _response in enumerate(self.mep_response) if _response == response][0]
                        curr_mep_matrix = mep_matrix[df_ind, :, mep_response_ind]
                        max_amplitude = curr_mep_matrix[..., within_mep_size_window].max()

                        for k in range(curr_mep_matrix.shape[0]):
                            x = (curr_mep_matrix[k, :] / max_amplitude) * (self.base // 2)
                            x += curr_df[self.intensity].values[k]
                            ax.plot(x, time, color=self.response_colors[r], alpha=.4)

                        if self.mep_size_window is not None:
                            ax.axhline(
                                y=self.mep_size_window[0], color="r", linestyle="--", alpha=.4, label="MEP Size Window"
                            )
                            ax.axhline(
                                y=self.mep_size_window[1], color="r", linestyle="--", alpha=.4
                            )
                            ax.set_ylim(bottom=-0.001, top=self.mep_size_window[1] + (self.mep_size_window[0] - (-0.001)))

                        ax.set_ylabel("Time")
                        ax.set_title(prefix + postfix)
                        ax.legend(loc="upper right")
                        ax.sharex(axes[i, 0])

                        if j // n_columns_per_response:
                            ax.get_legend().remove()

                        j += 1

                    """ Plots: Scatter Plot """
                    postfix = " - MEP Size"
                    ax = axes[i, j]
                    sns.scatterplot(data=curr_df, x=self.intensity, y=response, color=self.response_colors[r], ax=ax)

                    ax.set_ylabel(response)
                    ax.set_title(prefix + postfix)
                    ax.sharex(axes[i, 0])

                    j += 1

                    if posterior_samples is not None:
                        """ Plots: Scatter Plot and Recruitment curve """
                        postfix = "Recruitment Curve Fit"
                        ax = axes[i, j]
                        sns.scatterplot(data=curr_df, x=self.intensity, y=response, color=self.response_colors[r], ax=ax)
                        sns.lineplot(
                            x=curr_prediction_df[self.intensity],
                            y=curr_mu_posterior_predictive_map[:, r],
                            ax=ax,
                            **self.recruitment_curve_props,
                        )
                        sns.kdeplot(
                            x=curr_threshold_posterior[:, r],
                            ax=ax,
                            **self.threshold_posterior_props
                        )
                        ax.set_title(postfix)
                        ax.legend(loc="upper right")
                        ax.sharex(axes[i, 0])
                        ax.sharey(axes[i, j - 1])

                        j += 1

                        """ Plots: Threshold estimate """
                        ax = axes[i, j]
                        postfix = "Threshold Estimate"
                        sns.kdeplot(
                            x=curr_threshold_posterior[:, r],
                            ax=ax,
                            **self.threshold_posterior_props
                        )
                        ax.axvline(
                            curr_threshold[r],
                            linestyle="--",
                            color=self.response_colors[r],
                            label="Threshold"
                        )
                        ax.axvline(
                            curr_hpdi_interval[:, r][0],
                            linestyle="--",
                            color="black",
                            alpha=.4,
                            label="95% HPDI"
                        )
                        ax.axvline(
                            curr_hpdi_interval[:, r][1],
                            linestyle="--",
                            color="black",
                            alpha=.4
                        )

                        ax.set_xlabel(self.intensity)
                        ax.set_title(postfix)
                        ax.legend(loc="upper right")

                        if j // n_columns_per_response:
                            ax.get_legend().remove()
                            axes[i, j - 1].get_legend().remove()

                        j += 1

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {destination_path}")
        return

    @timing
    def _render_predictive_check(
        self,
        df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        predictive: dict,
        is_posterior_check: bool = True,
        encoder_dict: Optional[dict[str, LabelEncoder]] = None
    ):
        """ Posterior / Prior Predictive Check """
        destination_path = self.posterior_predictive_path

        if not is_posterior_check:
            destination_path = self.prior_predictive_path

        check_type = "Posterior" if is_posterior_check else "Prior"

        """ Predictions """
        obs_posterior_predictive = np.array(predictive[site.obs])
        mu_posterior_predictive = np.array(predictive[site.mu])

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
        pdf = PdfPages(destination_path)
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
                sharex="row",
                constrained_layout=True,
                squeeze=False
            )

            """ Iterate over combinations """
            for i in range(n_rows_current_page):
                curr_combination = combinations[combination_counter]
                curr_combination_inverse = ""

                if encoder_dict is not None:
                    curr_combination_inverse = self._invert_combination(
                        combination=curr_combination,
                        columns=self.combination_columns,
                        encoder_dict=encoder_dict
                    )
                    curr_combination_inverse = ", ".join(map(str, curr_combination_inverse))
                    curr_combination_inverse += "\n"

                """ Filter dataframe based on current combination """
                df_ind = df[self.combination_columns].apply(tuple, axis=1).isin([curr_combination])
                curr_df = df[df_ind].reset_index(drop=True).copy()

                """ Filter prediction dataframe based on current combination """
                prediction_df_ind = prediction_df[self.combination_columns].apply(tuple, axis=1).isin([curr_combination])
                curr_prediction_df = prediction_df[prediction_df_ind].reset_index(drop=True).copy()

                """ Predictions for current combination """
                curr_obs_posterior_predictive = obs_posterior_predictive[:, prediction_df_ind, :]
                curr_obs_posterior_predictive_map = curr_obs_posterior_predictive.mean(axis=0)

                curr_mu_posterior_predictive = mu_posterior_predictive[:, prediction_df_ind, :]
                curr_mu_posterior_predictive_map = curr_mu_posterior_predictive.mean(axis=0)

                """ HPDI intervals """
                hpdi_curr_obs_95 = hpdi(curr_obs_posterior_predictive, prob=.95)
                hpdi_curr_obs_85 = hpdi(curr_obs_posterior_predictive, prob=.85)
                hpdi_curr_obs_65 = hpdi(curr_obs_posterior_predictive, prob=.65)

                hpdi_curr_mu_95 = hpdi(curr_mu_posterior_predictive, prob=.95)

                if not is_posterior_check:
                    hpdi_curr_mu_85 = hpdi(curr_mu_posterior_predictive, prob=.85)
                    hpdi_curr_mu_65 = hpdi(curr_mu_posterior_predictive, prob=.65)

                """ Tickmarks """
                min_intensity, max_intensity_ = curr_df[self.intensity].agg([min, max])
                min_intensity = floor(min_intensity, base=self.base)
                max_intensity = ceil(max_intensity_, base=self.base)
                if max_intensity == max_intensity_:
                    max_intensity += self.base
                curr_x_ticks = np.arange(min_intensity, max_intensity, self.base)

                axes[i, 0].set_xticks(ticks=curr_x_ticks)
                axes[i, 0].set_xlim(left=min_intensity - (self.base // 2), right=max_intensity + (self.base // 2))
                axes[i, 0].tick_params(axis="x", rotation=90)

                """ Iterate over responses """
                j = 0
                for r, response in enumerate(self.response):
                    """ Plots: Scatter Plot and Recruitment curve """
                    ax = axes[i, j]
                    sns.scatterplot(data=curr_df, x=self.intensity, y=response, color=self.response_colors[r], ax=ax)
                    sns.lineplot(
                        x=curr_prediction_df[self.intensity],
                        y=curr_mu_posterior_predictive_map[:, r],
                        ax=ax,
                        **self.recruitment_curve_props,
                    )

                    if j == 0:
                        ax.set_title(f"{tuple(list(curr_combination)[::-1] + [r])}: {curr_combination_inverse}{response}")
                    else:
                        ax.set_title(f"{tuple(list(curr_combination)[::-1] + [r])}: {response}")

                    ax.sharex(axes[i, 0])
                    ax.legend(loc="upper right")
                    if j // n_columns_per_response:
                        ax.get_legend().remove()
                    j += 1

                    """ Plots: Observational predictive """
                    ax = axes[i, j]
                    sns.lineplot(
                        x=curr_prediction_df[self.intensity],
                        y=curr_obs_posterior_predictive_map[:, r],
                        color="black",
                        label=f"Mean Prediction",
                        ax=ax
                    )
                    ax.fill_between(
                        curr_prediction_df[self.intensity],
                        hpdi_curr_obs_95[0, :, r],
                        hpdi_curr_obs_95[1, :, r],
                        color="C1",
                        label="95% HPDI"
                    )
                    ax.fill_between(
                        curr_prediction_df[self.intensity],
                        hpdi_curr_obs_85[0, :, r],
                        hpdi_curr_obs_85[1, :, r],
                        color="C2",
                        label="85% HPDI"
                    )
                    ax.fill_between(
                        curr_prediction_df[self.intensity],
                        hpdi_curr_obs_65[0, :, r],
                        hpdi_curr_obs_65[1, :, r],
                        color="C3",
                        label="65% HPDI"
                    )
                    sns.scatterplot(
                        data=curr_df,
                        x=self.intensity,
                        y=response,
                        color="yellow",
                        edgecolor="black",
                        ax=ax
                    )

                    ax.sharex(axes[i, 0])
                    ax.sharey(axes[i, j - 1])
                    ax.set_title(f"{tuple(list(curr_combination)[::-1] + [r])}: {response} - Prediction")
                    ax.legend(loc="upper right")

                    if j // n_columns_per_response:
                        ax.get_legend().remove()
                    j += 1

                    """ Plots: Mean predictive """
                    ax = axes[i, j]
                    sns.lineplot(
                        x=curr_prediction_df[self.intensity],
                        y=curr_mu_posterior_predictive_map[:, r],
                        color="black",
                        label=f"Recruitment Curve",
                        ax=ax
                    )
                    ax.fill_between(
                        curr_prediction_df[self.intensity],
                        hpdi_curr_mu_95[0, :, r],
                        hpdi_curr_mu_95[1, :, r],
                        color="C1",
                        label="95% HPDI"
                    )
                    if not is_posterior_check:
                        ax.fill_between(
                            curr_prediction_df[self.intensity],
                            hpdi_curr_mu_85[0, :, r],
                            hpdi_curr_mu_85[1, :, r],
                            color="C2",
                            label="85% HPDI"
                        )
                        ax.fill_between(
                            curr_prediction_df[self.intensity],
                            hpdi_curr_mu_65[0, :, r],
                            hpdi_curr_mu_65[1, :, r],
                            color="C3",
                            label="65% HPDI"
                        )
                    sns.scatterplot(
                        data=curr_df,
                        x=self.intensity,
                        y=response,
                        color="yellow",
                        edgecolor="black",
                        ax=ax
                    )

                    ax.sharex(axes[i, 0])
                    ax.sharey(axes[i, j - 2])
                    ax.set_title(f"{tuple(list(curr_combination)[::-1] + [r])}: {response} - Fit")
                    ax.legend(loc="upper right")
                    if j // n_columns_per_response:
                        ax.get_legend().remove()
                    j += 1

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {destination_path}")
        return

    @timing
    def run_trace(self, df: pd.DataFrame):
        with numpyro.handlers.seed(rng_seed=self.random_state):
            trace = numpyro.handlers.trace(self._model).get_trace(
                *self._collect_regressors(df=df), *self._collect_response(df=df)
            )
        return trace

    @timing
    def run_inference(self, df: pd.DataFrame, sampler: MCMCKernel = None) -> tuple[MCMC, dict]:
        """ Set up sampler """
        if sampler is None: sampler = NUTS(self._model)
        mcmc = MCMC(sampler, **self.mcmc_params)

        """ Run MCMC inference """
        logger.info(f"Running inference with {self.LINK} ...")
        rng_key = jax.random.PRNGKey(self.random_state)
        mcmc.run(rng_key, *self._collect_regressors(df=df), *self._collect_response(df=df))

        posterior_samples = mcmc.get_samples()
        return mcmc, posterior_samples

    @timing
    def make_prediction_dataset(self, df: pd.DataFrame, num_points: int = 100):
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
            .apply(lambda x: np.linspace(x[0], x[1], num_points))
        pred_df = pred_df.explode(column=self.intensity)[self.regressors].copy()
        pred_df[self.intensity] = pred_df[self.intensity].astype(float)

        pred_df.reset_index(drop=True, inplace=True)
        return pred_df

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
    def plot(
        self,
        df: pd.DataFrame,
        encoder_dict: Optional[dict[str, LabelEncoder]] = None,
    ):
        return self._plot(df=df, encoder_dict=encoder_dict, destination_path=self.dataset_plot_path)

    @timing
    def render_recruitment_curves(
        self,
        df: pd.DataFrame,
        posterior_samples: dict = None,
        prediction_df: pd.DataFrame = None,
        posterior_predictive: dict = None,
        encoder_dict: Optional[dict[str, LabelEncoder]] = None
    ):
        return self._plot(
            df=df,
            destination_path=self.recruitment_curves_path,
            posterior_samples=posterior_samples,
            prediction_df=prediction_df,
            posterior_predictive=posterior_predictive,
            encoder_dict=encoder_dict
        )

    @timing
    def render_predictive_check(
        self,
        df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        prior_predictive: Optional[dict] = None,
        posterior_predictive: Optional[dict] = None,
        encoder_dict: Optional[dict[str, LabelEncoder]] = None
    ):
        assert (prior_predictive is not None) or (posterior_predictive is not None)

        if posterior_predictive is not None:
            predictive = posterior_predictive
            is_posterior_check = True
        else:
            predictive = prior_predictive
            is_posterior_check = False

        return self._render_predictive_check(
            df=df,
            prediction_df=prediction_df,
            predictive=predictive,
            is_posterior_check=is_posterior_check,
            encoder_dict=encoder_dict
        )

    @timing
    def simulate(
        self,
        n_subject=3,
        n_feature0=2,
        n_draws=5,
        n_repeats=10
    ):
        n_features = [n_subject, n_feature0]
        combinations = itertools.product(*[range(i) for i in n_features])
        combinations = list(combinations)
        combinations = sorted(combinations)

        logger.info("Simulating data ...")
        df = pd.DataFrame(combinations, columns=self.combination_columns)
        x_space = np.arange(0, 360, 4)

        df[self.intensity] = df.apply(lambda _: x_space, axis=1)
        df = df.explode(column=self.intensity).reset_index(drop=True).copy()
        df[self.intensity] = df[self.intensity].astype(float)

        pred_df = pd.concat([df] * n_repeats).reset_index(drop=True).copy()

        posterior_samples = self.predict(df=pred_df, num_samples=n_draws)
        posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}
        for u in {site.mu, site.obs}:
            posterior_samples[u] = posterior_samples[u].reshape(n_draws, n_repeats, -1, self.n_response)

        return df, posterior_samples

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
