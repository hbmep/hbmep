import os
import logging

import numpy as np
import pandas as pd
from numpyro.diagnostics import hpdi
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from hbmep.config import Config
from hbmep.dataset import Dataset
from hbmep.model.utils import Site as site
from hbmep.utils import timing, floor, ceil
from hbmep.utils.constants import (
    DATASET_PLOT,
    RECRUITMENT_CURVES,
    PRIOR_PREDICTIVE,
    POSTERIOR_PREDICTIVE,
)

logger = logging.getLogger(__name__)


class Plotter(Dataset):
    def __init__(self, config: Config):
        super(Plotter, self).__init__(config=config)
        self.response_colors = Plotter._get_colors(n=self.n_response)
        self.base = config.BASE
        self.subplot_cell_width = 5
        self.subplot_cell_height = 3
        self.recruitment_curve_props = {
            "label": "Recruitment Curve",
            "color": "black",
            "alpha": 0.4
        }
        self.threshold_posterior_props = {
            "color": "green",
            "alpha": 0.4
        }

    @staticmethod
    def _get_index_from_combination(combination: tuple[int]):
        ind = [slice(None)] + list(combination) + [slice(None)]
        return tuple(ind)

    @staticmethod
    def _get_samples_at_combination(
        samples: np.ndarray,
        combination: tuple[int],
    ):
        return samples[
            Plotter._get_index_from_combination(combination=combination)
        ]

    @staticmethod
    def _get_colors(n: int):
        return plt.cm.rainbow(np.linspace(0, 1, n))

    @staticmethod
    def mep_plot(
        ax: plt.Axes,
        mep_matrix: np.ndarray,
        intensity: np.ndarray,
        time: np.ndarray,
        **kwargs
    ):
        # intensity: (n_intensity,)
        # mep_matrix: (n_intensity, n_time)
        # mep_window: (start, end)
        for i in range(mep_matrix.shape[0]):
            x = mep_matrix[i, :]
            x = x + intensity[i]
            ax.plot(x, time, **kwargs)
        return ax

    def mep_renderer(
        self,
        df: pd.DataFrame,
        destination_path: str,
        posterior_samples: dict | None = None,
        prediction_df: pd.DataFrame | None = None,
        posterior_predictive: dict | None = None,
        encoder_dict: dict[str, LabelEncoder] | None = None,
        mep_matrix: np.ndarray | None = None,
        **kwargs
    ):
        """
        **kwargs:
            combination_columns: list[str]
            orderby: lambda function
            intensity: str
            response: list[str]
            response_colors: list[str] | np.ndarray
            base: int
            subplot_cell_width: int
            subplot_cell_height: int
            recruitment_curve_props: dict
            threshold_posterior_props: dict
        """
        combination_columns = kwargs.get("combination_columns", self.features)
        orderby = kwargs.get("orderby")
        intensity = kwargs.get("intensity", self.intensity)
        response = kwargs.get("response", self.response)
        response_colors = kwargs.get("response_colors", self.response_colors)
        base = kwargs.get("base", self.base)
        subplot_cell_width = kwargs.get("subplot_cell_width", self.subplot_cell_width)
        subplot_cell_height = kwargs.get("subplot_cell_height", self.subplot_cell_height)
        recruitment_curve_props = kwargs.get("recruitment_curve_props", self.recruitment_curve_props)
        threshold_posterior_props = kwargs.get("threshold_posterior_props", self.threshold_posterior_props)

        if mep_matrix is not None:
            assert mep_matrix.shape[0] == df.shape[0]
            a, b = self.mep_window
            time = np.linspace(a, b, mep_matrix.shape[1])
            is_within_mep_size_window = (time > self.mep_size_window[0]) & (time < self.mep_size_window[1])

        if posterior_samples is not None:
            assert (prediction_df is not None) and (posterior_predictive is not None)
            mu_posterior_predictive = posterior_predictive[site.mu]

        # Setup pdf layout
        combinations = self._get_combinations(df=df, columns=combination_columns, orderby=orderby)
        n_combinations = len(combinations)
        n_response = len(response)

        n_columns_per_response = 1
        if mep_matrix is not None: n_columns_per_response += 1
        if posterior_samples is not None: n_columns_per_response += 2

        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * n_response
        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1

        # Iterate over pdf pages
        pdf = PdfPages(destination_path)
        combination_counter = 0

        for page in range(n_pdf_pages):
            n_rows_current_page = min(
                n_fig_rows,
                n_combinations - page * n_fig_rows
            )
            fig, axes = plt.subplots(
                nrows=n_rows_current_page,
                ncols=n_fig_columns,
                figsize=(
                    n_fig_columns * subplot_cell_width,
                    n_rows_current_page * subplot_cell_height
                ),
                constrained_layout=True,
                squeeze=False
            )

            # Iterate over combinations
            for i in range(n_rows_current_page):
                curr_combination = combinations[combination_counter]
                curr_combination_inverse = ""

                if encoder_dict is not None:
                    curr_combination_inverse = self._get_combination_inverse(
                        combination=curr_combination,
                        columns=combination_columns,
                        encoder_dict=encoder_dict
                    )
                    curr_combination_inverse = ", ".join(map(str, curr_combination_inverse))
                    curr_combination_inverse += "\n"

                # Filter dataframe based on current combination
                df_ind = df[combination_columns].apply(tuple, axis=1).isin([curr_combination])
                curr_df = df[df_ind].reset_index(drop=True).copy()

                if posterior_samples is not None:
                    # Filter prediction dataframe based on current combination
                    prediction_df_ind = prediction_df[combination_columns].apply(tuple, axis=1).isin([curr_combination])
                    curr_prediction_df = prediction_df[prediction_df_ind].reset_index(drop=True).copy()

                    # Predictions for current combination
                    curr_mu_posterior_predictive = mu_posterior_predictive[:, prediction_df_ind, :]
                    curr_mu_posterior_predictive_map = curr_mu_posterior_predictive.mean(axis=0)

                    # Threshold estimate for current combination
                    curr_threshold_posterior = self._get_samples_at_combination(
                        combination=curr_combination, samples=posterior_samples[site.a]
                    )
                    curr_threshold_map = curr_threshold_posterior.mean(axis=0)
                    curr_threshold_hpdi = hpdi(curr_threshold_posterior, prob=0.95)

                # Tickmarks
                min_intensity, max_intensity_ = curr_df[intensity].agg(["min", "max"])
                min_intensity = floor(min_intensity, base=base)
                max_intensity = ceil(max_intensity_, base=base)
                if max_intensity == max_intensity_:
                    max_intensity += base
                curr_x_ticks = np.arange(min_intensity, max_intensity, base)

                axes[i, 0].set_xlabel(intensity)
                axes[i, 0].set_xticks(ticks=curr_x_ticks)
                axes[i, 0].set_xlim(left=min_intensity - (base // 2), right=max_intensity + (base // 2))

                # Iterate over responses
                j = 0
                for r, response_muscle in enumerate(response):
                    # Labels
                    prefix = f"{tuple(list(curr_combination) + [r])}: {response_muscle} - MEP"
                    if not j: prefix = curr_combination_inverse + prefix

                    # MEP data
                    if mep_matrix is not None:
                        postfix = " - MEP"
                        ax = axes[i, j]
                        mep_response_ind = [
                            i
                            for i, _response_muscle in enumerate(self.mep_response)
                            if _response_muscle == response_muscle
                        ]
                        mep_response_ind = mep_response_ind[0]
                        curr_mep_matrix = mep_matrix[df_ind, :, mep_response_ind]
                        # max_amplitude = curr_mep_matrix[..., is_within_mep_size_window].max()
                        max_amplitude = np.nanmax(curr_mep_matrix[..., is_within_mep_size_window])
                        curr_mep_matrix = (curr_mep_matrix / max_amplitude) * (base // 2)

                        ax = Plotter.mep_plot(
                            ax,
                            curr_mep_matrix,
                            curr_df[intensity],
                            time,
                            color=response_colors[r],
                            alpha=.4
                        )
                        ax.axhline(
                            y=self.mep_size_window[0], color="r", linestyle="--", alpha=.4, label="MEP Size Window"
                        )
                        ax.axhline(
                            y=self.mep_size_window[1], color="r", linestyle="--", alpha=.4
                        )
                        ax.set_ylim(bottom=-0.001, top=self.mep_size_window[1] + (self.mep_size_window[0] - (-0.001)))
                        ax.set_ylabel("Time")
                        ax.set_title(prefix + postfix)
                        ax.sharex(axes[i, 0])
                        ax.tick_params(axis="x", rotation=90)
                        if j > 0 and ax.get_legend() is not None: ax.get_legend().remove()
                        j += 1

                    # MEP Size scatter plot
                    postfix = " - MEP Size"
                    ax = axes[i, j]
                    sns.scatterplot(data=curr_df, x=intensity, y=response_muscle, color=response_colors[r], ax=ax)
                    ax.set_ylabel(response_muscle)
                    ax.set_title(prefix + postfix)
                    ax.sharex(axes[i, 0])
                    ax.tick_params(axis="x", rotation=90)
                    j += 1

                    if posterior_samples is not None:
                        # MEP Size scatter plot and recruitment curve
                        postfix = "Recruitment Curve Fit"
                        ax = axes[i, j]
                        sns.scatterplot(data=curr_df, x=intensity, y=response_muscle, color=response_colors[r], ax=ax)
                        sns.lineplot(
                            x=curr_prediction_df[intensity],
                            y=curr_mu_posterior_predictive_map[:, r],
                            ax=ax,
                            **recruitment_curve_props,
                        )
                        sns.kdeplot(
                            x=curr_threshold_posterior[:, r],
                            ax=ax,
                            **threshold_posterior_props
                        )
                        ax.set_title(postfix)
                        ax.sharex(axes[i, 0])
                        ax.sharey(axes[i, j - 1])
                        ax.tick_params(axis="x", rotation=90)
                        j += 1

                        # Threshold KDE
                        ax = axes[i, j]
                        postfix = "Threshold Estimate"
                        sns.kdeplot(
                            x=curr_threshold_posterior[:, r],
                            ax=ax,
                            **threshold_posterior_props
                        )
                        ax.axvline(
                            curr_threshold_map[r],
                            linestyle="--",
                            color=response_colors[r],
                            label="Threshold"
                        )
                        ax.axvline(
                            curr_threshold_hpdi[:, r][0],
                            linestyle="--",
                            color="black",
                            alpha=.4,
                            label="95% HPDI"
                        )
                        ax.axvline(
                            curr_threshold_hpdi[:, r][1],
                            linestyle="--",
                            color="black",
                            alpha=.4
                        )
                        ax.set_xlabel(intensity)
                        ax.set_title(postfix)
                        if j > 0 and ax.get_legend(): ax.get_legend().remove()
                        j += 1

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {destination_path}")
        return

    @timing
    def plot(
        self,
        df: pd.DataFrame,
        encoder_dict: dict[str, LabelEncoder] | None = None,
        destination_path: str | None = None,
        **kwargs
    ):
        if destination_path is None: destination_path = os.path.join(
            self.build_dir, DATASET_PLOT
        )
        logger.info("Rendering dataset ...")
        return self.mep_renderer(
            df=df,
            destination_path=destination_path,
            encoder_dict=encoder_dict,
            **kwargs
        )

    @timing
    def render_recruitment_curves(
        self,
        df: pd.DataFrame,
        posterior_samples: dict,
        prediction_df: pd.DataFrame,
        posterior_predictive: dict,
        encoder_dict: dict[str, LabelEncoder] | None = None,
        destination_path: str | None = None,
        **kwargs
    ):
        if destination_path is None: destination_path = os.path.join(
            self.build_dir, RECRUITMENT_CURVES
        )
        logger.info("Rendering recruitment curves ...")
        return self.mep_renderer(
            df=df,
            destination_path=destination_path,
            posterior_samples=posterior_samples,
            prediction_df=prediction_df,
            posterior_predictive=posterior_predictive,
            encoder_dict=encoder_dict,
            **kwargs
        )

    @timing
    def predictive_checks_renderer(
        self,
        df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        predictive: dict,
        destination_path: str,
        encoder_dict: dict[str, LabelEncoder] | None = None,
        **kwargs
    ):
        """
        **kwargs:
            combination_columns: list[str]
            orderby: lambda function
            base: int
            subplot_cell_width: float
            subplot_cell_height: float
        """
        combination_columns = kwargs.get("combination_columns", self.features)
        orderby = kwargs.get("orderby")
        base = kwargs.get("base", self.base)
        subplot_cell_width = kwargs.get("subplot_cell_width", self.subplot_cell_width)
        subplot_cell_height = kwargs.get("subplot_cell_height", self.subplot_cell_height)

        # Predictive samples
        obs, mu = predictive[site.obs], predictive[site.mu]

        # Setup pdf layout
        combinations = self._get_combinations(df=df, columns=combination_columns, orderby=orderby)
        n_combinations = len(combinations)

        n_columns_per_response = 3
        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * self.n_response

        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1

        # Iterate over pdf pages
        pdf = PdfPages(destination_path)
        combination_counter = 0

        for page in range(n_pdf_pages):
            n_rows_current_page = min(
                n_fig_rows,
                n_combinations - page * n_fig_rows
            )
            fig, axes = plt.subplots(
                nrows=n_rows_current_page,
                ncols=n_fig_columns,
                figsize=(
                    n_fig_columns * subplot_cell_width,
                    n_rows_current_page * subplot_cell_height
                ),
                sharex="row",
                constrained_layout=True,
                squeeze=False
            )

            # Iterate over combinations
            for i in range(n_rows_current_page):
                curr_combination = combinations[combination_counter]
                curr_combination_inverse = ""

                if encoder_dict is not None:
                    curr_combination_inverse = self._get_combination_inverse(
                        combination=curr_combination,
                        columns=combination_columns,
                        encoder_dict=encoder_dict
                    )
                    curr_combination_inverse = ", ".join(map(str, curr_combination_inverse))
                    curr_combination_inverse += "\n"

                # Filter dataframe based on current combination """
                df_ind = df[combination_columns].apply(tuple, axis=1).isin([curr_combination])
                curr_df = df[df_ind].reset_index(drop=True).copy()

                # Filter prediction dataframe based on current combination
                prediction_df_ind = prediction_df[combination_columns].apply(tuple, axis=1).isin([curr_combination])
                curr_prediction_df = prediction_df[prediction_df_ind].reset_index(drop=True).copy()

                # Predictive for current combination
                curr_obs = obs[:, prediction_df_ind, :]
                curr_obs_map = curr_obs.mean(axis=0)
                curr_mu = mu[:, prediction_df_ind, :]
                curr_mu_map = curr_mu.mean(axis=0)

                # HPDI intervals
                curr_obs_hpdi_95 = hpdi(curr_obs, prob=.95)
                curr_obs_hpdi_85 = hpdi(curr_obs, prob=.85)
                curr_obs_hpdi_65 = hpdi(curr_obs, prob=.65)
                curr_mu_hpdi_95 = hpdi(curr_mu, prob=.95)

                # Tickmarks
                min_intensity, max_intensity_ = curr_df[self.intensity].agg(["min", "max"])
                min_intensity = floor(min_intensity, base=base)
                max_intensity = ceil(max_intensity_, base=base)
                if max_intensity == max_intensity_:
                    max_intensity += base
                curr_x_ticks = np.arange(min_intensity, max_intensity, base)

                axes[i, 0].set_xticks(ticks=curr_x_ticks)
                axes[i, 0].set_xlim(left=min_intensity - (base // 2), right=max_intensity + (base // 2))

                # Iterate over responses
                j = 0
                for r, response in enumerate(self.response):
                    prefix = f"{tuple(list(curr_combination) + [r])}: {response} - MEP"
                    if not j: prefix = curr_combination_inverse + prefix

                    # MEP Size scatter plot and recruitment curve
                    postfix = ""
                    ax = axes[i, j]
                    sns.scatterplot(data=curr_df, x=self.intensity, y=response, color=self.response_colors[r], ax=ax)
                    sns.lineplot(
                        x=curr_prediction_df[self.intensity],
                        y=curr_mu_map[:, r],
                        ax=ax,
                        **self.recruitment_curve_props,
                    )
                    ax.set_title(prefix + postfix)
                    ax.sharex(axes[i, 0])
                    ax.tick_params(axis="x", rotation=90)
                    j += 1

                    # Observational predictive
                    postfix = "Prediction"
                    ax = axes[i, j]
                    sns.lineplot(
                        x=curr_prediction_df[self.intensity],
                        y=curr_obs_map[:, r],
                        color="black",
                        ax=ax
                    )
                    ax.fill_between(
                        curr_prediction_df[self.intensity],
                        curr_obs_hpdi_95[0, :, r],
                        curr_obs_hpdi_95[1, :, r],
                        color="C1"
                    )
                    ax.fill_between(
                        curr_prediction_df[self.intensity],
                        curr_obs_hpdi_85[0, :, r],
                        curr_obs_hpdi_85[1, :, r],
                        color="C2"
                    )
                    ax.fill_between(
                        curr_prediction_df[self.intensity],
                        curr_obs_hpdi_65[0, :, r],
                        curr_obs_hpdi_65[1, :, r],
                        color="C3"
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
                    ax.set_title(postfix)
                    ax.tick_params(axis="x", rotation=90)
                    j += 1

                    # Recruitment curve predictive
                    postfix = "Fit"
                    ax = axes[i, j]
                    sns.lineplot(
                        x=curr_prediction_df[self.intensity],
                        y=curr_mu_map[:, r],
                        color="black",
                        ax=ax
                    )
                    ax.fill_between(
                        curr_prediction_df[self.intensity],
                        curr_mu_hpdi_95[0, :, r],
                        curr_mu_hpdi_95[1, :, r],
                        color="C1"
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
                    ax.set_title(postfix)
                    ax.tick_params(axis="x", rotation=90)
                    j += 1

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {destination_path}")
        return

    @timing
    def render_predictive_check(
        self,
        df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        prior_predictive: dict | None = None,
        posterior_predictive: dict | None = None,
        encoder_dict: dict[str, LabelEncoder] | None = None,
        destination_path: str | None = None,
        **kwargs
    ):
        assert (prior_predictive is not None) or (posterior_predictive is not None)

        if posterior_predictive is not None:
            PREDICTIVE = POSTERIOR_PREDICTIVE
            predictive = posterior_predictive
            msg = "Rendering posterior predictive checks ..."
        else:
            PREDICTIVE = PRIOR_PREDICTIVE
            predictive = prior_predictive
            msg = "Rendering prior predictive checks ..."

        if destination_path is None:
            destination_path = os.path.join(self.build_dir, PREDICTIVE)

        logger.info(msg)
        return self.predictive_checks_renderer(
            df=df,
            prediction_df=prediction_df,
            predictive=predictive,
            destination_path=destination_path,
            encoder_dict=encoder_dict,
            **kwargs
        )

    @timing
    def render_diagnostics(
        self,
        df: pd.DataFrame,
        destination_path: str,
        posterior_samples: dict,
        var_names: list[str],
        encoder_dict: dict[str, LabelEncoder] | None = None,
        **kwargs
    ):
        """
        **kwargs:
            combination_columns: list[str]
            orderby: lambda function
            intensity: str
            response: list[str]
            response_colors: list[str] | np.ndarray
            base: int
            subplot_cell_width: int
            subplot_cell_height: int
            recruitment_curve_props: dict
            threshold_posterior_props: dict
        """
        combination_columns = kwargs.get("combination_columns", self.features)
        orderby = kwargs.get("orderby")
        response = kwargs.get("response", self.response)
        num_chains = kwargs.get("num_chains", self.mcmc_params["num_chains"])
        chain_colors = kwargs.get("chain_colors", Plotter._get_colors(n=num_chains))
        subplot_cell_width = kwargs.get("subplot_cell_width", self.subplot_cell_width)
        subplot_cell_height = kwargs.get("subplot_cell_height", self.subplot_cell_height)

        # Group by chain
        posterior_samples = {
            u: v.reshape(num_chains, -1, *v.shape[1:])
            for u, v in posterior_samples.items()
        }

        msg = "Rendering diagnostics ..."
        logger.info(msg)
        # Setup pdf layout
        combinations = self._get_combinations(df=df, columns=combination_columns, orderby=orderby)
        n_combinations = len(combinations)
        n_response = len(response)

        n_columns_per_response = 2 * len(var_names)

        n_fig_rows = 10
        n_fig_columns = n_columns_per_response * n_response
        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1

        # Iterate over pdf pages
        pdf = PdfPages(destination_path)
        combination_counter = 0

        for page in range(n_pdf_pages):
            n_rows_current_page = min(
                n_fig_rows,
                n_combinations - page * n_fig_rows
            )
            fig, axes = plt.subplots(
                nrows=n_rows_current_page,
                ncols=n_fig_columns,
                figsize=(
                    n_fig_columns * subplot_cell_width,
                    n_rows_current_page * subplot_cell_height
                ),
                constrained_layout=True,
                squeeze=False
            )

            # Iterate over combinations
            for i in range(n_rows_current_page):
                curr_combination = combinations[combination_counter]
                curr_combination_inverse = ""

                if encoder_dict is not None:
                    curr_combination_inverse = self._get_combination_inverse(
                        combination=curr_combination,
                        columns=combination_columns,
                        encoder_dict=encoder_dict
                    )
                    curr_combination_inverse = ", ".join(map(str, curr_combination_inverse))
                    curr_combination_inverse += "\n"

                # Iterate over responses
                j = 0
                for r, response_muscle in enumerate(response):
                    # Labels
                    prefix = f"{tuple(list(curr_combination) + [r])}: {response_muscle} - MEP"
                    if not j: prefix = curr_combination_inverse + prefix

                    for var_name in var_names:
                        postfix = f"{var_name} KDE"
                        for chain in range(num_chains):
                            samples = posterior_samples[var_name][chain, :, *curr_combination, r]
                            ax = axes[i, j]
                            sns.kdeplot(samples, color=chain_colors[chain], ax=ax, label=f"CH:{chain}")
                        ax.set_title(prefix + postfix)
                        ax.legend(loc="upper left")
                        j += 1

                        postfix = f"{var_name} Trace Plot"
                        for chain in range(num_chains):
                            samples = posterior_samples[var_name][chain, :, *curr_combination, r]
                            ax = axes[i, j]
                            ax.plot(samples, color=chain_colors[chain])
                        ax.set_title(postfix)
                        j += 1

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {destination_path}")
        return
