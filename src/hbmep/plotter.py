import logging

import numpy as np
import pandas as pd
from numpyro.diagnostics import hpdi
from sklearn.preprocessing import LabelEncoder

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import arviz as az

from hbmep.util import (
    timing, floor, ceil, invert_combination, generate_colors
)

logger = logging.getLogger(__name__)
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


def mepplot(
    ax: plt.Axes,
    mep_matrix: np.ndarray,
    intensity: np.ndarray,
    time: np.ndarray | None = None,
    **kwargs
):
    """
    Plot a matrix of Motor Evoked Potentials (MEP) on a given axis.

    Parameters
    ----------
    ax: matplotlib.axes.Axes
        The axis on which to plot the MEP.
    mep_matrix: numpy.ndarray
        The MEP matrix, where each row corresponds to a different intensity
        value, and each column corresponds to a different time point.
    intensity: numpy.ndarray
        The intensity values corresponding to each row of `mep_matrix`.
    time: numpy.ndarray
        The time points corresponding to each column of `mep_matrix`.
    **kwargs:
        Any additional keyword arguments to be passed to `ax.plot`.

    Returns
    -------
    ax: matplotlib.axes.Axes
        The axis on which the MEP was plotted.
    """
    if time is None: time = np.linspace(0, 1, mep_matrix.shape[1])
    for i in range(mep_matrix.shape[0]):
        x = mep_matrix[i, :]
        x = x + intensity[i]
        ax.plot(x, time, **kwargs)
    return ax


def plot(
    df: pd.DataFrame,
    *,
    intensity: str,
    features: list[str],
    response: list[str],
    output_path: str,
    mep_matrix: np.ndarray | None = None,
    mep_time: np.ndarray | None = None,
    mep_window: list[float] | None = None,
    mep_size_window: list[float] | None = None,
    df_pred: pd.DataFrame | None = None,
    response_pred: dict | None = None,
    threshold: np.ndarray | None = None,
    encoder_dict: dict[str, LabelEncoder] | None = None,
    **kwargs
):
    """
    **kwargs:
        combination_columns: list[str]
        hue: str | list[str]
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
    sort_key = kwargs.get("sort_key", None)
    response_colors = kwargs.get("response_colors", generate_colors(n=len(response)))
    base = kwargs.get("base", 10)
    rc_kwargs = kwargs.get(
        "rc_kwargs", {"label": "RC", "color": "k", "alpha": 0.4}
    )
    threshold_kwargs = kwargs.get(
        "threshold_kwargs", {"color": "green", "alpha": 0.4}
    )
    hue = kwargs.get("hue", [None] * len(response))

    ncols = 1
    if mep_matrix is not None:
        assert mep_matrix.shape[0] == df.shape[0]
        if mep_window is None: mep_window = [0, 1]

        if mep_size_window is None: mep_size_window = mep_window
        else: assert (mep_size_window[0] >= mep_window[0]) and (mep_size_window[1] <= mep_window[1])

        if mep_time is not None: assert mep_time.shape[0] == mep_matrix.shape[1]
        elif mep_window is not None: mep_time = np.linspace(*mep_window, mep_matrix.shape[1])
        within_window_size = (
            (mep_time > mep_size_window[0])
            & (mep_time < mep_size_window[1])
        )
        mep_scalar = kwargs.get("mep_scalar", 1.0)
        ncols += 1

    if threshold is not None:
        threshold_mean = threshold.mean(axis=0);
        threshold_hdi = hpdi(threshold, prob=0.95, axis=0)
        ncols += 1

    if df_pred is not None:
        assert response_pred is not None
        assert response_pred.ndim in {2, 3}
        if response_pred.ndim == 3: response_pred = response_pred.mean(axis=0)
        ncols +=1

    if hue is not None and isinstance(hue, str):
        hue = [hue] * len(response)

    # Setup pdf layout
    combinations = (
        df[features]
        .apply(tuple, axis=1)
        .unique()
        .tolist()
    )
    combinations = sorted(combinations, key=sort_key)
    ncombinations = len(combinations)
    nresponse = len(response)
    ncols *= nresponse
    nrows = 10
    npages = ncombinations // nrows + (ncombinations % nrows > 0)
    cell_width, cell_height = kwargs.get("cell_size", (5, 3))

    # Iterate over pdf pages
    counter = 0
    pdf = PdfPages(output_path)
    for page in range(npages):
        nrows_current = min(nrows, ncombinations - page * nrows)
        fig, axes = plt.subplots(
            nrows=nrows_current,
            ncols=ncols,
            figsize=(ncols * cell_width, nrows_current * cell_height),
            constrained_layout=True,
            squeeze=False
        )

        # Iterate over combinations
        for row in range(nrows_current):
            # Current combination (cc)
            cc = combinations[counter]
            cc_inverse = ""

            if encoder_dict is not None:
                cc_inverse = invert_combination(
                    cc, features, encoder_dict
                )
                cc_inverse = ", ".join(map(str, cc_inverse))
                cc_inverse += "\n"

            # Filter dataframe based on current combination
            df_idx = df[features].apply(tuple, axis=1).isin([cc])
            ccdf = df[df_idx].reset_index(drop=True).copy()

            if df_pred is not None:
                # Filter prediction dataframe based on current combination
                df_pred_idx = df_pred[features].apply(tuple, axis=1).isin([cc])
                ccdf_pred = df_pred[df_pred_idx].reset_index(drop=True).copy()

                # Predicted response for current combination
                ccresponse_pred = response_pred[df_pred_idx, :]

            if threshold is not None:
                # Threshold estimate for current combination
                ccthreshold = threshold[:, *cc, :]
                ccthreshold_mean = threshold_mean[*cc, :]
                ccthreshold_hdi = threshold_hdi[:, *cc, :]

            # Tickmarks
            min_intensity, max_intensity_ = ccdf[intensity].agg(["min", "max"])
            min_intensity = floor(min_intensity, base=base)
            max_intensity = ceil(max_intensity_, base=base)
            if max_intensity == max_intensity_:
                max_intensity += base
            curr_xticks = np.arange(min_intensity, max_intensity, base)
            axes[row, 0].set_xlabel(intensity)
            axes[row, 0].set_xticks(ticks=curr_xticks)
            axes[row, 0].set_xlim(
                left=min_intensity - (base // 2), right=max_intensity + base
            )

            # Iterate over responses
            j = 0
            for r, response_muscle in enumerate(response):
                # Labels
                prefix = f"{tuple(list(cc) + [r])}: {response_muscle} - MEP"
                if not j: prefix = cc_inverse + prefix

                # MEP data
                if mep_matrix is not None:
                    postfix = " - MEP"
                    ax = axes[row, j]
                    ccmatrix = mep_matrix[df_idx, :, r]
                    max_amplitude = ccmatrix[:, within_window_size].max()
                    ccmatrix = mep_scalar * (ccmatrix / max_amplitude) * (base // 2)

                    ax = mepplot(
                        ax,
                        ccmatrix,
                        ccdf[intensity],
                        mep_time,
                        color=response_colors[r],
                        alpha=.4,
                    )
                    ax.axhline(
                        y=mep_size_window[0],
                        color="r",
                        linestyle="--",
                        alpha=.4,
                        label="MEP Size Window"
                    )
                    ax.axhline(
                        y=mep_size_window[1],
                        color="r",
                        linestyle="--",
                        alpha=.4,
                        label="MEP Size Window"
                    )
                    ax.set_ylim(
                        bottom=-0.001,
                        top=mep_size_window[1] + (mep_size_window[0] - (-0.001))
                    )
                    ax.set_title(prefix + postfix)
                    ax.sharex(axes[row, 0])
                    if j > 0 and ax.get_legend() is not None: ax.get_legend().remove()
                    j += 1

                # MEP Size scatter plot
                postfix = " - MEP Size"
                ax = axes[row, j]
                sns.scatterplot(
                    data=ccdf,
                    x=intensity,
                    y=response_muscle,
                    color=response_colors[r],
                    ax=ax,
                    hue=hue[r]
                )
                ax.set_ylabel(response_muscle)
                ax.set_title(prefix + postfix)
                ax.sharex(axes[row, 0])
                if ax.get_legend() is not None: ax.get_legend().remove()
                j += 1

                if df_pred is not None:
                    if not np.all(np.isnan(ccdf[response_muscle].values)):
                        # MEP Size scatter plot and recruitment curve
                        postfix = "Recruitment Curve Fit"
                        ax = axes[row, j]
                        sns.scatterplot(data=ccdf, x=intensity, y=response_muscle, color=response_colors[r], ax=ax, hue=hue[r])
                        sns.lineplot(
                            x=ccdf_pred[intensity],
                            y=ccresponse_pred[:, r],
                            ax=ax,
                            **rc_kwargs,
                        )
                        if threshold is not None:
                            sns.kdeplot(
                            x=ccthreshold[:, r],
                            ax=ax,
                            **threshold_kwargs
                        )
                        ax.set_title(postfix)
                        ax.sharex(axes[row, 0])
                        ax.sharey(axes[row, j - 1])
                        ax.tick_params(axis="x", rotation=90)
                        if ax.get_legend() is not None: ax.get_legend().remove()
                    j += 1

                if threshold is not None:
                    # Threshold KDE
                    ax = axes[row, j]
                    postfix = "Threshold Estimate"
                    sns.kdeplot(
                        x=ccthreshold[:, r],
                        ax=ax,
                        **threshold_kwargs
                    )
                    ax.axvline(
                        ccthreshold_mean[r],
                        linestyle="--",
                        color=response_colors[r],
                        label="Threshold"
                    )
                    ax.axvline(
                        ccthreshold_hdi[:, r][0],
                        linestyle="--",
                        color="black",
                        alpha=.4,
                        label="95% HPDI"
                    )
                    ax.axvline(
                        ccthreshold_hdi[:, r][1],
                        linestyle="--",
                        color="black",
                        alpha=.4
                    )
                    ax.set_xlabel(intensity)
                    ax.set_title(postfix)
                    if j > 0 and ax.get_legend(): ax.get_legend().remove()
                    j += 1

            counter += 1

        logger.info(f"Page {page + 1} of {npages} done.")
        pdf.savefig(fig)
        plt.close()

    pdf.close()
    plt.show()

    logger.info(f"Saved to {output_path}")
    return


# def ppcplot(
#     df: pd.DataFrame,
#     *,
#     intensity: str,
#     features: list[str],
#     response: list[str],
#     output_path: str,
#     df_pred: pd.DataFrame,
#     predictive: dict,
#     mep_matrix: np.ndarray | None = None,
#     mep_time: np.ndarray | None = None,
#     mep_window: list[float] | None = None,
#     mep_size_window: list[float] | None = None,
#
#     response_pred: dict | None = None,
#     threshold: np.ndarray | None = None,
#     encoder_dict: dict[str, LabelEncoder] | None = None,
#     **kwargs
#     self,
#     df: pd.DataFrame,
#     prediction_df: pd.DataFrame,
#     predictive: dict,
#     destination_path: str,
#     encoder_dict: dict[str, LabelEncoder] | None = None,
#     **kwargs
# ):
#     """
#     **kwargs:
#         combination_columns: list[str]
#         orderby: lambda function
#         base: int
#         subplot_cell_width: float
#         subplot_cell_height: float
#     """
#     combination_columns = kwargs.get("combination_columns", self.features)
#     orderby = kwargs.get("orderby")
#     base = kwargs.get("base", self.base)
#     subplot_cell_width = kwargs.get("subplot_cell_width", self.subplot_cell_width)
#     subplot_cell_height = kwargs.get("subplot_cell_height", self.subplot_cell_height)
#
#     # Predictive samples
#     obs, mu = predictive[site.obs], predictive[site.mu]
#
#     # Setup pdf layout
#     combinations = self._get_combinations(df=df, columns=combination_columns, orderby=orderby)
#     n_combinations = len(combinations)
#
#     n_columns_per_response = 3
#     n_fig_rows = 10
#     n_fig_columns = n_columns_per_response * self.n_response
#
#     n_pdf_pages = n_combinations // n_fig_rows
#     if n_combinations % n_fig_rows: n_pdf_pages += 1
#
#     # Iterate over pdf pages
#     pdf = PdfPages(destination_path)
#     combination_counter = 0
#
#     for page in range(n_pdf_pages):
#         n_rows_current_page = min(
#             n_fig_rows,
#             n_combinations - page * n_fig_rows
#         )
#         fig, axes = plt.subplots(
#             nrows=n_rows_current_page,
#             ncols=n_fig_columns,
#             figsize=(
#                 n_fig_columns * subplot_cell_width,
#                 n_rows_current_page * subplot_cell_height
#             ),
#             sharex="row",
#             constrained_layout=True,
#             squeeze=False
#         )
#
#         # Iterate over combinations
#         for i in range(n_rows_current_page):
#             curr_combination = combinations[combination_counter]
#             curr_combination_inverse = ""
#
#             if encoder_dict is not None:
#                 curr_combination_inverse = self._get_combination_inverse(
#                     combination=curr_combination,
#                     columns=combination_columns,
#                     encoder_dict=encoder_dict
#                 )
#                 curr_combination_inverse = ", ".join(map(str, curr_combination_inverse))
#                 curr_combination_inverse += "\n"
#
#             # Filter dataframe based on current combination """
#             df_ind = df[combination_columns].apply(tuple, axis=1).isin([curr_combination])
#             curr_df = df[df_ind].reset_index(drop=True).copy()
#
#             # Filter prediction dataframe based on current combination
#             prediction_df_ind = prediction_df[combination_columns].apply(tuple, axis=1).isin([curr_combination])
#             curr_prediction_df = prediction_df[prediction_df_ind].reset_index(drop=True).copy()
#
#             # Predictive for current combination
#             curr_obs = obs[:, prediction_df_ind, :]
#             curr_obs_map = curr_obs.mean(axis=0)
#             curr_mu = mu[:, prediction_df_ind, :]
#             curr_mu_map = curr_mu.mean(axis=0)
#
#             # HPDI intervals
#             curr_obs_hpdi_95 = hpdi(curr_obs, prob=.95)
#             curr_obs_hpdi_85 = hpdi(curr_obs, prob=.85)
#             curr_obs_hpdi_65 = hpdi(curr_obs, prob=.65)
#             curr_mu_hpdi_95 = hpdi(curr_mu, prob=.95)
#
#             # Tickmarks
#             min_intensity, max_intensity_ = curr_df[self.intensity].agg(["min", "max"])
#             min_intensity = floor(min_intensity, base=base)
#             max_intensity = ceil(max_intensity_, base=base)
#             if max_intensity == max_intensity_:
#                 max_intensity += base
#             curr_x_ticks = np.arange(min_intensity, max_intensity, base)
#
#             axes[i, 0].set_xticks(ticks=curr_x_ticks)
#             axes[i, 0].set_xlim(left=min_intensity - (base // 2), right=max_intensity + (base // 2))
#
#             # Iterate over responses
#             j = 0
#             for r, response in enumerate(self.response):
#                 prefix = f"{tuple(list(curr_combination) + [r])}: {response} - MEP"
#                 if not j: prefix = curr_combination_inverse + prefix
#
#                 # MEP Size scatter plot and recruitment curve
#                 postfix = ""
#                 ax = axes[i, j]
#                 sns.scatterplot(data=curr_df, x=self.intensity, y=response, color=self.response_colors[r], ax=ax)
#                 sns.lineplot(
#                     x=curr_prediction_df[self.intensity],
#                     y=curr_mu_map[:, r],
#                     ax=ax,
#                     **self.recruitment_curve_props,
#                 )
#                 ax.set_title(prefix + postfix)
#                 ax.sharex(axes[i, 0])
#                 ax.tick_params(axis="x", rotation=90)
#                 j += 1
#
#                 # Observational predictive
#                 postfix = "Prediction"
#                 ax = axes[i, j]
#                 sns.lineplot(
#                     x=curr_prediction_df[self.intensity],
#                     y=curr_obs_map[:, r],
#                     color="black",
#                     ax=ax
#                 )
#                 ax.fill_between(
#                     curr_prediction_df[self.intensity],
#                     curr_obs_hpdi_95[0, :, r],
#                     curr_obs_hpdi_95[1, :, r],
#                     color="C1"
#                 )
#                 ax.fill_between(
#                     curr_prediction_df[self.intensity],
#                     curr_obs_hpdi_85[0, :, r],
#                     curr_obs_hpdi_85[1, :, r],
#                     color="C2"
#                 )
#                 ax.fill_between(
#                     curr_prediction_df[self.intensity],
#                     curr_obs_hpdi_65[0, :, r],
#                     curr_obs_hpdi_65[1, :, r],
#                     color="C3"
#                 )
#                 sns.scatterplot(
#                     data=curr_df,
#                     x=self.intensity,
#                     y=response,
#                     color="yellow",
#                     edgecolor="black",
#                     ax=ax
#                 )
#                 ax.sharex(axes[i, 0])
#                 ax.sharey(axes[i, j - 1])
#                 ax.set_title(postfix)
#                 ax.tick_params(axis="x", rotation=90)
#                 j += 1
#
#                 # Recruitment curve predictive
#                 postfix = "Fit"
#                 ax = axes[i, j]
#                 sns.lineplot(
#                     x=curr_prediction_df[self.intensity],
#                     y=curr_mu_map[:, r],
#                     color="black",
#                     ax=ax
#                 )
#                 ax.fill_between(
#                     curr_prediction_df[self.intensity],
#                     curr_mu_hpdi_95[0, :, r],
#                     curr_mu_hpdi_95[1, :, r],
#                     color="C1"
#                 )
#                 sns.scatterplot(
#                     data=curr_df,
#                     x=self.intensity,
#                     y=response,
#                     color="yellow",
#                     edgecolor="black",
#                     ax=ax
#                 )
#                 ax.sharex(axes[i, 0])
#                 ax.sharey(axes[i, j - 2])
#                 ax.set_title(postfix)
#                 ax.tick_params(axis="x", rotation=90)
#                 j += 1
#
#             combination_counter += 1
#
#         logger.info(f"Page {page + 1} of {n_pdf_pages} done.")
#         pdf.savefig(fig)
#         plt.close()
#
#     pdf.close()
#     plt.show()
#
#     logger.info(f"Saved to {destination_path}")
#     return
#
# # @timing
# # def render_predictive_check(
# #     self,
# #     df: pd.DataFrame,
# #     prediction_df: pd.DataFrame,
# #     prior_predictive: dict | None = None,
# #     posterior_predictive: dict | None = None,
# #     encoder_dict: dict[str, LabelEncoder] | None = None,
# #     destination_path: str | None = None,
# #     **kwargs
# # ):
# #     assert (prior_predictive is not None) or (posterior_predictive is not None)
# #
# #     if posterior_predictive is not None:
# #         PREDICTIVE = POSTERIOR_PREDICTIVE
# #         predictive = posterior_predictive
# #         msg = "Rendering posterior predictive checks ..."
# #     else:
# #         PREDICTIVE = PRIOR_PREDICTIVE
# #         predictive = prior_predictive
# #         msg = "Rendering prior predictive checks ..."
# #
# #     if destination_path is None:
# #         destination_path = os.path.join(self.build_dir, PREDICTIVE)
# #
# #     logger.info(msg)
# #     return self.predictive_checks_renderer(
# #         df=df,
# #         prediction_df=prediction_df,
# #         predictive=predictive,
# #         destination_path=destination_path,
# #         encoder_dict=encoder_dict,
# #         **kwargs
# #     )


# def trace_renderer(
#     posterior: dict,
#     var_names: list[str],
#     rhat_threshold: float,
#     destination_path: str,
#     **kwargs
# ):
#     """
#     **kwargs:
#         response_colors: list[str] | np.ndarray
#         subplot_cell_width: int
#         subplot_cell_height: int
#     """
#     num_chains = self.mcmc_params["num_chains"]
#     response_colors = kwargs.get("response_colors", generate_colors(len(response)))
#     chain_colors = kwargs.get("chain_colors", Plotter._get_colors(n=num_chains))
#     subplot_cell_width = kwargs.get("subplot_cell_width", self.subplot_cell_width / 1.5)
#     subplot_cell_height = kwargs.get("subplot_cell_height", self.subplot_cell_height / 1.5)
#
#     # Group by chain
#     posterior_samples = {
#         u: v.reshape(num_chains, -1, *v.shape[1:])
#         for u, v in posterior_samples.items()
#     }
#
#         posterior = {u: v for u, v in posterior.items() if u in var_names}
#
#     rhat
#
#     # Setup pdf layout
#     summary_df = az.summary(posterior_samples, var_names=var_names)
#     summary_df["site"] = summary_df.index; summary_df = summary_df.reset_index(drop=True).copy()
#     ind = summary_df.r_hat > rhat_threshold; summary_df = summary_df[ind].reset_index(drop=True).copy()
#     if not summary_df.shape[0]: logger.info(f"No site with rhat > {rhat_threshold}."); return
#
#     nrows = summary_df.shape[0]
#     n_fig_rows = 10
#     n_fig_columns = 2
#     n_pdf_pages = nrows // n_fig_rows
#     if nrows % n_fig_rows: n_pdf_pages += 1
#
#     # Iterate over pdf pages
#     pdf = PdfPages(destination_path)
#     row_counter = 0
#
#     for page in range(n_pdf_pages):
#         n_rows_current_page = min(
#             n_fig_rows,
#             nrows - page * n_fig_rows
#         )
#         fig, axes = plt.subplots(
#             nrows=n_rows_current_page,
#             ncols=n_fig_columns,
#             figsize=(
#                 n_fig_columns * subplot_cell_width,
#                 n_rows_current_page * subplot_cell_height
#             ),
#             constrained_layout=True,
#             squeeze=False
#         )
#
#         for i in range(n_rows_current_page):
#             curr_row = summary_df.iloc[row_counter]
#             curr_site = curr_row.site; c = ()
#             if "[" and "]" in curr_site:
#                 start, stop = curr_site.index("["), curr_site.index("]")
#                 c = curr_site[(start + 1):stop]
#                 c = tuple(map(int, c.split(",")))
#                 curr_site = curr_site[:start]
#
#             for chain in range(num_chains):
#                 samples = posterior_samples[curr_site][..., *c][chain, :]
#                 ax = axes[i, 0]
#                 sns.kdeplot(samples, color=chain_colors[chain], ax=ax, label=f"CH:{chain}")
#                 ax = axes[i, 1]
#                 sns.lineplot(x=np.arange(samples.shape[0]), y=samples, color=chain_colors[chain], ax=ax)
#
#             ax = axes[i, 0]
#             ax.set_title(f"{curr_site}" + (f" {c}" if c else ""))
#             ax.set_xlabel(""); ax.set_ylabel("")
#             ax.legend(loc="upper right")
#             if i and ax.get_legend(): ax.get_legend().remove()
#             ax = axes[i, 1]
#             ax.set_title(f"rhat: {curr_row.r_hat:.2f}")
#             ax.set_xlabel(""); ax.set_ylabel("")
#             if i and ax.get_legend(): ax.get_legend().remove()
#
#             row_counter += 1
#
#         logger.info(f"Page {page + 1} of {n_pdf_pages} done.")
#         pdf.savefig(fig)
#         plt.close()
#
#     pdf.close()
#     plt.show()
#
#     logger.info(f"Saved to {destination_path}")
#     return
#
# @timing
# def trace_plot(
#     self,
#     posterior_samples: dict,
#     var_names: list[str] | None = None,
#     rhat_threshold: float | None = None,
#     exclude_deterministic: bool = True,
#     destination_path: str | None = None,
#     **kwargs
# ):
#     if destination_path is None:
#         destination_path = os.path.join(self.build_dir, TRACE_PLOT)
#
#     if rhat_threshold is None:
#         rhat_threshold = 0; msg = "Rhat threshold not provided. All rhat values will be displayed."
#     else: msg = f"Rhat threshold: {rhat_threshold}"
#     logger.info(msg)
#
#     if var_names is None: var_names = (
#         self.sample_sites if exclude_deterministic
#         else self.sample_sites + self.deterministic_sites
#     )
#     msg = f"Rendering trace plots for {', '.join(var_names)} ..."
#     logger.info(msg)
#
#     return self.trace_renderer(
#         posterior_samples=posterior_samples,
#         var_names=var_names,
#         rhat_threshold=rhat_threshold,
#         destination_path=destination_path,
#         **kwargs
#     )


