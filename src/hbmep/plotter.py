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
CURVE_KW = {"label": "Curve", "color": "k", "alpha": 0.4}
THRESHOLD_KW = {"color": "green", "alpha": 0.4}


def plot_mep(
    ax: plt.Axes,
    *,
    mep_matrix: np.ndarray,
    intensity: np.ndarray,
    time: np.ndarray | None = None,
    **kwargs
):
    if time is None: time = np.linspace(0, 1, mep_matrix.shape[1])
    for i in range(mep_matrix.shape[0]):
        x = mep_matrix[i, :]
        x = x + intensity[i]
        if not np.isnan(x).all(): ax.plot(x, time, **kwargs)
    return ax


def plot(
    *,
    df: pd.DataFrame,
    intensity: str,
    features: list[str],
    response: list[str],
    output_path: str,
    encoder: dict[str, LabelEncoder] | None = None,
    sort_key = None,
    hue: str | list[str] | None = None,
    response_colors: list[str] | None = None,
    subplot_size: list[float] | None = None,
    mep_matrix: np.ndarray | None = None,
    mep_window: list[float] = [0, 1],
    mep_size_window: list[float] | None = None,
    mep_adjust: float = 1.,
    prediction_df: pd.DataFrame | None = None,
    response_pred: dict | None = None,
    threshold: np.ndarray | None = None,
    **kwargs
):
    """
    **kwargs:
        curve_kwargs: dict
        threshold_kwargs: dict
    """
    num_response = len(response)
    response_colors = response_colors or generate_colors(num_response)
    subplot_width, subplot_height = subplot_size or (5, 3)
    curve_kwargs = kwargs.get("curve_kwargs", CURVE_KW)
    threshold_kwargs = kwargs.get("threshold_kwargs", THRESHOLD_KW)

    df_features = (
        df[features].apply(tuple, axis=1) if len(features)
        else df[intensity].apply(lambda x: 0).astype(int).apply(lambda x: tuple([x]))
    )

    num_cols = 1
    if mep_matrix is not None:
        assert mep_matrix.shape[0] == df.shape[0]
        if mep_size_window is None: mep_size_window = mep_window
        else: assert (mep_size_window[0] >= mep_window[0]) and (mep_size_window[1] <= mep_window[1])
        mep_time = np.linspace(*mep_window, mep_matrix.shape[1])
        mep_time_offset = 10 / mep_matrix.shape[1]
        # within_size_window = (mep_time >= mep_size_window[0]) & (mep_time <= mep_size_window[1])
        mep_matrix = mep_matrix / np.nanmax(mep_matrix, axis=1, keepdims=True)
        # mep_matrix = mep_matrix / np.nanmax(mep_matrix[:, within_size_window, :], axis=1, keepdims=True)
        # mep_matrix = mep_matrix / np.nanstd(mep_matrix[:, within_size_window, :], axis=1, keepdims=True)
        mep_matrix = mep_adjust * mep_matrix
        num_cols += 1

    if hue is None or isinstance(hue, str): hue = [hue] * num_response

    if prediction_df is not None:
        pred_df_features = (
            prediction_df[features].apply(tuple, axis=1) if len(features)
            else prediction_df[intensity].apply(lambda x: 0).astype(int).apply(lambda x: tuple([x]))
        )
        assert response_pred is not None
        assert response_pred.ndim in {2, 3}
        if response_pred.ndim == 3: response_pred = response_pred.mean(axis=0)
        num_cols +=1

    if threshold is not None:
        if not len(features): threshold = threshold[:, None, ...]
        threshold_mean = threshold.mean(axis=0)
        threshold_hdi = hpdi(threshold, prob=0.95, axis=0)
        num_cols += 1

    # Setup pdf layout
    combinations = (
        df[features].apply(tuple, axis=1).unique().tolist() if len(features)
        else [(0,)]
    )
    combinations = sorted(combinations, key=sort_key)
    num_combinations = len(combinations)
    num_cols *= num_response
    num_rows = 10
    num_pages = num_combinations // num_rows + (num_combinations % num_rows > 0)

    # Iterate over pdf pages
    counter = 0
    pdf = PdfPages(output_path)
    for page in range(num_pages):
        num_rows_current = min(num_rows, num_combinations - page * num_rows)
        fig, axes = plt.subplots(
            nrows=num_rows_current,
            ncols=num_cols,
            figsize=(num_cols * subplot_width, num_rows_current * subplot_height),
            constrained_layout=True,
            squeeze=False
        )

        # Iterate over combinations
        for row in range(num_rows_current):
            # Current combination (cc)
            cc = combinations[counter]
            cc_inverse = ""

            if encoder is not None:
                cc_inverse = invert_combination(cc, features, encoder)
                cc_inverse = ", ".join(map(str, cc_inverse))
                cc_inverse += "\n"

            # Filter dataframe based on current combination
            df_idx = df_features.isin([cc])
            ccdf = df[df_idx].reset_index(drop=True).copy()

            if prediction_df is not None:
                # Filter prediction dataframe based on current combination
                prediction_df_idx = pred_df_features.isin([cc])
                ccprediction_df = prediction_df[prediction_df_idx].reset_index(drop=True).copy()
                # Predicted response for current combination
                ccresponse_pred = response_pred[prediction_df_idx, :]

            if threshold is not None:
                # Threshold estimate for current combination
                ccthreshold = threshold[:, *cc, :]
                ccthreshold_mean = threshold_mean[*cc, :]
                ccthreshold_hdi = threshold_hdi[:, *cc, :]

            axes[row, 0].set_xlabel(intensity)
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
                    ax = plot_mep(
                        ax,
                        mep_matrix=ccmatrix,
                        intensity=ccdf[intensity],
                        time=mep_time,
                        color=response_colors[r],
                        alpha=.4,
                    )
                    ax.axhline(mep_size_window[0], color="r", linestyle="--", alpha=.4, label="MEP size window")
                    ax.axhline(mep_size_window[1], color="r", linestyle="--", alpha=.4)
                    ax.set_ylim(
                        bottom=mep_size_window[0] - mep_time_offset,
                        top=mep_size_window[1] + mep_time_offset
                    )
                    ax.set_title(prefix + postfix)
                    ax.sharex(axes[row, 0])
                    if j > 0 and ax.get_legend() is not None: ax.get_legend().remove()
                    j += 1

                # MEP size scatter plot
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

                if prediction_df is not None:
                    if not np.all(np.isnan(ccdf[response_muscle].values)):
                        # MEP Size scatter plot and recruitment curve
                        postfix = "Recruitment curve fit"
                        ax = axes[row, j]
                        sns.scatterplot(data=ccdf, x=intensity, y=response_muscle, color=response_colors[r], ax=ax, hue=hue[r])
                        sns.lineplot(x=ccprediction_df[intensity], y=ccresponse_pred[:, r], ax=ax, **curve_kwargs)

                        if threshold is not None:
                            sns.kdeplot(x=ccthreshold[:, r], ax=ax, **threshold_kwargs)

                        ax.set_title(postfix)
                        ax.sharex(axes[row, 0])
                        ax.sharey(axes[row, j - 1])
                        ax.tick_params(axis="x", rotation=90)
                        if ax.get_legend() is not None: ax.get_legend().remove()
                    j += 1

                if threshold is not None:
                    # Threshold kde
                    ax = axes[row, j]
                    postfix = "Threshold estimate"
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
                    ax.axvline(ccthreshold_hdi[:, r][0], linestyle="--", color="black", alpha=.4, label="95% HPDI")
                    ax.axvline(ccthreshold_hdi[:, r][1], linestyle="--", color="black", alpha=.4)
                    ax.set_xlabel(intensity)
                    ax.set_title(postfix)
                    if j > 0 and ax.get_legend(): ax.get_legend().remove()
                    j += 1

            counter += 1

        logger.info(f"Page {page + 1} of {num_pages} done.")
        pdf.savefig(fig)
        plt.close()

    pdf.close()
    plt.show()
    logger.info(f"Saved to {output_path}")
    return
