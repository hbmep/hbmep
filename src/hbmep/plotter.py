import logging

import numpy as np
import pandas as pd
from numpyro.diagnostics import hpdi
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

import hbmep as mep
from hbmep.util import generate_response_colors, invert_combination

logger = logging.getLogger(__name__)
CURVE_KW = {"label": "Curve", "color": "k", "alpha": 0.4}
THRESHOLD_KW = {"color": "green", "alpha": 0.4}


def get_mep_data(
    mep_array: np.ndarray,
    *,
    response: list[str] = None,
    mep_response: list[str] = None,
    mep_window: list[float] = [0, 1],
    mep_size_window: list[float] | None = None,
    mep_adjust: float = 1.,
    **kw
):
    if not (response is None or mep_response is None) and mep_response != response:
        idx = [r for r, res in enumerate(mep_response) if res in response]
    else:
        idx = [r for r in range(mep_array.shape[-1])]
    mep_array = mep_array[..., idx]

    if mep_size_window is None: mep_size_window = mep_window
    assert (mep_size_window[0] >= mep_window[0]) and (mep_size_window[1] <= mep_window[1])

    mep_time = np.linspace(*mep_window, mep_array.shape[1])
    mep_array = mep_array / np.nanmax(mep_array, axis=1, keepdims=True)
    mep_array = mep_adjust * mep_array

    mep_time_offset = 10 / mep_array.shape[1]
    lo, hi = mep_size_window[0] - mep_time_offset, mep_size_window[1] + mep_time_offset
    idx = (mep_time >= lo) & (mep_time <= hi)
    mep_array = mep_array[:, idx, ...]
    mep_time = mep_time[idx]
    return mep_array, mep_time


def mep_plotter(
    mep_array: np.ndarray,
    intensity: np.ndarray,
    mep_time: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    **kwargs
):
    if ax is None: _, ax = plt.subplots(1, 1)
    if mep_time is None: mep_time = np.linspace(0, 1, mep_array.shape[1])
    for i in range(mep_array.shape[0]):
        x = mep_array[i, :]
        x = x + intensity[i]
        if not np.isnan(x).all(): ax.plot(x, mep_time, **kwargs)
    return ax


def plotter(
    df: pd.DataFrame,
    *,
    intensity: str,
    response: list[str],
    mep_array: np.ndarray | None = None,
    mep_time: np.ndarray | None = None,
    prediction_df: pd.DataFrame | None = None,
    prediction: np.ndarray | None = None,
    prediction_hdi: np.ndarray | None = None,
    prediction_prob: float = 0,
    threshold: np.ndarray | None = None,
    threshold_hdi: np.ndarray | None = None,
    threshold_prob: float = 0.95,
    axes: plt.Axes | None = None,
    yscale: str | None = None,
    **kw
):
    num_response = len(response)
    hue = kw.pop("hue", None)
    if hue is None or isinstance(hue, str): hue = [hue] * num_response
    colors = kw.pop("response_colors", None)
    if colors is None: colors = generate_response_colors(num_response)
    xoffset = kw.pop("xoffset", 0.5)
    curve_kwargs = kw.pop("curve_kwargs", CURVE_KW)
    threshold_kwargs = kw.pop("threshold_kwargs", THRESHOLD_KW)
    # if axes is None:

    if prediction_df is not None:
        assert prediction is not None
        if prediction_prob and prediction_hdi is None:
            prediction_hdi = hpdi(prediction, axis=0, prob=prediction_prob)
        prediction = prediction.mean(axis=0)

    if threshold is not None:
        if threshold_hdi is None:
            threshold_hdi = hpdi(threshold, axis=0, prob=threshold_prob)
        point_thresh = threshold.mean(axis=0)

    # Iterate over responses
    counter = 0
    for r in range(num_response):
        # MEP data
        if mep_array is not None:
            ax = axes[counter]
            ax = mep_plotter(
                mep_array=mep_array[..., r],
                intensity=df[intensity],
                mep_time=mep_time,
                ax=ax,
                color=colors[r],
                alpha=.4,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.sharex(axes[0])
            if counter > 0 and ax.get_legend(): ax.get_legend().remove()
            counter += 1

        # MEP size scatter plot
        ax = axes[counter]
        sns.scatterplot(ax=ax, data=df, x=intensity, y=response[r], color=colors[r], hue=hue[r])
        ax.set_xlabel(intensity)
        ax.set_ylabel(response[r])
        lo, hi = df[intensity].min(), df[intensity].max()
        ax.set_xlim(left=lo - xoffset, right=hi + xoffset)
        ax.sharex(axes[0])
        if ax.get_legend() is not None: ax.get_legend().remove()
        if yscale is not None:
            ax.set_yscale(yscale)
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        counter += 1

        # MEP size scatter plot and fitted curve
        if prediction_df is not None:
            if not np.all(np.isnan(df[response[r]].values)):
                ax = axes[counter]
                if prediction_hdi is not None:
                    ax.fill_between(
                        prediction_df[intensity],
                        prediction_hdi[0, :, r],
                        prediction_hdi[1, :, r],
                        color="cyan",
                        alpha=.4
                    )
                sns.scatterplot(ax=ax, data=df, x=intensity, y=response[r], color=colors[r], hue=hue[r])
                sns.lineplot(x=prediction_df[intensity], y=prediction[:, r], ax=ax, **curve_kwargs)
                if threshold is not None: sns.kdeplot(x=threshold[:, r], ax=ax, **threshold_kwargs)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.sharex(axes[counter - 1])
                ax.sharey(axes[counter - 1])
                ax.tick_params(axis="x", rotation=90)
                if ax.get_legend(): ax.get_legend().remove()
                if yscale is not None:
                    ax.set_yscale(yscale)
                    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            counter += 1

        # Threshold kde
        if threshold is not None:
            ax = axes[counter]
            sns.kdeplot(x=threshold[:, r], ax=ax, **threshold_kwargs)
            ax.axvline(point_thresh[r], linestyle="--", color=colors[r], label="Point estimate")
            ax.axvline(threshold_hdi[0, r], linestyle="--", color="black", alpha=.4, label="95% HPDI")
            ax.axvline(threshold_hdi[1, r], linestyle="--", color="black", alpha=.4)
            ax.set_xlabel(intensity)
            if ax.get_legend(): ax.get_legend().remove()
            counter += 1


def plot(
    df: pd.DataFrame,
    *,
    intensity: str,
    features: list[str],
    response: list[str],
    output_path: str,
    encoder: dict[str, LabelEncoder] | None = None,
    mep_array: np.ndarray | None = None,
    mep_response: list[str] | None = None,
    mep_window: list[float] = [0, 1],
    mep_size_window: list[float] | None = None,
    mep_adjust: float = 1.,
    prediction_df: pd.DataFrame | None = None,
    prediction: np.ndarray | None = None,
    prediction_hdi: np.ndarray | None = None,
    prediction_prob: float = 0,
    threshold: np.ndarray | None = None,
    threshold_hdi: np.ndarray | None = None,
    threshold_prob: float = 0.95,
    **kw
):
    """
    **kwargs:
        sort_key: Callable
        hue: str | list[str] | None
        response_colors: list[str]
        subplot_size: list[float]
        xaxis_offset: float
        curve_kwargs: dict
        threshold_kwargs: dict
    """
    sort_key = kw.pop("sort_key", None)
    subplot_width, subplot_height = kw.pop("subplot_size", (5, 3))
    annotation_offset = kw.pop("annotation_offset", 0)

    num_cols = 1
    mep_time = None
    if mep_array is not None:
        assert mep_array.shape[0] == df.shape[0]
        mep_array, mep_time = get_mep_data(
            mep_array,
            response=response,
            mep_response=mep_response,
            mep_window=mep_window,
            mep_size_window=mep_size_window,
            mep_adjust=mep_adjust
        )
        num_cols += 1
        annotation_offset += 1

    if prediction_df is not None:
        assert prediction is not None
        pred_features = mep.make_features(prediction_df, features)
        if prediction_prob and prediction_hdi is None:
            prediction_hdi = hpdi(prediction, prob=prediction_prob)
        prediction = prediction.mean(axis=0, keepdims=True)
        num_cols += 1

    if threshold is not None:
        if not len(features): threshold = threshold[:, None, ...]
        if threshold_hdi is None:
            threshold_hdi = hpdi(threshold, axis=0, prob=threshold_prob)
        num_cols += 1

    # Setup pdf layout
    df_features = mep.make_features(df, features)
    combinations = df_features.unique().tolist()
    combinations = sorted(combinations, key=sort_key)
    num_combinations = len(combinations)
    num_response = len(response)
    num_cols *= num_response
    num_rows = 10
    num_pages = num_combinations // num_rows + (num_combinations % num_rows > 0)

    logger.info(output_path)
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
            ccdf, ccmep_array = None, None
            ccpred_df, ccpred, ccpred_hdi = None, None, None
            ccthresh, ccthresh_hdi = None, None

            # Dataframe for current combination
            df_idx = df_features.isin([cc])
            ccdf = df[df_idx].reset_index(drop=True).copy()

            if mep_array is not None:
                ccmep_array = mep_array[df_idx, ...]

            # Prediction dataframe for current combination
            if prediction_df is not None:
                pred_idx = pred_features.isin([cc])
                ccpred_df = prediction_df[pred_idx].reset_index(drop=True).copy()
                # Prediction for current combination
                ccpred = prediction[:, pred_idx, :]
                if prediction_hdi is not None:
                    ccpred_hdi = prediction_hdi[:, pred_idx, :]

            # Threshold estimate for current combination
            if threshold is not None:
                ccthresh = threshold[:, *cc, :]
                ccthresh_hdi = threshold_hdi[:, *cc, :]

            plotter(
                ccdf,
                intensity=intensity,
                features=features,
                response=response,
                mep_array=ccmep_array,
                mep_time=mep_time,
                prediction_df=ccpred_df,
                prediction=ccpred,
                prediction_hdi=ccpred_hdi,
                threshold=ccthresh,
                threshold_hdi=ccthresh_hdi,
                axes=axes[row, :],
                **kw
            )

            annotation = ", ".join(map(str, cc))
            annotation_inverse = ""
            if encoder is not None:
                ccinverse = invert_combination(cc, features, encoder)
                ccinverse = ", ".join(map(str, ccinverse))
                annotation_inverse += f"\n{ccinverse}"

            for r in range(num_response):
                ax = axes[row, annotation_offset + r * (num_cols // num_response)]
                ax.set_title(f"({annotation}, {r})" + annotation_inverse)

            counter += 1

        logger.info(f"Page {page + 1} of {num_pages} done.")
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
    plt.close()
    logger.info(f"Saved to {output_path}")
    return
