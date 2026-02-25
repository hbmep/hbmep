import logging

import numpy as np
import pandas as pd
from numpyro.diagnostics import hpdi
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

import hbmep as mep
from hbmep.util import invert_combination

logger = logging.getLogger(__name__)
CURVE_KW = {"label": "Curve", "color": "k", "alpha": 0.4}
THRESHOLD_KW = {"color": "green", "alpha": 0.4}


def _build_hue_palette(
    df: pd.DataFrame,
    hue: list[str] | None,
    num_response: int
):
    if hue is None:
        return [hue] * num_response, {}
    if isinstance(hue, str):
        hue = [hue] * num_response

    unique = pd.unique(df[hue].values.ravel())
    n = len(unique)
    colors = sns.color_palette(palette="tab10", n_colors=n)
    palette = {u: v for u, v in zip(unique, colors)}
    return hue, palette


def get_mep_data(
    mep_array: np.ndarray,
    *,
    response: list[str] = None,
    mep_response: list[str] = None,
    mep_window: list[float] = [0, 1],
    mep_size_window: list[float] | None = None,
    **kw
):
    idx = [r for r in range(mep_array.shape[-1])]
    if (
        not (response is None or mep_response is None)
        and mep_response != response
    ):
        idx = [r for r, res in enumerate(mep_response) if res in response]
    mep_array = mep_array[..., idx]

    if mep_size_window is None:
        mep_size_window = mep_window

    assert (
        (mep_size_window[0] >= mep_window[0])
        and (mep_size_window[1] <= mep_window[1])
    )

    mep_time = np.linspace(*mep_window, mep_array.shape[1])
    mep_size_time = (mep_time > mep_size_window[0]) & (mep_time < mep_size_window[1])
    return mep_time, mep_size_time


def mep_plotter(
    mep_array: np.ndarray,
    intensity: np.ndarray,
    mep_time: np.ndarray | None = None,
    mep_size_time: np.ndarray | None = None,
    mep_adjust: list[float] = 1,
    mep_xoffset: list[float] | None = None,
    mep_yoffset: list[float] | None = None,
    ax: plt.Axes | None = None,
    **kwargs
):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    if mep_time is None:
        mep_time = np.linspace(0, 1, mep_array.shape[1])
    
    if mep_size_time is None:
        mep_size_time = True
    
    if mep_xoffset is None:
        mep_xoffset = [0, 0]

    if mep_yoffset is None:
        mep_yoffset = [0, 0]

    max_amplitude = np.nanmax(mep_array[:, mep_size_time], keepdims=True)
    mep_array /= max_amplitude
    mep_array *= mep_adjust

    for i in range(mep_array.shape[0]):
        x = mep_array[i, :]
        x = x + intensity[i]
        if not np.isnan(x).all():
            ax.plot(x, mep_time, **kwargs)
    
    lo, hi = (
        intensity.min() + mep_xoffset[0],
        intensity.max() + mep_xoffset[1]
    )
    ax.set_xlim(lo, hi)

    lo, hi = mep_time[mep_size_time].min(), mep_time[mep_size_time].max()
    ax.axhline(lo, color="r", zorder=int(1e9))
    ax.axhline(hi, color="r", zorder=int(1e9))
    ylims = lo + mep_yoffset[0], hi + mep_yoffset[1]
    ax.set_ylim(*ylims)

    return ax


def plotter(
    df: pd.DataFrame,
    *,
    intensity: str,
    response: list[str],
    mep_array: np.ndarray | None = None,
    mep_time: np.ndarray | None = None,
    mep_size_time: np.ndarray | None = None,
    mep_adjust: float = 1.,
    mep_xoffset: list[float] | None = None,
    mep_yoffset: list[float] | None = None,
    prediction_df: pd.DataFrame | None = None,
    prediction: np.ndarray | None = None,
    prediction_hdi: np.ndarray | None = None,
    prediction_prob: float = 0,
    threshold: np.ndarray | None = None,
    threshold_hdi: np.ndarray | None = None,
    threshold_prob: float = 0.95,
    axes: plt.Axes | None = None,
    colors: list | dict = None, 
    hue: str | list[str] = None,
    hue_palette: dict[str, tuple] = None,
    yscale: str | None = None,
    **kw
):
    xoffset = kw.pop("xoffset", [-0.5, 0.5])
    curve_kwargs = kw.pop("curve_kwargs", CURVE_KW)
    threshold_kwargs = kw.pop("threshold_kwargs", THRESHOLD_KW)

    if prediction_df is not None:
        assert prediction is not None
        if prediction_prob and prediction_hdi is None:
            prediction_hdi = hpdi(prediction, axis=0, prob=prediction_prob)
        prediction = prediction.mean(axis=0)

    if threshold is not None:
        if threshold_hdi is None:
            threshold_hdi = hpdi(threshold, axis=0, prob=threshold_prob)
        point_thresh = threshold.mean(axis=0)

    share_index = 0

    # Iterate over responses
    num_response = len(response)
    counter = 0
    for r in range(num_response):
        color = colors[response[r]]
        has_hue = hue[r] is not None and hue_palette is not None
        palette = [
            hue_palette[u] for u in sorted(hue_palette.keys())
            if u in df[hue[r]].tolist()
        ]
        # MEP data
        if mep_array is not None:
            ax = axes[counter]
            ax = mep_plotter(
                mep_array=mep_array[..., r],
                intensity=df[intensity],
                mep_time=mep_time,
                mep_size_time=mep_size_time,
                mep_adjust=mep_adjust,
                mep_xoffset=mep_xoffset,
                mep_yoffset=mep_yoffset,
                ax=ax,
                color=colors[response[r]],
                alpha=.4,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            if counter > 0 and ax.get_legend():
                ax.get_legend().remove()
            counter += 1

        # MEP size scatter plot
        ax = axes[counter]
        
        if has_hue:
            sns.scatterplot(
                ax=ax, data=df, x=intensity, y=response[r], hue=hue[r],
                palette=palette, legend=False
            )
        else:
            sns.scatterplot(
                ax=ax, data=df, x=intensity, y=response[r],
                color=color, legend=False
            )
        ax.set_xlabel(intensity)
        ax.set_ylabel(response[r])
        lo, hi = df[intensity].min(), df[intensity].max()
        ax.set_xlim(left=lo + xoffset[0], right=hi + xoffset[1])
        ax.sharex(axes[share_index])
        if yscale is not None:
            ax.set_yscale(yscale)
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel("")

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
                if has_hue:
                    sns.scatterplot(
                        ax=ax, data=df, x=intensity, y=response[r],
                        hue=hue[r], palette=hue_palette[r], legend=False
                    )
                else:
                    sns.scatterplot(
                        ax=ax, data=df, x=intensity, y=response[r],
                        color=color, legend=False
                    )
                sns.lineplot(
                    x=prediction_df[intensity], y=prediction[:, r], ax=ax,
                    **curve_kwargs
                )
                if threshold is not None:
                    ax2 = ax.twinx()
                    sns.kdeplot(x=threshold[:, r], ax=ax2, **threshold_kwargs)
                    ax2.set_ylim(0, ax2.get_ylim()[1])
                    ax2.set_yticks([])
                    ax2.set_ylabel("")
                    for spine in ax2.spines.values():
                        spine.set_visible(False)
                    ax2.patch.set_alpha(0)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis="x", rotation=90)
                if yscale is not None:
                    ax.set_yscale(yscale)
                    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        counter += 1

        # Threshold kde
        if threshold is not None:
            ax = axes[counter]
            sns.kdeplot(x=threshold[:, r], ax=ax, **threshold_kwargs)
            ax.axvline(
                point_thresh[r], linestyle="--", color=color,
                label="Point estimate"
            )
            ax.axvline(
                threshold_hdi[0, r], linestyle="--", color="black", alpha=.4,
                label="95% HPDI"
            )
            ax.axvline(
                threshold_hdi[1, r], linestyle="--", color="black", alpha=.4
            )
            ax.set_xlabel(intensity)
            if ax.get_legend():
                ax.get_legend().remove()
            counter += 1


def plot(
    df: pd.DataFrame,
    *,
    intensity: str,
    features: list[str],
    response: list[str],
    encoder: dict[str, LabelEncoder] | None = None,
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
    Generate multi-panel plots of MEP responses, optionally including raw waveforms,
    scatter plots, fitted recruitment curves, and threshold distributions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data frame containing stimulation intensity and response columns.
    intensity : str
        Column name in `df` specifying stimulation intensity.
    features : list of str
        Column names in `df` that define experimental conditions or grouping variables.
        Each unique combination of these features defines one row in the plot.
    response : list of str
        Column names in `df` specifying response variables (e.g., muscles).
        Each response gets its own column in the plot.
    encoder : dict of str -> LabelEncoder, optional
        Mapping of feature names to fitted encoders for converting integer-coded
        feature levels back to human-readable labels. Used for subplot titles.
    prediction_df : pandas.DataFrame, optional
        Data frame containing intensity and features values aligned with `prediction` output.
    prediction : ndarray, optional
        Posterior predictive of shape (draws, points, response).
        Here, points equal number of rows in prediction_df.
        This typically corresponds to the posterior of the curves, but it could also be the
        observational posterior.
    prediction_hdi : ndarray, optional
        Highest density intervals (HDI) for `prediction` of shape
        (2, points, response). If not given but `prediction_prob > 0`, computed
        internally from `prediction`.
    prediction_prob : float, default=0
        Credible interval probability (0-1) for predictive HDI bands. Ignored if 0.
    threshold : ndarray, optional
        Posterior samples of threshold estimates of shape (draws, *features, response).
    threshold_hdi : ndarray, optional
        Highest density intervals (HDI) for `threshold` of shape (2, *features, response).
        If not given, computed internally from `threshold` using `threshold_prob`.
    threshold_prob : float, default=0.95
        Credible interval probability (0-1) for threshold HPDIs.
    **kw
        Additional keyword arguments controlling plotting:
        - sort_key : callable, optional
            Sorting function for ordering unique feature combinations.
        - hue : str or list of str, optional
            Column(s) used for categorical coloring within responses.
        - response_colors : list of color or dict, optional
            Base colors for responses if `hue` is not provided.
            If a list is given, it is mapped onto `response` in order.
            If a dict is given, keys should be response names and values
            valid Matplotlib colors. If not provided, a default rainbow
            palette is generated.
        - subplot_size : (float, float), default (5, 3)
            Width and height of individual subplots.
        - xaxis_offset : float, default 0.5
            Padding added to x-axis limits.
        - curve_kwargs : dict, optional
            Extra keyword arguments for fitted curve line plots.
        - threshold_kwargs : dict, optional
            Extra keyword arguments for threshold KDE plots.
        - annotation_offset : int, optional
            Column offset for placing annotation titles.

        MEP data
        --------
        mep_array : ndarray, optional
            Raw MEP waveform data aligned with rows of `df`. Shape is
            (points, timepoints, response), where points equal number of rows in `df`.
        mep_response : list of str, optional
            Order of responses in `mep_array` (e.g., muscle names).
            Used to align responses in the `mep_array` with `response`.
        mep_window : list of float, default [0, 1]
            Time window for slicing MEP data.
        mep_size_window : list of float, optional
            Sub-window of `mep_window` that was used for computing MEP size present in `df`.
            The MEP waveforms will be zoomed to this window for plotting.
            Note, this won't compute any MEP size. If None, defaults to `mep_window`.
        mep_adjust : float, default 1.0
            Scaling factor applied to normalize and adjust waveform amplitudes.

    Returns
    -------
    figures : list of (matplotlib.figure.Figure, np.ndarray of Axes)
        List of figure handles and their corresponding axes arrays, one per page
        of results when the number of combinations exceeds the row limit.

    Notes
    -----
    - If predictions are not provided, it only plots the dataset `df`.
    - Each row of subplots corresponds to one unique feature combination.
    - Each response variable expands the row into multiple subplots: raw MEPs,
      scatter, fitted curves, and threshold KDEs (depending on inputs).
    - Legends are only retained if `hue` is provided.
    """
    sort_key = kw.pop("sort_key", None)
    subplot_width, subplot_height = kw.pop("subplot_size", (5, 3))
    annotation_offset = kw.pop("annotation_offset", 0)

    colors = kw.pop("response_colors", [])
    if not colors:
        colors = sns.color_palette(palette="rainbow", as_cmap=True)(np.linspace(0, 1, len(response))) 
        colors = list(colors)
    if isinstance(colors, list):
        colors = dict(zip(response, colors))
    missing = [r for r in response if r not in colors]
    if missing:
        raise ValueError(
            f"response_colors dict is missing entries for responses: {missing}"
        )

    num_response = len(response)
    hue = kw.pop("hue", None)
    hue, hue_palette = _build_hue_palette(df, hue, num_response)

    # MEP data
    mep_array = kw.pop("mep_array", None)
    mep_response = kw.pop("mep_response", None)
    mep_window = kw.pop("mep_window", [0, 1])
    mep_size_window = kw.pop("mep_size_window", None)
    mep_adjust = kw.pop("mep_adjust", 1.)
    mep_xoffset = kw.pop('mep_xoffset', [0, 0])
    mep_yoffset = kw.pop('mep_yoffset', [0, 0])

    num_cols = 1
    mep_time = None
    mep_size_time = None
    if mep_array is not None:
        assert mep_array.shape[0] == df.shape[0]
        mep_time, mep_size_time = get_mep_data(
            mep_array,
            response=response,
            mep_response=mep_response,
            mep_window=mep_window,
            mep_size_window=mep_size_window,
        )
        num_cols += 1
        annotation_offset += 1

    if prediction_df is not None:
        assert prediction is not None
        pred_features = mep.make_features(prediction_df, features=features)
        if prediction_prob and prediction_hdi is None:
            prediction_hdi = hpdi(prediction, prob=prediction_prob)
        prediction = prediction.mean(axis=0, keepdims=True)

    if threshold is not None:
        if not len(features): threshold = threshold[:, None, ...]
        if threshold_hdi is None:
            threshold_hdi = hpdi(threshold, axis=0, prob=threshold_prob)
        num_cols += 1

    # Setup layout
    df_features = mep.make_features(df, features=features)
    combinations = df_features.unique().tolist()
    combinations = sorted(combinations, key=sort_key)
    num_combinations = len(combinations)
    num_response = len(response)
    num_cols *= num_response
    num_rows = 10
    num_pages = num_combinations // num_rows + (num_combinations % num_rows > 0)

    # Iterate over pages
    counter = 0
    figures = []
    for page in range(num_pages):
        num_rows_current = min(num_rows, num_combinations - page * num_rows)
        fig, axes = plt.subplots(
            nrows=num_rows_current,
            ncols=num_cols,
            figsize=(
                num_cols * subplot_width,
                num_rows_current * subplot_height
            ),
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
                ccpred_df = (
                    prediction_df[pred_idx].reset_index(drop=True).copy()
                )
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
                mep_size_time=mep_size_time,
                mep_adjust=mep_adjust,
                mep_xoffset=mep_xoffset,
                mep_yoffset=mep_yoffset,
                prediction_df=ccpred_df,
                prediction=ccpred,
                prediction_hdi=ccpred_hdi,
                threshold=ccthresh,
                threshold_hdi=ccthresh_hdi,
                axes=axes[row, :],
                colors=colors,
                hue=hue,
                hue_palette=hue_palette,
                **kw
            )

            annotation = ", ".join(map(str, cc))
            annotation_inverse = ""
            if encoder is not None:
                ccinverse = invert_combination(cc, features, encoder)
                ccinverse = ", ".join(map(str, ccinverse))
                annotation_inverse += f"\n{ccinverse}"

            for r in range(num_response):
                ax = axes[
                    row, annotation_offset + r * (num_cols // num_response)
                ]
                ax.set_title(f"({annotation}, {r})" + annotation_inverse)

            counter += 1

        logger.info(f"Page {page + 1} of {num_pages} done.")
        figures.append([fig, axes])

    return figures
