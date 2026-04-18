import logging

import numpy as np
import pandas as pd
from numpyro.diagnostics import hpdi
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from hbmep.dataset import make_features
from hbmep.util import site, invert_combination

logger = logging.getLogger(__name__)

CURVE_KW = {"label": "Curve", "color": "k", "alpha": 0.9, "linewidth": 1.5}
THRESHOLD_KW = {"color": "green", "alpha": 0.4, "linewidth": 1.2}
THRESHOLD_LINE_KW = {"linestyle": "--", "alpha": 0.7, "linewidth": 1.2}
HDI_LINE_KW = {"linestyle": "--", "color": "black", "alpha": 0.4, "linewidth": 1.0}


def _build_hue_palette(
    df: pd.DataFrame,
    hue: list[str] | str | None,
    num_response: int,
):
    if hue is None:
        return [None] * num_response, {}

    if isinstance(hue, str):
        hue = [hue] * num_response

    unique = pd.unique(df[hue].values.ravel())
    colors = sns.color_palette(palette="tab10", n_colors=len(unique))
    palette = {u: v for u, v in zip(unique, colors)}
    return hue, palette


def _default_response_colors(response: list[str]):
    colors = sns.color_palette(
        palette="rainbow",
        as_cmap=True,
    )(np.linspace(0, 1, len(response)))
    return dict(zip(response, list(colors)))


def _get_prediction_chunk(
    predictive: dict | None,
    var: str | None,
    pred_idx,
):
    if predictive is None or var is None:
        return None
    if var not in predictive:
        return None
    return predictive[var][:, pred_idx, :]


def _get_threshold_chunk(
    posterior: dict | None,
    threshold_var: str | None,
    combination,
    num_features: int,
):
    if posterior is None or threshold_var is None:
        return None
    if threshold_var not in posterior:
        return None

    z = posterior[threshold_var]
    if num_features == 0:
        z = z[:, None, :]
        return z[:, 0, :]
    return z[:, *combination, :]


def get_mep_data(
    mep_array: np.ndarray,
    *,
    response: list[str] | None = None,
    mep_response: list[str] | None = None,
    mep_window: list[float] = [0, 1],
    mep_size_window: list[float] | None = None,
    **kw,
):
    idx = [r for r in range(mep_array.shape[-1])]
    if (
        response is not None
        and mep_response is not None
        and mep_response != response
    ):
        idx = [r for r, res in enumerate(mep_response) if res in response]
    mep_array = mep_array[..., idx]

    if mep_size_window is None:
        mep_size_window = mep_window

    assert (
        mep_size_window[0] >= mep_window[0]
        and mep_size_window[1] <= mep_window[1]
    )

    mep_time = np.linspace(*mep_window, mep_array.shape[1])
    mep_size_time = (
        (mep_time > mep_size_window[0])
        & (mep_time < mep_size_window[1])
    )
    return mep_time, mep_size_time


def mep_plotter(
    mep_array: np.ndarray,
    intensity: np.ndarray,
    mep_time: np.ndarray | None = None,
    mep_size_time: np.ndarray | None = None,
    mep_adjust: float = 1.0,
    mep_xoffset: list[float] | None = None,
    mep_yoffset: list[float] | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots(1, 1)

    if mep_time is None:
        mep_time = np.linspace(0, 1, mep_array.shape[1])

    if mep_size_time is None:
        mep_size_time = np.ones(mep_array.shape[1], dtype=bool)

    if mep_xoffset is None:
        mep_xoffset = [0, 0]

    if mep_yoffset is None:
        mep_yoffset = [0, 0]

    max_amplitude = np.nanmax(mep_array[:, mep_size_time], keepdims=True)
    if np.all(np.isnan(max_amplitude)) or np.all(max_amplitude == 0):
        max_amplitude = 1.0
    mep_array = mep_array / max_amplitude
    mep_array = mep_array * mep_adjust

    for i in range(mep_array.shape[0]):
        x = mep_array[i, :] + intensity[i]
        if not np.isnan(x).all():
            ax.plot(x, mep_time, **kwargs)

    lo, hi = intensity.min() + mep_xoffset[0], intensity.max() + mep_xoffset[1]
    ax.set_xlim(lo, hi)

    lo, hi = mep_time[mep_size_time].min(), mep_time[mep_size_time].max()
    ax.axhline(lo, color="r", zorder=int(1e9))
    ax.axhline(hi, color="r", zorder=int(1e9))
    ax.set_ylim(lo + mep_yoffset[0], hi + mep_yoffset[1])

    return ax


def _plot_main_panel(
    *,
    ax: plt.Axes,
    df: pd.DataFrame,
    intensity: str,
    response_name: str,
    color,
    hue_name: str | None,
    hue_palette: dict | None,
    xoffset: list[float],
    yscale: str | None,
    prediction_df: pd.DataFrame | None,
    curve_chunk: np.ndarray | None,
    hdi_chunk: np.ndarray | None,
    predictive_hdi_prob: float,
    threshold_chunk: np.ndarray | None,
    threshold_overlay: bool,
    curve_kwargs: dict,
    threshold_kwargs: dict,
):
    if hue_name is not None and hue_palette:
        present = sorted(pd.unique(df[hue_name]).tolist())
        palette = {k: hue_palette[k] for k in present if k in hue_palette}
        sns.scatterplot(
            ax=ax,
            data=df,
            x=intensity,
            y=response_name,
            hue=hue_name,
            palette=palette,
            legend=False,
        )
    else:
        sns.scatterplot(
            ax=ax,
            data=df,
            x=intensity,
            y=response_name,
            color=color,
            legend=False,
        )

    lo, hi = df[intensity].min(), df[intensity].max()
    ax.set_xlim(left=lo + xoffset[0], right=hi + xoffset[1])
    ax.set_xlabel("")
    ax.set_ylabel("")

    if yscale is not None:
        ax.set_yscale(yscale)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    if prediction_df is None or curve_chunk is None:
        return

    if predictive_hdi_prob > 0 and hdi_chunk is not None:
        band = hpdi(hdi_chunk, axis=0, prob=predictive_hdi_prob)
        ax.fill_between(
            prediction_df[intensity].values,
            band[0, :, 0] if band.ndim == 3 and band.shape[-1] == 1 else band[0, :,],
            band[1, :, 0] if band.ndim == 3 and band.shape[-1] == 1 else band[1, :,],
            color=color,
            alpha=0.15,
        )

    curve_mean = curve_chunk.mean(axis=0)
    if curve_mean.ndim == 2:
        y = curve_mean[:, 0]
    else:
        y = curve_mean
    sns.lineplot(
        x=prediction_df[intensity].values,
        y=y,
        ax=ax,
        **curve_kwargs,
    )

    if threshold_overlay and threshold_chunk is not None:
        point_thresh = threshold_chunk.mean(axis=0)
        if np.ndim(point_thresh) > 0:
            point_thresh = point_thresh[0]
        ax2 = ax.twinx()
        sns.kdeplot(
            x=(
                threshold_chunk[:, 0] if threshold_chunk.ndim == 2
                else threshold_chunk
            ),
            ax=ax2,
            **threshold_kwargs
        )
        ax2.set_ylim(0, ax2.get_ylim()[1])
        ax2.set_yticks([])
        ax2.set_ylabel("")
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.patch.set_alpha(0)


def _plot_threshold_panel(
    *,
    ax: plt.Axes,
    threshold_chunk: np.ndarray | None,
    threshold_hdi_prob: float,
    color,
    threshold_kwargs: dict,
):
    if threshold_chunk is None:
        ax.axis("off")
        return

    x = threshold_chunk[:, 0] if threshold_chunk.ndim == 2 else threshold_chunk
    sns.kdeplot(x=x, ax=ax, **threshold_kwargs)

    point_thresh = np.mean(x)
    ax.axvline(point_thresh, color=color, label="Point estimate", **THRESHOLD_LINE_KW)

    if threshold_hdi_prob > 0:
        hdi = hpdi(x, prob=threshold_hdi_prob)
        ax.axvline(hdi[0], label=f"{int(100 * threshold_hdi_prob)}% HPDI", **HDI_LINE_KW)
        ax.axvline(hdi[1], **HDI_LINE_KW)

    if ax.get_legend():
        ax.get_legend().remove()

    ax.set_xlabel("")
    ax.set_ylabel("")


def plot(
    df: pd.DataFrame,
    *,
    intensity: str,
    features: list[str],
    response: list[str],
    encoder: dict[str, LabelEncoder] | None = None,
    predictive: dict | None = None,
    posterior: dict | None = None,
    prediction_df: pd.DataFrame | None = None,
    predictive_var: str = site.mu,
    predictive_hdi_var: str | None = None,
    predictive_hdi_prob: float = 0.0,
    threshold_var: str | None = site.a,
    threshold_hdi_prob: float = 0.95,
    **kw,
):
    """
    Plot MEP sizes, optional traces, estimated curves, predictive HDIs,
    and threshold posteriors.

    Layout is determined from the keyword arguments.
    """
    sort_key = kw.pop("sort_key", None)
    subplot_width, subplot_height = kw.pop("subplot_size", (4, 2.8))
    num_page_rows = kw.pop("num_page_rows", 10)

    xoffset = kw.pop("xoffset", [-0.5, 0.5])
    curve_kwargs = kw.pop("curve_kwargs", CURVE_KW.copy())
    threshold_kwargs = kw.pop("threshold_kwargs", THRESHOLD_KW.copy())
    trace_kwargs = kw.pop("trace_kwargs", {})
    yscale = kw.pop("yscale", None)

    # Optional MEP trace data
    mep_array = kw.pop("mep_array", None)
    mep_response = kw.pop("mep_response", None)
    mep_window = kw.pop("mep_window", [0, 1])
    mep_size_window = kw.pop("mep_size_window", None)
    mep_adjust = kw.pop("mep_adjust", 1.0)
    mep_xoffset = kw.pop("mep_xoffset", [0, 0])
    mep_yoffset = kw.pop("mep_yoffset", [0, 0])

    # Optional hue
    hue = kw.pop("hue", None)

    # Colors
    colors = kw.pop("response_colors", None)
    if colors is None:
        colors = _default_response_colors(response)
    elif isinstance(colors, list):
        colors = dict(zip(response, colors))

    missing = [r for r in response if r not in colors]
    if missing:
        raise ValueError(
            f"response_colors dict is missing entries for responses: {missing}"
        )

    num_response = len(response)
    hue, hue_palette = _build_hue_palette(df, hue, num_response)

    # Determine what to show
    show_traces = mep_array is not None
    show_curves = (
        prediction_df is not None
        and predictive is not None
        and predictive_var in predictive
    )
    show_threshold = (
        posterior is not None
        and threshold_var is not None
        and threshold_var in posterior
    )

    # Layout
    if show_traces and show_threshold:
        kc = 3
        width_ratios = [0.5, 0.5, 0.5]
    elif show_traces:
        kc = 2
        width_ratios = [0.5, 0.5]
    elif show_threshold:
        kc = 2
        width_ratios = [0.5, 0.5]
    else:
        kc = 1
        width_ratios = [1.0]

    kr = 1
    ncols = num_response * kc

    # MEP traces preprocessing
    mep_time = None
    mep_size_time = None
    if show_traces:
        assert mep_array.shape[0] == df.shape[0]
        mep_time, mep_size_time = get_mep_data(
            mep_array,
            response=response,
            mep_response=mep_response,
            mep_window=mep_window,
            mep_size_window=mep_size_window,
        )

    # Combinations
    df_features = make_features(df, features=features)
    combinations = sorted(df_features.unique().tolist(), key=sort_key)
    num_combinations = len(combinations)
    num_pages = num_combinations // num_page_rows + (num_combinations % num_page_rows > 0)

    pred_features = None
    if prediction_df is not None:
        pred_features = make_features(prediction_df, features=features)

    figures = []
    counter = 0

    for page in range(num_pages):
        num_rows_current = min(num_page_rows, num_combinations - page * num_page_rows)

        fig, axes = plt.subplots(
            nrows=num_rows_current * kr,
            ncols=ncols,
            figsize=(ncols * subplot_width, num_rows_current * subplot_height),
            constrained_layout=True,
            squeeze=False,
            gridspec_kw={"width_ratios": width_ratios * num_response},
        )

        for row in range(num_rows_current):
            combination = combinations[counter]

            df_idx = df_features.isin([combination])
            curr_df = df[df_idx].reset_index(drop=True).copy()

            curr_mep = None
            if show_traces:
                curr_mep = mep_array[df_idx, ...]

            curr_pred_df = None
            pred_idx = None
            if prediction_df is not None:
                pred_idx = pred_features.isin([combination])
                curr_pred_df = prediction_df[pred_idx].reset_index(drop=True).copy()

            row_main_axes = []

            for j, response_name in enumerate(response):
                base = j * kc
                color = colors[response_name]
                hue_name = hue[j] if hue is not None else None

                if show_curves:
                    curve_chunk = _get_prediction_chunk(
                        predictive,
                        predictive_var,
                        pred_idx,
                    )
                    if curve_chunk is not None:
                        curve_chunk = curve_chunk[..., j:j+1]
                else:
                    curve_chunk = None

                if show_curves:
                    hdi_chunk = _get_prediction_chunk(
                        predictive,
                        predictive_hdi_var,
                        pred_idx,
                    )
                    if hdi_chunk is not None:
                        hdi_chunk = hdi_chunk[..., j:j+1]
                else:
                    hdi_chunk = None

                threshold_chunk = _get_threshold_chunk(
                    posterior,
                    threshold_var,
                    combination,
                    num_features=len(features),
                )
                if threshold_chunk is not None:
                    threshold_chunk = threshold_chunk[..., j:j+1]

                # Traces panel
                if show_traces:
                    ax_trace = axes[row * kr, base]
                    mep_plotter(
                        mep_array=curr_mep[..., j],
                        intensity=curr_df[intensity].values,
                        mep_time=mep_time,
                        mep_size_time=mep_size_time,
                        mep_adjust=mep_adjust,
                        mep_xoffset=mep_xoffset,
                        mep_yoffset=mep_yoffset,
                        ax=ax_trace,
                        color=color,
                        alpha=0.4,
                        **trace_kwargs,
                    )
                    ax_trace.set_xlabel("")
                    ax_trace.set_ylabel(response_name)

                # Main panel
                main_col = base + 1 if show_traces else base
                ax_main = axes[row * kr, main_col]
                _plot_main_panel(
                    ax=ax_main,
                    df=curr_df,
                    intensity=intensity,
                    response_name=response_name,
                    color=color,
                    hue_name=hue_name,
                    hue_palette=hue_palette,
                    xoffset=xoffset,
                    yscale=yscale,
                    prediction_df=curr_pred_df,
                    curve_chunk=curve_chunk,
                    hdi_chunk=hdi_chunk,
                    predictive_hdi_prob=predictive_hdi_prob,
                    threshold_chunk=threshold_chunk,
                    threshold_overlay=show_threshold,
                    curve_kwargs=curve_kwargs,
                    threshold_kwargs=threshold_kwargs,
                )
                ax_main.set_xlabel("")
                ax_main.set_ylabel(response_name if not show_traces else "")
                ax_main.tick_params(axis="x", rotation=90)
                row_main_axes.append(ax_main)

                # Threshold panel
                if show_threshold:
                    thresh_col = base + 2 if show_traces else base + 1
                    ax_thresh = axes[row * kr, thresh_col]
                    _plot_threshold_panel(
                        ax=ax_thresh,
                        threshold_chunk=threshold_chunk,
                        threshold_hdi_prob=threshold_hdi_prob,
                        color=color,
                        threshold_kwargs=threshold_kwargs,
                    )
                    ax_thresh.set_xlabel("")
                    ax_thresh.set_ylabel("")

            annotation = ", ".join(map(str, combination))
            annotation_inverse = ""
            if encoder is not None:
                try:
                    ccinverse = invert_combination(combination, features, encoder)
                    ccinverse = ", ".join(map(str, ccinverse))
                    annotation_inverse = f"\n{ccinverse}"
                except Exception:
                    pass

            for j in range(num_response):
                base = j * kc
                main_col = base + 1 if show_traces else base
                ax = axes[row * kr, main_col]
                ax.set_title(f"({annotation}, {j})" + annotation_inverse)

            if len(row_main_axes) > 1:
                ymin, ymax = np.inf, -np.inf
                for ax in row_main_axes:
                    lo, hi = ax.get_ylim()
                    ymin = min(ymin, lo)
                    ymax = max(ymax, hi)
                for ax in row_main_axes[1:]:
                    ax.sharey(row_main_axes[0])
                row_main_axes[0].set_ylim(ymin, ymax)

            counter += 1

        logger.info(f"Page {page + 1} of {num_pages} done.")
        figures.append([fig, axes])

    return figures
