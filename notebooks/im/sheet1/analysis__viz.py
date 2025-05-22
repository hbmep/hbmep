import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from jax import random, numpy as jnp
from numpyro.diagnostics import hpdi
import hbmep as mep
from hbmep.util import (
    timing,
    generate_response_colors,
    invert_combination
)

from hbmep.notebooks.util import (
    clear_axes,
    make_pdf,
)
from models import nHB
from util import Site as site, load_model
from constants import (
    BUILD_DIR,
    MAPPING,
)

# NUM_POINTS = 4000
NUM_POINTS = 200


def load(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
        *_,
    ) = load_model(model_dir)
    prediction_df = model.make_prediction_dataset(df, num_points=NUM_POINTS)
    if site.outlier_prob in posterior.keys(): posterior[site.outlier_prob] *= 0
    predictive = model.predict(prediction_df, posterior=posterior)
    df_features = df[model.features].apply(tuple, axis=1)
    pred_features = prediction_df[model.features].apply(tuple, axis=1)
    combinations = df_features.unique().tolist()
    combinations = sorted(combinations)
    combinations_inverse = [
        invert_combination(c, model.features, encoder)
        for c in combinations
    ]
    combinations = list(zip(combinations, combinations_inverse))
    return (
        df,
		encoder,
		model,
		posterior,
		prediction_df,
		predictive,
		df_features,
		pred_features,
		combinations,
    )


def viz_group_by_plate(model_dir, point_estimates=True):
    (
        df,
		encoder,
		model,
		posterior,
		prediction_df,
		predictive,
		df_features,
		pred_features,
		combinations,
    ) = load(model_dir)
    mu = predictive[site.mu][..., 0]
    colors = sns.color_palette(palette="viridis", n_colors=3)

    t = sorted(df[model.features[0]].unique())
    t_inv = encoder[model.features[0]].inverse_transform(t)
    # t_inv = (
    #     pd.Series(t_inv).replace(MAPPING).to_numpy()
    # )
    feature0 = list(zip(t, t_inv))

    nr, nc = 5, 4
    size = (4 * nc, 1.5 * nr)
    sharey = "row" if point_estimates else None
    sharex = None if not point_estimates else None
    fig, axes = plt.subplots(
        nr, nc, figsize=size, constrained_layout=True, squeeze=False, height_ratios=[1., .5, .5, .5, .5], sharey=sharey, sharex=sharex
    )

    statistic = np.mean
    statistic = np.median
    named_params = [site.b1, site.b2, site.b3.log, site.b4]

    def body_map(contam, colors):
        if contam == "0":
            color = colors[0]
            label = "0:1"
        else:
            num_zeros = int(contam.split("-")[1])
            if num_zeros % 2: color = colors[2]
            else: color = colors[1]
            if num_zeros >= 6:
                label = f"{int(10 ** num_zeros // 1e6)}M"
            elif num_zeros >= 3:
                label = f"{int(10 ** num_zeros // 1e3)}K"
            else:
                label = f"{int(10 ** num_zeros)}"
        return label, color

    clear_axes(axes)
    for f0_idx, f0_inv in feature0:
        combination = (f0_idx,)
        contam = f0_inv.split("__")[1]
        plate = f0_inv.split("__")[2]
        label, color = body_map(contam, colors)
        ax = axes[0, int(plate[-1]) - 1]
        idx = df_features.isin([combination])
        ccdf = df[idx].reset_index(drop=True).copy()
        idx = pred_features.isin([combination])
        ccpred_df = prediction_df[idx].reset_index(drop=True).copy()
        ccpred = mu[:, idx]
        sns.scatterplot(x=ccdf[model.intensity], y=ccdf[model.response[0]], color=color, edgecolor="w", ax=ax)
        sns.lineplot(x=ccpred_df[model.intensity], y=statistic(ccpred, axis=0), ax=ax, color=color, label=label)
        ax.sharex(axes[0, 0])

    for i, named_param in enumerate(named_params):
        for plate in ["P1", "P2", "P3", "P4"]:
            ax = axes[i + 1, int(plate[-1]) - 1]; ax.clear()
            idx = [(u, v.split("__")[1]) for u, v in feature0 if plate in v]
            assert len(idx) == 3
            idx, contam = map(list, zip(*idx))
            param = statistic(posterior[named_param][..., 0], axis=0)[idx]
            contam = [body_map(u, colors) for u in contam]
            contam, color = map(list, zip(*contam))
            if point_estimates:
                sns.lineplot(x=contam, y=param, ax=ax, color="k", alpha=.4, linestyle="--")
                sns.scatterplot(x=contam, y=param, ax=ax, color=color, zorder=10)
                if plate == "P1": ax.set_ylabel(f"{named_param}\n({statistic.__name__})")
                ax.sharex(axes[1, int(plate[-1]) - 1])
            else:
                raise ValueError

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.set_xlabel("")
            if not point_estimates and i > 0:
                ax.tick_params(axis="y", left=False, labelleft=False)
            if not point_estimates and i in {1, 4}:
                ax.sharex(axes[i, 0])
            ax.autoscale(axis="x")
            # sides = ["top", "right"]
            # ax.spines[sides].set_visible(False)

    for j in range(nc):
        ax = axes[0, j]
        ax.legend(loc="upper left")
        if not j: ax.legend(title="Contaminant")
        ax.set_title(f"Plate {j + 1}")

    ax = axes[0, 0]
    ax.autoscale(axis="x")
    ax.set_xscale("log")
    ax.set_ylim(top=3)
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_ylabel("Optical density")
    ax.set_xlabel("Der p 1 concentration (ng/ml)")
    fig.align_xlabels()
    fig.align_ylabels()

    rest = model.run_id.split("-")[0]
    match rest:
        case "unrest": rest = "full range"
        case "rest": rest = "common range"
    title = f"grouped by plate, {model._model.__name__} model, with {rest} of concentration, use mixture: {model.use_mixture}"
    fig.suptitle(title)
    return (fig, axes),


@timing
def main():
    model_dirs = [
        "/home/vishu/reports/hbmep/notebooks/im/sheet1/unrest-notlogconc/10000w_10000s_4c_10t_15d_95a_tm/non_hierarchical",
    ]

    out = []
    out += [viz_group_by_plate(model_dir)[0][0] for model_dir in model_dirs]
    
    output_path = os.path.join(BUILD_DIR, "out.pdf")
    make_pdf(out, output_path)
    return


if __name__ == "__main__":
    main()
