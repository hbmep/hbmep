import os
import pickle
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from jax import random, numpy as jnp
from numpyro.diagnostics import hpdi

import hbmep as mep
from hbmep.util import timing, setup_logging, generate_response_colors, invert_combination

from models import ImmunoModel
import functional as RF
from utils import Site as site
from constants import (
    HOME,
    DATA_PATH,
    BUILD_DIR,
)

logger = logging.getLogger(__name__)


def viz_nhb_l4(model):
    src = os.path.join(model.build_dir, "inf.pkl")
    with open(src, "rb") as f:
        df, encoder, posterior = pickle.load(f)

    prediction_df = model.make_prediction_dataset(df)
    predictive = model.predict(prediction_df, posterior=posterior)
    predictive.keys()

    odf = df.copy()
    odf[model.intensity] = 2 ** odf[model.intensity]
    opred_df = prediction_df.copy()
    opred_df[model.intensity] = 2 ** opred_df[model.intensity]

    output_path = os.path.join(model.build_dir, "original_curves.pdf")
    model.plot_curves(
        odf, prediction_df=opred_df, predictive=predictive, encoder=encoder, prediction_prob=.95, output_path=output_path
    )

    df_features = df[model.features].apply(tuple, axis=1)
    pred_features = prediction_df[model.features].apply(tuple, axis=1)
    combinations = df_features.unique().tolist()
    combinations = sorted(combinations)
    combinations_inverse = [
        invert_combination(c, model.features, encoder)
        for c in combinations
    ]
    combinations_inverse
    num_combinations = len(combinations)
    colors = generate_response_colors(num_combinations)

    mu = predictive[site.mu][..., 0]
    mu_hdi = hpdi(mu, axis=0, prob=.95)

    nr, nc = 4, 4
    fig, axes = plt.subplots(
        nr, nc, figsize=(5 * nc, 3 * nr), constrained_layout=True, squeeze=False
    )
    counter = 0
    named_params = [site.b1, site.b2, site.b3, site.b4]
    for c, combination in enumerate(combinations):
        ax = axes[counter // nc, counter % nc]
        ax.clear()
        idx = df_features.isin([combination])
        ccdf = df[idx].reset_index(drop=True).copy()
        idx = pred_features.isin([combination])
        ccpred_df = prediction_df[idx].reset_index(drop=True).copy()
        ccpred = mu[:, idx]
        ccpred_hdi = mu_hdi[:, idx]
        # ax.fill_between(
        #     ccpred_df[model.intensity],
        #     ccpred_hdi[0, :],
        #     ccpred_hdi[1, :],
        #     color="cyan",
        #     alpha=.4
        # )
        sns.scatterplot(x=ccdf[model.intensity], y=ccdf[model.response[0]], color="b", ax=ax)
        sns.lineplot(x=ccpred_df[model.intensity], y=ccpred.mean(axis=0), ax=ax, color=colors[c])
        ax.sharex(axes[0, 0])
        ax.sharey(axes[0, 0])
        ax.set_title(
            f"contaminant 1:{combinations_inverse[c][0]}" if not c
            else f"1:{combinations_inverse[c][0]}"
        )
        counter += 1
    xdilution = [c[0] for c in combinations]
    assert xdilution == sorted(xdilution)
    xdilution_inverse = [c[0] for c in combinations_inverse]
    xdilution_inverse
    xdilution_inverse = [int(v[0]) if "P" in v else int(v[-1]) for v in xdilution_inverse]
    posterior[site.b1].shape
    for i, named_param in enumerate(named_params):
        ax = axes[counter // nc, counter % nc]
        ax.clear()
        param = posterior[named_param][..., 0].mean(axis=0)
        sns.scatterplot(x=xdilution_inverse, y=param, ax=ax, color=colors)
        ax.set_title(named_param)
        ax.sharex(axes[counter // nc, 0])
        ax.set_xticks(xdilution_inverse)
        counter += 1
    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.set_xlabel("")
            ax.set_ylabel("")
    ax = axes[0, 0]
    ax.set_ylim(top=3)
    ax.set_xlabel("log2(concentration)")
    ax.set_ylabel("optical density")
    ax = axes[-1, 0]
    ax.set_xlabel("log10(dilution)")
    # ax.set_xticks(xdilution)
    # ax.set_xticklabels([1 + xdil for xdil in xdilution])
    fig.show()

    plt.close()
    output_path = os.path.join(model.build_dir, "scatter.png")
    fig.savefig(output_path, dpi=600)
    logger.info(f"Saved to {output_path}")
    return


def viz_hb1_l4(model):
    src = os.path.join(model.build_dir, "inf.pkl")
    with open(src, "rb") as f:
        df, encoder, posterior = pickle.load(f)

    prediction_df = model.make_prediction_dataset(df)
    predictive = model.predict(prediction_df, posterior=posterior)
    predictive.keys()

    odf = df.copy()
    odf[model.intensity] = 2 ** odf[model.intensity]
    opred_df = prediction_df.copy()
    opred_df[model.intensity] = 2 ** opred_df[model.intensity]

    output_path = os.path.join(model.build_dir, "original_curves.pdf")
    model.plot_curves(
        odf, prediction_df=opred_df, predictive=predictive, encoder=encoder, prediction_prob=.95, output_path=output_path
    )

    df_features = df[model.features].apply(tuple, axis=1)
    pred_features = prediction_df[model.features].apply(tuple, axis=1)
    combinations = df_features.unique().tolist()
    combinations = sorted(combinations)
    combinations_inverse = [
        invert_combination(c, model.features, encoder)
        for c in combinations
    ]
    combinations_inverse
    num_combinations = len(combinations)
    colors = generate_response_colors(num_combinations)

    mu = predictive[site.mu][..., 0]
    mu_hdi = hpdi(mu, axis=0, prob=.95)

    plt.close()
    nr, nc = 5, 4
    fig, axes = plt.subplots(
        nr, nc, figsize=(5 * nc, 3 * nr), constrained_layout=True, squeeze=False
    )
    counter = 0
    named_params = [site.b1, site.b2, site.b3, site.b4]
    for c, combination in enumerate(combinations):
        ax = axes[counter // nc, counter % nc]
        ax.clear()
        idx = df_features.isin([combination])
        ccdf = df[idx].reset_index(drop=True).copy()
        idx = pred_features.isin([combination])
        ccpred_df = prediction_df[idx].reset_index(drop=True).copy()
        ccpred = mu[:, idx]
        ccpred_hdi = mu_hdi[:, idx]
        # ax.fill_between(
        #     ccpred_df[model.intensity],
        #     ccpred_hdi[0, :],
        #     ccpred_hdi[1, :],
        #     color="cyan",
        #     alpha=.4
        # )
        sns.scatterplot(x=ccdf[model.intensity], y=ccdf[model.response[0]], color="b", ax=ax)
        sns.lineplot(x=ccpred_df[model.intensity], y=ccpred.mean(axis=0), ax=ax, color=colors[c])
        ax.sharex(axes[0, 0])
        ax.sharey(axes[0, 0])
        ax.set_title(
            f"contaminant 1:{combinations_inverse[c][0]}" if not c
            else f"1:{combinations_inverse[c][0]}"
        )
        counter += 1
    xdilution = [c[0] for c in combinations]
    assert xdilution == sorted(xdilution)
    xdilution_inverse = [c[0] for c in combinations_inverse]
    xdilution_inverse
    xdilution_inverse = [int(v[0]) if "P" in v else int(v[-1]) for v in xdilution_inverse]
    posterior[site.b1].shape
    posterior.keys()
    for i, named_param in enumerate(named_params):
        ax = axes[counter // nc, counter % nc]
        ax.clear()
        param = posterior[named_param][..., 0].mean(axis=0)
        sns.scatterplot(x=xdilution_inverse, y=param, ax=ax, color=colors)
        ax.set_title(named_param)
        ax.sharex(axes[counter // nc, 0])
        ax.set_xticks(xdilution_inverse)
        counter += 1
    named_params = [site.b1.scale, site.b2.loc, site.b3.loc, site.b4.scale]
    for i, named_param in enumerate(named_params):
        ax = axes[counter // nc, counter % nc]
        ax.clear()
        param = posterior[named_param]
        sns.kdeplot(param, ax=ax)
        lo, hi = hpdi(param, prob=.95)
        ax.axvline(lo, linestyle="--", label="95% HDI")
        ax.axvline(hi, linestyle="--")
        ax.set_title(named_param)
        # ax.sharey(axes[counter // nc, 0])
        if not i: ax.legend(loc="upper right")
        if i and ax.get_legend(): ax.get_legend().remove()
        counter += 1
    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.set_xlabel("")
            ax.set_ylabel("")
    ax = axes[0, 0]
    ax.set_ylim(top=3)
    ax.set_xlabel("log2(concentration)")
    ax.set_ylabel("optical density")
    ax = axes[-2, 0]
    ax.set_xlabel("log10(dilution)")
    # ax.set_xticks(xdilution)
    # ax.set_xticklabels([1 + xdil for xdil in xdilution])
    fig.show()

    plt.close()
    output_path = os.path.join(model.build_dir, "hdi.png")
    fig.savefig(output_path, dpi=600)
    logger.info(f"Saved to {output_path}")
    return


@timing
def main(model):
    match model._model.__name__:
        case "nhb_l4": viz_nhb_l4(model)
        case "hb1_l4": viz_hb1_l4(model)
        case _: raise ValueError("Invalid model")
    return


if __name__ == "__main__":
    # MODEL_DIR = f"{HOME}/reports/hbmep/notebooks/im/sheet1/nhb_l4"
    MODEL_DIR = f"{HOME}/reports/hbmep/notebooks/im/sheet1/hb1_l4"
    MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
    with open(MODEL_FILE, "rb") as f: model, = pickle.load(f)
    setup_logging(model.build_dir)
    main(model)
