import os
import pickle
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from jax import random, numpy as jnp

import hbmep as mep
from hbmep.util import timing, setup_logging, get_response_colors

from models import ImmunoModel
import functional as RF
from utils import Site as site
from constants import (
    HOME,
    DATA_PATH,
    BUILD_DIR,
)

MODEL_DIR = f"{HOME}/reports/hbmep/notebooks/im/sheet1/nhb_logistic4"
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
logger = logging.getLogger(__name__)


@timing
def main(model):
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

    # output_path = os.path.join(model.build_dir, "original_curves.pdf")
    # model.plot_curves(
    #     odf, prediction_df=opred_df, predictive=predictive, encoder=encoder, prediction_prob=.95, output_path=output_path
    # )

    df_features = df[model.features].apply(tuple, axis=1)
    pred_features = prediction_df[model.features].apply(tuple, axis=1)
    combinations = df_features.unique().tolist()
    combinations = sorted(combinations)
    combinations_inverse = [
        tuple(
            encoder[model.features[i]].inverse_transform([c])[0]
            for i, c in enumerate(combination)
        )
        for combination in combinations
    ]
    combinations_inverse
    num_combinations = len(combinations)
    colors = get_response_colors(num_combinations)

    mu = predictive[site.mu][..., 0]
    from numpyro.diagnostics import hpdi
    mu_hdi = hpdi(mu, axis=0, prob=.95)

    nr, nc = 3, 4
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
        ax.fill_between(
            ccpred_df[model.intensity],
            ccpred_hdi[0, :],
            ccpred_hdi[1, :],
            color="cyan",
            alpha=.4
        )
        sns.scatterplot(x=ccdf[model.intensity], y=ccdf[model.response[0]], color="b", ax=ax)
        sns.lineplot(x=ccpred_df[model.intensity], y=ccpred.mean(axis=0), ax=ax, color=colors[c])
        ax.sharex(axes[0, 0])
        ax.sharey(axes[0, 0])
        counter += 1
    x = [c[0] for c in combinations]
    assert x == sorted(x)
    xlabels = [c_inv[0][2:] for c_inv in combinations_inverse]
    xlabels = [u.replace(",", "") for u in xlabels]
    xlabels = list(map(int, xlabels))
    xlabels
    x = np.log10(xlabels)
    x
    posterior[site.b1].shape
    for i, named_param in enumerate(named_params):
        ax = axes[-1, i]
        ax.clear()
        param = posterior[named_param][..., 0].mean(axis=0)
        sns.scatterplot(x=x, y=param, ax=ax, color=colors)
        ax.set_title(named_param)
        if not i: ax.set_xlabel("log10(dilution)")
        ax.sharex(axes[-1, 0])
    for c, combination in enumerate(combinations):
        ax = axes[counter // nc, counter % nc]
        ax.clear()
        idx = df_features.isin([combination])
        occdf = odf[idx].reset_index(drop=True).copy()
        idx = pred_features.isin([combination])
        occpred_df = opred_df[idx].reset_index(drop=True).copy()
        occpred = omu[:, idx]
        occpred_hdi = omu_hdi[:, idx]
        ax.fill_between(
            occpred_df[model.intensity],
            occpred_hdi[0, :],
            occpred_hdi[1, :],
            color="cyan",
            alpha=.4
        )
        sns.scatterplot(x=occdf[model.intensity], y=occdf[model.response[0]], color="b", ax=ax)
        sns.lineplot(x=occpred_df[model.intensity], y=ccpred.mean(axis=0), ax=ax, color=colors[c])
        ax.sharex(axes[0, 0])
        ax.sharey(axes[0, 0])
        counter += 1
    fig.show()

    plt.close()
    # output_path = os.path.join(model.build_dir, "scatter.png")
    # fig.savefig(output_path, dpi=600)
    # logger.info(f"Saved to {output_path}")
    return


if __name__ == "__main__":
    with open(MODEL_FILE, "rb") as f: model, = pickle.load(f)
    setup_logging(model.build_dir)
    main(model)
