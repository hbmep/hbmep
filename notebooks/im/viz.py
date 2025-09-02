import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpyro.diagnostics import hpdi
from hbmep.model import BaseModel

from constants import BUILD_DIR, DATA_PATH, CONFIG
from hbmep.notebooks.util import (
    make_pdf, load_model, clear_axes, turn_off_ax
)

from util import Site as site

BUILD_DIR = os.path.join(BUILD_DIR, "out")
os.makedirs(BUILD_DIR, exist_ok=True)

NUM_POINTS = 4_000
# NUM_POINTS = 200


def plot(model_dir):
    df, encoder, posterior, model, mcmc, = load_model(model_dir)
    if site.outlier_prob in posterior.keys():
        posterior[site.outlier_prob] *= 0
    pred_df = model.make_prediction_dataset(df, num_points=NUM_POINTS)
    predictive = model.predict(pred_df, posterior=posterior)
    mu = predictive[site.mu]
    mu_hdi = hpdi(mu, axis=0, prob=.95)

    features = df[model.features].apply(tuple, axis=1)
    pred_features = pred_df[model.features].apply(tuple, axis=1)
    combinations = features.unique().tolist()
    combinations = sorted(combinations)
    combinations_inv = [
        (encoder[model.features[0]].inverse_transform([u])[0],)
        for u, in combinations
    ]
    combinations_inv = [(f"{u * 100}%",) for u, in combinations_inv]
    num_combinations = len(combinations)
    colors = sns.color_palette(palette="viridis", n_colors=num_combinations)

    nr, nc = 6, 4
    fig, axes = plt.subplots(
        *(nr, nc), figsize=(3 * nc, 2.2 * nr), squeeze=False,
        constrained_layout=True
    )

    clear_axes(axes)
    counter = 0
    for i, combination in enumerate(combinations):
        combination_inv = combinations_inv[i][0]
        idx = (counter // nc, counter % nc)
        ax = axes[*idx]
        idx = features.isin([combination])
        ccdf = df[idx].reset_index(drop=True).copy()
        x = ccdf.conc
        y = ccdf.y
        color = colors[counter]
        sns.scatterplot(x=x, y=y, ax=ax, color=color)
        idx = pred_features.isin([combination])
        ccpreddf = pred_df[idx].reset_index(drop=True).copy()
        x_pred = ccpreddf[model.intensity]
        y_pred = mu[..., 0][..., idx]
        y_pred = np.mean(y_pred, axis=0)
        y_pred_hdi = mu_hdi[..., 0][:, idx]
        linestyle = "--" if counter % 2 else "-"
        sns.lineplot(
            x=x_pred, y=y_pred, ax=ax, color=color, linestyle=linestyle
        )
        ax.fill_between(
            x_pred, y_pred_hdi[0], y_pred_hdi[1], alpha=.4, color=color
        )
        ax.set_title(combination_inv)
        ax.sharex(axes[0, 0])
        ax.sharey(axes[0, 0])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xscale("log")
        counter += 1
    ax = axes[0, 0]
    ax.set_ylim(top=1.1 * df[model.response[0]].max())
    while counter % nc:
        idx = (counter // nc, counter % nc)
        ax = axes[*idx]
        turn_off_ax(ax)
        counter += 1

    x = [u for u, in combinations]
    xinv = [encoder[model.features[0]].inverse_transform([u])[0] for u in x]
    xinv_labels = [f"{u * 100}%" for u in xinv]
    clear_axes(axes[2:3, :])
 

    def body_plot(axes, named_params):
        for j, named_param in enumerate(named_params):
            if named_param in posterior:
                param = posterior[named_param][..., 0]
            elif named_param in predictive:
                param = predictive[named_param][..., 0]
            else:
                raise ValueError
            # param_hdi = hpdi(param, axis=0, prob=.95)
            param = np.median(param, axis=0)
            # err_lower = param - param_hdi[0]
            # err_upper = param_hdi[1] - param
            # sns.scatterplot(x=xinv, y=param, ax=ax)
            ax = axes[0, j]
            sns.scatterplot(x=x, y=param, ax=ax)
            ax.set_xticks(x)
            ax.set_xticklabels(xinv_labels, rotation=90, fontsize=6)
            ax.sharex(axes[0, 0])
            ax.set_title(named_param)

            ax = axes[1, j]
            sns.scatterplot(x=xinv, y=param, ax=ax)
            ax.set_xticks(xinv)
            ax.set_xticklabels(xinv_labels, rotation=90, fontsize=6)
            ax.sharex(axes[1, 0])
            ax.set_title(named_param)
   
    clear_axes(axes[2:])
    named_params = [site.b1.log, site.b2.log, site.b3.log, site.b4.log]
    body_plot(axes[2:4], named_params)   
    named_params = [site.b1, site.b2, site.b3, site.b4]
    body_plot(axes[4:6], named_params)

    title = model.run_id
    title += f", {model._model.__name__}"
    fig.suptitle(title)
    return (fig, axes),
    

def main():
    model_dirs = [
        "/home/vishu/reports/hbmep/notebooks/im/2025/070825_plate/10000w_10000s_4c_10t_15d_95a_fm/non_hierarchical",
        "/home/vishu/reports/hbmep/notebooks/im/2025/070825_plate/10000w_10000s_4c_10t_15d_95a_fm/hierarchical_sharedb1b4"
    ]
    out = []
    for model_dir in model_dirs:
        (fig, axes), = plot(model_dir)
        out.append(fig)

    output_path = os.path.join(BUILD_DIR, "out.pdf")
    make_pdf(out, output_path)


if __name__ == "__main__":
    main()
