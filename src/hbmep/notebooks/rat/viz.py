import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpyro.diagnostics import hpdi

import hbmep as mep
from hbmep import functional as F
from hbmep.util import site

from hbmep.notebooks.rat.util import get_response_colors


def _viz(
    *,
    df,
    encoder,
    model,
    combinations,
    x,
    y_mean,
    y_hdi=None,
    g=None,
    scatterplot=None,
    auc=None
):
    subjects = sorted(df[model.features[0]].unique())
    df_features = df[model.features].apply(tuple, axis=1)
    colors = get_response_colors(model.response)
    nr, nc = len(subjects), len(combinations)

    fig, axes = plt.subplots(
        nr, nc, figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True, sharex=True, sharey="row"
    )

    for i, subject in enumerate(subjects):
        subject_inv = encoder[model.features[0]].inverse_transform([subject])[0]
        for j, combination in enumerate(combinations):
            combination_inv = [encoder[feature].inverse_transform([f])[0] for feature, f in zip(model.features[1:], combination)]
            combination_inv = tuple(combination)
            ax = axes[i, j]
            ax.clear()
            idx = df_features.isin([(subject, *combination)])
            df_temp = df[idx].reset_index(drop=True).copy()
            for response_idx, response in enumerate(model.response):
                # if response_idx not in test: continue
                if scatterplot:
                    assert g is not None
                    offset_response = np.nanmean(g[:, :, subject, *combination, response_idx], axis=1)
                    sns.scatterplot(x=df_temp[model.intensity], y=df_temp[model.response[response_idx]] - offset_response, ax=ax, color=colors[response_idx], s=8)

                # idx = x < df_temp[model.intensity].max()
                idx = x <= x.max(); assert idx.sum() == x.shape[0]
                x_response = x[idx]
                y_response_mean = y_mean[:, subject, *combination, response_idx]
                y_response_mean = y_response_mean[idx]
                sns.lineplot(x=x_response, y=y_response_mean, ax=ax, color=colors[response_idx], label=response)
                if y_hdi is not None:
                    y_response_hdi = y_hdi[:, :, subject, *combination, response_idx]
                    y_response_hdi = y_response_hdi[idx, :]
                    ax.fill_between(x_response, y_response_hdi[:, 0], y_response_hdi[:, 1], color=colors[response_idx], alpha=.4)
            title = f"{combination_inv}"
            if not j: title = f"{subject_inv}, {combination_inv}"
            if auc is not None: title = f"{title}: {auc[subject, *combination]:.3f}"
            ax.set_title(title)
            if not i and not j: ax.legend(loc="upper left")
            elif ax.get_legend(): ax.get_legend().remove()

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelbottom=True)
        
    return fig


def viz_selectivity(
    *,
    df,
    encoder, model, combinations, x,
    y_unnorm,
    y_norm,
    g,
    auc
):
    out = []

    y_mean = np.nanmean(y_unnorm, axis=1)
    y_hdi = hpdi(y_unnorm, axis=1, prob=.95)
    scatterplot = True
    print("Plotting y_unnorm...")
    fig = _viz(
        df=df,
        encoder=encoder,
        model=model,
        combinations=combinations,
        x=x,
        y_mean=y_mean,
        y_hdi=y_hdi,
        scatterplot=scatterplot,
        g=g,
    )
    fig.suptitle("Unnormalized")
    out.append(fig)
    
    y_mean = y_norm.copy()
    y_hdi = None
    scatterplot = False
    print("Plotting y_norm...")
    fig = _viz(
        df=df,
        encoder=encoder,
        model=model,
        combinations=combinations,
        x=x,
        y_mean=y_mean,
        y_hdi=y_hdi,
        scatterplot=scatterplot,
        g=g,
        auc=auc
    )
    fig.suptitle("Normalized")
    out.append(fig)
    return out
