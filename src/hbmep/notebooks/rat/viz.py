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
    model,
    subjects,
    subjects_inv,
    combinations,
    x,
    y_mean,
    y_hdi=None,
    g=None,
    scatterplot=None,
    auc=None
):
    df_features = df[model.features].apply(tuple, axis=1)
    colors = get_response_colors(model.response)
    response_absent = False
    if y_mean.shape[-1] != model.num_response:
        response_absent = True
        colors = ["k"] * y_mean.shape[-1]
    nr, nc = len(subjects), len(combinations)

    fig, axes = plt.subplots(
        nr, nc, figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True, sharex=True, sharey="row"
    )

    for i, subject_idx in enumerate(subjects):
        subject_inv = subjects_inv[i]
        for j, (combination_idx, combination_inv) in enumerate(combinations):
            ax = axes[i, j]
            ax.clear()
            combination = (subject_idx, combination_idx)
            idx = df_features.isin([combination])
            df_temp = df[idx].reset_index(drop=True).copy()
            for response_idx, response in enumerate(model.response):
                if response_idx >= y_mean.shape[-1]: break
                if scatterplot:
                    assert g is not None
                    offset_response = np.nanmean(
                        g[..., *combination, response_idx]
                    )
                    sns.scatterplot(
                        x=df_temp[model.intensity],
                        y=df_temp[response] - offset_response,
                        ax=ax,
                        color=colors[response_idx],
                        s=8
                    )
                idx = x <= x.max(); assert idx.sum() == x.shape[0]
                x_response = x[idx]
                y_response_mean = y_mean[:, *combination, response_idx]
                y_response_mean = y_response_mean[idx]
                sns.lineplot(
                    x=x_response,
                    y=y_response_mean,
                    ax=ax,
                    color=colors[response_idx],
                    label=response if not response_absent else None
                )
                if y_hdi is not None:
                    y_response_hdi = y_hdi[..., *combination, response_idx]
                    y_response_hdi = y_response_hdi[idx, ...]
                    ax.fill_between(
                        x_response,
                        y_response_hdi[:, 0],
                        y_response_hdi[:, 1],
                        color=colors[response_idx],
                        alpha=.4
                    )
            title = f"{combination_inv}"
            if not j: title = f"{subject_inv}, {combination_inv}"
            if auc is not None: title = f"{title}, auc:{auc[*combination]:.3f}"
            ax.set_title(title)
            if not i and not j and (not response_absent):
                ax.legend(loc="upper left")
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
    model,
    subjects,
    subjects_inv,
    combinations,
    x,
    y_unnorm,
    y_unnorm_mean,
    y_norm,
    p,
    plogp,
    entropy,
    g,
    auc
):
    out = []

    y_mean = y_unnorm_mean.copy()
    y_hdi = hpdi(y_unnorm, axis=1, prob=.95)
    print("Plotting y_unnorm...")
    fig = _viz(
        df=df,
        model=model,
        subjects=subjects,
        subjects_inv=subjects_inv,
        combinations=combinations,
        x=x,
        y_mean=y_mean,
        y_hdi=y_hdi,
        scatterplot=True,
        g=g,
        auc=auc
    )
    fig.suptitle("1. Unnormalized")
    out.append(fig)
    
    print("Plotting y_norm...")
    fig = _viz(
        df=df,
        model=model,
        subjects=subjects,
        subjects_inv=subjects_inv,
        combinations=combinations,
        x=x,
        y_mean=y_norm.copy(),
        y_hdi=None,
        scatterplot=False,
        g=None,
        auc=auc
    )
    fig.suptitle("2. Normalized")
    out.append(fig)

    print("Plotting p...")
    fig = _viz(
        df=df,
        model=model,
        subjects=subjects,
        subjects_inv=subjects_inv,
        combinations=combinations,
        x=x,
        y_mean=p.copy(),
        y_hdi=None,
        scatterplot=False,
        g=None,
        auc=auc
    )
    fig.suptitle("3. Proportion curves")
    out.append(fig)

    print("Plotting plogp...")
    fig = _viz(
        df=df,
        model=model,
        subjects=subjects,
        subjects_inv=subjects_inv,
        combinations=combinations,
        x=x,
        y_mean=plogp.copy(),
        y_hdi=None,
        scatterplot=False,
        g=None,
        auc=auc
    )
    fig.suptitle("4. plogp curves")
    out.append(fig)

    print("Plotting entropy...")
    fig = _viz(
        df=df,
        model=model,
        subjects=subjects,
        subjects_inv=subjects_inv,
        combinations=combinations,
        x=x,
        y_mean=entropy[..., None],
        y_hdi=None,
        scatterplot=False,
        g=None,
        auc=auc
    )
    fig.suptitle("5. Entropy curves")
    out.append(fig)

    return out
