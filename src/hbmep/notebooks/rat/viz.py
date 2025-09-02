import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpyro.diagnostics import hpdi

import hbmep as mep
from hbmep import functional as F
from hbmep.util import site

from hbmep.util import generate_response_colors
from hbmep.notebooks.rat.util import get_response_colors


def _viz(
    *,
    df,
    model,
    subjects,
    combinations,
    x,
    yme,
    yhdi,
    g,
    auc,
    scatter=False,
    set_yticks=True,
    **kw
):
    a_min = kw.get("a_min", None)

    # plt.close("all")
    nr, nc = len(subjects), len(combinations)
    fig, axes = plt.subplots(
        *(nr, nc),
        figsize=(5 * nc, 3 * nr),
        squeeze=False,
        constrained_layout=True,
        # sharex=True,
        sharey="row"
    )

    response_absent = False
    # colors = get_response_colors(model.response)
    colors = generate_response_colors(model.num_response)
    if yme.shape[-1] != model.num_response:
        response_absent = True
        colors = ["k"] * yme.shape[-1]

    if scatter:
        assert g is not None
        df_features = df[model.features].apply(tuple, axis=1)

    for i, (subject_idx, subject_inv) in enumerate(subjects):
        for j, (partial_combination, partial_combination_inv) in enumerate(combinations):
            ax = axes[i, j]
            ax.clear()
            combination = (subject_idx, partial_combination)
            if scatter:
                idx = df_features.isin([combination])
                df_temp = df[idx].reset_index(drop=True).copy()
            for response_idx, response in enumerate(model.response):
                if response_idx >= yme.shape[-1]: break
                if scatter:
                    offset_response = g[0, 0, *combination, response_idx]
                    sns.scatterplot(
                        x=df_temp[model.intensity],
                        y=df_temp[response] - offset_response,
                        ax=ax,
                        color=colors[response_idx],
                        s=8
                    )
                x_response = x[:, 0, *combination, 0]
                y_response_mean = yme[:, *combination, response_idx]
                sns.lineplot(
                    x=x_response,
                    y=y_response_mean,
                    ax=ax,
                    color=colors[response_idx],
                    label=response if not response_absent else None
                )
                if yhdi is not None:
                    y_response_hdi = yhdi[..., *combination, response_idx]
                    ax.fill_between(
                        x_response,
                        y_response_hdi[:, 0],
                        y_response_hdi[:, 1],
                        color=colors[response_idx],
                        alpha=.4
                    )
            title = f"{partial_combination_inv}"
            if not j: title = f"{subject_inv}, {partial_combination_inv}"
            if auc is not None: title = f"{title}, auc:{auc[:, *combination].mean():.3f}"
            ax.set_title(title)
            if not i and not j and (not response_absent): ax.legend(loc="upper left")
            elif ax.get_legend(): ax.get_legend().remove()
            if a_min is not None:
                ax.axvline(x=a_min[0, 0, *combination, 0], color="orange", linestyle="--")

    for i in range(nr):
        ax = axes[i, 0]
        if set_yticks: ax.set_yticks([.2 * i for i in range(6)])
        for j in range(nc):
            ax = axes[i, j]
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelbottom=True)
            if not (i == 0 and j == 0): ax.sharex(axes[0, 0])
    
    return fig, axes

        
def viz_entropy(
    *,
    df,
    model,
    subjects,
    combinations,
    x,
    y_unnorm,
    y_norm,
    p,
    e,
    g,
    auc,
    **kw
):
    out = []
    axes = None

    print(f"Plotting y_unnorm...")
    yme = np.mean(y_unnorm, axis=1); print(yme.shape)
    # yhdi = hpdi(y_unnorm, axis=1, prob=.95); print(yhdi.shape)
    yhdi = None
    fig, curr_axes = _viz(
        df=df,
        model=model,
        subjects=subjects,
        combinations=combinations,
        x=x,
        yme=yme,
        yhdi=yhdi,
        g=g,
        auc=auc,
        scatter=True,
        set_yticks=False,
        **kw
    )
    axes = curr_axes
    fig.suptitle("response curves (y_unnorm)")
    out.append(fig)

        # print(f"Plotting y_unnorm without scatter...")
        # yme = np.mean(y_unnorm, axis=1); print(yme.shape)
        # # yhdi = hpdi(y_unnorm, axis=1, prob=.95); print(yhdi.shape)
        # yhdi = None
        # fig, curr_axes = _viz(
        #     df=df,
        #     model=model,
        #     subjects=subjects,
        #     combinations=combinations,
        #     x=x,
        #     yme=yme,
        #     yhdi=yhdi,
        #     g=g,
        #     auc=auc,
        #     scatter=False,
        #     set_yticks=False,
        #     **kw
        # )
        # curr_axes[0, 0].sharex(axes[0, 0])
        # fig.suptitle("response curves (y_unnorm)")
        # out.append(fig)

    if y_norm is not None:
        print(f"Plotting y_norm...")
        yme = np.mean(y_norm, axis=1); print(yme.shape)
        yhdi = None
        fig, curr_axes = _viz(
            df=df,
            model=model,
            subjects=subjects,
            combinations=combinations,
            x=x,
            yme=yme,
            yhdi=yhdi,
            g=g,
            auc=auc,
            **kw
        )
        curr_axes[0, 0].sharex(axes[0, 0])
        fig.suptitle("normalized curves (y_norm)")
        out.append(fig)

    if p is not None:
        print(f"Plotting p...")
        yme = np.mean(p, axis=1); print(yme.shape)
        yhdi = None
        fig, curr_axes = _viz(
            df=df,
            model=model,
            subjects=subjects,
            combinations=combinations,
            x=x,
            yme=yme,
            yhdi=yhdi,
            g=g,
            auc=auc,
            **kw
        )
        curr_axes[0, 0].sharex(axes[0, 0])
        fig.suptitle("proportion curves (p)")
        out.append(fig)

    if e is not None:
        print(f"Plotting e...")
        yme = np.mean(e, axis=1); print(yme.shape)
        yme = yme[..., None]
        yhdi = None
        fig, curr_axes = _viz(
            df=df,
            model=model,
            subjects=subjects,
            combinations=combinations,
            x=x,
            yme=yme,
            yhdi=yhdi,
            g=g,
            auc=auc,
            **kw
        )
        curr_axes[0, 0].sharex(axes[0, 0])
        fig.suptitle("entropy")
        out.append(fig)

    return out
