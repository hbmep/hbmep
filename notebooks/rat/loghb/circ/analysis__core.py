import os
import pickle
import inspect

import pandas as pd
from jax import random
import numpy as np
from scipy import stats
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import hbmep as mep
from hbmep import functional as F
from hbmep.util import site

from hbmep.notebooks.rat.viz import viz_selectivity
from hbmep.notebooks.rat.analysis import evaluate_response, evaluate_entropy
from hbmep.notebooks.rat.util import make_compare, make_pdf, compare_less_than

from constants import BUILD_DIR


def load(model_dir):
    src = os.path.join(model_dir, "inf.pkl")
    with open(src, "rb") as f:
        df, encoder, posterior = pickle.load(f)
    src = os.path.join(model_dir, "model.pkl")
    with open(src, "rb") as f:
        model, = pickle.load(f)
    subjects = sorted(df[model.features[0]].unique())
    subjects
    subjects_inv = encoder[model.features[0]].inverse_transform(subjects)
    subjects_inv
    positions = sorted(df[model.features[1]].unique())
    positions
    positions_inv = encoder[model.features[1]].inverse_transform(positions)
    positions_inv
    num_features = df[model.features].max().to_numpy() + 1
    return (
        df,
		encoder,
		model,
		posterior,
		subjects,
		subjects_inv,
		positions,
		positions_inv,
		num_features
    )


def threshold_analysis(model_dir, correction=False):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
        subjects_inv,
		positions,
		positions_inv,
        num_features,
        *_
    ) = load(model_dir)
    posterior.keys()

    model.features
    a = np.nanmean(posterior[site.a], axis=0)
    print(a.shape)

    print(f"np.isnan(a).any() {np.isnan(a).any()}")
    a_mean = np.nanmean(a, axis=-1)
    print(a_mean.shape)

    diff = a_mean[..., None] - a_mean[:, None, :]
    print(diff.shape)
    labels = positions_inv.copy()
    print(labels)


    def body_check_1():
        z = a[..., None, :] - a[..., None, :, :]
        z.shape
        z = np.nanmean(z, axis=-1)
        z.shape
        np.testing.assert_almost_equal(z, diff)
        import inspect
        print(f"{inspect.currentframe().f_code.co_name} success.")
    body_check_1()


    # +ve row, -ve column
    diff = -diff

    plt.close("all")
    pvalue, statistic, deg, me, eff, fig, axes = make_compare(diff, labels, correction=correction)
    fig.suptitle(f"{'/'.join(model.build_dir.split('/')[-2:])}\ncorrection:{correction}")
    return fig


def estimation_analysis(model_dir, correction=False):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
        subjects_inv,
		positions,
		positions_inv,
        num_features,
        *_
    ) = load(model_dir)
    posterior.keys()

    posterior.keys()
    param = posterior["a_delta_loc"]
    print(param.shape)

    nr, nc = 1, 1
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 5 * nr), squeeze=False, constrained_layout=True)

    ax = axes[0, 0]
    ax.clear()
    ax.axvline(x=0, label=positions_inv[0][1:], color="k", linestyle="--")
    for i in range(param.shape[-1]):
        label = f"[{i}]{positions_inv[1:][i]}"
        samples = param[:, i]
        sns.kdeplot(samples, ax=ax, label=label)
    ax.legend(loc="upper right")

    reference_idx = 2
    reference = positions_inv[1:][reference_idx]
    counter = 1
    key = random.key(0)
    key, prob = compare_less_than(key, param[:, reference_idx], np.array([0.]))
    title = f"[{reference_idx}]{reference} < {positions_inv[0][1:]}:{prob: .3f}, "
    for i in range(param.shape[-1]):
        if i == reference_idx: continue
        key, prob = compare_less_than(key, param[:, reference_idx], param[:, i])
        title += f"[{i}]{positions_inv[1:][i]}:{prob: .2f}, "
        counter += 1
        if not counter % 2 and i != param.shape[-1]: title += f"\n"

    build_dir = model.build_dir.split('/')
    build_dir = np.array(build_dir)[[-3, -1]].tolist()
    title = f"{'/'.join(build_dir)}\n\n{title}"
    fig.suptitle(title)

    return fig

    
def figure(model_dirs):
    model_dir = model_dirs[0]
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
        subjects_inv,
		positions,
		positions_inv,
        num_features,
        *_
    ) = load(model_dir)
    posterior.keys()

    model.features
    a = np.nanmean(posterior[site.a], axis=0)
    print(a.shape)

    print(f"np.isnan(a).any() {np.isnan(a).any()}")
    a_mean = np.nanmean(a, axis=-1)
    print(a_mean.shape)

    positions = list(zip(positions, positions_inv))
    print(positions)

    if "diam" in model_dir: reference_idx = 2
    if "radii" in model_dir: reference_idx = 4
    if "vertices" in model_dir: reference_idx = 5
    diff = a_mean[..., reference_idx: reference_idx + 1] - a_mean
    xme = np.nanmean(diff, axis=0)
    xerr = stats.sem(diff, axis=0, nan_policy="omit")
    xme, xerr, y = zip(*sorted(zip(xme, xerr, positions), key=lambda x: (x[0], x[1], x[2][0])))

    colors = sns.color_palette(palette="viridis", n_colors=len(y))
    # colors = sns.color_palette(palette="muted", n_colors=len(y))
    colors_map = dict(zip([u[1] for u in y], colors))

    # plt.close("all")
    nr, nc = 1, 3
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True)

    ax = axes[0, 0]
    ax.clear()
    for pos_idx, pos in y:
        sns.lineplot(
            x=subjects_inv,
            y=a_mean[:,
            pos_idx],
            label=pos,
            ax=ax,
            color=colors_map[pos],
            marker="o",
        )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(-.05, 1), loc="upper right", reverse=True)

    ax = axes[0, 1]
    ax.clear()
    for i, (pos_idx, pos) in enumerate(y):
        ax.errorbar(
            x=xme[i],
            xerr=xerr[i],
            y=[u[1] for u in y][i],
            fmt="o",
            ecolor=colors_map[pos],
            color=colors_map[pos],
        )
    ax.tick_params(axis="x", rotation=45)

    model_dir = model_dirs[1]
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
        subjects_inv,
		positions,
		positions_inv,
        num_features,
        *_
    ) = load(model_dir)
    posterior.keys()

    param = posterior["a_delta_loc"]
    print(param.shape)

    ax = axes[0, 2]
    ax.clear()
    try:
        ax.axvline(x=0, color=colors_map[positions_inv[0][1:]], linestyle="--")
    except KeyError:
        print("Reference color not found")
        ax.axvline(x=0, color="k", linestyle="--")

    for i in range(param.shape[-1]):
        samples = param[:, i]
        pos_charge_inv = positions_inv[1:][i]
        sns.kdeplot(samples, color=colors_map[pos_charge_inv])
    ax.tick_params(axis="both", labelleft=False, left=False)
    ax.set_ylabel("")

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            sides = ["right", "top"]
            ax.spines[sides].set_visible(False)

    plt.close("all")
    return fig


def main():
    out = []

    # ## Threshold
    # model_dirs = [
    #     "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/diam/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked",
    #     "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/radii/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked",
    #     "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/vertices/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked",
    # ]
    # # out += [threshold_analysis(model_dir) for model_dir in model_dirs]

    model_dirs = [
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/diam/4000w_4000s_4c_4t_15d_95a_tm/hb_mvn_rl_masked",
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/radii/4000w_4000s_4c_4t_15d_95a_tm/hb_mvn_rl_masked",
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/vertices/4000w_4000s_4c_4t_15d_95a_tm/hb_mvn_rl_masked",
    ]
    # out += [threshold_analysis(model_dir) for model_dir in model_dirs]

    model_dirs = [
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/estimation/4000w_4000s_4c_4t_15d_95a_tm/diam/circ_est_mvn_reference_rl_masked",
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/estimation/4000w_4000s_4c_4t_15d_95a_tm/radii/circ_est_mvn_reference_rl_masked",
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/estimation/4000w_4000s_4c_4t_15d_95a_tm/vertices/circ_est_mvn_reference_rl_masked",
    ]
    # out += [estimation_analysis(model_dir) for model_dir in model_dirs]


    model_dirs = [
        (
            "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/diam/4000w_4000s_4c_4t_15d_95a_tm/hb_mvn_rl_masked",
            "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/estimation/4000w_4000s_4c_4t_15d_95a_tm/diam/circ_est_mvn_reference_rl_masked",
        ),
        (
            "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/radii/4000w_4000s_4c_4t_15d_95a_tm/hb_mvn_rl_masked",
            "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/estimation/4000w_4000s_4c_4t_15d_95a_tm/radii/circ_est_mvn_reference_rl_masked",
        ),
        (
            "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/vertices/4000w_4000s_4c_4t_15d_95a_tm/hb_mvn_rl_masked",
            "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/estimation/4000w_4000s_4c_4t_15d_95a_tm/vertices/circ_est_mvn_reference_rl_masked",
        )
    ]
    out += [figure(mdirs) for mdirs in model_dirs]

    output_path = os.path.join(BUILD_DIR, "out.pdf")
    make_pdf(out, output_path)
    return


if __name__ == "__main__":
    main()
