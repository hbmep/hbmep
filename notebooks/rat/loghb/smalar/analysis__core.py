import os
import pickle

import pandas as pd
import numpy as np
from jax import random
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from hbmep import functional as F
from hbmep.util import site

from hbmep.notebooks.rat.analysis import load_smalar as load
from hbmep.notebooks.rat.util import (
    make_test,
    make_compare3p,
    make_compare3p_bar,
    make_pdf,
    make_dump,
    compare_less_than
)
from hbmep.notebooks.rat.testing import (
    checknans,
    check1,
    check2
)

BUILD_DIR = "/home/vishu/reports/hbmep/notebooks/rat/loghb/out"
os.makedirs(BUILD_DIR, exist_ok=True)


def threshold_analysis_lat(model_dir, correction=False, fig=None, dump=False):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
		positions,
        degrees,
        sizes,
        num_features,
        mask_features,
        *_,
    ) = load(model_dir)
    assert sizes is None

    print(model.features)
    num_nans = (~mask_features).sum()
    a = posterior[site.a].copy(); assert np.isnan(a[0, ..., 0]).sum() == num_nans
    a = np.mean(posterior[site.a], axis=0, keepdims=True); print(a.shape)
    assert np.isnan(a[~mask_features[..., 0]]).all()
    assert not np.isnan(a[mask_features[..., 0]]).any()
    check2(a)

    a = np.nanmean(a, axis=-3); print(a.shape)
    a_mean = np.mean(a, axis=-1); print(a_mean.shape)

    (
        fig, diff_positions, diff_mean, diff_err, colors, *_
    ) = make_compare3p(a_mean, degrees, negate=True, correction=correction, fig=fig)
    fig, axes = fig
    ax = axes[0, 0]; ax.set_ylabel("← is more effective")
    ax = axes[0, 1]; ax.set_xlabel("→ is more effective")
    fig.suptitle(f"{'/'.join(model.build_dir.split('/')[-2:])}")

    output_path = os.path.join(BUILD_DIR, f"{model.run_id}.pkl")
    if dump: make_dump((diff_positions, colors, diff_mean, diff_err,), output_path)

    if "big" in model.build_dir:
        a_mean = np.nanmean(a, axis=(-1, 1))
        test = stats.ttest_ind(a=a_mean[:, 2], b=a_mean[:, 1], nan_policy="omit")
        print(test)

    return (fig, axes), model, degrees, a, a_mean, diff_positions, diff_mean, diff_err, colors


def estimation_analysis_lat(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
		positions,
        degrees,
        sizes,
        num_features,
        mask_features,
        *_,
    ) = load(model_dir, estimation=True)

    param = posterior["a_delta_loc"]; print(param.shape)
    nr, nc = 1, 1
    fig, axes = plt.subplots(
        nr, nc, figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True
    )

    positions = degrees.copy()
    ax = axes[0, 0]
    ax.clear()
    ax.axvline(x=0, label=positions[0][1][1:], color="k", linestyle="--")
    for i in range(param.shape[-1]):
        label = f"[{i}]{positions[1:][i][1]}"
        samples = param[:, i]
        sns.kdeplot(samples, ax=ax, label=label)
    ax.legend(loc="upper right")

    if "small" in model.run_id: reference_idx = 2
    else: reference_idx = 2
    reference = positions[1:][reference_idx][1]

    counter = 1
    key = random.key(0)
    key, prob = compare_less_than(key, param[:, reference_idx], np.array([0.]))
    title = f"[{reference_idx}]{reference} < {positions[0][1][1:]}:{prob: .3f}, "
    for i in range(param.shape[-1]):
        if i == reference_idx: continue
        key, prob = compare_less_than(key, param[:, reference_idx], param[:, i])
        title += f"[{i}]{positions[1:][i][1]}:{prob: .2f}, "
        counter += 1
        if not counter % 4 and i != param.shape[-1]: title += f"\n"

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    build_dir = model.build_dir.split('/')
    build_dir = np.array(build_dir)[[-3, -1]].tolist()
    title = f"{'/'.join(build_dir)}\n\n{title}"
    fig.suptitle(title)
    return (fig, axes), param, model, positions


def figure_lat(model_dirs, correction = False):
    nr, nc = 1, 4
    fig, axes = plt.subplots(
        nr, nc, figsize=(5 * nc, 3.4 * nr), squeeze=False, constrained_layout=True
    )

    model_dir = model_dirs[0]
    (
        _, model, positions, a, a_mean, diff_positions, diff_mean, diff_err, colors, *_
    ) = threshold_analysis_lat(
        model_dir, correction=correction, fig=(fig, axes)
    )
    suptitle = f"{model.run_id}/{model._model.__name__}/mix:{model.use_mixture}"

    model_dir = model_dirs[1]
    _, param, model, positions, *_ = estimation_analysis_lat(model_dir)
    suptitle += f"\n{model.run_id}/{model._model.__name__}/mix:{model.use_mixture}"

    ax = axes[0, 3]; ax.clear()
    _, v = zip(*positions)
    param_inv = dict(zip(v[1:], range(len(v[1:])))); print(param_inv)
    for _, pos_inv in diff_positions:
        try:
            samples = param[:, param_inv[pos_inv]]
            sns.kdeplot(samples, color=colors[pos_inv], label=pos_inv, ax=ax)
        except KeyError:
            assert pos_inv == positions[0][1][1:]
            try:
                label = positions[0][1][1:]
                ax.axvline(x=0, color=colors[label], linestyle="--", label=label)
            except KeyError:
                print("Reference color not found")
                ax.axvline(x=0, color="k", linestyle="--", label=label)
    ax.tick_params(axis="both", labelleft=False, left=False)
    ax.set_ylabel("")

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            sides = ["right", "top"]
            ax.spines[sides].set_visible(False)
            if ax.get_legend(): ax.get_legend().remove()

    ax = axes[-1, -1]
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", reverse=True)
    fig.suptitle(suptitle)
    return (fig, axes),


def threshold_analysis_size(model_dir, correction=False, fig=None, dump=False):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
		positions,
        degrees,
        sizes,
        num_features,
        mask_features,
        *_,
    ) = load(model_dir)
    assert sizes is not None

    print(model.features)
    num_nans = (~mask_features).sum()
    a = posterior[site.a].copy(); assert np.isnan(a[0, ..., 0]).sum() == num_nans
    a = np.mean(posterior[site.a], axis=0, keepdims=True); print(a.shape)
    assert np.isnan(a[~mask_features[..., 0]]).all()
    assert not np.isnan(a[mask_features[..., 0]]).any()

    assert sizes[0][1] == "B"; assert sizes[1][1] == "S"
    diff = a[..., 0, :] - a[..., 1, :]; print(diff.shape)   # (B - S)
    diff = np.mean(diff, axis=-1); print(diff.shape)
    diff = np.nanmean(diff, axis=-2); print(diff.shape)
    diff = -diff    # (S - B)

    fig, diff_positions, diff_mean, diff_err, colors, *_, = make_compare3p_bar(
        diff, degrees, correction=correction, fig=fig
    )
    fig, axes = fig
    ax = axes[0, 0]; ax.set_ylabel("log( S/B ) > 0 ⇒ Big is more effective")
    fig.suptitle(f"{'/'.join(model.build_dir.split('/')[-2:])}")

    output_path = os.path.join(BUILD_DIR, f"{model.run_id}.pkl")
    if dump: make_dump((diff_positions, colors, diff_mean, diff_err,), output_path)
    return (fig, axes), model, degrees, diff_positions, diff_mean, diff_err, colors


def estimation_analysis_size(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
		positions,
        degrees,
        sizes,
        num_features,
        mask_features,
        *_,
    ) = load(model_dir, estimation=True)

    param = posterior["a_delta_loc"][:, 0, ...]; print(param.shape)
    nr, nc = 1, 1
    fig, axes = plt.subplots(
        nr, nc, figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True
    )

    positions = degrees.copy()
    ax = axes[0, 0]
    ax.clear()
    ax.axvline(x=0, label=sizes[0][1][1:], color="k", linestyle="--")
    for i in range(param.shape[-1]):
        label = f"[{i}]{positions[i][1]}"
        samples = -param[:, i]
        sns.kdeplot(samples, ax=ax, label=label)
    ax.legend(loc="upper right")

    title = f"Pr( {sizes[1][1]} < {sizes[0][1][1:]} ):\n"
    key = random.key(0)
    for i in range(param.shape[-1]):
        key, prob = compare_less_than(key, param[:, i], np.array([0.]))
        title += f"[{i}]{positions[i][1]}:{prob: .2f}, "

    build_dir = model.build_dir.split('/')
    build_dir = np.array(build_dir)[[-3, -1]].tolist()
    title = f"{'/'.join(build_dir)}\n\n{title}"
    fig.suptitle(title)
    return (fig, axes), param, model, positions, sizes


def figure_size(model_dirs, correction=False):
    nr, nc = 1, 3
    fig, axes = plt.subplots(
        nr, nc, figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True
    )

    model_dir = model_dirs[0]
    (
        _, model, positions, diff_positions, diff_mean, diff_err, colors, *_
    ) = threshold_analysis_size(model_dir, correction=correction, fig=(fig, axes))
    suptitle = f"{model.run_id}/{model._model.__name__}/mix:{model.use_mixture}"

    model_dir = model_dirs[1]
    _, param, model, positions, sizes, *_ = estimation_analysis_size(model_dir)
    suptitle += f"\n{model.run_id}/{model._model.__name__}/mix:{model.use_mixture}"

    ax = axes[0, 2]; ax.clear()
    ax.axvline(x=0, label=sizes[0][1][1:], color="k", linestyle="--")
    u, v = zip(*positions)
    param_inv = dict(zip(v, range(len(v)))); print(param_inv)

    for _, pos_inv in diff_positions:
        samples = -param[:, param_inv[pos_inv]]
        sns.kdeplot(samples, ax=ax, label=pos_inv, color=colors[pos_inv])
    ax.tick_params(axis="both", labelleft=False, left=False)
    ax.set_ylabel("")
    ax.set_xlabel("log( S/B ) > 0 ⇒ Big is more effective")

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            sides = ["right", "top"]
            ax.spines[sides].set_visible(False)
            if ax.get_legend(): ax.get_legend().remove()

    ax = axes[-1, -1]
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", reverse=True)
    fig.suptitle(suptitle)
    return (fig, axes),


def main_lat():
    out = []

    # Threshold laterality
    model_dirs = [
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-small-ground/hb_mvn_rl_masked",
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-big-ground/hb_mvn_rl_masked",

    ]
    out += [threshold_analysis_lat(model_dir, dump=True)[0][0] for model_dir in model_dirs]

    model_dirs = [
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-small-ground/lat_est_mvn_block_reference_rl_masked",
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-big-ground/lat_est_mvn_block_reference_rl_masked",

    ]
    out += [estimation_analysis_lat(model_dir)[0][0] for model_dir in model_dirs]

    model_dirs = [
        (
            "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-small-ground/hb_mvn_rl_masked",
            "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-small-ground/lat_est_mvn_block_reference_rl_masked",
        ),
        (
            "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-big-ground/hb_mvn_rl_masked",
            "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-big-ground/lat_est_mvn_block_reference_rl_masked",
        )
    ]
    out += [figure_lat(mdirs, correction=True)[0][0] for mdirs in model_dirs]
    return out


def main_size():
    out = []

    # Threshold size
    model_dirs = [
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/size-ground/hb_mvn_rl_masked",
    ]
    out += [threshold_analysis_size(model_dir, dump=True)[0][0] for model_dir in model_dirs]

    model_dirs = [
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/size-ground/size_est_mvn_block_reference_rl_masked"
    ]
    out += [estimation_analysis_size(model_dir)[0][0] for model_dir in model_dirs]

    model_dirs = [
        (
            "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/size-ground/hb_mvn_rl_masked",
            "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/size-ground/size_est_mvn_block_reference_rl_masked"
        ),
    ]
    out += [figure_size(mdirs, correction=True)[0][0] for mdirs in model_dirs]
    return out


def main():
    out = []
    out += main_lat()
    out += main_size()
    return out


if __name__ == "__main__":
    out = main()
    output_path = os.path.join(BUILD_DIR, "smalar.pdf")
    make_pdf(out, output_path)
