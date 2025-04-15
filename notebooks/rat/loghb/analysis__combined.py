# def selectivity_with_auc():
import os
import pickle

import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import integrate as sio

from hbmep import functional as F
from hbmep.util import site

from hbmep.notebooks.rat.model import HB
from hbmep.notebooks.rat.viz import viz_selectivity
from hbmep.notebooks.rat.analysis import cap_response, evaluate_response, evaluate_entropy
from hbmep.notebooks.rat.util import make_compare, make_pdf

from analysis__util import load

SEPARATOR = "___"
ignore_warnings = np.errstate(divide='ignore', invalid='ignore')


def viz_hmax(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
		subjects,
		subjects_inv,
		f1,
		f1_inv,
		num_features,
		mask_features
    ) = load(model_dir)
    posterior.keys()

    h_max = posterior["h_max"]
    print(h_max.shape)

    h_max_mean = np.nanmean(h_max, axis=0)
    print(h_max_mean.shape)

    from numpyro.diagnostics import hpdi
    h_max_hdi = hpdi(h_max, axis=0)
    print(h_max_hdi.shape)

    h_max_global = posterior["h_max_global"]
    print(h_max_global.shape)

    h_max_global_mean = np.nanmean(h_max_global, axis=0)
    print(h_max_global_mean.shape)

    h_max_global_hdi = hpdi(h_max_global, axis=0)
    print(h_max_global_hdi.shape)

    plt.close("all")
    nr, nc = len(subjects), model.num_response
    nr, nc = 1, model.num_response
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True, sharex=True, sharey=True)

    for response_idx, response in enumerate(model.response):
        ax = axes[0, response_idx]
        ax.clear()
        x = range(len(subjects))
        yme = h_max_mean[:, 0, response_idx]
        yhdi = h_max_hdi[:, :, 0, response_idx]
        sns.lineplot(x=x, y=yme, linestyle="-", ax=ax, marker="o", color="k")
        sns.lineplot(x=x, y=yhdi[0], linestyle="--", color="r", ax=ax)
        sns.lineplot(x=x, y=yhdi[1], linestyle="--", color="r", ax=ax)
        ax.axhline(y=h_max_global_mean[response_idx], linestyle="-", color="k")
        ax.axhline(y=h_max_global_hdi[0, response_idx], linestyle="--", color="b")
        ax.axhline(y=h_max_global_hdi[1, response_idx], linestyle="--", color="b")
        ax.set_title(response)
        ax.set_xticks(x)
        ax.set_xlabel("subject")
        ax.set_ylabel("h_max")

    fig.suptitle(f"Prior Beta({model.concentration1}, 1)")
    fig.show()

    h_max_fraction = posterior["h_max_fraction"]
    print(h_max_fraction.shape)

    plt.close("all")
    nr, nc = len(subjects), model.num_response
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True, sharex=True)

    for response_idx, response in enumerate(model.response):
        for subject_idx, subject in enumerate(subjects_inv):
            ax = axes[subject_idx, response_idx]
            ax.clear()
            samples = h_max_fraction[:, subject_idx, 0, response_idx]
            sns.kdeplot(samples, ax=ax)
            if not response_idx: ax.set_ylabel(subject_idx)
            else: ax.set_ylabel("")
            if not subject_idx: ax.set_title(response)
            ax.tick_params(axis="y", labelleft=False, left=False)
            # ax = axes[subject_idx, 2 * response_idx + 1]
            # samples = 
    ax = axes[0, 0]
    ax.set_xlabel("h_max_fraction")
    fig.suptitle(f"Prior Beta({model.concentration1}, 1)")

    fig.show()
    return


def selectivity_with_auc(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
		subjects,
		subjects_inv,
		f1,
		f1_inv,
		num_features,
		mask_features
    ) = load(model_dir)
    posterior.keys()
 
    min_intensity, max_intensity, intensity75th = (
        df[model.intensity].describe()
    )[["min", "max", "75%"]]
    # max_intensity = 9.344295907915816
    num_points = 200
    x = np.linspace(min_intensity, max_intensity, num_points)

    func = F.logistic4
    named_params = [site.a, site.b, site.g, site.h]
    y_unnorm, g, params = evaluate_response(
        func=func,
        named_params=named_params,
        x=x,
        posterior=posterior,
    )
    print(f"y_unnorm.shape {y_unnorm.shape}")

    y = np.nanmean(y_unnorm, axis=1)
    print(y.shape)

    print("Evaluating y_norm...")
    y_max = np.nanmax(y, axis=(0, -2), keepdims=True)
    assert not np.isnan(y_max).any()
    y_norm = np.where(y_max, y / y_max, 0.)
    print(f"y_norm.shape {y_norm.shape}")

    (
        p,
        plogp,
        entropy,
        auc
    ) = evaluate_entropy(x=x, y_norm=y_norm)


    def body_compare(subset, remove_str):
        idx, labels = zip(*subset)
        labels = [v.replace(remove_str, "") for (u, v) in subset]
        arr = entropy[..., idx]
        diff = arr[..., None] - arr[..., None, :]
        diff = np.trapz(y=diff, x=x, axis=0)
        return make_compare(diff, labels)


    def body_get_experiment(experiment):
        remove_str = SEPARATOR + experiment
        subset = [(u, v) for u, v in zip(f1, f1_inv) if v.split(SEPARATOR)[-1] == experiment]
        return subset, remove_str


    def body_compare_2(subset, remove_str, r=1/12):
        idx, labels = zip(*subset)
        labels = [v.replace(remove_str, "") for (u, v) in subset]
        arr = entropy[..., idx]
        diff = arr[..., None] - arr[..., None, :]
        cumdiff = sio.cumulative_trapezoid(diff, x=x, axis=0)
        low, high = intensity75th, max_intensity
        assert low < high
        idx = (x[1:] >= low) & (x[1:] <= high)
        diff = cumdiff[idx, ...]
        # diff = np.mean(diff, axis=0)
        n = diff.shape[0]
        weights = np.array([r ** i for i in range(n)])
        weights = weights / np.sum(weights)
        # assert np.sum(weights) == 1.0
        diff = np.average(diff, axis=0, weights=weights)
        return make_compare(diff, labels)


    experiment = "L_CIRC"
    circ, remove_str = body_get_experiment(experiment)
    vertices = [(u, v) for u, v in circ if v.split("-")[0] == ""]
    print(vertices)
    radii  = [(u, v) for u, v in circ if (u, v) not in vertices and v.replace(SEPARATOR + experiment, "")[-1] == "C"]
    print(radii)
    diam = [(u, v) for u, v in circ if (u, v) not in vertices + radii]
    print(diam)

    out = []
    plt.close("all")
    # pvalue, statistic, deg, me, eff, fig, axes = body_compare(diam, remove_str=remove_str)
    pvalue, statistic, deg, me, eff, fig, axes = body_compare_2(diam, remove_str=remove_str)
    out.append(fig)

    plt.close("all")
    # pvalue, statistic, deg, me, eff, fig, axes = body_compare(radii, remove_str=remove_str)
    pvalue, statistic, deg, me, eff, fig, axes = body_compare_2(radii, remove_str=remove_str)
    out.append(fig)

    plt.close("all")
    # pvalue, statistic, deg, me, eff, fig, axes = body_compare(vertices, remove_str=remove_str)
    pvalue, statistic, deg, me, eff, fig, axes = body_compare_2(vertices, remove_str=remove_str)
    out.append(fig)

    experiment = "L_SHIE"
    shie, remove_str = body_get_experiment(experiment)
    plt.close("all")
    # pvalue, statistic, deg, me, eff, fig, axes = body_compare(shie, remove_str=remove_str)
    pvalue, statistic, deg, me, eff, fig, axes = body_compare_2(shie, remove_str=remove_str)

    output_path = os.path.join(model_dir, "out.pdf")
    make_pdf(out, output_path)

    return out


def selectivity_with_auc_hfraction(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
		subjects,
		subjects_inv,
		f1,
		f1_inv,
		num_features,
		mask_features
    ) = load(model_dir)
    posterior.keys()
 
    min_intensity, max_intensity, intensity75th = (
        df[model.intensity].describe()
    )[["min", "max", "75%"]]
    # max_intensity = 9.344295907915816
    num_points = 200
    x = np.linspace(min_intensity, max_intensity, num_points)

    func = F.logistic4
    named_params = [site.a, site.b, site.g, "h_fraction"]
    y_norm_samples, g, params = evaluate_response(
        func=func,
        named_params=named_params,
        x=x,
        posterior=posterior,
    )
    print(f"y_norm_samples.shape {y_norm_samples.shape}")

    y_norm = np.nanmean(y_norm_samples, axis=1)
    print(y_norm.shape)
    (
        p,
        plogp,
        entropy,
        auc
    ) = evaluate_entropy(x=x, y_norm=y_norm)


    def body_compare(subset, remove_str):
        idx, labels = zip(*subset)
        labels = [v.replace(remove_str, "") for (u, v) in subset]
        arr = entropy[..., idx]
        diff = arr[..., None] - arr[..., None, :]
        diff = np.trapz(y=diff, x=x, axis=0)
        return make_compare(diff, labels)


    def body_get_experiment(experiment):
        remove_str = SEPARATOR + experiment
        subset = [(u, v) for u, v in zip(f1, f1_inv) if v.split(SEPARATOR)[-1] == experiment]
        return subset, remove_str


    def body_compare(subset, remove_str):
        idx, labels = zip(*subset)
        labels = [v.replace(remove_str, "") for (u, v) in subset]
        arr = entropy[..., idx]
        diff = arr[..., None] - arr[..., None, :]
        cumdiff = sio.cumulative_trapezoid(diff, x=x, axis=0)
        low, high = intensity75th, max_intensity
        assert low < high
        idx = (x[1:] >= low) & (x[1:] <= high)
        diff = cumdiff[idx, ...]
        diff = np.nanmean(diff, axis=0)
        return make_compare(diff, labels)


    experiment = "L_CIRC"
    circ, remove_str = body_get_experiment(experiment)
    vertices = [(u, v) for u, v in circ if v.split("-")[0] == ""]
    print(vertices)
    radii  = [(u, v) for u, v in circ if (u, v) not in vertices and v.replace(SEPARATOR + experiment, "")[-1] == "C"]
    print(radii)
    diam = [(u, v) for u, v in circ if (u, v) not in vertices + radii]
    print(diam)

    a = np.nanmean(posterior[site.a], axis=0)
    print(a.shape)

    subset = diam.copy()   
    idx, labels = zip(*subset)
    labels = [v.replace(remove_str, "") for (u, v) in subset]
    arr = a[:, idx, :]
    arr = np.nanstd(arr, axis=-1)
    diff = arr[..., None] - arr[..., None, :]
    make_compare(diff, labels=labels)

    out = []
    plt.close("all")
    pvalue, statistic, deg, me, eff, fig, axes = body_compare(diam, remove_str=remove_str)
    out.append(fig)

    plt.close("all")
    pvalue, statistic, deg, me, eff, fig, axes = body_compare(radii, remove_str=remove_str)
    out.append(fig)

    plt.close("all")
    pvalue, statistic, deg, me, eff, fig, axes = body_compare(vertices, remove_str=remove_str)
    out.append(fig)

    experiment = "L_SHIE"
    shie, remove_str = body_get_experiment(experiment)
    plt.close("all")
    pvalue, statistic, deg, me, eff, fig, axes = body_compare(shie, remove_str=remove_str)

    output_path = os.path.join(model_dir, "out.pdf")
    make_pdf(out, output_path)

    return out


def main():
    out = []

    model_dirs = [
        "/home/vishu/reports/hbmep/notebooks/rat/combined_data/4000w_4000s_4c_4t_15d_95a_tm/hb_l4_masked_mmax0/L_CIRC___L_SHIE___C_SMA_LAR___J_RCML/h_prior_0.1__conc1_10",
        # "/home/vishu/reports/hbmep/notebooks/rat/combined_data/4000w_4000s_4c_4t_15d_95a_tm/hb_l4_masked/L_CIRC___L_SHIE___C_SMA_LAR___J_RCML"
    ]
    out = [selectivity_with_auc(model_dir) for model_dir in model_dirs]
    # out = [selectivity_with_auc_hfraction(model_dir) for model_dir in model_dirs]
    # out = [viz_hmax(model_dir) for model_dir in model_dirs]

    return


if __name__ == "__main__":
    main()
