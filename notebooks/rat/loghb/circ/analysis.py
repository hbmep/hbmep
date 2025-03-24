import os
import pickle

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import cumulative_trapezoid

from hbmep import functional as F
from hbmep.util import site, setup_logging, generate_response_colors

from hbmep.notebooks.rat.util import annotate_heatmap, mask_upper

ignore_warnings = np.errstate(divide='ignore',invalid='ignore')


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
    return (
        df, encoder, model, posterior, subjects, subjects_inv, positions, positions_inv
    )


def make_test(diff):
    test = stats.wilcoxon(diff, axis=0)
    pvalue = test.pvalue
    pvalue = mask_upper(pvalue)
    pvalue.shape
    _test = stats.ttest_1samp(diff, popmean=0)
    statistic = _test.statistic
    statistic = mask_upper(statistic)
    statistic.shape
    deg = _test.df.astype(float)
    deg = mask_upper(deg)
    deg.shape
    return pvalue, statistic, deg


def make_plot(pvalue, statistic, deg, labels):
    num_labels = len(labels)
    fig, axes = plt.subplots(1, 1, constrained_layout=True, squeeze=False, figsize=(1.5 * num_labels, .8 * num_labels))
    ax = axes[0, 0]
    sns.heatmap(pvalue, annot=False, ax=ax, cbar=False)
    # Annotate
    pvalue_annot_kws = {"ha": 'center', "va": 'center'}
    annotate_heatmap(ax, pvalue,  np.round(pvalue, 3), 0.5, 0.5, star=True, star_arr=pvalue, **pvalue_annot_kws)
    deg_annot_kws = {"ha": 'left', "va": 'bottom'}
    annotate_heatmap(ax, pvalue, deg.astype(int), 0, 1, **deg_annot_kws)
    statistic_annot_kws = {"ha": 'center', "va": 'top'}
    annotate_heatmap(ax, pvalue, np.round(statistic, 3), 0.5, 0, **statistic_annot_kws)
    ax.set_xticklabels(labels=labels, rotation=15, ha="right");
    ax.set_yticklabels(labels=labels, rotation=0);
    plt.show()
    return fig, axes


def make_compare(diff, labels):
    pvalue, statistic, deg = make_test(diff)
    axes = make_plot(pvalue, statistic, deg, labels)
    return pvalue, statistic, deg, axes


# def selectivity_with_auc(model_dir):
#     (
#         df,
#         encoder,
#         model,
#         posterior,
#         positions,
#         positions_inv,
#         charges,
#         charges_inv
#     ) = load(model_dir)
#     posterior.keys()

#     a = posterior[site.a].mean(axis=0)
#     a.shape

#     diff = a[..., None] - a[..., None, :]
#     diff = diff.mean(axis=-1)
#     t = diff.mean(axis=-1)
#     np.testing.assert_almost_equal(t, np.zeros(t.shape))

#     diff.shape
#     diff = diff[..., None, :] - diff[:, None, :, :]
#     diff.shape

#     r = 5
#     plt.close("all")
#     pvalue, statistic, deg, axes = make_compare(diff[..., r], positions_inv)

#     test = stats.wilcoxon(diff, axis=0)
#     pvalue = test.pvalue
#     pvalue.shape

#     pvalue = mask_upper(pvalue)
#     pvalue.shape
#     _test = stats.ttest_1samp(diff, popmean=0)
#     statistic = _test.statistic
#     statistic = mask_upper(statistic)
#     statistic.shape
#     deg = _test.df.astype(float)
#     deg = mask_upper(deg)
#     deg.shape
#     return


def selectivity_with_auc(model_dir):
    (
        df,
        encoder,
        model,
        posterior,
        subjects,
        subjects_inv,
        positions,
        positions_inv,
        charges,
        charges_inv
    ) = load(model_dir)
    posterior.keys()

    MAX_INTENSITY = 10 # 2 ** 9 = 512
    named_params = [site.a, site.b, site.g, site.h]
    params = [np.nanmean(posterior[param], axis=0) for param in named_params]
    params = [p[None, ...] for p in params]
    params[2] = params[2] * 0 # get rid of noise floor (baseline value)

    a_shape = params[0].shape
    a_ndim = len(a_shape)
    print(a_shape, a_ndim)

    x = np.arange(0, MAX_INTENSITY, .05)

    # Response
    y = np.array(F.logistic4(
        x[:, *(None for _ in range(a_ndim - 1))], *params
    ))
    print(y.shape)

    # Get max
    y_max = np.nanmax(y, axis=(0, -1, -2), keepdims=True)
    print(y_max.shape)

    # Normalize
    y_norm = np.where(y_max, y / y_max, 0.)
    print(y_norm.shape)

    # Calculate selectivity
    p = np.nansum(y, axis=-1, keepdims=True)
    print(p.shape)

    with ignore_warnings:
        p = np.where(p, y / p, 1 / y.shape[-1])
        p.shape

    with ignore_warnings:
        plogp = np.where(p, p * np.log(p), 0)
        plogp.shape

    entropy = 1 + (plogp.sum(axis=-1) / np.log(y.shape[-1]))
    entropy.shape

    print(f"entropy.isnan.sum: {np.isnan(entropy).sum()}")
    auc = np.trapz(y=entropy, x=x, axis=0)
    print(auc.shape)

    cum_auc = cumulative_trapezoid(y=entropy, x=x, axis=0)
    print(cum_auc.shape)

    # Transpose trick to get pairwise diffs
    def body_make_compare(arr):
        print(f"arr.shape {arr.shape}")
        arr_reshaped = arr.reshape(*arr.shape[:-2], -1)
        print(f"arr_reshaped.shape: {arr_reshaped.shape}")
        labels = []
        for pos in positions_inv:
            for ch in charges_inv:
                labels.append(f"{pos}__{ch}")
        labels = np.array(labels)
        print(f"labels: {labels}")
        diff = arr_reshaped[..., None] - arr_reshaped[..., None, :]
        return diff, labels

    diff, labels = body_make_compare(cum_auc)
    print(diff.shape, labels)

    arr = cum_auc.copy()
    arr.shape
    arr = arr.reshape(*arr.shape[:-2], -1)
    arr.shape
    subset = list(range(4))
    curr_arr = arr[..., subset]
    curr_arr.shape
    curr_labels = labels[subset]
    curr_labels
    curr_colors = generate_response_colors(n=len(subset), palette="viridis")
    curr_arr.shape
    curr_diff = curr_arr[..., None] - curr_arr[..., None, :]

    nr, nc = 3, 4
    plt.close("all")
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3 * nr), constrained_layout=True, squeeze=False)
    plt.show()

    reference = 0
    counter = 0
    ax = axes[-1, 0]
    ax.clear()
    ax.axhline(0, linestyle="--", color=curr_colors[reference], label=curr_labels[reference])
    for subject_idx, subject in enumerate(subjects_inv):
        ax = axes[counter // nc, counter % nc]
        ax.clear()
        for label_idx, label in enumerate(curr_labels):
            if label_idx == reference: continue
            # sns.lineplot(x=x[1:], y=curr_arr[:, subject_idx, label_idx], label=label, ax=ax, color=curr_colors[label_idx]);
            sns.lineplot(x=x[1:], y=curr_diff[:, subject_idx, reference, label_idx], label=label, ax=ax, color=curr_colors[label_idx]);
            if subject_idx == 0:
                # sns.lineplot(x=x[1:], y=curr_arr.mean(axis=1)[:, label_idx], label=label, ax=axes[-1, -1], color=curr_colors[label_idx]);
                sns.lineplot(x=x[1:], y=curr_diff.mean(axis=1)[:, reference, label_idx], label=label, ax=axes[-1, 0], color=curr_colors[label_idx]);
        ax.set_title(subject);
        ax.axhline(0, linestyle="--", color=curr_colors[reference], label=curr_labels[reference])
        if counter > 0:
            if ax.get_legend(): ax.get_legend().remove()
        counter += 1
    ax = axes[0, 0]
    ax.set_xlabel("log2(intensity)")

    return


def selectivity_with_threshold(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
        subjects_inv,
		positions,
		positions_inv,
    ) = load(model_dir)
    posterior.keys()

    model.features
    a = np.nanmean(posterior[site.a], axis=0)
    a.shape

    a = np.nanmean(posterior[site.a], axis=0)
    a.shape

    a = a - np.nanmin(a, axis=-1, keepdims=True)
    a.shape

    np.isnan(a).any()
    a_mean = np.nanmean(a, axis=-1)
    a_mean.shape

    diff = a_mean[..., None] - a_mean[:, None, :]
    print(diff.shape)
    labels = positions_inv.copy()
    print(labels)


    plt.close("all")
    pvalue, statistic, deg, axes = make_compare(diff, labels)

    return


def threshold_analysis(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
        subjects_inv,
		positions,
		positions_inv,
    ) = load(model_dir)
    posterior.keys()

    model.features
    a = posterior[site.a].mean(axis=0)
    a.shape

    np.isnan(a).any()
    a_mean = np.nanmean(a, axis=-1)
    a_mean.shape

    diff = a_mean[..., None] - a_mean[:, None, :]
    print(diff.shape)
    labels = positions_inv.copy()
    print(labels)

    plt.close("all")
    pvalue, statistic, deg, axes = make_compare(diff, labels)

    return


def threshold_estimation(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
        subjects_inv,
		positions,
		positions_inv,
    ) = load(model_dir)
    posterior.keys()

    a_fixed_loc = posterior["a_fixed_loc"]
    a_fixed_loc.shape

    a_delta_loc = posterior["a_delta_loc"]
    a_delta_loc.shape
    
    param = a_fixed_loc[:, None] + a_delta_loc
    param.shape

    plt.close("all")
    for pos, pos_inv in enumerate(positions_inv):
        sns.kdeplot(param[:, pos], label=pos_inv)
    # sns.kdeplot(a_delta_loc)
    plt.legend()
    plt.show()



    return

def main():
    ## Threshold
    model_dirs = [
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/diam/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked",
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/radii/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked",
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/vertices/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked",
    ]
    # threshold_analysis(model_dir)
    # [selectivity_with_threshold(model_dir) for model_dir in model_dirs]

    # model_dir = "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/all/4000w_4000s_4c_4t_15d_95a_tm/hb_mvn_l4_masked"
    # # # model_dir = "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/no-ground/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked"
    # # selectivity_analysis(model_dir)

    # model_dir = "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/all/4000w_4000s_4c_4t_15d_95a_tm/hb_mvn_l4_masked"
    # selectivity_analysis(model_dir)

    model_dir = "/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/estimation/400w_400s_4c_1t_15d_95a_fm/diam/circ_ln_est_mvn_reference_rl_nov_masked/test_run"
    threshold_estimation(model_dir)

    return


if __name__ == "__main__":
    main()
