import os
import pickle

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from hbmep.util import site, setup_logging, generate_response_colors

from hbmep.notebooks.rat.util import annotate_heatmap, mask_upper


def load(model_dir):
    src = os.path.join(model_dir, "inf.pkl")
    with open(src, "rb") as f:
        df, encoder, posterior = pickle.load(f)
    src = os.path.join(model_dir, "model.pkl")
    with open(src, "rb") as f:
        model, = pickle.load(f)
    positions = sorted(df[model.features[1]].unique())
    positions
    positions_inv = encoder[model.features[1]].inverse_transform(positions)
    positions_inv
    charges = sorted(df[model.features[2]].unique())
    charges
    charges_inv = encoder[model.features[2]].inverse_transform(charges)
    charges_inv
    return (
        df, encoder, model, posterior, positions, positions_inv, charges, charges_inv
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


def selectivity_analysis(model_dir):
    (
        df,
        encoder,
        model,
        posterior,
        positions,
        positions_inv,
        charges,
        charges_inv
    ) = load(model_dir)
    posterior.keys()

    a = posterior[site.a].mean(axis=0)
    a.shape

    diff = a[..., None] - a[..., None, :]
    diff = diff.mean(axis=-1)
    t = diff.mean(axis=-1)
    np.testing.assert_almost_equal(t, np.zeros(t.shape))

    diff.shape
    diff = diff[..., None, :] - diff[:, None, :, :]
    diff.shape

    r = 5
    plt.close("all")
    pvalue, statistic, deg, axes = make_compare(diff[..., r], positions_inv)

    test = stats.wilcoxon(diff, axis=0)
    pvalue = test.pvalue
    pvalue.shape

    pvalue = mask_upper(pvalue)
    pvalue.shape
    _test = stats.ttest_1samp(diff, popmean=0)
    statistic = _test.statistic
    statistic = mask_upper(statistic)
    statistic.shape
    deg = _test.df.astype(float)
    deg = mask_upper(deg)
    deg.shape
    return


def threshold_analysis(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
		positions,
		positions_inv,
        charges,
        charges_inv
    ) = load(model_dir)
    posterior.keys()

    model.features
    a = posterior[site.a].mean(axis=0)
    a.shape
    a = a.reshape(a.shape[0], -1, a.shape[-1])
    a.shape

    position_charges = []
    for pos in positions_inv:
        for ch in charges_inv:
            position_charges.append(f"{pos}__{ch}")

    np.isnan(a).any()
    a_mean = np.nanmean(a, axis=-1)
    a_mean.shape

    diff = a_mean[..., None] - a_mean[:, None, :]
    print(diff.shape)
    labels = position_charges.copy()
    print(labels)

    plt.close("all")
    pvalue, statistic, deg, axes = make_compare(diff, labels)

    return


def main():
    # model_dir = "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/ground/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked"
    # # model_dir = "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/no-ground/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked"
    # threshold_analysis(model_dir)

    model_dir = "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/ground/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked"
    # model_dir = "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/no-ground/4000w_4000s_4c_4t_15d_95a_fm/hb_mvn_rl_nov_masked"
    selectivity_analysis(model_dir)
    return


if __name__ == "__main__":
    main()
