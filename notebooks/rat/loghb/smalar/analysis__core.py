import os
import pickle

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import cumulative_trapezoid

from hbmep import functional as F
from hbmep.util import site

from hbmep.notebooks.rat.util import make_compare, make_pdf, compare_less_than

from constants import BUILD_DIR


def load(model_dir, estimation=False):
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
    feature_pos_idx, feature_deg_idx = 1, 2
    if estimation: feature_pos_idx, feature_deg_idx = 2, 1
    positions = sorted(df[model.features[feature_pos_idx]].unique())
    positions
    positions_inv = encoder[model.features[feature_pos_idx]].inverse_transform(positions)
    positions_inv
    degrees = sorted(df[model.features[feature_deg_idx]].unique())
    degrees
    degrees_inv = encoder[model.features[feature_deg_idx]].inverse_transform(degrees)
    degrees_inv

    try: size_feature = model.features[3]
    except IndexError as e: 
        print(f"Encoutered {e}. Possbily size feature is missing.")
        sizes, sizes_inv = None, None
    else:
        sizes = sorted(df[size_feature].unique())
        sizes_inv = encoder[size_feature].inverse_transform(sizes)

    num_features, mask_features = None, None
    if not estimation:
        named_params = [site.a, site.b, site.g, site.h, site.v]
        posterior = {u: posterior[u] for u in named_params if u in posterior.keys()}
        for u, v in posterior.items(): print(u, v.shape)

        num_features = df[model.features].max().to_numpy() + 1
        mask_features = np.full((*num_features,), False)
        _, features = model.get_regressors(df)
        mask_features[*features.T] = True
        mask_features = mask_features[None, ..., None]
        print(mask_features.shape)


        def body_mask(named_param):
            masked_arr = np.where(mask_features, posterior[named_param], np.nan)
            return masked_arr


        def body_check_1(named_params):
            for u in named_params:
                try: v = posterior[u]
                except KeyError: print(f"Skipping {u}..."); continue
                mask_on = v[:, mask_features[0, ..., 0], :]
                mask_off = v[:, ~mask_features[0, ..., 0], :]
                bool_mask_on = np.isnan(mask_on).any()
                bool_mask_off = np.isnan(mask_off).all()
                assert (not bool_mask_on) and bool_mask_off
                print(f"{u} ok.")
            import inspect
            print(f"{inspect.currentframe().f_code.co_name} success.")
            return


        named_params = [site.a, site.b, site.g, site.h, site.v]
        posterior = {u: body_mask(u) if u in named_params else v for u, v in posterior.items()}
        for u, v in posterior.items(): print(u, v.shape)
        body_check_1(named_params)

    return (
        df,
		encoder,
		model,
		posterior,
        subjects,
        subjects_inv,
		positions,
		positions_inv,
        degrees,
        degrees_inv,
        sizes,
        sizes_inv,
        num_features,
        mask_features
    )


def threshold_analysis_lat(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
        subjects_inv,
		positions,
		positions_inv,
        degrees,
        degrees_inv,
        sizes,
        sizes_inv,
        num_features,
        mask_features
    ) = load(model_dir)
    assert sizes is None
    assert sizes_inv is None
    posterior.keys()
    # for u, v in posterior.items(): print(u, v.shape)

    print(model.features)
    a = np.nanmean(posterior[site.a], axis=0)
    print(a.shape)

    assert not np.isnan(a[mask_features[0, ..., 0]]).any()
    assert np.isnan(a[~mask_features[0, ..., 0]]).all()

    # a_mean = np.nanmean(a, axis=(-1, -3))
    diff = a[..., None, :] - a[..., None, :, :]
    print(diff.shape)
    diff = np.nanmean(diff, axis=(-1, 1))
    print(diff.shape)

    labels = degrees_inv.copy()
    print(labels)


    # def body_check_1():
    #     z = a[..., :, None, :] - a[..., None, :, :]
    #     z = np.nanmean(z, axis=(-1, -4))
    #     z.shape
    #     diff.shape
    #     np.testing.assert_almost_equal(z, diff)
    #     import inspect
    #     print(f"{inspect.currentframe().f_code.co_name} success.")
    #     return
    # # body_check_1()


    def body_check_2():
        a_mean = np.nanmean(a, axis=-1)
        d = a_mean[..., None] - a_mean[..., None, :]
        d = np.nanmean(d, axis=1)
        np.testing.assert_almost_equal(diff, d)
        import inspect
        print(f"{inspect.currentframe().f_code.co_name} success.")
        return
    body_check_2()


    # +ve row, -ve column
    diff = -diff

    plt.close("all")
    pvalue, statistic, deg, me, eff, fig, axes = make_compare(diff, labels)
    build_dir = model.build_dir.split("/")[-2:]
    build_dir = "/".join(build_dir)
    fig.suptitle(build_dir)

    return fig


def figure_lat(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
        subjects_inv,
		positions,
		positions_inv,
        degrees,
        degrees_inv,
        sizes,
        sizes_inv,
        num_features,
        mask_features
    ) = load(model_dir)
    posterior.keys()
    suptitle = f"{model.run_id}/{model._model.__name__}/mix:{model.use_mixture}"

    print(model.features)
    a = np.nanmean(posterior[site.a], axis=0)
    print(a.shape)
    a_mean = np.nanmean(a, axis=-1)
    print(a_mean.shape)

    degrees = list(zip(degrees, degrees_inv))
    print(degrees)

    reference_idx = 2
    diff = a_mean[..., reference_idx: reference_idx + 1] - a_mean
    print(diff.shape)
    diff = np.nanmean(diff, axis=1)
    print(diff.shape)

    print(a_mean.shape)
    a_mean = np.nanmean(a_mean, axis=1)
    print(a_mean.shape)
    temp = a_mean[:, reference_idx: reference_idx + 1] - a_mean
    np.testing.assert_almost_equal(temp, diff)

    xme = np.nanmean(diff, axis=0)
    xerr = stats.sem(diff, axis=0, nan_policy="omit")
    xme, xerr, y = zip(*sorted(zip(xme, xerr, degrees), key=lambda x: (x[0], x[1], x[2][0])))

    colors = sns.color_palette(palette="viridis", n_colors=len(y))
    # colors = sns.color_palette(palette="muted", n_colors=len(y))
    colors_map = dict(zip([u[1] for u in y], colors))

    # plt.close("all")
    nr, nc = 1, 3
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True)

    ax = axes[0, 0]
    ax.clear()
    for deg_idx, deg in y:
        # if pos_charge not in ["X-C__Pseudo-Mono", '-C__Pseudo-Mono']: continue
        sns.lineplot(
            x=subjects_inv,
            y=a_mean[:,
            deg_idx],
            label=deg,
            ax=ax,
            color=colors_map[deg],
            marker="o"
        )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(-.05, 1), loc="upper right", reverse=True)

    ax = axes[0, 1]
    ax.clear()
    for i, (deg_idx, deg_inv) in enumerate(y):
        ax.errorbar(
            x=xme[i],
            xerr=xerr[i],
            y=[u[1] for u in y][i],
            fmt="o",
            ecolor=colors_map[deg_inv],
            color=colors_map[deg_inv],
            # markerfacecolor=[colors_map[u[1]] for u in y]
        )
        if deg_idx == reference_idx:
            ax.vlines(xme[i], linestyle="--", color=colors_map[deg], ymax=len(y) - 1, ymin=0)
    ax.tick_params(axis="x", rotation=45)

    # model_dir = "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/estimation/4000w_4000s_4c_4t_15d_95a_tm/all/circ_est_mvn_reference_rl_masked"
    # (
    #     df,
	# 	encoder,
	# 	model,
	# 	posterior,
	# 	subjects,
	# 	subjects_inv,
	# 	position_charges,
	# 	position_charges_inv,
    #     *_,
    # ) = load(model_dir)
    # posterior.keys()
    # suptitle += f"\n{model.run_id}/{model._model.__name__}/mix:{model.use_mixture}"

    # param = posterior["a_delta_loc"]
    # print(param.shape)

    # ax = axes[0, 2]
    # ax.clear()
    # ax.axvline(x=0, color=colors_map[position_charges_inv[0][1:]], linestyle="--")
    # for i in range(param.shape[-1]):
    #     samples = param[:, i]
    #     pos_charge_inv = position_charges_inv[1:][i]
    #     sns.kdeplot(samples, color=colors_map[pos_charge_inv])
    # ax.tick_params(axis="both", labelleft=False, left=False)
    # ax.set_ylabel("")

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            sides = ["right", "top"]
            ax.spines[sides].set_visible(False)

    build_dir = model.build_dir.split("/")[-2:]
    build_dir = "/".join(build_dir)
    fig.suptitle(suptitle)
    return fig


def threshold_analysis_size(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
        subjects_inv,
		# positions,
		# positions_inv,
        degrees,
        degrees_inv,
        sizes,
        sizes_inv,
        num_features,
        mask_features
    ) = load(model_dir)
    assert sizes is not None
    assert sizes_inv is not None
    posterior.keys()

    print(model.features)
    a = np.nanmean(posterior[site.a], axis=0)
    print(a.shape)

    diff = a[..., 0, :] - a[..., 1, :]
    print(diff.shape)

    diff = np.nanmean(diff, axis=(-1,))
    print(diff.shape)
 
    pvalue, statistic, deg, me, eff = make_test(diff, mask=False)
    print(f"Me {sizes_inv[0]} - {sizes_inv[1]}: {list(zip(degrees_inv, np.round(me, 3)))}")
    print(f"p: {list(zip(degrees_inv, np.round(pvalue, 3)))}")
    print(f"df: {list(zip(degrees_inv, deg))}")

    labels = degrees_inv.copy()
    print(labels)

    plt.close("all")
    pvalue, statistic, deg, axes = make_compare(diff, labels)

    return


def threshold_estimation_lat(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
        subjects,
        subjects_inv,
		positions,
		positions_inv,
        degrees,
        degrees_inv,
        sizes,
        sizes_inv,
        num_features,
        mask_features
    ) = load(model_dir, estimation=True)

    param = posterior["a_delta_loc"]
    print(param.shape)

    plt.close("all")
    nr, nc = 1, 1
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True)

    ax = axes[0, 0]
    ax.clear()
    for i in range(param.shape[-1]):
        deg = degrees_inv[1:][i]
        samples = param[:, i]
        sns.kdeplot(samples, ax=ax, label=deg)
    ax.legend()

    return


def main():
    out = []

    # Threshold laterality
    model_dirs = [
        # rl_nov, with mixture
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-small-ground/hb_mvn_rl_nov_masked",
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-big-ground/hb_mvn_rl_nov_masked",

        # rl_nov, without mixture
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_fm/big-no-ground/hb_mvn_rl_nov_masked",
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_fm/big-ground/hb_mvn_rl_nov_masked/",
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_fm/small-no-ground/hb_mvn_rl_nov_masked",
        # "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_fm/small-ground/hb_mvn_rl_nov_masked",
    ]

    # out += [threshold_analysis_lat(model_dir) for model_dir in model_dirs]
    # out += [figure_lat(model_dir) for model_dir in model_dirs]
    # output_path = os.path.join(BUILD_DIR, "out-lat.pdf")
    # make_pdf(out, output_path)

    model_dirs = [
        "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-small-ground/lat_est_mvn_reference_rl_masked/test_run",
    ]
    out += [threshold_estimation_lat(model_dir) for model_dir in model_dirs]

    # model_dirs = [
    #     "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-small-ground/hb_mvn_rl_masked",
    #     "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/lat-big-ground/hb_mvn_rl_masked",
    # ]
    # [threshold_analysis_lat(model_dir) for model_dir in model_dirs]

    # # Threshold size
    # model_dirs = [
    #     "/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/size-ground/hb_mvn_rl_masked",
    # ]
    # [threshold_analysis_size(model_dir) for model_dir in model_dirs]

    # [selectivity_with_threshold(model_dir) for model_dir in model_dirs]

    return


if __name__ == "__main__":
    main()


    # def save_for_andres():
    #     a_mean = np.nanmean(a, axis=-1)
    #     print(a_mean.shape)
    #     output_path = os.path.join(os.getenv("HOME"), f"threshold__{model._model.__name__}__lat-" + model.run_id + ".pkl")
    #     with open(output_path, "wb") as f:
    #         pickle.dump((a_mean, positions, positions_inv, degrees, degrees_inv,), f)
    #     print(f"Saved to {output_path}")

    # def load_smalar_lat(src):
    #     with open(src, "rb") as f:
    #         a_mean, positions, positions_inv, degrees, degrees_inv, = pickle.load(f)
    #     return a_mean, positions, positions_inv, degrees, degrees_inv

    # # Compare laterality in small electrodes with ground return
    # src = "/home/andres/reports/hbmep/notebooks/rat/loghb/data_for_figures/threshold__hb_mvn_rl_nov_masked__lat-small-ground.pkl"
    # a_mean, positions, positions_inv, degrees, degrees_inv = load_smalar_lat(src)

    # # Compare laterality in big electrodes with ground return
    # src = "/home/andres/reports/hbmep/notebooks/rat/loghb/data_for_figures/threshold__hb_mvn_rl_nov_masked__lat-big-ground.pkl"
    # a_mean, positions, positions_inv, degrees, degrees_inv = load_smalar_lat(src)
