import os
import pickle
import warnings
from functools import partial

import numpy as np
from scipy import stats
from hbmep.util import site
import matplotlib.pyplot as plt
import seaborn as sns
from hbmep import functional as F

from hbmep.notebooks.rat.util import (
	make_compare3p,
	make_compare3p_bar,
    get_response_colors,
)
from analysis__constants import SEPARATOR, MODEL_DIR, NAMED_PARAMS


def get_experiment(f1, experiment):
    remove_str = SEPARATOR + experiment
    subset = [
        (u, v.replace(remove_str, ""))
        for u, v in f1
        if v.split(SEPARATOR)[-1] == experiment
    ]
    subset = [(u, v.replace("___", "__")) for u, v in subset]
    return subset


def get_circ(f1):
    experiment = "L_CIRC"
    circ = get_experiment(f1, experiment); assert len(circ) == 21
    vertices = [(u, v) for u, v in circ if v.split("-")[0] == ""]; assert len(vertices) == 9
    radii  = [(u, v) for u, v in circ if (u, v) not in vertices and v.replace(SEPARATOR + experiment, "")[-1] == "C"]; assert len(radii) == 8
    diam = [(u, v) for u, v in circ if (u, v) not in vertices + radii]; assert len(diam) == 4
    assert set(diam).isdisjoint(set(radii)) and set(diam).isdisjoint(set(vertices)) and set(radii).isdisjoint(set(vertices))
    assert diam == sorted(diam, key=lambda x: x[0])
    assert radii == sorted(radii, key=lambda x: x[0])
    assert vertices == sorted(vertices, key=lambda x: x[0])
    return circ, diam, radii, vertices


def get_shie(f1):
    experiment = "L_SHIE"
    shie = get_experiment(f1, experiment); assert len(shie) == 8
    assert shie == sorted(shie, key=lambda x: x[0])
    return shie


def get_smalar(f1):
    experiment = "C_SMA_LAR"
    smalar = get_experiment(f1, experiment)
    smalar = [(u, v.split("__")) for u, v in smalar]
    ground = [(u, v) for u, v in smalar if v[0].split("-")[0]==""]
    no_ground = [u for u in smalar if u not in ground]
    sort = partial(sorted, key=lambda x: (x[1][0], x[1][1], x[1][2]))
    ground = sort(ground)
    no_ground = sort(no_ground)
    lat_small_ground = [(u, v) for u, v in ground if v[-1] == "S"]
    lat_big_ground = [(u, v) for u, v in ground if v[-1] == "B"]
    assert lat_small_ground == sort(lat_small_ground)
    assert lat_big_ground == sort(lat_big_ground)
    assert sorted(ground) == sorted(lat_small_ground + lat_big_ground)
    return smalar, ground, no_ground, lat_small_ground, lat_big_ground


def get_rcml(f1):
    experiment = "J_RCML"
    from analysis__util import get_experiment
    rcml = get_experiment(f1, experiment)
    rcml_ground = [(u, v) for u, v in rcml if v.split("-")[0] == ""]
    rcml_no_ground = [u for u in rcml if u not in rcml_ground]
    return rcml, rcml_ground, rcml_no_ground


def load(run_id):
    model_dir = MODEL_DIR; named_params = NAMED_PARAMS
    src = os.path.join(model_dir, "combined_inf.pkl")
    with open(src, "rb") as f: df, encoder, posterior = pickle.load(f)
    src = os.path.join(model_dir, "combined_model.pkl")
    with open(src, "rb") as f: model, = pickle.load(f)
    src = os.path.join(model_dir, "combined_mask.pkl")
    with open(src, "rb") as f: num_features, mask_features, = pickle.load(f)

    subjects = sorted(df[model.features[0]].unique())
    subjects_inv = encoder[model.features[0]].inverse_transform(subjects)
    subjects = list(zip(subjects, subjects_inv))

    f1 = sorted(df[model.features[1]].unique())
    f1_inv = encoder[model.features[1]].inverse_transform(f1)
    f1 = list(zip(f1, f1_inv))

    circ, diam, radii, vertices = get_circ(f1)
    shie = get_shie(f1)
    smalar, ground, no_ground, lat_small_ground, lat_big_ground = get_smalar(f1)
    rcml, rcml_ground, rcml_no_ground = get_rcml(f1)
    comparisons = {
        "diam": diam,
        "radii": radii,
        "vertices": vertices,
        "shie": shie,
        "lat-small-ground": lat_small_ground,
        "lat-big-ground": lat_big_ground,
        "size-ground": ground,
        "rcml-ground": rcml_ground,
        "rcml-no-ground": rcml_no_ground,
    }
    assert run_id in comparisons.keys()

    positions = comparisons[run_id]
    idx, labels = zip(*positions)
    posterior = {u: posterior[u] for u in named_params} # (S, P, C, M)
    h_max = posterior.pop("h_max")
    for u, v in posterior.items(): assert v.shape == posterior[site.a].shape
    posterior = {u: v[..., idx, :] for u, v in posterior.items()}
    positions = list(zip(range(len(labels)), labels))
    
    df_idx = df[model.features[1]].isin(idx)
    df = df[df_idx].reset_index(drop=True).copy()
    df[model.features[1]] = df[model.features[1]].map(dict(zip(idx, range(len(labels)))))
    return df, model, posterior, h_max, subjects, positions,


def arrange(run_id, measure, positions):
    # measure (S, P, C)
    if run_id in {"lat-small-ground", "lat-big-ground"}:
        idx, positions = zip(*positions)
        t = [u[0] for u in positions]; t = np.array(t).reshape(2, -1);
        np.testing.assert_equal(np.unique(t, axis=1), np.array([['-C5'], ['-C6']]))
        t = [u[-1] for u in positions]; assert len(set(t)) == 1
        t = [u[1] for u in positions]; t = np.array(t).reshape(2, -1);
        np.testing.assert_equal(t[0], t[1])
        labels = t[0].tolist()
        labels = list(zip(range(len(labels)), labels))
        arr = measure[..., idx].copy(); arr = arr.reshape(*arr.shape[:-1], 2, -1)
        t = measure[..., np.array(idx).reshape(2, -1)].copy()
        np.testing.assert_almost_equal(arr, t)

        diff = arr[..., None] - arr[..., None, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            diff = np.nanmean(diff, axis=-3)
            t = np.nanmean(arr, axis=-2)
        t = t[..., None] - t[..., None, :]
        np.testing.assert_almost_equal(diff, t)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            arr = np.nanmean(arr, axis=-2)
        return arr, labels

    if run_id in {"size-ground"}:
        positions = [(u, v) for u, v in positions if "-LM2" not in v]
        idx, positions = zip(*positions)
        t = [u[0] for u in positions]; t = np.array(t).reshape(2, -1, 2)
        assert np.unique(t[0]) == np.array(["-C5"])
        assert np.unique(t[1]) == np.array(["-C6"])
        t = [u[-1] for u in positions]; t = np.array(t).reshape(2, -1, 2)
        assert np.unique(t[..., 0]) == np.array(["B"])
        assert np.unique(t[..., 1]) == np.array(["S"])
        sizes = t[0, 0, :].tolist(); sizes = list(zip(range(len(sizes)), sizes))
        t = [u[1] for u in positions]; t = np.array(t).reshape(2, -1, 2)
        np.testing.assert_equal(t[0, ..., 0], t[1, ..., 0])
        np.testing.assert_equal(t[0, ..., 1], t[1, ..., 1])
        np.testing.assert_equal(t[0, ..., 0], t[0, ..., 1])
        np.testing.assert_equal(t[1, ..., 0], t[1, ..., 1])
        labels = t[0, ..., 0].tolist()
        labels = list(zip(range(len(labels)), labels))
        arr = measure[..., idx].copy(); arr = arr.reshape(*arr.shape[:-1], 2, -1, 2)
        t = measure[..., np.array(idx).reshape(2, -1, 2)].copy()
        np.testing.assert_almost_equal(arr, t)
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=RuntimeWarning)
        #     arr = np.nanmean(arr, axis=0, keepdims=True)

        assert sizes[0][1] == "B"; assert sizes[1][1] == "S"
        diff = arr[..., 0] - arr[..., 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            diff = np.nanmean(diff, axis=-2)
        return diff, labels

    idx, positions = zip(*positions)
    measure = measure[..., idx].copy()
    positions = list(zip(range(len(positions)), positions))
    return measure, positions


def plot2d(run_id, measure, positions, negate, correction=False, palette="viridis", consistent_colors=False):
    measure, positions = arrange(run_id, measure, positions)
    if "size" in run_id: return plot2d_bar(run_id, measure, positions, negate, correction=correction, palette=palette, consistent_colors=consistent_colors)

    nr, nc = 1, 4
    fig, axes = plt.subplots(
        *(nr, nc), figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True
    )
    _, diff_positions, diff_mean, diff_err, colors, negate, *_, = make_compare3p(
        measure,
		positions,
		negate=negate,
		fig=(fig, axes),
		correction=correction,
        palette=palette,
		consistent_colors=consistent_colors,
    )

    src = f"/home/vishu/reports/hbmep/notebooks/rat/loghb/out/{run_id}.pkl"
    with open(src, "rb") as f:
        th_diff_positions, _, th_diff_mean, th_diff_err, th_negate = pickle.load(f)
    u, v = zip(*diff_positions); inv = dict(zip(v, u))
    u, v = zip(*th_diff_positions); th_inv = dict(zip(v, u))
    assert sorted(inv.keys()) == sorted(th_inv.keys())
    reference_idx = diff_positions[-1][0]
    th_reference_idx = th_inv[diff_positions[-1][1]]
    ax = axes[0, -1]; ax.clear()
    for pos_idx, pos_inv in diff_positions:
        th_pos_idx = th_inv[pos_inv]
        th_indexer = (th_pos_idx, th_reference_idx)
        if th_negate: th_indexer = (th_reference_idx, th_pos_idx)
        xme = th_diff_mean[*th_indexer]
        xerr = th_diff_err[*th_indexer]
        indexer = (pos_idx, reference_idx)
        if negate: indexer = (reference_idx, pos_idx)
        yme = diff_mean[*indexer]
        yerr = diff_err[*indexer]
        ax.errorbar(
            x=xme,
            xerr=xerr,
            y=yme,
            yerr=yerr,
            fmt="o",
            ecolor=colors[pos_inv],
            color=colors[pos_inv],
            label=pos_inv,
        )
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            sides = ["top", "right"]
            sides = ["right", "top"]
            ax.spines[sides].set_visible(False)
            if ax.get_legend(): ax.get_legend().remove()
            ax.set_xlabel(""); ax.set_ylabel("")
    ax = axes[0, 0]
    ax.legend(bbox_to_anchor=(-.2, 1), loc="upper right", reverse=True, fontsize="x-small")
    arrow = "←" if negate else "→"
    ax.set_ylabel(f"{arrow} is better")
    ax = axes[0, 1]
    ax.set_xlabel(f"→ is better")
    ax = axes[0, -1]
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", reverse=True, fontsize="x-small")
    ax.set_xlabel("→ is more effective")
    ax.set_ylabel("→ is more selective")
    return (fig, axes),


def oneway(run_id):

    nr, nc = 1, 1
    fig, axes = plt.subplots(
        *(nr, nc), figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True
    )
    fig.show()


    src = f"/home/vishu/reports/hbmep/notebooks/rat/loghb/out/{run_id}.pkl"
    with open(src, "rb") as f:
        positions, _, diff_mean, diff_err, negate = pickle.load(f)
    u, v = zip(*positions); inv = dict(zip(v, u))
    reference_idx = positions[-1][0]

    ax = axes[0, 0]; ax.clear()
    for pos_idx, pos_inv in positions:
        if negate:
            xme = diff_mean[-1, :]
            xerr = diff_err[..., -1, :]
        else:
            xme = diff_mean[:, -1]
            xerr = diff_err[..., :, -1]
        xme = xme[pos_idx]
        xerr = xerr[pos_idx]
        ax.errorbar(
            x=xme,
            xerr=xerr,
            y=pos_inv,
            fmt="o",
            # ecolor=colors[pos_inv],
            # color=colors[pos_inv],
        )


    def draw_electrode_array(ax, position, label, radius=0.5, electrode_radius=0.1):
        """
        Draw a circular electrode array at specified position with labeled electrodes colored.
        
        Parameters:
        - ax: matplotlib axes object
        - position: (x, y) tuple for array center position
        - label: string indicating which electrodes to highlight (e.g., "S-N", "SE-NW", etc.)
        - radius: radius of the electrode array circle
        - electrode_radius: radius of each individual electrode
        """
        x, y = position
        center = (x, y)
        
        # Define electrode positions (8 electrodes in a circle)
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        electrode_positions = []
        for angle in angles:
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            electrode_positions.append((x + dx, y + dy))
        
        # Draw the central circle (empty)
        ax.add_patch(Circle(center, radius, fill=False, color='black', lw=1))
        
        # Determine which electrodes to highlight based on label
        highlight_indices = []
        if label == "S-N":
            highlight_indices = [0, 4]  # South and North
        elif label == "SE-NW":
            highlight_indices = [1, 5]  # Southeast and Northwest
        elif label == "E-W":
            highlight_indices = [2, 6]  # East and West
        elif label == "NE-SW":
            highlight_indices = [3, 7]  # Northeast and Southwest
        
        # Draw all electrodes
        for i, (ex, ey) in enumerate(electrode_positions):
            if i in highlight_indices:
                # Highlighted electrode
                ax.add_patch(Circle((ex, ey), electrode_radius, fill=True, color='red'))
            else:
                # Normal electrode
                ax.add_patch(Circle((ex, ey), electrode_radius, fill=True, color='gray', alpha=0.3))
        
        # Add connection lines between highlighted electrodes
        if len(highlight_indices) == 2:
            i1, i2 = highlight_indices
            x1, y1 = electrode_positions[i1]
            x2, y2 = electrode_positions[i2]
            ax.plot([x1, x2], [y1, y2], 'r--', lw=1, alpha=0.5)

# Example usage:
fig, ax = plt.subplots(figsize=(8, 6))

# Add your main plot content here...

# Place electrode arrays next to y-axis labels
y_labels = ["S-N", "SE-NW", "NE-SW", "E-W"]
y_positions = np.linspace(0.8, 0.2, len(y_labels))

for label, y_pos in zip(y_labels, y_positions):
    # Position to the left of the y-axis label
    draw_electrode_array(ax, position=(-0.5, y_pos), label=label)
    
    # Add the label text
    ax.text(-0.3, y_pos, label, va='center', ha='left')

ax.set_xlim(-1, 1)
ax.set_ylim(0, 1)
ax.axis('off')  # Turn off axes for cleaner look
plt.tight_layout()
plt.show()
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            sides = ["top", "right"]
            sides = ["right", "top"]
            ax.spines[sides].set_visible(False)
            if ax.get_legend(): ax.get_legend().remove()
            ax.set_xlabel(""); ax.set_ylabel("")
    ax = axes[0, 0]
    ax.legend(bbox_to_anchor=(-.2, 1), loc="upper right", reverse=True, fontsize="x-small")
    arrow = "←" if negate else "→"
    ax.set_ylabel(f"{arrow} is better")
    ax = axes[0, 1]
    ax.set_xlabel(f"→ is better")
    ax = axes[0, -1]
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", reverse=True, fontsize="x-small")
    ax.set_xlabel("→ is more effective")
    ax.set_ylabel("→ is more selective")
    return (fig, axes),

        
        


def plot2d_bar(run_id, measure, positions, negate=False, correction=False, palette="viridis", consistent_colors=False):
    nr, nc = 1, 3
    fig, axes = plt.subplots(
        *(nr, nc),
        figsize=(5 * nc, 3 * nr),
        squeeze=False,
        constrained_layout=True
    )

    (
        fig,
        diff_positions,
        diff_mean,
        diff_err,
        colors,
        *_,
    ) = make_compare3p_bar(
        measure, positions, correction=correction, fig=(fig, axes), negate=negate, palette=palette, consistent_colors=consistent_colors
    )
    fig, axes = fig

    src = f"/home/vishu/reports/hbmep/notebooks/rat/loghb/out/{run_id}.pkl"
    with open(src, "rb") as f:
        th_diff_positions, _, th_diff_mean, th_diff_err, th_negate = pickle.load(f)

    u, v = zip(*diff_positions); inv = dict(zip(v, u))
    u, v = zip(*th_diff_positions); th_inv = dict(zip(v, u))
    assert sorted(inv.keys()) == sorted(th_inv.keys())

    ax = axes[0, -1]; ax.clear()
    for pos_idx, pos_inv in diff_positions:
        th_pos_idx = th_inv[pos_inv]
        xme = th_diff_mean[th_pos_idx]
        xerr = th_diff_err[th_pos_idx]
        yme = diff_mean[pos_idx]
        yerr = diff_err[pos_idx]
        ax.errorbar(
            x=xme,
            xerr=xerr,
            y=yme,
            yerr=yerr,
            fmt="o",
            ecolor=colors[pos_inv],
            color=colors[pos_inv],
            label=pos_inv,
        )

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            sides = ["top", "right"]
            sides = ["right", "top"]
            ax.spines[sides].set_visible(False)
            if ax.get_legend(): ax.get_legend().remove()
            ax.set_xlabel(""); ax.set_ylabel("")

    ax = axes[0, 0]
    ax.legend(bbox_to_anchor=(-.2, 1), loc="upper right", reverse=True)
    ax.set_ylabel("log( B/S )")
    if negate: ax.set_ylabel("log( S/B )")
    ax = axes[0, -1]
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", reverse=True)
    ax.set_xlabel("log( B/S )")
    if th_negate: ax.set_xlabel("log( S/B )")
    ax.set_ylabel("log( B/S )")
    if negate: ax.set_ylabel("log( S/B )")
    ax.axvline(0, linestyle="--", color="k")
    ax.axhline(0, linestyle="--", color="k")
    return (fig, axes),


def plot_rc_group_by_muscle(run_id):
    (
        df,
        model,
        posterior,
        h_max,
        subjects,
        positions,
        *_,
    ) = load(run_id)
    df_features = df[model.features].apply(tuple, axis=1)
    colors = get_response_colors(model.response)
    min_intensity, max_intensity = df[model.intensity].min(), df[model.intensity].max()
    out = []


    def body_plot(posterior, response):

        nr, nc = len(subjects), len(positions)
        fig, axes = plt.subplots(
            *(nr, nc), figsize=(2.5 * nc, 1.2 * nr), squeeze=True, constrained_layout=True,
            sharex="row"
        )

        for subject_idx, subject in subjects:
            for pos_idx, pos in positions:
                idx = df_features.isin([(subject_idx, pos_idx,)])
                ccdf = df[idx].reset_index(drop=True).copy()
                x = ccdf[model.intensity]
                x_pred = np.linspace(x.min(), x.max(), 200)
                ax = axes[subject_idx, pos_idx]; ax.clear()
                for response_idx, res in enumerate(response):
                    # if response_idx not in {2}: continue
                    color = colors[response_idx]
                    y = F.rectified_logistic(x_pred[:, None], **{
                        u: v[:, subject_idx, pos_idx, response_idx][None]
                        for u, v in posterior.items()
                    }); y = np.mean(y, axis=-1)
                    sns.lineplot(x=x_pred, y=y, ax=ax, color=color, linewidth=1)
                    ax.axvline(
                        posterior[site.a][:, subject_idx, pos_idx, response_idx].mean(),
                        color=color,
                        linestyle="--",
                        linewidth=1
                    )
                    y = ccdf[res]
                    sns.scatterplot(x=x, y=y, ax=ax, color=color, s=5, zorder=20)
                if not subject_idx: ax.set_title(pos)
                if not pos_idx: ax.set_ylabel(subject)

        for i in range(nr):
            for j in range(nc):
                ax = axes[i, j]
                ax.tick_params(axis="both", labelbottom=True, labelsize="x-small")
                ax.set_xlabel("")
                if j: ax.set_ylabel("")
                # sides = ["right", "top"]
                # ax.spines[sides].set_visible(False)

        fig.align_labels()
        return (fig, axes),


    (fig, axes), = body_plot(response=model.response, posterior=posterior)
    fig.suptitle(f"Estimated curves")
    out.append(fig)
    posterior[site.h] /= h_max; posterior[site.v] /= h_max
    (fig, axes), = body_plot(response=[r + "_norm" for r in model.response], posterior=posterior)
    fig.suptitle(f"Normalized curves")
    # for i in range(axes.shape[0]):
    #     for j in range(axes.shape[1]):
    #         ax = axes[i, j]
    #         ax.sharey(axes[0, 0])
    out.append(fig)
    return out


def plot_rc(run_id, group_by_muscle):
    if group_by_muscle: return plot_rc_group_by_muscle(run_id)
    (
        df,
        model,
        posterior,
        h_max,
        subjects,
        positions,
    ) = load(run_id)
    colors = get_response_colors(model.response)
    combinations = df[model.features].apply(tuple, axis=1).unique()
    combinations = sorted(combinations, key=lambda x: (x[1], x[0]))
    num_combinations = len(combinations)
    num_rows = 10
    num_pages = num_combinations // num_rows + (num_combinations % num_rows > 0)
    num_columns = 2 * model.num_response

    df_features = df[model.features].apply(tuple, axis=1)
    func = F.rectified_logistic
    named_params = [site.a, site.b, site.g, site.h, site.v]

    out = []; counter = 0
    for page in range(num_pages):
        num_rows_current = min(num_rows, num_combinations - page * num_rows)
        fig, axes = plt.subplots(
            *(num_rows_current, num_columns),
            figsize=(3.5 * num_columns, 2 * num_rows_current),
            squeeze=True,
            constrained_layout=True
        )
        for row in range(num_rows_current):
            combination = combinations[counter]
            subject_idx, pos_idx = combination
            idx = df_features.isin([combination])
            ccdf = df[idx].reset_index(drop=True).copy()
            for response_idx, response in enumerate(model.response):
                ax = axes[row, 2 * response_idx]; ax.clear()
                x = ccdf[model.intensity]; y = ccdf[response].to_numpy()
                sns.scatterplot(x=x, y=y, ax=ax, color=colors[response_idx])
                x_pred = np.linspace(x.min(), x.max(), 100)
                curr_posterior = {u: v.copy() for u, v in posterior.items()}
                curr_posterior = {u: curr_posterior[u][..., *combination, response_idx][None] for u in named_params}
                y_pred = func(x=x_pred[:, None], **curr_posterior)
                y_pred_mean = np.mean(y_pred, axis=-1)
                sns.lineplot(x=x_pred, y=y_pred_mean, ax=ax, color="k")
                pos_inv = [(u, v) for u, v in positions if u == pos_idx]; assert len(pos_inv) == 1; pos_inv = pos_inv[0][1]
                if not response_idx: ax.set_title(f"amap0{subject_idx + 1}, {pos_inv}")
                ax.set_xlabel(""); ax.set_ylabel("")
                ax = axes[row, 2 * response_idx + 1]; ax.clear()
                y = ccdf[response + "_norm"].to_numpy()
                curr_h_max = h_max[:, subject_idx, 0, response_idx][None]
                curr_posterior[site.h] /= curr_h_max; curr_posterior[site.v] /= curr_h_max
                sns.scatterplot(x=x, y=y, ax=ax, color=colors[response_idx])
                y_pred = func(x=x_pred[:, None], **curr_posterior)
                y_pred_mean = np.mean(y_pred, axis=-1)
                sns.lineplot(x=x_pred, y=y_pred_mean, ax=ax, color="k")
                ax.set_xlabel(""); ax.set_ylabel("")
            counter += 1
        print(f"Page {page + 1} of {num_pages} done.")
        out.append(fig)
    return out



def diag_circ():
    nr, nc = 1, 1
    fig, axes = plt.subplots(
        *(nr, nc), figsize=(5 * nc, 3 * nr), squeeze=False, constrained_layout=True,
    )
    fig.show()

    ax = axes[0, 0]

    r = 1
    a, b = 0, 0
    x, y = [], []
    ts = [(np.pi / 4) * i for i in range(8)] 

    for t in ts:
        x.append(a + r * np.cos(t))
        y.append(b + r * np.sin(t))

    sns.scatterplot(x=x, y=y, ax=ax)
    
    ax.clear()
    plot_circle_on_ax(ax, 0, 0, .05, color="grey")


