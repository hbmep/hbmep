import os
import warnings

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from hbmep.util import site

from hbmep.notebooks.rat.util import load_model
from hbmep.notebooks.rat.util import (
    make_compare3p,
    make_pdf,
    make_dump,
)
from hbmep.notebooks.rat.testing import (
    checknans,
    check1,
    check2,
)

SEPARATOR = "___"
MODEL_DIR = "/home/vishu/reports/hbmep/notebooks/rat/combined_data/4000w_4000s_4c_4t_15d_95a_tm/hb_rl_masked_hmaxPooled/L_CIRC___L_SHIE___C_SMA_LAR___J_RCML"
BUILD_DIR = "/home/vishu/reports/hbmep/notebooks/rat/loghb/combined_data/out-efficacy"
os.makedirs(BUILD_DIR, exist_ok=True)


def run(
    run_id,
    model_dir,
    correction=True,
    fig=None,
    dump=False,
):
    df, model, posterior, subjects, positions, = arrange(run_id, model_dir)
    a = posterior[site.a].copy()
    # a = np.mean(a, axis=0, keepdims=True)
    a = np.median(a, axis=0, keepdims=True)

    if run_id in {"diam", "radii", "vertices", "shie"}:
        checknans(a)
        a_mean = np.mean(a, axis=-1)
        check1(a, a_mean)
    
    elif run_id in {"lat-small-ground", "lat-big-ground"}:
        check2(a)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            a = np.nanmean(a, axis=2)
        a_mean = np.mean(a, axis=-1)

    (
        fig,
        positions,
        measure_mean,
        diff,
        diff_mean,
        diff_err,
        negate,
        *_
    ) = make_compare3p(a_mean, positions, negate=True, correction=correction, fig=fig)
    fig, axes = fig

    if "big" in run_id:
        a_mean = np.mean(a_mean, axis=0)
        u, v = 2, 1
        t = stats.ttest_ind(a=a_mean[:, u], b=a_mean[:, v], nan_policy="omit").pvalue
        u, v = positions[u][1], positions[v][1]
        title = f"Indep. ttest {u} vs {v}: {t:.3f}"
        fig.suptitle(title)

    output_path = os.path.join(BUILD_DIR, f"{run_id}.pkl")
    if dump: make_dump((positions, measure_mean, diff, diff_mean, diff_err, negate,), output_path)
    return (fig, axes),


def load(model_dir):
    df, encoder, posterior, model, _ = load_model(model_dir)
    subjects = sorted(df[model.features[0]].unique())
    t = encoder[model.features[0]].inverse_transform(subjects)
    subjects = list(zip(subjects, t))
    f1 = sorted(df[model.features[1]].unique())
    f1_inv = encoder[model.features[1]].inverse_transform(f1)
    f1 = list(zip(f1, f1_inv))
    return df, encoder, posterior, model, subjects, f1


def get_experiment(f1, experiment):
    remove_str = "___" + experiment
    subset = [
        (u, v.replace(remove_str, ""))
        for u, v in f1
        if v.split(SEPARATOR)[-1] == experiment
    ]
    subset = [(u, v.replace("___", "__")) for u, v in subset]
    return subset


def arrange(run_id, model_dir):
    df, encoder, posterior, model, subjects, f1 = load(model_dir)
    h_max = posterior.pop("h_max")
    if run_id in {"diam", "radii", "vertices"}:
        experiment = "L_CIRC"
        experiment = get_experiment(f1, experiment)
        if run_id == "diam":
            experiment = [
                (u, v) for u, v in experiment
                if "" not in v.split("-")
                and "C" not in v.split("-")
            ]
            assert len(experiment) == 4
        elif run_id == "radii":
            experiment = [
                (u, v) for u, v in experiment
                if "" not in v.split("-")
                and "C" in v.split("-")
            ]
            assert len(experiment) == 8
        else:
            experiment = [(u, v) for u, v in experiment if "" in v.split("-")]
            assert len(experiment) == 9
        assert sorted(experiment) == experiment
        
    elif run_id == "shie":
        experiment = "L_SHIE"
        experiment = get_experiment(f1, experiment)
        assert len(experiment) == 8
        assert sorted(experiment) == experiment

    elif run_id in {"lat-small-ground", "lat-big-ground", "size-ground"}:
        experiment = "C_SMA_LAR"
        experiment = get_experiment(f1, experiment)
        experiment = [(u, v.split("__")) for u, v in experiment]
        if "ground" in run_id:
            experiment = [
                (u, v) for u, v in experiment
                if v[0].split("-")[0] == ""
            ]
        else:
            raise ValueError("Logic for bipolar not implemented for C_SMA_LAR")
        experiment = sorted(
            experiment,
            key=lambda x: (x[1][0], x[1][1], x[1][2])
        )
        if "small" in run_id or "big" in run_id:
            if "small" in run_id: size = "S"
            elif "big" in run_id: size = "B"
            experiment = [(u, v) for u, v in experiment if v[-1] == size]

    elif run_id in {
        "rcml-ground", "rcml-ML", "rcml-RC", "rcml-RA", "rcml"
    }:
        experiment = "J_RCML"
        experiment = get_experiment(f1, experiment)
        if "ground" in run_id:
            experiment = [(u, v) for u, v in experiment if "" in v.split("-")]
            experiment = [(u, v) for u, v in experiment if v[-1] == "L"]
        elif "ML" in run_id:
            experiment = [
                (u, v) for u, v in experiment
                if "" not in v.split("-")                           # not ground
                and v.split("-")[0][:-1] == v.split("-")[1][:-1]    # same segment
            ]
        elif "RC" in run_id:
            experiment = [
                (u, v) for u, v in experiment
                if "" not in v.split("-")                           # not ground
                and v.split("-")[0][-1] == v.split("-")[1][-1]      # same degree
                and v.split("-")[0][-1] == "L"
            ]
        elif "RA" in run_id:
            experiment = [
                (u, v) for u, v in experiment
                if "" not in v.split("-")                           # not ground
                and v.split("-")[0][:-1] != v.split("-")[1][:-1]    # not same segment
                and v.split("-")[0][-1] != v.split("-")[1][-1]      # not same degree
            ]
        # else:
        #     raise ValueError

    else:
        raise ValueError

    idx, positions = zip(*experiment)
    named_params = [u for u in posterior.keys() if u != "h_max"]
    posterior = {
        u: v[..., idx, :]
        for u, v in posterior.items() if u in named_params
    }
    positions = list(zip(range(len(positions)), positions))

    df = df[df[model.features[1]].isin(idx)].reset_index(drop=True).copy()
    df[model.features[1]] = (
        df[model.features[1]].map(dict(zip(idx, range(len(positions)))))
    )

    if run_id in {"size-ground"}:
        positions = [(u, v) for u, v in positions if "-LM2" not in v]
        idx, positions = zip(*positions)
        posterior = {
            u: v[..., idx, :]
            for u, v in posterior.items() if u in named_params
        }
        positions = list(zip(range(len(positions)), positions))
        df = df[df[model.features[1]].isin(idx)].reset_index(drop=True).copy()
        df[model.features[1]] = (
            df[model.features[1]].map(dict(zip(idx, range(len(positions)))))
        )

    if run_id in {"lat-small-ground", "lat-big-ground"}:
        idx, positions = zip(*positions)
        t = [u[0] for u in positions]
        t = np.array(t).reshape(2, -1)
        np.testing.assert_equal(
            np.unique(t, axis=1), np.array([['-C5'], ['-C6']])
        )
        t = [u[-1] for u in positions]
        assert len(set(t)) == 1
        t = [u[1] for u in positions]
        t = np.array(t).reshape(2, -1)
        np.testing.assert_equal(t[0], t[1])
        positions = t[0].tolist()
        positions = list(zip(range(len(positions)), positions))
        posterior = {
            u: v.reshape(*v.shape[:2], 2, -1, v.shape[-1])
            for u, v in posterior.items()
            if u in named_params
        }
        h_max = h_max[..., None, :, :]
    
    elif run_id in {"size-ground"}:
        idx, positions = zip(*positions)
        t = [u[0] for u in positions]
        t = np.array(t).reshape(2, -1, 2)
        assert np.unique(t[0]) == np.array(["-C5"])
        assert np.unique(t[1]) == np.array(["-C6"])
        t = [u[-1] for u in positions]
        t = np.array(t).reshape(2, -1, 2)
        assert np.unique(t[..., 0]) == np.array(["B"])
        assert np.unique(t[..., 1]) == np.array(["S"])
        sizes = t[0, 0, :].tolist()
        sizes = list(zip(range(len(sizes)), sizes))
        t = [u[1] for u in positions]
        t = np.array(t).reshape(2, -1, 2)
        np.testing.assert_equal(t[0, ..., 0], t[1, ..., 0])
        np.testing.assert_equal(t[0, ..., 1], t[1, ..., 1])
        np.testing.assert_equal(t[0, ..., 0], t[0, ..., 1])
        np.testing.assert_equal(t[1, ..., 0], t[1, ..., 1])
        positions = t[0, ..., 0].tolist()
        positions = list(zip(range(len(positions)), positions))
        posterior = {
            u: v.reshape(*v.shape[:-2], 2, -1, 2, v.shape[-1])
            for u, v in posterior.items()
            if u in named_params
        }
        # t = measure[..., np.array(idx).reshape(2, -1, 2)].copy()
        # np.testing.assert_almost_equal(arr, t)
        # arr = np.nanmean(arr, axis=0, keepdims=True); print(arr.shape)
        h_max = h_max[:, :, :, None, None, :]
        assert sizes[0][1] == "B"; assert sizes[1][1] == "S"
        # diff = arr[..., 0] - arr[..., 1]; print(diff.shape)
        # diff = np.nanmean(diff, axis=-2); print(diff.shape)

    # elif run_id in {"rcml-ground"}:
    #     idx, positions = zip(*positions)
    #     t = [u[1:-1] for u in positions]
    #     t = np.array(t).reshape(-1, 2)
    #     np.testing.assert_equal(
    #         np.unique(t, axis=1),
    #         np.array([['C5'], ['C6'], ['C7'], ['C8']])
    #     )
    #     t = [u[-1] for u in positions]
    #     t = np.array(t).reshape(-1, 2)
    #     np.testing.assert_equal(
    #         np.unique(t, axis=0),
    #         np.array([['L', 'M']])
    #     )

    posterior["h_max"] = h_max
    return df, model, posterior, subjects, positions,


def main():
    model_dir = MODEL_DIR

    out = []
    run_ids = [
        "diam",
        "radii",
        "vertices",
        "shie",
        "lat-small-ground",
        "lat-big-ground",
        # "size-ground",
    ]
    out += [
        run(run_id, model_dir, dump=True)[0][0]
        for run_id in run_ids
    ]
    output_path = os.path.join(BUILD_DIR, "out.pdf")
    make_pdf(out, output_path)

    return


if __name__ == "__main__":
    main()
