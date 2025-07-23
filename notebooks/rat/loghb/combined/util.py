import os
import pickle

import numpy as np
from hbmep.util import site

from hbmep.notebooks.rat.analysis import load_circ, load_shie, load_smalar

combined_dir = "/home/vishu/reports/hbmep/notebooks/rat/combined_data/4000w_4000s_4c_4t_15d_95a_tm/hb_rl_masked_hmaxPooled/L_CIRC___L_SHIE___C_SMA_LAR___J_RCML"

named_params = [site.a, site.b, site.g, site.h, site.v]


def make_model_dir(run_id):
    if run_id in {"diam", "radii", "vertices"}:
        model_dir = f"/home/vishu/reports/hbmep/notebooks/rat/loghb/circ/{run_id}/4000w_4000s_4c_4t_15d_95a_tm/hb_mvn_rl_masked"
        load_fn = load_circ
    elif run_id in {"shie"}:
        model_dir = "/home/vishu/reports/hbmep/notebooks/rat/loghb/shie/all/4000w_4000s_4c_4t_15d_95a_tm/robust_hb_mvn_rl_masked"
        load_fn = load_shie
    elif run_id in {"lat-small-ground", "lat-big-ground"}:
        model_dir = f"/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/{run_id}/robust_hb_mvn_rl_masked"
        load_fn = load_smalar
    elif run_id in {"size-ground", "size-no-ground"}:
        model_dir = f"/home/vishu/reports/hbmep/notebooks/rat/loghb/smalar/4000w_4000s_4c_4t_15d_95a_tm/{run_id}/robust_hb_mvn_rl_masked"
        load_fn = load_smalar
    else: raise ValueError(f"Incorrect run_id {run_id}")
    return model_dir, load_fn,


def make_h_max():
    run_ids = [
        "diam", "radii", "vertices", "shie",
        "lat-small-ground", "lat-big-ground",
        "size-ground"
    ]

    h = None
    for run_id in run_ids:
        model_dir, load_fn = make_model_dir(run_id)
        (
            df,
            encoder,
            model,
            posterior,
            *_
        ) = load_fn(model_dir)
        curr_h = posterior[site.h].copy()
        
        if run_id in {"shie", "lat-small-ground", "lat-big-ground", "size-ground", "size-no-ground"}:
            curr_h = curr_h.reshape(*curr_h.shape[:2], -1, curr_h.shape[-1])

        if h is None: h = curr_h.copy()
        else: h = np.concatenate([h, curr_h], axis=-2)
    
    h_max = np.nanmax(h, axis=-2, keepdims=True)
    return h_max

# make_h_max()

def load(run_id):
    model_dir, load_fn = make_model_dir(run_id)
    if run_id in {"diam", "radii", "vertices"}:
        (
            df,
            encoder,
            model,
            posterior,
            subjects,
            positions,
            num_features,
            *_
        ) = load_fn(model_dir)
        posterior = {u: posterior[u] for u in named_params}
    
    elif run_id in {"shie"}:
        (
            df,
            encoder,
            model,
            posterior,
            subjects,
            positions,
            charges,
            num_features,
            *_,
        ) = load_fn(model_dir)
        posterior = {u: posterior[u] for u in named_params}
        posterior = {
            u: v.reshape(*v.shape[:-3], -1, v.shape[-1])
            for u, v in posterior.items()
        }
        # for u, v in posterior.items(): print(u, v.shape)

        position_charges = []
        counter = 0
        for _, pos_inv in positions:
            for _, ch_inv in charges:
                position_charges.append((counter, f"{pos_inv}__{ch_inv}"))
                counter += 1
        positions = position_charges
        # print(positions)

    # model_dir = combined_dir
    # src = os.path.join(model_dir, "combined_inf.pkl")
    # with open(src, "rb") as f: _, _, combined_post = pickle.load(f)
    # h_max = combined_post["h_max"]; h_max = np.max(h_max, axis=0, keepdims=True)
    # print(h_max.shape)
    # np.sum(posterior[site.h] > h_max)

    # h_max = make_h_max()
    # assert not np.any(posterior[site.h] > h_max)

    h_max = posterior[site.h].copy()
    h_max = np.max(h_max, axis=-2, keepdims=True)

    return (
            df,
            model,
            posterior,
            h_max,
            subjects,
            positions,
    )

# load("shie")
