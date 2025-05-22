import os
import pickle
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from jax import random, numpy as jnp
from numpyro.diagnostics import hpdi
import hbmep as mep
from hbmep.util import (
    timing,
    generate_response_colors,
    invert_combination
)

from hbmep.notebooks.util import (
    clear_axes,
    make_pdf,
)
from models import nHB
from util import Site as site, load_model, make_serial
from constants import (
    SIMULATION_FACTOR_DIR as BUILD_DIR,
    MAPPING,
    FACTOR,
    FACTORS_SPACE,
    REP,
    TOTAL_REPS,
)


def simulate_data(model_dir):
    (
        df,
		encoder,
		model,
		posterior,
        *_,
    ) = load_model(model_dir)
    posterior.keys()

    if site.outlier_prob in posterior.keys():
        posterior[site.outlier_prob] *= 0

    # model.n_jobs = 1
    [posterior.pop(u) for u in [site.obs, site.mu]]
    print(posterior.keys())

    feature0 = sorted(df[model.features[0]].unique())
    t = encoder[model.features[0]].inverse_transform(feature0)
    feature0 = list(zip(feature0, t))

    # idx = df[model.features[0]].isin([5])
    # x = df[idx].reset_index(drop=True)[model.intensity]
    # x = sorted(x.unique())
    # print(x, len(x))
    # print([x[i] / x[i + 1] for i in range(len(x) - 1)])
    df.groupby(by=model.features[0]).agg({model.intensity: [np.min, np.max]})
    max_conc = df[model.intensity].max()
    min_conc = df[model.intensity].min()
    print(max_conc, min_conc)
    num_features = np.max(df[model.features].to_numpy(), axis=0) + 1
    print(num_features)


    def body_make_prediction_dataset(key, dilution_factor):
        x = make_serial(dilution_factor)
        curr_df = pd.DataFrame(
            list(itertools.product(
                np.arange(0, num_features[0]),
                sorted(x)
            )),
            columns=model.regressors[::-1]
        )
        arr = []
        for i in range(TOTAL_REPS):
            arr += [i] * curr_df.shape[0]
        curr_df = (
            pd.concat([curr_df] * TOTAL_REPS, ignore_index=True)
            .reset_index(drop=True)
            .copy()
        )
        curr_df[REP] = arr
        curr_df[FACTOR] = dilution_factor
        key, subkey = random.split(key)
        curr_predictive = model.predict(
            curr_df, posterior, key=subkey,
            # return_sites=[site.obs, site.mu],
        )
        # for u, v in curr_predictive.items(): print(u, v.shape)
        for u in [site.b1, site.b2, site.b3, site.b4]:
            np.testing.assert_almost_equal(curr_predictive[u], posterior[u])
        curr_predictive = {u: curr_predictive[u] for u in [site.obs, site.mu, site.obs.log]}
        return key, curr_df, curr_predictive


    sim_df = None; sim_ppd = None
    key = random.key(0)
    for factor in FACTORS_SPACE:
        key, curr_df, curr_predictive = body_make_prediction_dataset(key, factor)
        if sim_df is None:
            sim_df = curr_df.copy()
            sim_ppd = {u: v for u, v in curr_predictive.items()}
        else:
            sim_df = pd.concat([sim_df, curr_df], ignore_index=True)
            sim_df = sim_df.reset_index(drop=True).copy()
            for u, v in sim_ppd.items():
                sim_ppd[u] = np.concatenate([v, curr_predictive[u]], axis=-2)

    print(sim_df.shape)
    for u, v in sim_ppd.items(): print(u, v.shape)
    for u, v in posterior.items():
        assert u not in sim_ppd.keys()
        sim_ppd[u] = v.copy()

    key, subkey = random.split(key)
    idx = np.arange(0, sim_ppd[site.obs].shape[0], 1)
    idx = random.permutation(subkey, idx); idx = np.array(idx)
    sim_ppd = {u: v[idx, ...] for u, v in sim_ppd.items()}

    # Save
    output_path = os.path.join(BUILD_DIR, "inf.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((sim_df, encoder, sim_ppd,), f)
    print(f"Saved to {output_path}")

    output_path = os.path.join(BUILD_DIR, "model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model,), f)
    print(f"Saved to {output_path}")

    return


@timing
def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    model_dir = "/home/vishu/reports/hbmep/notebooks/im/sheet1/unrest-notlogconc/10000w_10000s_4c_10t_15d_95a_tm/non_hierarchical"
    simulate_data(model_dir)
    return


if __name__ == "__main__":
    main()
