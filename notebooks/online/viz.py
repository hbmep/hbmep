import os
import gc
import sys
import pickle
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import jax
from jax import random, numpy as np
from numpyro.diagnostics import hpdi
from joblib import Parallel, delayed
from hbmep import functional as F
from hbmep.util import (
    timing,
    setup_logging,
    Site as site
)

from model import Simulator, HB
from constants import (
    BUILD_DIR,
    TOML_PATH,
    SIMULATION_DF_PATH,
    SIMULATION_PPD_PATH,
    N_PULSES_SPACE,
)

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)

BUILD_DIR = os.path.join(BUILD_DIR, "viz")
logger = logging.getLogger(__name__)
setup_logging(os.path.join(BUILD_DIR, "viz.log"))

NUM_SUBJECTS = 1


@timing
def main(num_pulses, draw):
    simulator = Simulator(toml_path=TOML_PATH)
    with open(SIMULATION_PPD_PATH, "rb") as f:
        simulation_ppd, = pickle.load(f)

    named_params = [site.a, site.b, site.L, site.ell, site.H]
    simulation_ppd = {u: v for u, v in simulation_ppd.items() if u in named_params}
    simulation_params = [simulation_ppd[named_param][draw: draw + 1, :NUM_SUBJECTS, ...] for named_param in named_params]


    def run_offline(key):
        x = np.linspace(0, 100, num_pulses)
        f = np.zeros(x.shape)
        df = pd.DataFrame(np.array([x, f]).T, columns=simulator.regressors)
        df[simulator.intensity] = df[simulator.intensity].astype(np.float64)
        df[simulator.features] = df[simulator.features].astype(int)

        key, subkey = random.split(key)
        prediction_obs = simulator.predict(
            df,
            num_samples=1,
            posterior={u: v for u, v in zip(named_params, simulation_params)},
            key=subkey,
            return_sites=[site.obs]
        )[site.obs][0, ...]
        df[simulator.response] = prediction_obs

        # Build model
        M = HB
        model = M(toml_path=TOML_PATH)
        model.features = []
        model.build_dir = os.path.join(
            BUILD_DIR,
            f"p{num_pulses}",
            f"d{draw}",
            "offline"
        )
        os.makedirs(model.build_dir, exist_ok=True)
        _, posterior = model.run(df=df)
        prediction_df = model.make_prediction_dataset(df=df)
        predictive = model.predict(prediction_df, posterior=posterior)
        model.plot_curves(
            df=df,
            posterior=posterior,
            prediction_df=prediction_df,
            predictive=predictive,
        )
        return


    def run_online(key):
        # Build model
        M = HB
        model = M(toml_path=TOML_PATH)
        model.features = []
        model.build_dir = os.path.join(
            BUILD_DIR,
            f"p{num_pulses}",
            f"d{draw}",
            "online"
        )
        os.makedirs(model.build_dir, exist_ok=True)

        df = None
        proposal = [20, 50, 80]

        for iter in range(num_pulses // len(proposal)):
            curr_df = pd.DataFrame(
                np.array([proposal, [0] * len(proposal)]).T, columns=simulator.regressors
            )
            curr_df[simulator.intensity] = curr_df[simulator.intensity].astype(np.float64)
            curr_df[simulator.features] = curr_df[simulator.features].astype(int)

            key, subkey = random.split(key)
            prediction_obs = simulator.predict(
                curr_df,
                num_samples=1,
                posterior={u: v for u, v in zip(named_params, simulation_params)},
                key=subkey,
                return_sites=[site.obs]
            )[site.obs][0, ...]
            curr_df[simulator.response] = prediction_obs

            if df is None: df = curr_df.copy()
            else: df = pd.concat([df, curr_df]).reset_index(drop=True).copy()

            _, posterior = model.run_svi(df=df)
            curr_a = posterior[site.a]
            curr_a_mean = curr_a.mean()

            ## naive_hdi
            l, r = hpdi(curr_a.reshape(-1,), prob=.9)
            proposal = [l, r, (l + r) / 2]
            proposal = [max(0, p) for p in proposal]
            proposal = [min(p, 100) for p in proposal]

            output_path = os.path.join(model.build_dir, f"iter{iter:02}.pdf")
            # Predictions and recruitment curves
            prediction_df = model.make_prediction_dataset(df=df)
            predictive = model.predict(prediction_df, posterior=posterior)
            model.plot_curves(
                df=df,
                posterior=posterior,
                prediction_df=prediction_df,
                predictive=predictive,
                output_path=output_path
            )
        return

    key, subkey = random.split(simulator.key)
    run_offline(key)
    key, subkey = random.split(simulator.key)
    run_online(key)
    return


if __name__ == "__main__":
    # num_pulses, draw = list(map(int, sys.argv[1:]))
    num_pulses = 16
    [main(num_pulses, draw) for draw in range(20)]
