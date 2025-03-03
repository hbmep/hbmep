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
logger = logging.getLogger(__name__)
N_SUBJECTS = 1
N_REPS = 1
global simulation_ppd


@timing
def main(draws_space, num_pulses_space, models, n_jobs=-1):
    global simulation_ppd
    with open(SIMULATION_PPD_PATH, "rb") as f:
        simulation_ppd, = pickle.load(f)
    
    named_params = [site.a, site.b, site.L, site.ell, site.H]
    simulation_ppd = {u: v for u, v in simulation_ppd.items() if u in named_params}


    def body_run(num_reps, num_pulses, num_subjects, draw, M):
        global simulation_ppd
        # Build model
        model = M(toml_path=TOML_PATH)
        model.features = []
        model.build_dir = os.path.join(
            BUILD_DIR,
            os.path.splitext(os.path.basename(__file__))[0],
            f"p{num_pulses}",
            f"d{draw}",
            model.name
        )
        os.makedirs(model.build_dir, exist_ok=True)
        setup_logging(os.path.join(model.build_dir, "logs.log"))

        # Simulator
        simulator = Simulator(toml_path=TOML_PATH)
        params = [simulation_ppd[named_param][draw: draw + 1, :num_subjects, ...] for named_param in named_params]
        key, subkey = random.split(simulator.key)

        df = None
        proposal = [20, 50, 80]

        for _ in range(num_pulses // len(proposal)):
            curr_df = pd.DataFrame(
                np.array([proposal, [0] * len(proposal)]).T, columns=simulator.regressors
            )
            curr_df[simulator.intensity] = curr_df[simulator.intensity].astype(np.float64)
            curr_df[simulator.features] = curr_df[simulator.features].astype(int)

            key, subkey = random.split(key)
            obs = simulator.predict(
                curr_df,
                num_samples=1,
                posterior={u: v for u, v in zip(named_params, params)},
                key=subkey,
                return_sites=[site.obs]
            )
            curr_df[model.response] = obs[site.obs][0, ...]

            if df is None: df = curr_df.copy()
            else: df = pd.concat([df, curr_df]).reset_index(drop=True).copy()

            _, posterior = model.run_svi(df=df)
            curr_a = posterior[site.a]
            curr_a_mean = curr_a.mean()

            ## core__online_naive_pm1
            # proposal = [curr_a_mean - 1, curr_a_mean, curr_a_mean + 1]

            # ## core__online_naive_hdi_unbounded
            # l, r = hpdi(curr_a.reshape(-1,), prob=.9)
            # proposal = [l, r, (l + r) / 2]

            ## core__online_naive_hdi
            l, r = hpdi(curr_a.reshape(-1,), prob=.9)
            proposal = [l, r, (l + r) / 2]
            proposal = [max(0, p) for p in proposal]
            proposal = [min(p, 100) for p in proposal]

        # Compute error and save results
        a_true = params[0][0]
        a_pred = posterior[site.a][:, None, ...]
        assert a_pred.mean(axis=0).shape == a_true.shape
        np.save(os.path.join(model.build_dir, "a_true.npy"), a_true)
        np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)

        # Predictions and recruitment curves
        prediction_df = model.make_prediction_dataset(df=df)
        predictive = model.predict(prediction_df, posterior=posterior)
        model.plot_curves(
            df=df,
            posterior=posterior,
            prediction_df=prediction_df,
            predictive=predictive,
        )

        model, simulator, params, key, subkey = None, None, None, None, None
        df, proposal, curr_df, obs, _, posterior = None, None, None, None, None, None
        curr_a, a_true, a_pred, prediction_df, predictive = None, None, None, None, None
        del model, simulator, params, key, subkey
        del df, proposal, curr_df, obs, _, posterior 
        del curr_a, a_true, a_pred, prediction_df, predictive 
        gc.collect()
        return


    logger.info("Number of pulses experiment...")
    logger.info(f"num_pulses_space: {', '.join(map(str, num_pulses_space))}")
    logger.info(f"models: {', '.join([m().name for m in models])}")
    logger.info(f"Running draws {draws_space.start} to {draws_space.stop - 1}.")
    logger.info(f"n_jobs: {n_jobs}")

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(body_run)(N_REPS, n_pulses, N_SUBJECTS, draw, M)
            for draw in draws_space
            for n_pulses in num_pulses_space
            for M in models
        )
    return


if __name__ == "__main__":
    # # Usage: python -m core 0 100
    lo, hi, NUM_PULSES = list(map(int, sys.argv[1:]))
    # lo, hi = 0, 4000
    # lo, hi = 2000, 4000
    # lo, hi = 0, 200
    # lo, hi = 247, 1000
    # # lo, hi = 0, 100
    # # lo, hi = 0, 1
    n_jobs = -10
    # # n_jobs = 1

    # Experiment space
    models = [HB]
    draws_space = range(lo, hi)
    num_pulses_space = [NUM_PULSES]
    main(
        draws_space=draws_space,
		num_pulses_space=num_pulses_space,
		models=models,
		n_jobs=n_jobs
    )
