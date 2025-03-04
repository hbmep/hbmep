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

from models import Simulator, HB
from constants import (
    BUILD_DIR,
    CONFIG,
    SIMULATION_DF_PATH,
    SIMULATION_PPD_PATH,
    N_PULSES_SPACE,
)

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
logger = logging.getLogger(__name__)
global simulation_ppd


@timing
def main(draws_space, num_pulses_space, methods, n_jobs=-1):
    # for method in methods:
        # assert method in {"off-mcmc", "off-svi", "on-hdi90", "on-hdi99", "on-hdi50", "on-hdi90-2"}

    global simulation_ppd
    with open(SIMULATION_PPD_PATH, "rb") as f:
        simulation_ppd, = pickle.load(f)
    
    named_params = [site.a, site.b, site.L, site.ell, site.H]
    simulation_ppd = {u: v for u, v in simulation_ppd.items() if u in named_params}
    setup_logging(os.path.join(
        BUILD_DIR,
        f"{os.path.splitext(os.path.basename(__file__))[0]}.log"
    ))


    def body_run(num_pulses, draw, method):
        global simulation_ppd
        # Build model
        model = HB(config=CONFIG)
        model.features = []
        model.build_dir = os.path.join(
            BUILD_DIR,
            os.path.splitext(os.path.basename(__file__))[0],
            f"p{num_pulses}",
            f"d{draw}",
            method
        )
        os.makedirs(model.build_dir, exist_ok=True)
        setup_logging(os.path.join(model.build_dir, "logs.log"))

        # Simulator
        simulator = Simulator(config=CONFIG)
        simulation_params = [simulation_ppd[named_param][draw: draw + 1, :1, ...] for named_param in named_params]
        key, subkey = random.split(simulator.key)

        if method.startswith("off"):
            df = pd.DataFrame(
                np.array([np.linspace(0, 100, num_pulses), np.zeros((num_pulses,))]).T,
                columns=simulator.regressors
            )
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

            match method:
                case "off-mcmc": _, posterior = model.run(df=df)
                case "off-svi": _, posterior = model.run_svi(df=df)
                case _: raise ValueError("Invalid offline method")

        else:
            df = None
            proposal = [20, 50, 80]
            if method in {"on-hdi90-2", "on-hdi95"}:
                proposal = [5, 50, 90]

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
                output_path = os.path.join(model.build_dir, f"iter{iter:02}.pdf")
                prediction_df = model.make_prediction_dataset(df=df)
                predictive = model.predict(prediction_df, posterior=posterior)
                model.plot_curves(
                    df=df,
                    posterior=posterior,
                    prediction_df=prediction_df,
                    predictive=predictive,
                    output_path=output_path
                )

                # Update proposal
                match method:
                    case "on-hdi90" | "on-hdi90-2":
                        l, r = hpdi(posterior[site.a].reshape(-1,), prob=.9)
                        proposal = [l, r, (l + r) / 2]
                        proposal = [max(0, p) for p in proposal]
                        proposal = [min(p, 100) for p in proposal]

                    case "on-hdi95":
                        l, r = hpdi(posterior[site.a].reshape(-1,), prob=.95)
                        proposal = [l, r, (l + r) / 2]
                        proposal = [max(0, p) for p in proposal]
                        proposal = [min(p, 100) for p in proposal]

                    case "on-hdi99":
                        l, r = hpdi(posterior[site.a].reshape(-1,), prob=.99)
                        proposal = [l, r, (l + r) / 2]
                        proposal = [max(0, p) for p in proposal]
                        proposal = [min(p, 100) for p in proposal]

                    case "on-hdi50":
                        l, r = hpdi(posterior[site.a].reshape(-1,), prob=.5)
                        proposal = [l, r, (l + r) / 2]
                        proposal = [max(0, p) for p in proposal]
                        proposal = [min(p, 100) for p in proposal]

                    case _: raise ValueError("Invalid online method")
                
            del curr_df, proposal

        # Compute error and save results
        a_true = simulation_params[0][0]
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

        # key, subkey = None, None
        # model, simulator, simulation_params, key, subkey = None, None, None, None, None
        # df, proposal, curr_df, obs, _, posterior = None, None, None, None, None, None
        # curr_a, a_true, a_pred, prediction_df, predictive = None, None, None, None, None
        del _, key, subkey, model, simulator
        del simulation_params, prediction_obs, posterior
        del df, prediction_df, predictive
        del a_true, a_pred
        gc.collect()
        return


    logger.info(f"num_pulses_space: {', '.join(map(str, num_pulses_space))}")
    logger.info(f"methods: {', '.join(methods)}")
    logger.info(f"Running draws {draws_space.start} to {draws_space.stop - 1}.")
    logger.info(f"n_jobs: {n_jobs}")

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(body_run)(n_pulses, draw, method)
            for draw in draws_space
            for n_pulses in num_pulses_space
            for method in methods
        )
    return


if __name__ == "__main__":
    args = sys.argv[1:]
    method = args[-1]
    lo, hi, NUM_PULSES, = list(map(int, args[:-1]))

    draws_space = range(lo, hi)
    num_pulses_space = [NUM_PULSES]
    methods = [method]
    n_jobs = -10

    # draws_space = range(1)
    # num_pulses_space = [18]
    # methods = ["on-hdi95"]
    # n_jobs = 1

    main(
        draws_space=draws_space,
		num_pulses_space=num_pulses_space,
		methods=methods,
		n_jobs=n_jobs
    )
