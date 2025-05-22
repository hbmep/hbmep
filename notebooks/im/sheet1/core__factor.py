import os
import pickle
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from jax import random, numpy as jnp
from numpyro.diagnostics import hpdi
import hbmep as mep
from hbmep.util import timing, setup_logging
from joblib import Parallel, delayed

from hbmep.notebooks.util import make_pdf
from models import nHB
from util import Site as site, load_model, make_serial
from constants import (
    TOML_PATH,
    SIMULATION_FACTOR_DIR,
    FACTOR,
    FACTORS_SPACE,
    REP,
    TOTAL_REPS,
)

logger = logging.getLogger(__name__)

BUILD_DIR = os.path.join(SIMULATION_FACTOR_DIR, "experiments")


def run(draws_space, factors_space, num_reps_space, models, n_jobs=-1):
    (
        sim_df, sim_encoder, sim, sim_ppd, *_,
    ) = load_model(SIMULATION_FACTOR_DIR)

    # sim_df_features = sim_df[sim.features].apply(tuple, axis=1)
    sim_obs = sim_ppd[site.obs].copy()
    sim_ppd = None
    

    def body_run(
        draw,
        factor,
        num_reps,
        M,
    ):
        idx = (
            (sim_df[REP] < num_reps)
            & sim_df[FACTOR].isin([factor])
        )
        df = sim_df[idx].reset_index(drop=True).copy()
        df[sim.response] = sim_obs[draw, idx, ...]
        model = M()
        model.intensity = sim.intensity
        model.features = sim.features.copy()
        model.response = sim.response.copy()
        model._model = model.non_hierarchical
        model.mcmc_params = {
            "num_warmup": 2000,
            "num_samples": 1000,
            "num_chains": 4,
            "thinning": 1,
        }
        model.nuts_params = {
            "max_tree_depth": (15, 15),
            "target_accept_prob": .95,
        }
        model.use_mixture = False
        # model.n_jobs = -1
        model.n_jobs = 1
        model.build_dir = os.path.join(
            BUILD_DIR,
            f"reps{num_reps}",
            f"draw{draw}",
            f"factor{factor}",
            model._model.__name__,
            model.name,
        )
        os.makedirs(model.build_dir, exist_ok=True)
        setup_logging(model.build_dir)
        _, posterior = model.run(df=df)

        prediction_df = model.make_prediction_dataset(df=df)
        if site.outlier_prob in posterior.keys():
            posterior[site.outlier_prob] = 0 * posterior[site.outlier_prob]
        predictive = model.predict(prediction_df, posterior=posterior)
        model.plot_curves(
            df,
            prediction_df=prediction_df,
            predictive=predictive,
            encoder=sim_encoder,
            prediction_prob=.95
        )
        named_params = [site.b1, site.b2, site.b3, site.b4]
        posterior = {u: posterior[u] for u in named_params}
        output_path = os.path.join(model.build_dir, "inf.pkl")
        with open(output_path, "wb") as f:
            pickle.dump((posterior,), f)
        logger.info(f"Saved to {output_path}")
        idx, df, model, posterior, = [None] * 4
        return


    logger.info(f"Running draws {draws_space.start} to {draws_space.stop - 1}.")
    logger.info(f"Factors: {', '.join(map(str, factors_space))}")
    logger.info(f"Reps: {', '.join(map(str, num_reps_space))}")
    logger.info(f"n_jobs: {n_jobs}")

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(body_run)(
                draw, factor, num_reps, M
            )
            for draw in draws_space
            for factor in factors_space
            for num_reps in num_reps_space
            for M in models
        )


@timing
def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    setup_logging(BUILD_DIR)

    draws_space = range(500)
    factors_space = FACTORS_SPACE
    num_reps_space = [2]
    models = [nHB]
    n_jobs = -8

    run(
        draws_space,
		factors_space,
		num_reps_space,
		models,
		n_jobs
    )
    return


if __name__ == "__main__":
    main()
