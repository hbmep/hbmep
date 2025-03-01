import os
import gc
import logging
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import jax
from jax import random, numpy as np
from joblib import Parallel, delayed

from numpyro import optim
from numpyro.infer import (
    autoguide,
	Predictive,
	SVI,
	Trace_ELBO,
	TraceEnum_ELBO,
	TraceMeanField_ELBO
)

from hbmep.util import (
    timing,
    setup_logging,
    Site as site
)

from model import HB
from util import generate_nested_pulses
from constants import (
    BUILD_DIR,
    TOML_PATH,
    SIM_DF_PATH,
    SIM_PPD_PATH,
    N_PULSES_SPACE,
)

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
logger = logging.getLogger(__name__)
N_SUBJECTS = 1
N_REPS = 1


def get_subdir(num_reps, num_pulses, num_subjects, draw):
    return os.path.join(f"p{num_pulses}", f"d{draw}")


def run(
    df,
    model,
    lr=1e-2,
    steps=2000,
    PROGRESS_BAR=True,
):
    optimizer = optim.ClippedAdam(step_size=lr)
    _guide = autoguide.AutoLowRankMultivariateNormal(model._model)
    svi = SVI(
        model._model,
        _guide,
        optimizer,
        loss=Trace_ELBO(num_particles=20)
    )
    svi_result = svi.run(
        model.key,
        steps,
        *model._get_regressors(df=df),
        *model._get_response(df=df),
        progress_bar=PROGRESS_BAR
    )
    predictive = Predictive(
        _guide,
        params=svi_result.params,
        num_samples=4000
    )
    posterior_samples = predictive(model.key, *model._get_regressors(df=df))
    posterior_samples = {u: np.array(v) for u, v in posterior_samples.items()}
    return svi_result, posterior_samples


@timing
def main(draws_space, num_pulses_space, models, n_jobs=-1):
    sdf = pd.read_csv(SIM_DF_PATH)
    with open(SIM_PPD_PATH, "rb") as f:
        sppd, = pickle.load(f)
    
    sobs = sppd[site.obs]
    sthreshold = sppd[site.a]
    sppd = None
    del sppd

    setup_logging(BUILD_DIR)
    pulses_map = generate_nested_pulses(sdf)


    def body_run(num_reps, num_pulses, num_subjects, draw, M):
        # Build model
        model = M(toml_path=TOML_PATH)
        model.build_dir = os.path.join(
            BUILD_DIR,
            "experiments",
            get_subdir(num_reps, num_pulses, num_subjects, draw),
            model.name
        )
        setup_logging(model.build_dir)
        
        # Load data
        ind = (
            (sdf["participant"] < num_subjects) &
            (sdf["rep"] < num_reps) &
            (sdf["TMSInt"].isin(pulses_map[num_pulses]))
        )
        df = sdf[ind].reset_index(drop=True).copy()
        df["PKPK_APB"] = sobs[draw, ind, 0]

        ind = df["PKPK_APB"] > 0
        df = df[ind].reset_index(drop=True).copy()

        # Run inference
        df, encoder = model.load(df=df)
        model.plot(df=df)
        return
        # _, posterior_samples = model.run(df=df)
        logger.info(f"Running {model.name}...")
        svi_result, posterior = run(df=df, model=model)
        losses = svi_result.losses
        # sns.lineplot(x=range(len(losses[-2000:])), y=losses[-2000:])
        # output_path = os.path.join(model.build_dir, "loss.png")
        # plt.savefig(output_path)
        # plt.close()
        # logger.info(f"Saved to {output_path}")

        # Compute error and save results
        a_true = sthreshold[draw, :num_subjects, ...]
        a_pred = posterior[site.a]
        assert a_pred.mean(axis=0).shape == a_true.shape
        np.save(os.path.join(model.build_dir, "a_true.npy"), a_true)
        np.save(os.path.join(model.build_dir, "a_pred.npy"), a_pred)

        # Predictions and recruitment curves
        prediction_df = model.make_prediction_dataset(df=df)
        predictive = model.predict(prediction_df, posterior=posterior)
        model.plot_curves(
            df=df,
            encoder=encoder,
            posterior=posterior,
            prediction_df=prediction_df,
            predictive=predictive,
        )

        ind, df, encoder, model = [None] * 4
        svi_result, posterior, losses, output_path, prediction_df, predictive = [None] * 6
        a_true, a_pred = None, None
        del ind, df, encoder, model
        del svi_result, posterior, losses, output_path, prediction_df, predictive
        del a_true, a_pred
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
    # lo, hi = list(map(int, sys.argv[1:]))
    lo, hi = 0, 4000
    # lo, hi = 2000, 4000
    # lo, hi = 0, 100
    # lo, hi = 0, 1
    # n_jobs = -8
    n_jobs = -1
    # n_jobs = 1

    # Experiment space
    models = [HB]
    draws_space = range(lo, hi)
    # num_pulses_space = N_PULSES_SPACE
    num_pulses_space = [24]
    main(
        draws_space=draws_space,
		num_pulses_space=num_pulses_space,
		models=models,
		n_jobs=n_jobs
    )
