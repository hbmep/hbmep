import os
import pickle
import logging

import numpy as np
from scipy import stats
from hbmep.util import timing, setup_logging

from model import HB
from core import get_subdir
from constants import (
    BUILD_DIR,
    TOML_PATH,
    SIM_DF_PATH,
    SIM_PPD_PATH,
    N_PULSES_SPACE,
)

logger = logging.getLogger(__name__)
N_REPS = 1
N_SUBJECTS = 1


@timing
def main(draws_space, n_pulses_space, models):
    num_reps = N_REPS
    num_subjects = N_SUBJECTS

    num_draws_processed = 0
    draws_not_processed = []
    mae, mse = [], []

    for draw in draws_space:
        curr_mae, curr_mse = [], []

        try:
            for M in models:
                for num_pulses in n_pulses_space:
                    subdir = get_subdir(num_reps, num_pulses, num_subjects, draw)

                    match M().name:
                        case name if name == HB().name:
                            src = os.path.join(
                                BUILD_DIR,
                                "experiments",
                                subdir,
                                M().name
                            )
                            a_true = np.load(os.path.join(src, "a_true.npy"))
                            a_true = a_true.reshape(-1,)
                            a_pred = np.load(os.path.join(src, "a_pred.npy"))
                            a_pred = a_pred.mean(axis=0).reshape(-1,)

                        case _:
                            raise ValueError(f"Invalid model {M.NAME}.")

                    curr_mae.append(np.abs(a_true - a_pred).mean())
                    curr_mse.append(np.square(a_true - a_pred).mean())

        except FileNotFoundError:
            draws_not_processed.append(draw)
            logger.info(f"Draw: {draw} - Missing")

        else:
            logger.info(f"Draw: {draw}")
            mae += curr_mae
            mse += curr_mse
            num_draws_processed += 1

    mae = np.array(mae)
    mae = mae.reshape(num_draws_processed, len(models), len(n_pulses_space))
    mse = np.array(mse)
    mse = mse.reshape(num_draws_processed, len(models), len(n_pulses_space))

    logger.info(f"MAE: {mae.shape}\n{mae.mean(axis=0)}")
    logger.info(f"\n{mae.mean(axis=0) - 1.96 * stats.sem(mae, axis=0)}")

    model_names = [M().name for M in models]
    dest = os.path.join(BUILD_DIR, "results.pkl")
    with open (dest, "wb") as f:
        pickle.dump((model_names, n_pulses_space, mae, mse), f)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    setup_logging(BUILD_DIR)
    draws_space = range(4000)
    models = [HB]
    n_pulses_space = N_PULSES_SPACE
    n_pulses_space = [24]

    main(
        draws_space=draws_space,
        n_pulses_space=n_pulses_space,
        models=models
    )
