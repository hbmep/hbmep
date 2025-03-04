import os
import sys
import pickle
import logging

import numpy as np
from scipy import stats
from hbmep.util import timing, setup_logging

from constants import BUILD_DIR

logger = logging.getLogger(__name__)
N_REPS = 1
N_SUBJECTS = 1


@timing
def main(draws_space, n_pulses_space, methods):
    num_draws_processed = 0
    draws_not_processed = []
    mae, mse = [], []

    for draw in draws_space:
        curr_mae, curr_mse = [], []

        try:
            for method in methods:
                for num_pulses in n_pulses_space:
                    src = os.path.join(
                        BUILD_DIR,
                        "core__online_naive",
                        f"p{num_pulses}",
                        f"d{draw}",
                        method
                    )
                    a_true = np.load(os.path.join(src, "a_true.npy"))
                    a_true = a_true.reshape(-1,)
                    a_pred = np.load(os.path.join(src, "a_pred.npy"))
                    a_pred = a_pred.mean(axis=0).reshape(-1,)

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
    mae = mae.reshape(num_draws_processed, len(methods), len(n_pulses_space))
    mse = np.array(mse)
    mse = mse.reshape(num_draws_processed, len(methods), len(n_pulses_space))

    logger.info(f"MAE: {mae.shape}\n{mae.mean(axis=0)}")
    logger.info(f"\n{mae.mean(axis=0) - 1.96 * stats.sem(mae, axis=0)}")

    dest = os.path.join(BUILD_DIR, "results.pkl")
    with open (dest, "wb") as f:
        pickle.dump((methods, n_pulses_space, mae, mse), f)
    logger.info(f"Saved to {dest}")
    return


if __name__ == "__main__":
    setup_logging(BUILD_DIR)
    NUM_PULSES, = list(map(int, sys.argv[1:]))
    n_pulses_space = [NUM_PULSES]
    draws_space = range(1000)
    methods = [
        # "off-svi",
        # "off-mcmc",
        # "on-hdi50",
        # "on-hdi90",
        # "on-hdi90-2",
        "on-hdi95",
    ]
    main(
        draws_space=draws_space,
        n_pulses_space=n_pulses_space,
        methods=methods
    )
