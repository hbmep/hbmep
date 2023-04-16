import os
import logging

import jax
import numpy as np
import pandas as pd

from hb_mep.config import HBMepConfig
from hb_mep.experiments import Experiment
from hb_mep.utils import timing

logger = logging.getLogger(__name__)


class SparseDataExperiment(Experiment):
    def __init__(self, config: HBMepConfig):
        super(SparseDataExperiment, self).__init__(config=config)
        self.name = "Sparse_Data_Experiment"

    @timing
    def run(self):
        n_participant = 50
        n_segment = 5

        n_trials = 150
        n_sparse_factors = 10

        sparse_factors = [i/10 for i in range(n_sparse_factors)]
        trials = jax.random.choice(
            jax.random.PRNGKey(self.random_state),
            np.array(range(1000)),
            shape=(n_trials,),
            replace=False
        )

        logger.info(f" === Running {self.name} ===")
        logger.info(f" === {n_participant} Participants ===")
        logger.info(f" === {n_segment} Segments ===")
        logger.info(f" === {n_trials} Trials ===")
        logger.info(f" === {n_sparse_factors} Sparse Factors ===")

        columns = ["sparse_factor", "random_seed", "mae_hb", "mae_nhb", "mae_mle"]
        initial_iteration = True

        for sparse_factor in sparse_factors:
            for j, random_seed in enumerate(trials):
                logger.info(f" === {sparse_factor} Sparse Factor, {j + 1}/{n_trials} Trial ===")

                df, a, _ = self.simulate(
                    random_seed=random_seed,
                    n_participant=n_participant,
                    n_segment=n_segment,
                    sparse_factor=sparse_factor
                )

                # HB Model
                model = SaturatedReLU_HB(config)
                _, posterior_samples = model.sample(df=df)
                error = posterior_samples["a"].mean(axis=0).reshape(-1,) - a.reshape(-1,)
                hb_error = np.abs(error).mean()

                # NHB Model
                model = SaturatedReLU_NHB(config)
                _, posterior_samples = model.sample(df=df)
                error = posterior_samples["a"].mean(axis=0).reshape(-1,) - a.reshape(-1,)
                nhb_error = np.abs(error).mean()

                # MLE Model
                model = SaturatedReLU_MLE(config)
                _, posterior_samples = model.sample(df=df)
                error = posterior_samples["a"].mean(axis=0).reshape(-1,) - a.reshape(-1,)
                mle_error = np.abs(error).mean()

                logger.info(f" === HB: {hb_error}, NHB: {nhb_error}, MLE: {mle_error} ===")
                logger.info(f" === NHB-HB: {nhb_error - hb_error} ===")
                logger.info(f" === MLE-HB: {mle_error - hb_error} ===")
                logger.info(f" === MLE-NHB: {mle_error - nhb_error} ===")

                arr = np.array([
                    sparse_factor, random_seed, hb_error, nhb_error, mle_error
                ]).reshape(-1, 1).T

                res_df = pd.DataFrame(arr, columns=columns)
                res_df.random_seed = res_df.random_seed.astype(int)

                save_path = os.path.join(self.reports_path, f"{self.name}.csv")

                if initial_iteration:
                    res_df.to_csv(save_path, index=False)
                    initial_iteration = False
                else:
                    res_df.to_csv(save_path, mode="a", header=False, index=False)
