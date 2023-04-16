import os
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
import numpyro.distributions as dist

from hb_mep.config import HBMepConfig
from hb_mep.utils import timing
from hb_mep.utils.constants import (
    DATA_DIR,
    REPORTS_DIR,
    INTENSITY,
    RESPONSE,
    PARTICIPANT,
    FEATURES
)

logger = logging.getLogger(__name__)


class Experiment():
    def __init__(self, config: HBMepConfig):
        self.config = config
        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.data_path = Path(os.path.join(self.current_path, DATA_DIR))
        self.reports_path = Path(os.path.join(self.current_path, REPORTS_DIR))

        self.name = "Experiment"
        self.random_state = 0

    @timing
    def simulate(
        self,
        random_seed: int = 0,
        n_participant: int = 10,
        n_segment: int = 3,
        sparse_factor: float = 0,
        xMax: int = 350,
        n_points: int = 50,
        save_to_disk: bool = False
    ):
        seed = jax.random.PRNGKey(random_seed)
        logger.info(f"Random seed: {random_seed}")
        logger.info(f"Simulating for {n_participant} participants, {n_segment} segments ...")

        a_mean = dist.TruncatedDistribution(dist.Normal(150, 100), low=0).sample(seed, sample_shape=(n_segment,))

        sigma = 1
        a_scale = jnp.array([sigma] * n_segment)

        a = dist.TruncatedDistribution(dist.Normal(a_mean, a_scale), low=0).sample(seed, (n_participant, ))

        b = dist.Uniform(0.2, 0.3).sample(seed, (n_participant, n_segment))
        lo = dist.Uniform(0.2, 0.35).sample(seed, (n_participant, n_segment))
        g = dist.HalfCauchy(0.0001).sample(seed, (n_participant, n_segment))
        noise_offset = dist.Uniform(0.1, 0.15).sample(seed, (n_participant, n_segment))
        noise_slope = dist.Uniform(0.21, 0.25).sample(seed, (n_participant, n_segment))

        columns = [PARTICIPANT, FEATURES[0], INTENSITY, RESPONSE]
        x = jnp.linspace(0, xMax, n_points)
        df = None

        for i in range(n_participant):
            for j in range(n_segment):
                participant = jnp.repeat(i, n_points)
                segment = jnp.repeat(j, n_points)

                mean = \
                    -jnp.log(jnp.maximum(g[i, j], jnp.exp(-jnp.maximum(lo[i, j], b[i, j] * (x - a[i, j])))))

                noise = noise_offset[i, j] + noise_slope[i, j] * mean

                y = dist.TruncatedNormal(mean, noise, low=0).sample(seed)

                arr = jnp.array([participant, segment, x, y]).T
                temp_df = pd.DataFrame(arr, columns=columns)

                # Make data sparse
                temp_df = temp_df.sample(frac=1).copy()
                temp_df = temp_df.sample(frac=(1 - sparse_factor)).copy()

                if df is None:
                    df = temp_df.copy()
                else:
                    df = pd.concat([df, temp_df], ignore_index=True).copy()

        df[PARTICIPANT] = df[PARTICIPANT].astype(int)
        df[FEATURES[0]] = df[FEATURES[0]].astype(int)
        logger.info(f"Finished simulating data ...")

        if save_to_disk:
            save_path = os.path.join(self.data_path, "simulated_data.csv")
            df.to_csv(save_path, index=False)
            logger.info(f"Saved to {save_path}")

        return df, a, a_mean
