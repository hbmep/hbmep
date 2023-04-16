import os
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

import jax.numpy as jnp
from jax import random
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


class DataClass:
    def __init__(self, config: HBMepConfig):
        self.config = config
        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.data_path = Path(os.path.join(self.current_path, DATA_DIR))
        self.reports_path = Path(os.path.join(self.current_path, REPORTS_DIR))

    def make_dirs(self):
        dirs = {
            DATA_DIR: self.data_path,
            REPORTS_DIR: self.reports_path
        }
        for dir in dirs:
            dirs[dir].mkdir(parents=True, exist_ok=True)
            logger.info(f"Created {dir} directory {dirs[dir]}")

    def preprocess(
        self,
        df: pd.DataFrame,
        min_observations: int = 25,
        scalar_intensity: float = 1000,
        scalar_response: float = 1
        ) -> tuple[pd.DataFrame, dict, dict[str,  LabelEncoder]]:
        """
        Preprocess data
        """
        # Scale data
        df[INTENSITY] = df[INTENSITY].apply(lambda x: x * scalar_intensity)
        df[RESPONSE] = df[RESPONSE].apply(lambda x: x * scalar_response)

        # Mininum observations constraint
        temp_df = df \
                .groupby(by=[PARTICIPANT] + FEATURES) \
                .size() \
                .to_frame('counts') \
                .reset_index().copy()
        temp_df = temp_df[temp_df.counts >= min_observations].copy()
        keep = list(temp_df[[PARTICIPANT] + FEATURES].apply(tuple, axis=1))
        idx = df[[PARTICIPANT] + FEATURES].apply(tuple, axis=1).isin(keep)
        df = df[idx].copy()

        # Encode participants and features
        encoder_dict = defaultdict(LabelEncoder)
        df[[PARTICIPANT] + FEATURES] = \
            df[[PARTICIPANT] + FEATURES] \
            .apply(lambda x: encoder_dict[x.name].fit_transform(x)).copy()

        df.reset_index(inplace=True, drop=True)
        return df, encoder_dict

    @timing
    def build(self, df: pd.DataFrame = None):
        if df is None:
            fpath = os.path.join(self.data_path, self.config.FNAME)
            logger.info(f"Reading data from {fpath}...")
            df = pd.read_csv(fpath)

        df[f"raw_{RESPONSE}"] = df[RESPONSE]
        logger.info('Processing data ...')
        return self.preprocess(df, **self.config.PREPROCESS_PARAMS)

    @timing
    def simulate(
        self,
        random_seed: int = 0,
        n_participant: int = 10,
        n_segment: int = 3,
        sparse_factor: float = 0,
        xMax: int = 400,
        n_points: int = 40
    ):
        seed = random.PRNGKey(random_seed)
        logger.info(f"Random seed: {random_seed}")
        logger.info(f"Simulating for {n_participant} participants, {n_segment} segments ...")

        a_mean = dist.TruncatedDistribution(dist.Normal(3, 2), low=0).sample(seed, sample_shape=(n_segment,))

        sigma = 1
        a_scale = jnp.array([sigma] * n_segment)

        a = dist.TruncatedDistribution(dist.Normal(a_mean, a_scale), low=0).sample(seed, (n_participant, ))

        b = dist.Uniform(2, 3).sample(seed, (n_participant, n_segment))
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
        return df, a, a_mean
