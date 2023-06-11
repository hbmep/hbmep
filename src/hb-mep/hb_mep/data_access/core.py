import os
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from hb_mep.config import HBMepConfig
from hb_mep.utils import timing
from hb_mep.utils.constants import (
    DATA_DIR,
    REPORTS_DIR,
    INTENSITY,
    RESPONSE,
    PARTICIPANT,
    FEATURES,
    AUC_MAP
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

    @timing
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
        idx = df[RESPONSE].isin([0])
        if idx.sum():
            df = df[~idx].copy()
            logger.info(f"Removed {idx.sum()} observation(s) with zero AUC response.")

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
