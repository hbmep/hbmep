import os
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from hb_mep.config import HBMepConfig
from hb_mep.utils import timing
from hb_mep.utils.constants import (
    DATA_DIR,
    REPORTS_DIR,
    INTENSITY,
    RESPONSE_MUSCLES,
    PARTICIPANT,
    INDEPENDENT_FEATURES,
    PARTICIPANT_ENCODER,
    SEGMENT_ENCODER,
    NUM_PARTICIPANTS,
    NUM_SEGMENTS,
    SEGMENTS_PER_PARTICIPANT,
    TOTAL_COMBINATIONS
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
        scalar_mep: float = 1
        ) -> tuple[pd.DataFrame, dict, dict[str,  LabelEncoder]]:
        """
        Preprocess cleaned data

        Args:
            df (pd.DataFrame): Data to be preprocessed.
            min_observations (int, optional): (participant, segment) combinations with less than
            min_observations observations will be removed. Defaults to 25.
            scalar_intensity (float, optional): Scaling constant for intensity column. Defaults to 1000.
            scalar_mep (float, optional): Scaling constant for mep size. Defaults to 1.

        Returns:
            tuple[pd.DataFrame, dict, dict[str,  LabelEncoder]]: Processed data, data dictionary to be used for
            inference algorithms such as MCMC and other utility functions such as plotting, Label encoders for
            participants and segments
        """
        # Scale
        df[INTENSITY] = df[INTENSITY].apply(lambda x: x * scalar_intensity)
        df[RESPONSE_MUSCLES] = df[RESPONSE_MUSCLES].apply(lambda x: x * scalar_mep, axis=1)

        # Zero-One transformation
        df[RESPONSE_MUSCLES] = \
            df[RESPONSE_MUSCLES] \
            .apply(lambda x: [y if y > self.config.ZERO_ONE_THRESHOLDS[i] else 0 for i, y in enumerate(list(x))], axis=1) \
            .apply(pd.Series).copy()

        # Mininum observations constraint
        temp = df \
                .groupby(by=[PARTICIPANT] + INDEPENDENT_FEATURES) \
                .size() \
                .to_frame('counts') \
                .reset_index().copy()
        temp = temp[temp.counts >= min_observations].copy()
        keep = list(temp[[PARTICIPANT] + INDEPENDENT_FEATURES].apply(tuple, axis=1))
        df = \
            df[df[[PARTICIPANT] + INDEPENDENT_FEATURES] \
            .apply(tuple, axis=1) \
            .isin(keep)] \
            .copy()

        # Encode participants and independent features
        encoder_dict = defaultdict(LabelEncoder)
        df[[PARTICIPANT] + INDEPENDENT_FEATURES] = \
            df[[PARTICIPANT] + INDEPENDENT_FEATURES] \
            .apply(lambda x: encoder_dict[x.name].fit_transform(x)).copy()

        # Reorder
        df = df.sort_values(by=[PARTICIPANT] + INDEPENDENT_FEATURES + [INTENSITY], ascending=True).copy()
        df.reset_index(inplace=True, drop=True)

        return df, encoder_dict

    @timing
    def build(self):
        logger.info('Reading data ....')
        df = pd.read_csv(os.path.join(self.data_path, self.config.FNAME))
        df = df[[INTENSITY] + RESPONSE_MUSCLES + [PARTICIPANT] + INDEPENDENT_FEATURES]
        for muscle in RESPONSE_MUSCLES:
            df['raw_' + muscle] = df[muscle]
        logger.info('Processing data ...')
        df, encoder_dict = self.preprocess(df, **self.config.PREPROCESS_PARAMS)
        return df, encoder_dict
