import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from hb_mep.config import HBMepConfig
from hb_mep.utils import timing
from hb_mep.utils.constants import (
    DATA_DIR,
    REPORTS_DIR,
    INTENSITY,
    MEP_SIZE,
    PARTICIPANT,
    SEGMENT,
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
        df[MEP_SIZE] = df[MEP_SIZE].apply(lambda x: tuple(scalar_mep * np.array(list(x))))

        # Mininum observations constraint
        temp = df \
                .groupby(by=[PARTICIPANT, SEGMENT]) \
                .size() \
                .to_frame('counts') \
                .reset_index().copy()
        temp = temp[temp.counts > min_observations].copy()
        keep = list(zip(temp[PARTICIPANT], temp[SEGMENT]))
        df = \
            df[df[[PARTICIPANT, SEGMENT]] \
            .apply(tuple, 1) \
            .isin(keep)] \
            .copy()

        # Encode participants
        participant_encoder = LabelEncoder()
        df[PARTICIPANT] = participant_encoder.fit_transform(df[PARTICIPANT])

        # Encode levels
        segment_encoder = LabelEncoder()
        df[SEGMENT] = segment_encoder.fit_transform(df[SEGMENT])

        # Reorder
        df = df.sort_values(by=[PARTICIPANT, SEGMENT, INTENSITY], ascending=True).copy()
        df.reset_index(inplace=True, drop=True)

        n_participants = df[PARTICIPANT].nunique()
        n_segments = df[SEGMENT].nunique()
        segments_per_participant = \
            [sorted(df[df[PARTICIPANT] == i][SEGMENT].unique()) for i in range(n_participants)]

        total_combinations = 0
        for row in segments_per_participant:
            total_combinations += len(row)

        mep_size = df[MEP_SIZE].apply(lambda x: x[0]).to_numpy().reshape(-1,)
        participant = df[PARTICIPANT].to_numpy().reshape(-1,)
        segment = df[SEGMENT].to_numpy().reshape(-1,)
        intensity = df[INTENSITY].to_numpy().reshape(-1,)

        # Dictionary of encoders
        encoders_dict = {
            PARTICIPANT_ENCODER: participant_encoder,
            SEGMENT_ENCODER: segment_encoder
        }

        # Data dictionary for MCMC
        data_dict = {
            NUM_PARTICIPANTS: n_participants,
            NUM_SEGMENTS: n_segments,
            SEGMENTS_PER_PARTICIPANT: segments_per_participant,
            TOTAL_COMBINATIONS: total_combinations,
            INTENSITY: intensity,
            MEP_SIZE: mep_size,
            PARTICIPANT: participant,
            SEGMENT: segment
        }
        return df, data_dict, encoders_dict

    @timing
    def build(self):
        logger.info('Reading data ....')
        df = pd.read_csv(os.path.join(self.data_path, self.config.FNAME))
        df = df[[INTENSITY, MEP_SIZE, PARTICIPANT, SEGMENT]]
        df[MEP_SIZE] = df[MEP_SIZE].apply(lambda x: (x,))
        logger.info('Processing data ...')
        df, data_dict, encoders_dict = self.preprocess(df, **self.config.PREPROCESS_PARAMS)
        return df, data_dict, encoders_dict
