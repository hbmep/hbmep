import shutil
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from hbmep.config import Config
from hbmep.utils import timing

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, config: Config):
        self.toml_path = config.TOML_PATH
        self.csv_path = config.CSV_PATH
        self.build_dir = config.BUILD_DIR

        self.subject = config.SUBJECT
        self.features = config.FEATURES
        self.intensity = config.INTENSITY
        self.response = config.RESPONSE

        self.n_features = len(self.features)
        self.n_response = len(self.response)
        self.combination_columns = None
        self.regressors = [self.subject] + self.features + [self.intensity]

        self.mep_matrix_path = config.MEP_MATRIX_PATH
        self.mep_response = config.MEP_RESPONSE
        self.mep_window = config.MEP_TIME_RANGE
        self.mep_size_window = config.MEP_SIZE_TIME_RANGE

    def _make_dir(self, dir: str):
        Path(dir).mkdir(parents=True, exist_ok=True)
        return

    def _copy(self, src: str, dst: str):
        try:
            shutil.copy(src, dst)
        except shutil.SameFileError:
            pass
        return

    def _make_combinations(self, df: pd.DataFrame, columns: list[str]) -> list[tuple[int]]:
        combinations = df \
            .groupby(by=columns) \
            .size() \
            .to_frame("counts") \
            .reset_index() \
            .copy()
        combinations = combinations[columns] \
            .apply(tuple, axis=1) \
            .tolist()
        combinations = sorted(combinations)
        return combinations

    def _invert_combination(
        self,
        combination: tuple[int],
        columns: list[str],
        encoder_dict: dict[str, LabelEncoder]
    ) -> tuple:
        combination_inverse = []
        for (column, value) in zip(columns, combination):
            combination_inverse.append(
                encoder_dict[column].inverse_transform(np.array([value]))[0]
            )
        return tuple(combination_inverse)

    def _preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        """ Encode """
        encoder_dict = defaultdict(LabelEncoder)
        df[self.combination_columns] = \
            df[self.combination_columns] \
            .apply(lambda x: encoder_dict[x.name].fit_transform(x)) \
            .copy()
        df.reset_index(inplace=True, drop=True)
        return df, encoder_dict

    @timing
    def load(self, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        self._make_dir(dir=self.build_dir)
        logger.info(f"Artefacts will be stored here - {self.build_dir}")
        self._copy(src=self.toml_path, dst=self.build_dir)
        logger.info(f"Copied config to {self.build_dir}")

        if df is None:
            csv_path = self.csv_path
            logger.info(f"Reading data from {csv_path} ...")
            df = pd.read_csv(csv_path)

        """ Positive response constraint """
        num_non_positive_observation = (df[self.response] <= 0).any(axis=1).sum()
        logger.warning(f"Total non-positive observations: {num_non_positive_observation}")
        assert not num_non_positive_observation

        """ Missing response constraint """
        num_missing_observation = df[self.response].isna().any(axis=1).sum()
        logger.warning(f"Total missing observations: {num_missing_observation}")
        assert not num_missing_observation

        logger.info("Processing data ...")
        df, encoder_dict = self._preprocess(df=df)
        return df, encoder_dict
