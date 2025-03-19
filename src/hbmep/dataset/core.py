import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from hbmep.config import Config
from hbmep.utils import timing
from hbmep.utils import constants as const

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, config: Config):
        self.toml_path = config.TOML_PATH
        self.csv_path = config.CSV_PATH
        self.build_dir = config.BUILD_DIR

        self.intensity = config.INTENSITY
        self._features = config.FEATURES
        self.features = self._init_features(self._features)
        self.regressors = [self.intensity] + self.features
        self.response = config.RESPONSE

        self.n_features = len(self.features)
        self.n_regressors = len(self.regressors)
        self.n_response = len(self.response)

        self.mep_data = config.MEP_DATA

    @staticmethod
    def _init_features(_features: list[str]) -> list[str]:
        features = []
        for feature in _features:
            if isinstance(feature, list):
                feature = const.SEP.join(feature)
            features.append(feature)
        return features

    @staticmethod
    def _make_dir(dir: str):
        # TODO: try and except
        Path(dir).mkdir(parents=True, exist_ok=True)
        return

    @staticmethod
    def _get_combinations(
        df: pd.DataFrame,
        columns: list[str],
        orderby=None
    ) -> list[tuple[int]]:
        combinations = (
            df[columns]
            .apply(tuple, axis=1)
            .unique()
            .tolist()
        )
        combinations = sorted(combinations, key=orderby)
        return combinations

    @staticmethod
    def _get_combination_inverse(
        combination: tuple[int],
        columns: list[str],
        encoder_dict: dict[str, LabelEncoder]
    ) -> tuple:
        return tuple(
            encoder_dict[column].inverse_transform(np.array([value]))[0]
            for (column, value) in zip(columns, combination)
        )

    @staticmethod
    def _preprocess(
        df: pd.DataFrame,
        columns: list[str]
    ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        # Encode
        encoder_dict = defaultdict(LabelEncoder)
        df[columns] = (
            df[columns]
            .apply(lambda x: encoder_dict[x.name].fit_transform(x))
        )
        return df, encoder_dict

    def preprocess(
        self,
        df: pd.DataFrame,
        encoder_dict: dict[str, LabelEncoder] = None
    ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        # Concatenate (necessary) features
        for i, feature in enumerate(self._features):
            if isinstance(feature, list):
                df[self.features[i]] = (
                    df[feature]
                    .apply(lambda x: const.SEP.join(x), axis=1)
                )

        # Positive response constraint
        num_non_positive_observation = (
            (df[self.response] <= 0)
            .any(axis=1)
            .sum()
        )
        if num_non_positive_observation:
            logger.info(
                "Total non-positive observations: ",
                f"{num_non_positive_observation}"
            )
        assert not num_non_positive_observation

        # Missing response constraint
        num_missing_observation = (
            df[self.response]
            .isna()
            .any(axis=1)
            .sum()
        )
        if num_missing_observation:
            logger.info(
                "Total missing observations: ",
                f"{num_missing_observation}"
            )
        assert not num_missing_observation

        # Encode
        if encoder_dict is None:
            df, encoder_dict = self._preprocess(df=df, columns=self.features)
        else:
            df[self.features] = (
                df[self.features]
                .apply(lambda x: encoder_dict[x.name].transform(x))
            )
        return df, encoder_dict

    @timing
    def load(
        self,
        df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        self._make_dir(dir=self.build_dir)
        logger.info(f"Artefacts will be stored here - {self.build_dir}")

        # Read data if not provided
        if df is None:
            logger.info(f"Reading data from {self.csv_path} ...")
            df = pd.read_csv(self.csv_path)

        # Concatenate (necessary) features
        for i, feature in enumerate(self._features):
            if isinstance(feature, list):
                df[self.features[i]] = (
                    df[feature]
                    .apply(lambda x: const.SEP.join(x), axis=1)
                )

        # Positive response constraint
        num_non_positive_observation = (
            (df[self.response] <= 0)
            .any(axis=1)
            .sum()
        )
        if num_non_positive_observation:
            logger.info(
                "Total non-positive observations: ",
                f"{num_non_positive_observation}"
            )
        assert not num_non_positive_observation

        # Missing response constraint
        num_missing_observation = (
            df[self.response]
            .isna()
            .any(axis=1)
            .sum()
        )
        if num_missing_observation:
            logger.info(f"Total missing observations: {num_missing_observation}")
        # assert not num_missing_observation

        logger.info("Processing data ...")
        df, encoder_dict = self._preprocess(df=df, columns=self.features)
        return df, encoder_dict
