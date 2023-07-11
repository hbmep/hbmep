import os
import shutil
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from hbmep.config import MepConfig
from hbmep.utils import (timing, floor, ceil)
from hbmep.utils.constants import (RAW, DATASET_PLOT)

logger = logging.getLogger(__name__)


class MepDataset:
    def __init__(self, config: MepConfig):
        self.toml_path = config.TOML_PATH
        self.csv_path = config.CSV_PATH
        self.build_dir = config.BUILD_DIR
        self.run_id = config.RUN_ID
        self.run_dir = os.path.join(self.build_dir, self.run_id)

        self.subject = config.SUBJECT
        self.features = config.FEATURES
        self.intensity = config.INTENSITY
        self.response = config.RESPONSE

        self.n_features = len(self.features)
        self.n_response = len(self.response)
        self.columns = [self.subject] + self.features

        self.preprocess_params = config.PREPROCESS_PARAMS
        self.dataset_plot_path = os.path.join(self.run_dir, DATASET_PLOT)

        self.base = config.BASE
        self.subplot_cell_width = 5
        self.subplot_cell_height = 3

    def _make_dir(self, dir: str):
        Path(dir).mkdir(parents=True, exist_ok=True)
        return

    def _copy(self, src: str, dst: str):
        shutil.copy(src, dst)
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
            encoder_dict: dict[str,  LabelEncoder]
    ) -> tuple:
        combination_inverse = []
        for (column, value) in zip(columns, combination):
            value_inverse = encoder_dict[column].inverse_transform(np.array([value]))[0]
            combination_inverse.append(value_inverse)

        combination_inverse = tuple(combination_inverse)
        return combination_inverse

    def _preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str,  LabelEncoder]]:
        """ Positive observation constraint  """
        ind = (df[self.response] <= 0).any(axis=1)
        if ind.sum():
            df = df[~ind].copy()
            logger.info(
                f"Removed {ind.sum()} non-positive observation(s)"
            )

        """ Encode """
        encoder_dict = defaultdict(LabelEncoder)
        df[self.columns] = df[self.columns] \
            .apply(lambda x: encoder_dict[x.name].fit_transform(x)) \
            .copy()

        df.reset_index(inplace=True, drop=True)
        return df, encoder_dict

    @timing
    def build(self, df: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, dict[str,  LabelEncoder]]:
        self._make_dir(dir=self.run_dir)
        logger.info(f"Initialized {self.run_dir} for storing artefacts")

        self._copy(src=self.toml_path, dst=self.run_dir)
        logger.info(f"Copied config to {self.run_dir}")

        if df is None:
            csv_path = self.csv_path
            logger.info(f"Reading data from {csv_path} ...")
            df = pd.read_csv(csv_path)

        logger.info("Processing data ...")
        df, encoder_dict = self._preprocess(df=df)
        return df, encoder_dict

    @timing
    def plot(
        self,
        df: pd.DataFrame,
        encoder_dict: dict[str,  LabelEncoder]
    ):
        """ Setup pdf layout """
        combinations = self._make_combinations(df=df, columns=self.columns)
        n_combinations = len(combinations)

        n_columns_per_response = 1
        n_fig_rows = 10
        n_fig_columns = 2 + n_columns_per_response * self.n_response

        n_pdf_pages = n_combinations // n_fig_rows
        if n_combinations % n_fig_rows: n_pdf_pages += 1

        """ Iterate over pdf pages """
        pdf = PdfPages(self.dataset_plot_path)
        combination_counter = 0

        for page in range(n_pdf_pages):
            n_rows_current_page = min(
                n_fig_rows,
                n_combinations - page * n_fig_rows
            )

            fig, axes = plt.subplots(
                n_rows_current_page,
                n_fig_columns,
                figsize=(
                    n_fig_columns * self.subplot_cell_width,
                    n_rows_current_page * self.subplot_cell_height
                ),
                constrained_layout=True,
                squeeze=False
            )

            """ Iterate over combinations """
            for i in range(n_rows_current_page):
                combination = combinations[combination_counter]

                """ Filter dataframe """
                ind = df[self.columns].apply(tuple, axis=1).isin([combination])
                temp_df = df[ind].reset_index(drop=True).copy()

                """ Tickmarks """
                min_intensity = temp_df[self.intensity].min()
                min_intensity = floor(min_intensity, base=self.base)
                max_intensity = temp_df[self.intensity].max()
                max_intensity = ceil(max_intensity, base=self.base)
                x_ticks = np.arange(min_intensity, max_intensity, self.base)

                """ Response KDE """
                sns.kdeplot(temp_df[self.response], ax=axes[i, 0])

                """ Log Response KDE """
                sns.kdeplot(np.log(temp_df[self.response]), ax=axes[i, 1])

                """ Labels """
                title = f"{tuple(self.columns)} - encoded: {combination}"
                axes[i, 0].set_title(title)
                combination_inverse = self._invert_combination(
                    combination=combination,
                    columns=self.columns,
                    encoder_dict=encoder_dict
                )
                title = f"decoded: {combination_inverse}"
                axes[i, 1].set_title(title)

                """ Legends """
                axes[i, 0].legend(loc="upper right", labels=self.response)
                axes[i, 1].legend(
                    loc="upper right",
                    labels=["log " + r for r in self.response]
                )

                j = 2
                for response in self.response:
                    """ Plots """
                    ax = axes[i, j]
                    sns.scatterplot(data=temp_df, x=self.intensity, y=response, ax=ax)

                    ax.set_xticks(ticks=x_ticks)
                    ax.tick_params(axis="x",rotation=90)
                    ax.set_xlim(left=min_intensity, right=max_intensity)
                    ax.set_xlabel(f"{self.intensity}")
                    ax.set_ylabel(f"{response}")
                    ax.set_title(f"{response} - MEP Size")

                    j += 1

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {self.dataset_plot_path}")
        return
