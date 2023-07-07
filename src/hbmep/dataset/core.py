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
        self.csv_path = config.CSV_PATH
        self.build_dir = config.BUILD_DIR
        self.run_id = config.RUN_ID
        self.run_dir = os.path.join(self.build_dir, self.run_id)

        self._make_dir(dir=self.run_dir)
        logger.info(f"Initialized {self.run_dir} for storing artefacts")
        self._copy(src=config.TOML_PATH, dst=self.run_dir)
        logger.info(f"Copied config to {self.run_dir}")

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

    def _make_dir(self, dir: str):
        Path(dir).mkdir(parents=True, exist_ok=True)
        return

    def _copy(self, src: str, dst: str):
        shutil.copy(src, dst)
        return

    def _make_combinations(self, df: pd.DataFrame, columns: list[str]):
        assert set(columns) <= set(df.columns)
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
    ):
        combination_inverse = []
        for (column, value) in zip(columns, combination):
            value_inverse = encoder_dict[column].inverse_transform(np.array([value]))[0]
            combination_inverse.append(value_inverse)

        combination_inverse = tuple(combination_inverse)
        return combination_inverse

    @timing
    def preprocess(
        self,
        df: pd.DataFrame,
        scalar_intensity: float,
        scalar_response: list[float],
        min_observations: int,
        mat: Optional[np.ndarray] = None
        ) -> tuple[pd.DataFrame, dict, dict[str,  LabelEncoder]]:
        assert len(self.response) == len(scalar_response)

        """ Remove zero AUC response """
        ind = df[self.response].isin([0]).any(axis=1)
        if ind.sum():
            df = df[~ind].copy()
            if mat is not None: mat = mat[ind, :, :]
            logger.info(
                f"Removed {ind.sum()} observation(s) containing zero response"
            )

        """ Rescale data """
        df[self.intensity] = df[self.intensity].apply(lambda x: x * scalar_intensity)
        df[self.response] = df[self.response] * scalar_response

        """ Mininum-observations constraint """
        temp_df = df \
            .groupby(by=[self.subject] + self.features) \
            .size() \
            .to_frame('counts') \
            .reset_index() \
            .copy()

        temp_df = temp_df[temp_df.counts >= min_observations].copy()
        keep = list(temp_df[[self.subject] + self.features].apply(tuple, axis=1))
        ind = df[[self.subject] + self.features].apply(tuple, axis=1).isin(keep)
        df = df[ind].copy()
        if mat is not None: mat = mat[ind, :, :]

        """ Encode participants and features """
        encoder_dict = defaultdict(LabelEncoder)
        df[[self.subject] + self.features] = df[[self.subject] + self.features] \
            .apply(lambda x: encoder_dict[x.name].fit_transform(x)) \
            .copy()

        df.reset_index(inplace=True, drop=True)
        return df, encoder_dict, mat

    @timing
    def build(
        self,
        df: Optional[pd.DataFrame] = None,
        mat: Optional[np.ndarray] = None
    ):
        if df is None:
            csv_path = self.csv_path
            logger.info(f"Reading data from {csv_path} ...")
            df = pd.read_csv(csv_path)

        df[[f"{RAW}" + r for r in self.response]] = df[self.response]
        logger.info("Processing data ...")
        df, encoder_dict, mat = self.preprocess(df=df, **self.preprocess_params, mat=mat)
        return df, encoder_dict, mat

    @timing
    def plot(
        self,
        df: pd.DataFrame,
        encoder_dict: Optional[dict] = None,
        mat: Optional[np.ndarray] = None,
        time: Optional[np.ndarray] = None,
        auc_window: Optional[list[float]] = None
    ):
        if mat is not None:
            assert time is not None
            assert auc_window is not None

        """ Setup pdf layout """
        combinations = self._make_combinations(df=df, columns=self.columns)
        n_combinations = len(combinations)

        n_columns_per_response = 1
        if mat is not None: n_columns_per_response += 1

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
                figsize=(n_fig_columns * 5, n_rows_current_page * 3),
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

                title = f"{self.columns} - {combination}"
                axes[i, 0].set_title(title)
                axes[i, 0].legend(loc="upper right", labels=self.response)

                """ Log Response KDE """
                sns.kdeplot(np.log(temp_df[self.response]), ax=axes[i, 1])
                axes[i, 1].legend(
                    loc="upper right",
                    labels=["log " + r for r in self.response]
                )

                """ Inverted labels """
                if encoder_dict is not None:
                    combination_inverse = self._invert_combination(
                        combination=combination,
                        columns=self.columns,
                        encoder_dict=encoder_dict
                    )
                    axes[i, 1].set_title(combination_inverse)

                j = 2
                for response in self.response:
                    """ EEG data """
                    if mat is not None:
                        ax = axes[i, j]
                        temp_mat = mat[ind, :, r]

                        for k in range(temp_mat.shape[0]):
                            x = temp_mat[k, :]/60 + temp_df[self.intensity].values[k]
                            ax.plot(x, time, color="green", alpha=.4)

                        ax.axhline(
                            y=auc_window[0],
                            color="red",
                            linestyle='--',
                            alpha=.4,
                            label=f"AUC Window {auc_window}"
                        )
                        ax.axhline(
                            y=auc_window[1],
                            color="red",
                            linestyle='--',
                            alpha=.4
                        )

                        ax.set_xticks(ticks=x_ticks)
                        ax.tick_params(axis="x", rotation=90)
                        ax.set_xlim(left=min_intensity, right=max_intensity)
                        ax.set_ylim(bottom=-0.001, top=auc_window[1] + .005)

                        ax.set_xlabel(f"{self.intensity}")
                        ax.set_ylabel(f"Time")
                        ax.legend(loc="upper right")
                        ax.set_title(f"Motor Evoked Potential")

                        j += 1

                    """ Plots """
                    ax = axes[i, j]
                    sns.scatterplot(data=temp_df, x=self.intensity, y=response, ax=ax)

                    ax.set_xticks(ticks=x_ticks)
                    ax.tick_params(axis="x",rotation=90)
                    ax.set_xlim(left=min_intensity, right=max_intensity)
                    ax.set_xlabel(f"{self.intensity}")
                    ax.set_ylabel(f"{response}")
                    ax.set_title("MEP Size (AUC)")

                    j += 1

                combination_counter += 1

            pdf.savefig(fig)
            plt.close()

        pdf.close()
        plt.show()

        logger.info(f"Saved to {self.dataset_plot_path}")
        return
