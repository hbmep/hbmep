import os
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from hbmep.config import Config
from hbmep.utils import timing
from hbmep.utils.constants import REPORTS_DIR

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, config: Config):
        self.here = Path(os.getcwd())
        if config.HERE is not None: self.here = config.HERE
        self.csv_path = config.CSV_PATH
        self.reports_path = Path(os.path.join(config.HERE, REPORTS_DIR))

        self.subject = config.SUBJECT
        self.features = config.FEATURES
        self.intensity = config.INTENSITY
        self.response = config.RESPONSE

        self.preprocess_params = config.PREPROCESS_PARAMS

        self.n_response = len(config.RESPONSE)
        self.columns = [config.SUBJECT] + config.FEATURES

        self._make_dir(self.reports_path)

    def _make_dir(self, dir: Path):
        dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Artefacts directory - {dir}")
        return

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
        df[[self.subject] + self.features] = \
            df[[self.subject] + self.features] \
                .apply(
                    lambda x: encoder_dict[x.name].fit_transform(x)
                ) \
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
            fpath = self.data_path
            logger.info(f"Reading data from {fpath}...")
            df = pd.read_csv(fpath)

        df[["raw_" + r for r in self.response]] = df[self.response]
        logger.info("Processing data ...")
        df, encoder_dict, mat = self.preprocess(df=df, **self.preprocess_params, mat=mat)
        return df, encoder_dict, mat

    def _make_combinations(self, df: pd.DataFrame):
        assert set(self.columns) <= set(df.columns)
        combinations = df \
            .groupby(by=self.columns) \
            .size() \
            .to_frame("counts") \
            .reset_index() \
            .copy()
        combinations = combinations[self.columns] \
            .apply(tuple, axis=1) \
            .tolist()
        combinations = sorted(combinations)
        return combinations

    # @timing
    # def plot(
    #     df: pd.DataFrame,
    #     save_path: Path,
    #     encoder_dict: Optional[dict] = None,
    #     pred: Optional[pd.DataFrame] = None,
    #     mat: Optional[np.ndarray] = None,
    #     time: Optional[np.ndarray] = None,
    #     auc_window: Optional[list[float]] = None
    # ):
    #     if pred is not None:
    #         assert encoder_dict is not None

    #     if mat is not None:
    #         assert time is not None
    #         assert auc_window is not None

    #     columns = [PARTICIPANT] + FEATURES
    #     combinations = make_combinations(df, columns)

    #     n_combinations = len(combinations)
    #     n_response = len(RESPONSE)

    #     n_fig_columns = 2 + n_response
    #     if mat is not None: n_fig_columns += n_response

    #     n_rows = 10
    #     n_pages = n_combinations // n_rows

    #     if n_combinations % n_rows:
    #         n_pages += 1

    #     pdf = PdfPages(save_path)
    #     combination_counter = 0

    #     for page in range(n_pages):
    #         n_rows_current_page = min(n_rows, n_combinations - page * n_rows)

    #         fig, axes = plt.subplots(
    #             n_rows_current_page,
    #             n_fig_columns,
    #             figsize=(n_fig_columns * 5, n_rows_current_page * 3),
    #             constrained_layout=True,
    #             squeeze=False
    #         )

    #         for i in range(n_rows_current_page):
    #             combination = combinations[combination_counter]

    #             idx = df[columns].apply(tuple, axis=1).isin([combination])
    #             temp_df = df[idx].reset_index(drop=True).copy()

    #             """ Response KDE """
    #             sns.kdeplot(temp_df[RESPONSE], ax=axes[i, 0])

    #             title = f"{columns} - {combination}"
    #             axes[i, 0].set_title(title)
    #             axes[i, 0].legend(loc="upper right", labels=RESPONSE)


    #             """ Log Response KDE """
    #             sns.kdeplot(np.log(temp_df[RESPONSE]), ax=axes[i, 1])
    #             axes[i, 1].legend(
    #                 loc="upper right",
    #                 labels=["log " + r for r in RESPONSE]
    #             )

    #             """ Inverted labels """
    #             if encoder_dict is not None:
    #                 combination_inverse = []
    #                 for (column, value) in zip(columns, combination):
    #                     value_inverse = encoder_dict[column].inverse_transform(np.array([value]))[0]
    #                     combination_inverse.append(value_inverse)

    #                 title_inverted = f"{tuple(combination_inverse)}"
    #                 axes[i, 1].set_title(title_inverted)

    #             j = 2
    #             for response in RESPONSE:
    #                 """ EEG data """
    #                 if mat is not None:
    #                     ax = axes[i, j]

    #                     muscle_idx = int(response.split("_")[1]) - 1
    #                     temp_mat = mat[idx, :, muscle_idx]

    #                     for k in range(temp_mat.shape[0]):
    #                         x = temp_mat[k, :]/60 + temp_df[INTENSITY].values[k]
    #                         ax.plot(x, time, color="green", alpha=.4)

    #                     ax.axhline(
    #                         y=auc_window[0], color="red", linestyle='--', alpha=.4, label=f"AUC Window {auc_window}"
    #                     )
    #                     ax.axhline(
    #                         y=auc_window[1], color="red", linestyle='--', alpha=.4
    #                     )

    #                     ax.set_ylim(bottom=-0.001, top=0.02)

    #                     ax.set_xlabel(f"{INTENSITY}")
    #                     ax.set_ylabel("Time")

    #                     ax.legend(loc="upper right")
    #                     axes[i, j].set_title("Motor Evoked Potential")

    #                     if encoder_dict is None:
    #                         axes[i, j].set_title(f"{response} - " + title)
    #                     else:
    #                         axes[i, j].set_title(f"{response} - " + title_inverted)

    #                     j += 1

    #                 """ Scatter plot """
    #                 sns.scatterplot(data=temp_df, x=INTENSITY, y=response, ax=axes[i, j])

    #                 axes[i, j].set_xlabel(f"{INTENSITY}")
    #                 axes[i, j].set_ylabel(f"{response}")
    #                 axes[i, j].set_title("MEP Size (AUC)")

    #                 j += 1
    #             combination_counter += 1
    #         pdf.savefig(fig)
    #         plt.close()

    #     pdf.close()
    #     plt.show()
