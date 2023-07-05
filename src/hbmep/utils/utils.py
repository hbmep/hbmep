import logging
from time import time
from pathlib import Path
from typing import Optional
from functools import wraps

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from numpyro.diagnostics import hpdi

from hb_mep.utils.constants import (
    INTENSITY,
    RESPONSE,
    PARTICIPANT,
    FEATURES
)

logger = logging.getLogger(__name__)


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        time_taken = te - ts
        hours_taken = time_taken // (60 * 60)
        time_taken %= (60 * 60)
        minutes_taken = time_taken // 60
        time_taken %= 60
        seconds_taken = time_taken % 60
        if hours_taken:
            message = \
                f"func:{f.__name__} took: {hours_taken:0.0f} hr and " + \
                f"{minutes_taken:0.0f} min"
        elif minutes_taken:
            message = \
                f"func:{f.__name__} took: {minutes_taken:0.0f} min and " + \
                f"{seconds_taken:0.2f} sec"
        else:
            message = f"func:{f.__name__} took: {seconds_taken:0.2f} sec"
        logger.info(message)
        return result
    return wrap


def ceil(x: float, base: int = 10):
    return base * np.ceil(x / base)


def make_combinations(df: pd.DataFrame, columns: list[str]):
    assert set(columns) <= set(df.columns)
    combinations = \
        df \
        .groupby(by=columns) \
        .size() \
        .to_frame("counts") \
        .reset_index().copy()
    combinations = combinations[columns].apply(tuple, axis=1).tolist()
    combinations = sorted(combinations)
    return combinations


def evaluate_posterior_mean(posterior_samples, prob: float = .95):
    posterior_mean = posterior_samples.mean(axis=0)
    return posterior_mean


def evaluate_hpdi_interval(posterior_samples, prob: float = .95):
    hpdi_interval = hpdi(posterior_samples, prob=prob)
    return hpdi_interval


@timing
def plot(
    df: pd.DataFrame,
    save_path: Path,
    encoder_dict: Optional[dict] = None,
    pred: Optional[pd.DataFrame] = None,
    mat: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
    auc_window: Optional[list[float]] = None
):
    if pred is not None:
        assert encoder_dict is not None

    if mat is not None:
        assert time is not None
        assert auc_window is not None

    columns = [PARTICIPANT] + FEATURES
    combinations = make_combinations(df, columns)

    n_combinations = len(combinations)
    n_response = len(RESPONSE)

    n_fig_columns = 2 + n_response
    if mat is not None: n_fig_columns += n_response

    n_rows = 10
    n_pages = n_combinations // n_rows

    if n_combinations % n_rows:
        n_pages += 1

    pdf = PdfPages(save_path)
    combination_counter = 0

    for page in range(n_pages):
        n_rows_current_page = min(n_rows, n_combinations - page * n_rows)

        fig, axes = plt.subplots(
            n_rows_current_page,
            n_fig_columns,
            figsize=(n_fig_columns * 5, n_rows_current_page * 3),
            constrained_layout=True,
            squeeze=False
        )

        for i in range(n_rows_current_page):
            combination = combinations[combination_counter]

            idx = df[columns].apply(tuple, axis=1).isin([combination])
            temp_df = df[idx].reset_index(drop=True).copy()

            """ Response KDE """
            sns.kdeplot(temp_df[RESPONSE], ax=axes[i, 0])

            title = f"{columns} - {combination}"
            axes[i, 0].set_title(title)
            axes[i, 0].legend(loc="upper right", labels=RESPONSE)


            """ Log Response KDE """
            sns.kdeplot(np.log(temp_df[RESPONSE]), ax=axes[i, 1])
            axes[i, 1].legend(
                loc="upper right",
                labels=["log " + r for r in RESPONSE]
            )

            """ Inverted labels """
            if encoder_dict is not None:
                combination_inverse = []
                for (column, value) in zip(columns, combination):
                    value_inverse = encoder_dict[column].inverse_transform(np.array([value]))[0]
                    combination_inverse.append(value_inverse)

                title_inverted = f"{tuple(combination_inverse)}"
                axes[i, 1].set_title(title_inverted)

            j = 2
            for response in RESPONSE:
                """ EEG data """
                if mat is not None:
                    ax = axes[i, j]

                    muscle_idx = int(response.split("_")[1]) - 1
                    temp_mat = mat[idx, :, muscle_idx]

                    for k in range(temp_mat.shape[0]):
                        x = temp_mat[k, :]/60 + temp_df[INTENSITY].values[k]
                        ax.plot(x, time, color="green", alpha=.4)

                    ax.axhline(
                        y=auc_window[0], color="red", linestyle='--', alpha=.4, label=f"AUC Window {auc_window}"
                    )
                    ax.axhline(
                        y=auc_window[1], color="red", linestyle='--', alpha=.4
                    )

                    ax.set_ylim(bottom=-0.001, top=0.02)

                    ax.set_xlabel(f"{INTENSITY}")
                    ax.set_ylabel("Time")

                    ax.legend(loc="upper right")
                    axes[i, j].set_title("Motor Evoked Potential")

                    if encoder_dict is None:
                        axes[i, j].set_title(f"{response} - " + title)
                    else:
                        axes[i, j].set_title(f"{response} - " + title_inverted)

                    j += 1

                """ Scatter plot """
                sns.scatterplot(data=temp_df, x=INTENSITY, y=response, ax=axes[i, j])

                axes[i, j].set_xlabel(f"{INTENSITY}")
                axes[i, j].set_ylabel(f"{response}")
                axes[i, j].set_title("MEP Size (AUC)")

                j += 1
            combination_counter += 1
        pdf.savefig(fig)
        plt.close()

    pdf.close()
    plt.show()