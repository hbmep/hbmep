import logging
from time import time
from functools import wraps

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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


@timing
def plot(
    df: pd.DataFrame,
    encoder_dict: dict = None,
    pred: pd.DataFrame = None,
    mat: np.ndarray = None,
    time: np.ndarray = None
):
    if pred is not None:
        assert encoder_dict is not None

    if mat is not None:
        assert time is not None

    columns = [PARTICIPANT] + FEATURES
    combinations = \
        df \
        .groupby(by=columns) \
        .size() \
        .to_frame("counts") \
        .reset_index().copy()
    combinations = combinations[columns].apply(tuple, axis=1).tolist()
    n_combinations = len(combinations)

    n_columns = 2 if mat is None else 3

    fig, axes = plt.subplots(
        n_combinations,
        n_columns,
        figsize=(n_columns * 5, n_combinations * 3),
        constrained_layout=True
    )

    for i, combination in enumerate(combinations):
        idx = df[columns].apply(tuple, axis=1).isin([combination])

        temp_df = df[idx].reset_index(drop=True).copy()

        sns.kdeplot(temp_df[RESPONSE], ax=axes[i, 0], color="b", label=f"{RESPONSE}")
        sns.kdeplot(np.log(temp_df[RESPONSE]), ax=axes[i, 0], color="g", label=f"log {RESPONSE}")

        title = f"{columns} - {combination}"
        axes[i, 0].set_title(title)
        axes[i, 0].legend(loc="upper left")

        ax = axes[i] if mat is None else axes[i][0]
        sns.scatterplot(data=temp_df, x=INTENSITY, y=RESPONSE, ax=axes[i, 1])

        axes[i, 1].set_xlabel(f"{INTENSITY}")
        axes[i, 1].set_ylabel(f"{RESPONSE}")

        if encoder_dict is not None:
            c_inverse = []
            for (column, value) in zip(columns, combination):
                value_inverse = encoder_dict[column].inverse_transform(np.array([value]))[0]
                c_inverse.append(value_inverse)
            title = f"{tuple(c_inverse)}"

        axes[i, 1].set_title(title)

        # """ Ahmet's method """
        # if pred is not None:
        #     temp_pred = pred[pred[columns].apply(tuple, axis=1).isin([(c0, c1, c2)])]
        #     prediction = temp_pred[RESPONSE].values
        #     assert len(prediction) == 1
        #     ax.axvline(
        #         x=prediction[0],
        #         color="red",
        #         linestyle='--',
        #         alpha=.4,
        #         label=f"Ahmet's prediction: {prediction[0]}"
        #     )
        #     ax.legend(loc="upper right")

        if mat is not None:
            ax = axes[i, 2]
            temp_mat = mat[idx, :]

            for j in range(temp_mat.shape[0]):
                x = temp_mat[j, :]/60 + temp_df[INTENSITY].values[j]
                ax.plot(x, time, color="green", alpha=.4)

            ax.axhline(
                y=0.003, color="red", linestyle='--', alpha=.4, label="AUC Window"
            )
            ax.axhline(
                y=0.015, color="red", linestyle='--', alpha=.4
            )

            ax.set_ylim(bottom=-0.001, top=0.02)

            ax.set_xlabel(f"{INTENSITY}")
            ax.set_ylabel(f"Time")

            ax.legend(loc="upper right")
            ax.set_title(f"Motor Evoked Potential")

    return fig
