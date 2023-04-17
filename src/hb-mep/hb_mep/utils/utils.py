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
def plot(df: pd.DataFrame):
    columns = [PARTICIPANT] + FEATURES
    combinations = \
        df \
        .groupby(by=columns) \
        .size() \
        .to_frame("counts") \
        .reset_index().copy()
    combinations = combinations[columns].apply(tuple, axis=1).tolist()
    n_combinations = len(combinations)

    fig, axes = plt.subplots(
        n_combinations, 1, figsize=(8, n_combinations * 3), constrained_layout=True
    )

    for i, c in enumerate(combinations):
        idx = df[columns].apply(tuple, axis=1).isin([c])
        temp_df = df[idx].reset_index(drop=True).copy()

        sns.scatterplot(data=temp_df, x=INTENSITY, y=RESPONSE, ax=axes[i])
        axes[i].set_title(f'Actual: Combination:{c}, {RESPONSE}')

    return fig
