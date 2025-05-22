import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from jax import random, numpy as jnp
from numpyro.diagnostics import hpdi
import hbmep as mep
from hbmep.util import timing, setup_logging

from hbmep.notebooks.util import make_pdf
from models import nHB
from util import Site as site, load_model, make_serial
from constants import (
    SIMULATION_FACTOR_DIR as BUILD_DIR,
    FACTOR,
    FACTORS_SPACE,
    REP,
    TOTAL_REPS,
)

logger = logging.getLogger(__name__)


def viz(model_dir, num_reps, dilution_factor):
    (
        df,
		encoder,
		model,
		posterior,
        *_,
    ) = load_model(model_dir)
    posterior.keys()

    obs = posterior[site.obs].copy()
    mu = posterior[site.mu].copy()

    num_draws = 30
    combinations = df[model.features].apply(tuple, axis=1).unique()
    combinations = sorted(combinations)
    num_rows = 10
    num_pages = num_draws // num_rows + (num_draws % num_rows > 0)
    num_columns = 12
    df_features = df[model.features].apply(tuple, axis=1)

    out = []; counter = 0
    for page in range(num_pages):
        num_rows_current = min(num_rows, num_draws - page * num_rows)
        fig, axes = plt.subplots(
            *(num_rows_current, num_columns),
            figsize=(3.5 * num_columns, 2 * num_rows_current),
            squeeze=True,
            constrained_layout=True
        )
        for row in range(num_rows_current):
            for combination in combinations:
                f0_idx, = combination
                idx = (
                    df_features.isin([combination])
                    & (df[REP] < num_reps)
                    & df[FACTOR].isin([dilution_factor])
                )
                ccdf = df[idx].reset_index(drop=True).copy()
                ax = axes[row, f0_idx]; ax.clear()
                x = ccdf[model.intensity]
                y = obs[:, idx, ...][counter, ..., 0]
                sns.scatterplot(x=x, y=y, ax=ax)
                y = mu[:, idx, ...][counter, ..., 0]
                sns.lineplot(x=x, y=y, ax=ax, color="k")
                if not row: ax.set_title(f"{combination}")
            counter += 1
        for i in range(num_rows_current): 
            for j in range(num_columns):
                ax = axes[i, j]
                ax.set_xscale("log")
        print(f"Page {page + 1} of {num_pages} done.")
        out.append(fig)

    output_path = os.path.join(
        model_dir,
        f"simulation_data_r{num_reps}_f{dilution_factor}.pdf"
    )
    make_pdf(out, output_path=output_path)
    return


@timing
def main():
    os.makedirs(BUILD_DIR, exist_ok=True)
    for num_reps in range(TOTAL_REPS):
        for factor in FACTORS_SPACE:
            print(f"Plotting {num_reps + 1} reps with dilution factor {factor}...")
            viz(BUILD_DIR, num_reps + 1, factor)
    return


if __name__ == "__main__":
    main()
