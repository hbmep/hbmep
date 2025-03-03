import os
import gc
import sys
import pickle
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import jax
from jax import random, numpy as np
from numpyro.diagnostics import hpdi
from joblib import Parallel, delayed
from hbmep import functional as F
from hbmep.util import (
    timing,
    setup_logging,
    Site as site
)

from models import Simulator, HB
from constants import (
    BUILD_DIR,
    CONFIG,
    SIMULATION_DF_PATH,
    SIMULATION_PPD_PATH,
    N_PULSES_SPACE,
)

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
BUILD_DIR = os.path.join(BUILD_DIR, "viz")
logger = logging.getLogger(__name__)
setup_logging(BUILD_DIR)
plt.rcParams["svg.fonttype"] = "none"
VLINE_KW = {"alpha": .4, "ymax": .78, "linestyle": "--"}


def main(num_pulses, draw):
    with open(SIMULATION_PPD_PATH, "rb") as f:
        simulation_ppd, = pickle.load(f)

    simulator = Simulator(config=CONFIG)
    key, subkey = random.split(simulator.key)

    named_params = [site.a, site.b, site.L, site.ell, site.H]
    simulation_ppd = {u: v for u, v in simulation_ppd.items() if u in named_params}
    simulation_params = [simulation_ppd[named_param][draw: draw + 1, :1, ...] for named_param in named_params]
    a_true = simulation_params[0][0].reshape(-1,)

    model = HB(config=CONFIG)
    model.features = []
    model.build_dir = BUILD_DIR

    df = None
    proposal = [5, 50, 90]
    num_iters = num_pulses // len(proposal)
    nr, nc = 2, (num_iters // 2) * 2
    fig, axes = plt.subplots(nr, nc, figsize=(3.5 * nc, 3 * nr), squeeze=False, constrained_layout=True)

    ax_pos = [
        (0, 0), (0, 1), (1, 0), (1, 1),
        (0, 2), (0, 3), (1, 2), (1, 3),
        (0, 4), (0, 5), (1, 4), (1, 5),
    ]
    counter = 0
    for iter in range(num_iters):
        curr_df = pd.DataFrame(
            np.array([proposal, [0] * len(proposal)]).T, columns=simulator.regressors
        )
        curr_df[simulator.intensity] = curr_df[simulator.intensity].astype(np.float64)
        curr_df[simulator.features] = curr_df[simulator.features].astype(int)

        key, subkey = random.split(key)
        prediction_obs = simulator.predict(
            curr_df,
            num_samples=1,
            posterior={u: v for u, v in zip(named_params, simulation_params)},
            key=subkey,
            return_sites=[site.obs]
        )[site.obs][0, ...]
        curr_df[simulator.response] = prediction_obs

        if df is None: df = curr_df.copy()
        else: df = pd.concat([df, curr_df]).reset_index(drop=True).copy()

        _, posterior = model.run_svi(df=df)
        prediction_df = model.make_prediction_dataset(df=df)
        predictive = model.predict(prediction_df, posterior=posterior)

        a_pred = posterior[site.a][:, None, ...].reshape(-1,)
        a_mae = np.abs(a_pred.mean() - a_true).item()

        ax = axes[*ax_pos[counter]]
        sns.scatterplot(df, x=model.intensity, y=model.response[0], ax=ax, color="b")
        ax.axvline(a_true.mean(), **VLINE_KW, color="g", label="True threshold")
        if iter: sns.scatterplot(curr_df, x=model.intensity, y=model.response[0], ax=ax, color="r", label="Proposal")
        ax.set_title(f"Iteration {iter:02}")
        counter += 1

        ax = axes[*ax_pos[counter]]
        sns.scatterplot(df, x=model.intensity, y=model.response[0], ax=ax, color="b")
        sns.lineplot(prediction_df, x=model.intensity, y=predictive[site.mu].mean(axis=0)[..., 0], ax=ax)
        l, r = hpdi(a_pred, prob=.95)
        sns.kdeplot(a_pred, ax=ax, color="g", alpha=.4)
        ax.axvline(a_true.mean(), **VLINE_KW, color="g")
        ax.axvline(a_pred.mean(), **VLINE_KW, color="r", label="Estimated threshold")
        ax.axvline(l, **VLINE_KW, color="k")
        ax.axvline(r, **VLINE_KW, color="k")
        ax.set_title(f"Mean absolute error: {a_mae:.3f}")
        counter += 1

        proposal = [l, r, (l + r) / 2]
        proposal = [max(0, p) for p in proposal]
        proposal = [min(p, 100) for p in proposal]


    ax = axes[-1, -1]
    ax.set_xticks(np.arange(0, 110, 20))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_xlim(left=0, right=100)
    ax.set_ylim(bottom=-.2)

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.tick_params(axis="both", labelbottom=True, labelleft=True if not j % 2 else False)
            if ax.get_legend(): ax.get_legend().remove()
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.sharex(axes[-1, -1])
            ax.sharey(axes[-1, -1])
            if not j % 2:
                ax.set_xlabel("TMS intensity (% MSO)")
                ax.set_ylabel("MEP size (pk-pk)")
            
    # # fig.show()
    # ax = axes[0, 0]
    # ax.set_xlabel("TMS intensity (% MSO)")
    # ax.set_ylabel("MEP size (pk-pk)")

    # ax = axes[0, 0]
    # ax.legend(bbox_to_anchor=(-.1, 1), loc='upper left')
    # ax = axes[0, 1]
    # ax.legend(loc="upper left")
    # ax = axes[1, 0]
    # ax.legend(loc="upper left")

    fig.align_xlabels()
    fig.align_ylabels()
    output_path = os.path.join(model.build_dir, f"online__p{num_pulses}_d{draw}.svg")
    fig.savefig(output_path, dpi=600); logger.info(f"Saved to {output_path}")
    output_path = os.path.join(model.build_dir, f"online__p{num_pulses}_d{draw}.png")
    fig.savefig(output_path, dpi=600); logger.info(f"Saved to {output_path}")
    plt.close(fig)
    return


if __name__ == "__main__":
    # num_pulses = 18
    # draws_space = range(100)
    # n_jobs = -10
    # with Parallel(n_jobs=n_jobs) as parallel:
    #     parallel(
    #         delayed(main)(num_pulses, draw)
    #         for draw in draws_space
    #     )

    # Success
    main(18, 8)
    # Failure
    main(18, 9)
