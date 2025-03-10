import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

from jax import random, numpy as jnp
from hbmep.util import timing, setup_logging, get_response_colors

import functional as RF
from utils import Site as site
from constants import (
    DATA_PATH,
    BUILD_DIR,
)

num_iter = 5
colors = get_response_colors(num_iter)


def main():
    key = random.key(0)
    key, subkey = random.split(key)
    b1 = np.array(random.uniform(subkey, shape=(100,)))
    key, subkey = random.split(key)
    b2 = np.array(random.uniform(subkey, shape=(100,)))
    key, subkey = random.split(key)
    b3 = np.array(random.uniform(subkey, shape=(100,)))
    # b3 = b3 * 5
    key, subkey = random.split(key)
    b4 = np.array(random.uniform(subkey, shape=(100,)))

    params = [b1, b2, b3, b4]
    x = np.arange(0, 10, .05)
    y = RF.function(
        x[:, None],
        *(b[None, :] for b in params)
    )
    y = np.array(y)
    y.shape

    nr, nc = 1, 2
    fig, axes = plt.subplots(nr, nc, constrained_layout=True, squeeze=False, figsize=(5 * nc, 3 * nr))
    ax = axes[0, 0]
    ax.clear()
    for i in range(num_iter):
        sns.lineplot(x=x, y=y[:, i], ax=ax, color=colors[i])
    ax = axes[0, 1]
    for i in range(num_iter):
        sns.lineplot(x=x, y=y[:, i], ax=ax, color=colors[i])
        ax.axvline(b3[i], colo=colors[i], linestyle="--")
    ax.set_xscale("log")
    fig.show()

    plt.close()
    return


if __name__ == "__main__":
    main()
