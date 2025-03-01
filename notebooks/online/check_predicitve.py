import os
import logging
import pickle

import pandas as pd
import numpy as np
import jax
from jax import random
from hbmep.util import timing, setup_logging, Site as site

from model import Simulator, HB
from util import generate_nested_pulses
from constants import (
    BUILD_DIR, TOML_PATH, SIM_DF_PATH, SIM_PPD_PATH, POSTERIOR_PATH
)

PLATFORM = "cpu"
jax.config.update("jax_platforms", PLATFORM)
logger = logging.getLogger(__name__)


@timing
def main():
    sdf = pd.read_csv(SIM_DF_PATH)
    with open(SIM_PPD_PATH, "rb") as f:
        sppd, = pickle.load(f)

    src = POSTERIOR_PATH
    with open(src, "rb") as g:
        posterior, = pickle.load(g) 

    sim = Simulator(toml_path=TOML_PATH)
    sim.build_dir = os.path.join(BUILD_DIR, sim.name)
    setup_logging(sim.build_dir)
    
    logger.info(f"Checking predictive...")
    simulation_ppd = sim.predict(sdf, posterior=posterior)
    ind = np.arange(0, simulation_ppd[site.a].shape[0], 1)
    _, rng_key = random.split(sim.key)
    ind = random.permutation(rng_key, ind)
    ind = np.array(ind)
    simulation_ppd = {
        u: v[ind, ...] for u, v in simulation_ppd.items()
    }    
    np.testing.assert_almost_equal(
        simulation_ppd[site.obs],
        sppd[site.obs]
    )
    logger.info("Predictive check: ok")
    return


if __name__ == "__main__":
    main()
