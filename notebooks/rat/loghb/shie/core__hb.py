import os
import sys
import logging

import pandas as pd
import numpy as np
from hbmep.util import timing, setup_logging

from hbmep.notebooks.rat.model import HB
from hbmep.notebooks.rat.util import load_shie, run
from constants import BUILD_DIR, TOML_PATH

logger = logging.getLogger(__name__)


@timing
def main(model):
    run_id = model.run_id
    df = load_shie(**model.variables, run_id=run_id)

    if model.test_run:
        subset = ["amap01", "amap02"]
        idx = df[model.features[0]].isin(subset)
        df = df[idx].reset_index(drop=True).copy()
        model.response = model.response[:3]
        model.mcmc_params = {
            "num_chains": 4,
            "thinning": 1,
            "num_warmup": 400,
            "num_samples": 400,
        }

    logger.info(f"*** run id: {run_id} ***")
    logger.info(f"*** model: {model._model.__name__} ***")
    run(df, model, extra_fields=["num_steps"])
    return


if __name__ == "__main__":
    model = HB(toml_path=TOML_PATH)
    model.use_mixture = False
    # model.test_run = True

    model._model = model.hb_mvn_rl_nov_masked
    model.use_mixture = True
    model.run_id = "all"
    # model.run_id = "ground"
    # model.run_id = "no-ground"

    # model._model = model.hb_mvn_l4_masked
    # model.run_id = "all"
    # model.use_mixture = True

    # model._model = model.hb_mvn_rl_masked
    # model.run_id = "all"
    # model.use_mixture = True

    model.mcmc_params = {
        "num_chains": 4,

        "thinning": 4,
        "num_warmup": 4000,
        "num_samples": 4000,

        # "thinning": 1,
        # "num_warmup": 1000,
        # "num_samples": 1000,

        # "thinning": 1,
        # "num_warmup": 400,
        # "num_samples": 400,

    }
    model.nuts_params = {
        "max_tree_depth": (15, 15),
        "target_accept_prob": .95,
    }

    model.build_dir = os.path.join(BUILD_DIR, model.run_id, model.name, model._model.__name__)
    setup_logging(model.build_dir)
    main(model)
