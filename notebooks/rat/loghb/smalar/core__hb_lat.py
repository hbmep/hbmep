import os
import sys
import logging

import pandas as pd
import numpy as np
from hbmep.util import timing, setup_logging

from hbmep.notebooks.rat.model import HB
from hbmep.notebooks.rat.util import run, log_transform_intensity
from constants import (
    BUILD_DIR,
    TOML_PATH,
    DATA_PATH_FILTERED,
    GROUND_BIG,
    GROUND_SMALL,
    NO_GROUND_BIG,
    NO_GROUND_SMALL,
)

logger = logging.getLogger(__name__)


@timing
def main(model):
    # Load data
    src = DATA_PATH_FILTERED
    data = pd.read_csv(src)
    df = log_transform_intensity(data, model.intensity)
    
    run_id = model.run_id
    assert run_id in {"small-ground", "big-ground", "small-no-ground", "big-no-ground"}
    subset = []
    match run_id:
        case "small-ground": subset = GROUND_SMALL
        case "big-ground": subset = GROUND_BIG
        case "small-no-ground": subset = NO_GROUND_SMALL
        case "big-no-ground": subset = NO_GROUND_BIG
        case _: raise ValueError
    assert len(set(subset)) == len(subset)
    cols = ["lat", "segment", "compound_size"]
    assert set(subset) <= set(df[cols].apply(tuple, axis=1).tolist())
    idx = df[cols].apply(tuple, axis=1).isin(subset)
    df = df[idx].reset_index(drop=True).copy()
    df[model.features[-1]] = df[model.features[-1]].replace(
        {"-LM1": "-LM", "M-LM1": "M-LM"}
    )
    # df[model.features[-2]] = df[model.features[-3]].replace(
    #     {"C5-C5": "-C5", "C6-C6": "-C6"}
    # )

    if model.test_run:
        os.makedirs(model.build_dir, exist_ok=True)
        model.build_dir = os.path.join(model.build_dir, "test_run")
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

    run(df, model, extra_fields=["num_steps"])
    return


if __name__ == "__main__":
    model = HB(toml_path=TOML_PATH)
    model.features = ["participant", "segment", "lat"]
    model.use_mixture = False
    # model.test_run = True

    model._model = model.hb_mvn_rl_nov_masked
    model.run_id = "small-ground"
    # model.run_id = "big-ground"
    # model.run_id = "small-no-ground"
    # model.run_id = "big-no-ground"

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

    model.build_dir = os.path.join(BUILD_DIR, model.name, model.run_id, model._model.__name__)
    setup_logging(model.build_dir)
    main(model)
