import os
import sys
import logging

import pandas as pd
import numpy as np
from hbmep.util import timing, setup_logging

from hbmep.notebooks.rat.model import Estimation
from hbmep.notebooks.rat.util import load_size, run
from constants import BUILD_DIR, TOML_PATH

logger = logging.getLogger(__name__)


@timing
def main(model, remove_c5=False):
    run_id = model.run_id

    set_reference = False
    if "reference" in model._model.__name__:
        set_reference = True

    df = load_size(**model.variables, run_id=run_id, set_reference=set_reference)
    # idx = df["segment"].apply(lambda x: "C6" in x)
    # df = df[idx].reset_index(drop=True).copy()
    # model.features = ["participant", "lat"]

    model.features = ["participant", "compound_size", "lat", "segment"]
    t = df.groupby(model.features[1:], as_index=True).agg({model.features[0]: [np.unique, lambda x: x.nunique()]})
    print(t)

    if model.test_run:
        model.build_dir = os.path.join(model.build_dir, "test_run")
        os.makedirs(model.build_dir, exist_ok=True)
        subset = ["amap03", "amap04"]
        idx = df[model.features[0]].isin(subset)
        df = df[idx].reset_index(drop=True).copy()
        model.response = model.response[:3]
        model.mcmc_params = {
            "num_chains": 4,
            "thinning": 1,
            "num_warmup": 400,
            "num_samples": 400,
        }

    t = df.groupby(model.features[1:], as_index=True).agg({model.features[0]: [np.unique, lambda x: x.nunique()]})
    print(t)
    run(df, model, extra_fields=["num_steps"])
    return


if __name__ == "__main__":
    model = Estimation(toml_path=TOML_PATH)
    model.features = ["participant", "segment", "lat", "compound_size"]
    model.use_mixture = False
    model.test_run = True

    # model._model = model.size_est_mvn_reference_rl_masked
    model._model = model.size_est_mvn_block_reference_rl_masked
    model.use_mixture = True
    model.run_id = "size-ground"
    # model.run_id = "size-no-ground"

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
    model.nuts_params["target_accept_prob"] = .95
    model.nuts_params["max_tree_depth"] = (15, 15)

    model.build_dir = os.path.join(BUILD_DIR, model.name, model.run_id, model._model.__name__)
    setup_logging(model.build_dir)
    main(model)
