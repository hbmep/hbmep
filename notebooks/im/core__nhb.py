import os
import logging

import numpy as np
import pandas as pd
from hbmep.util import timing, setup_logging

from models import nHB, HB
from util import Site as site, run
from constants import BUILD_DIR, DATA_PATH, CONFIG
from models import EPS

logger = logging.getLogger(__name__)


@timing
def _run(model):
    run_id = model.run_id
    df = pd.read_csv(DATA_PATH)
    df.conc = df.conc.replace({0: EPS})
    # idx = df.conc > 0
    # df = df[idx].reset_index(drop=True).copy()

    idx = df.plate == run_id
    df = df[idx].reset_index(drop=True).copy()

    if run_id == "070825_plate":
        idx = (
            (df[model.response[0]] > .7)
            & (df[model.features[0]].isin([0.0063, 0.0125]))
        )
        df = df[~idx].reset_index(drop=True).copy()

    if model.test_run:
        if "non_hierarchical" in model._model.__name__:
            model.n_jobs = 1
        df_features = df[model.features].apply(tuple, axis=1)
        combinations = sorted(df_features.unique().tolist())
        idx = df_features.isin(combinations[:3])
        df = df[idx].reset_index(drop=True).copy()
        model.build_dir = os.path.join(model.build_dir, "test_run")
        os.makedirs(model.build_dir, exist_ok=True)
        model.mcmc_params = {
            "num_warmup": 400,
            "num_samples": 400,
            "num_chains": 4,
            "thinning": 1,
        }

    logger.info(f"run_id {run_id}")
    logger.info(f"Running {model._model.__name__}...")
    run(
        df,
		model,
		extra_fields=["num_steps"]
    )
    return


def main():
    # model = nHB(config=CONFIG)
    # model._model = model.non_hierarchical

    model = HB(config=CONFIG)
    model._model = model.hierarchical_sharedb1b4

    model.mcmc_params = {
        "num_warmup": 10_000,
        "num_samples": 10_000,
        "num_chains": 4,
        "thinning": 10,
    }
    model.nuts_params = {
        "max_tree_depth": (15, 15),
        "target_accept_prob": .95,
    }
    model.use_mixture = False
    model.test_run = False

    # ['070825_plate', '070925_plate']
    model.run_id = "070825_plate"

    model.build_dir = os.path.join(
        BUILD_DIR,
		model.run_id,
		model.name,
		model._model.__name__
    )
    setup_logging(model.build_dir)
    _run(model)
    return


if __name__ == "__main__":
    main()
