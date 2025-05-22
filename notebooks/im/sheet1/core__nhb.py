import os
import logging

import pandas as pd
from hbmep.util import timing, setup_logging

from models import nHB
from util import Site as site, load, run
from constants import DATA_PATH, TOML_PATH, BUILD_DIR

logger = logging.getLogger(__name__)


@timing
def main(model):
    model.b3_var = (
        site.b3.log
        if "logistic" in model._model.__name__
        else site.b3
    )
    run_id = model.run_id
    assert run_id in {
        "rest-notlogconc", "rest-logconc", 
        "unrest-notlogconc", "unrest-logconc"
    }

    rest_id, log_conc_id = run_id.split("-")
    rest = False
    if rest_id == "rest": rest = True
    log_conc = False
    if log_conc_id == "logconc": log_conc = True

    data = pd.read_csv(DATA_PATH)
    df = load(data, log_conc=log_conc)
    model.features = [["contam_mapped", "plate"]]

    if model.test_run:
        model.n_jobs = 1
        idx = df[model.features[0]].apply(lambda x: "0_" in x)
        df = df[idx].reset_index(drop=True).copy()
        model.build_dir = os.path.join(model.build_dir, "test_run")
        os.makedirs(model.build_dir, exist_ok=True)
        model.mcmc_params = {
            "num_warmup": 400,
            "num_samples": 400,
            "num_chains": 4,
            "thinning": 1,
        }

    logger.info(f"Range restricted {rest}")
    logger.info(f"Log concentration {log_conc}")
    logger.info(f"Running {model._model.__name__}...")
    run(
        df,
		model,
		extra_fields=["num_steps"]
    )
    return


if __name__ == "__main__":
    model = nHB(toml_path=TOML_PATH)
    model.mcmc_params = {
        "num_warmup": int(1e4),
        "num_samples": int(1e4),
        "num_chains": 4,
        "thinning": 10,
    }
    model.nuts_params = {
        "max_tree_depth": (15, 15),
        "target_accept_prob": .95,
    }
    model.use_mixture = True
    # model.test_run = True

    model._model = model.non_hierarchical
    model.run_id = "unrest-notlogconc"

    model.build_dir = os.path.join(
        BUILD_DIR,
		model.run_id,
		model.name,
		model._model.__name__
    )
    setup_logging(model.build_dir)
    main(model)
