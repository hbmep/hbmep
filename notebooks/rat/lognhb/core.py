import os
import logging

import pandas as pd
from hbmep.util import timing, setup_logging

from hbmep.notebooks.rat.model import nHB
from hbmep.notebooks.rat.util import run, log_transform_intensity
from constants import HOME

logger = logging.getLogger(__name__)


@timing
def main(model, data_path):
    # Load data
    data = pd.read_csv(data_path)
    ## This logic was used for lcirc and lshie
    # idx = data[model.intensity] > 0
    # data = data[idx].reset_index(drop=True).copy()
    # data[model.intensity] = np.log2(data[model.intensity])

    ## This logic was used for csmalar
    data = log_transform_intensity(data, model.intensity)

    if model.test_run:
        idx = data[model.features[0]].isin(["amap01"])
        data = data[idx].reset_index(drop=True).copy()
        model.response = model.response[:1]

    run(data, model)
    return


if __name__ == "__main__":
    experiment = "csmalar"
    match experiment:
        case "lcirc": exp = "L_CIRC"
        case "lshie": exp = "L_SHIE"
        case "csmalar": exp = "C_SMA_LAR"
        case _: raise ValueError("Invalid experiment")

    toml_path = f"{HOME}/repos/refactor/hbmep/configs/rat/{exp}.toml"
    model = nHB(toml_path=toml_path)
    model.use_mixture = True
    model.mcmc_params = {
        "num_warmup": 4000,
        "num_samples": 1000,
        "num_chains": 4,
        "thinning": 1,
    }
    model.nuts_params = {
        "max_tree_depth": (20, 20),
        "target_accept_prob": .95,
    }
    model._model = model.rectified_logistic
    # model.test_run = True

    build_dir = f"{HOME}/reports/hbmep/notebooks/rat/lognhb/{model.name}/{experiment}/"
    model.build_dir = os.path.join(build_dir, model._model.__name__)
    setup_logging(model.build_dir)

    data_path = f"{HOME}/data/hbmep-processed/rat/{exp}/data.csv"
    if experiment == "csmalar":
        data_path = f"{HOME}/data/hbmep-processed/rat/{exp}/data_filtered.csv"
    main(model, data_path)
