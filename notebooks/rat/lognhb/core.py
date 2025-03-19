import os
import sys
import logging

import pandas as pd
import numpy as np
from hbmep.util import timing, setup_logging

from hbmep.notebooks.rat.model import nHB
from hbmep.notebooks.rat.util import run
from constants import HOME

logger = logging.getLogger(__name__)


@timing
def main(model, data_path):
    # Load data
    data = pd.read_csv(data_path)
    idx = data[model.intensity] > 0
    data = data[idx].reset_index(drop=True).copy()
    data[model.intensity] = np.log2(data[model.intensity])

    # idx = data[model.features[0]].isin(["amap01"])
    # data = data[idx].reset_index(drop=True).copy()
    # model.response = model.response[:1]

    run(data, model)
    return


if __name__ == "__main__":
    experiment = "lcirc"
    match experiment:
        case "lcirc": exp = "L_CIRC"
        case "lshie": exp = "L_SHIE"
        case "csmalar": exp = "C_SMA_LAR"
        case _: raise ValueError("Invalid experiment")

    use_mixture = True
    toml_path = f"{HOME}/repos/refactor/hbmep/configs/rat/{exp}.toml"
    model = nHB(toml_path=toml_path)

    build_dir = f"{HOME}/reports/hbmep/notebooks/rat/lognhb/{model.name}/{experiment}/"
    model.build_dir = os.path.join(build_dir, model._model.__name__)
    setup_logging(model.build_dir)

    data_path = f"{HOME}/data/hbmep-processed/rat/{exp}/data.csv"
    main(model, data_path)
