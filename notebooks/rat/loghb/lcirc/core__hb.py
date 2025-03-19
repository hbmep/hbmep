import os
import sys
import logging

import pandas as pd
import numpy as np
from hbmep.util import timing, setup_logging

from models import HB
from hbmep.notebooks.rat.util import run
from constants import (
    BUILD_DIR,
    TOML_PATH,
    DATA_PATH,
    MAP,
    DIAM,
    VERTICES,
    RADII
)

logger = logging.getLogger(__name__)


@timing
def main(model):
    # Load data
    src = DATA_PATH
    data = pd.read_csv(src)
    data[model.intensity] = 1 + data[model.intensity]
    # idx = data[model.intensity] > 0
    # data = data[idx].reset_index(drop=True).copy()
    data[model.intensity] = np.log2(data[model.intensity])

    cats = data[model.features[1]].unique().tolist()
    mapping = {}
    for cat in cats:
        assert cat not in mapping
        l, r = cat.split("-")
        mapping[cat] = l[3:] + "-" + r[3:]
    assert mapping == MAP
    data[model.features[1]] = data[model.features[1]].replace(mapping)
    cats = set(data[model.features[1]].tolist())
    assert set(DIAM) <= cats
    assert set(VERTICES) <= cats
    assert set(RADII) <= cats
    df = data.copy()

    run_id = model.run_id
    assert run_id in {"diam", "radii", "vertices", "all"}
    match run_id:
        case "diam": subset = DIAM
        case "radii": subset = RADII
        case "vertices": subset = VERTICES
        case "all": subset = DIAM + RADII + VERTICES
        case _: raise ValueError
    assert set(subset) <= set(df[model.features[1]].values.tolist())
    ind = df[model.features[1]].isin(subset)
    df = df[ind].reset_index(drop=True).copy()

    # subset = ["amap01", "amap02"]
    # idx = df[model.features[0]].isin(subset)
    # df = df[idx].reset_index(drop=True).copy()
    # model.response = model.response[:3]

    # model.features = [model.features]
    logger.info(f"*** run id: {run_id} ***")
    logger.info(f"*** model: {model._model.__name__} ***")
    run(df, model, extra_fields=["num_steps"])
    return


if __name__ == "__main__":
    model = HB(toml_path=TOML_PATH)
    model.build_dir = os.path.join(BUILD_DIR, model.run_id, model.name, model._model.__name__)
    setup_logging(model.build_dir)
    main(model)
