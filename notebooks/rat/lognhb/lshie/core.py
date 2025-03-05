import os
import sys
import logging

import pandas as pd
import numpy as np
from hbmep.util import timing, setup_logging

from hbmep.notebooks.rat.model import nHB
from hbmep.notebooks.rat.util import run
from constants import (
    BUILD_DIR,
    TOML_PATH,
    DATA_PATH,
)

logger = logging.getLogger(__name__)


@timing
def main(model):
    # Load data
    src = DATA_PATH
    data = pd.read_csv(src)
    idx = data[model.intensity] > 0
    data = data[idx].reset_index(drop=True).copy()
    data[model.intensity] = np.log2(data[model.intensity])

    # subset = ["amap01"]
    # idx = data[model.features[0]].isin(subset)
    # data = data[idx].reset_index(drop=True).copy()

    run(data, model)
    return


if __name__ == "__main__":
    use_mixture = bool(int(sys.argv[1]))
    # Build model
    model = nHB(use_mixture=use_mixture, toml_path=TOML_PATH, n_jobs=-4)
    model.build_dir = os.path.join(BUILD_DIR, model.name)
    setup_logging(model.build_dir)
    main(model)
