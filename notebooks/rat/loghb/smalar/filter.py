import os
import logging

import pandas as pd
import numpy as np

import hbmep as mep
from hbmep.model import BaseModel
from hbmep.util import timing, setup_logging

from hbmep.notebooks.rat.model import HB
from hbmep.notebooks.rat.util import run, log_transform_intensity, load_csmalar_data
from constants import (
    BUILD_DIR,
    TOML_PATH,
    DATA_PATH,
    NO_GROUND,
    GROUND,
    NO_GROUND_SMALL,
    NO_GROUND_BIG,
    GROUND_BIG,
    GROUND_SMALL
)

logger = logging.getLogger(__name__)


@timing
def main(model):
    # Load data
    src = DATA_PATH
    data = pd.read_csv(src)
    df = load_csmalar_data(data)
    
    subset = (
        NO_GROUND
        + GROUND
        + NO_GROUND_SMALL
        + NO_GROUND_BIG
        + GROUND_SMALL
        + GROUND_BIG
    )
    subset = list(set(subset))
    cols = ["lat", "segment", "compound_size"]
    assert set(subset) <= set(df[cols].apply(tuple, axis=1).tolist())
    idx = df[cols].apply(tuple, axis=1).isin(subset)
    df = df[idx].reset_index(drop=True).copy()
    # output_path = os.path.join(model.build_dir, "unfiltered.pdf")
    # model.plot(df, output_path=output_path)

    hue = np.full((df.shape[0], model.num_response), False)
    hue_columns = [f"hue_{response}" for response in model.response]
    idx = (
        (df.subdir == "/mnt/hdd1/acute_mapping/proc/physio/amap04/2023-03-24_C_SMA2_000")
        & (df.time >= 937.26834688)
    )
    hue[idx, :] = True
    df[hue_columns] = hue
    # output_path = os.path.join(model.build_dir, "flagged.pdf")
    # model.plot(df, output_path=output_path, hue=hue_columns)

    idx = hue.any(axis=1)
    logger.info(f"filter: {idx.sum()}")
    df = df[~idx].reset_index(drop=True).copy()
    # output_path = os.path.join(model.build_dir, "filtered.pdf")
    # model.plot(df, output_path=output_path)

    root, _ = os.path.splitext(DATA_PATH)
    output_path = f"{root}_filtered.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved filtered data to {output_path}")
    return


if __name__ == "__main__":
    model = BaseModel(toml_path=TOML_PATH)
    model.features = ["participant", "segment", "lat", "compound_size"]
    model.build_dir = os.path.join(BUILD_DIR, "filter")
    setup_logging(model.build_dir)
    main(model)
