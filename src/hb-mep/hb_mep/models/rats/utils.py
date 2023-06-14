import os
import glob
import logging
from pathlib import Path
from typing import Optional

import mat73
import numpy as np
import pandas as pd

from hb_mep.data_access import DataClass
from hb_mep.utils import timing
from hb_mep.utils.constants import (
    PARTICIPANT
)

logger = logging.getLogger(__name__)


@timing
def load_data(
    dir: Path,
    subdirs_pattern: list[str] = ["*L_CIRC*"],
    participants: list[int] = range(1, 7),
):
    df = None

    for p in participants:
        participant = f"amap{p:02}"

        for subdir_pattern in subdirs_pattern:
            PREFIX = f"{dir}/{participant}/{subdir_pattern}"

            fpath = glob.glob(f"{PREFIX}/*auc_table.csv")[0]
            temp_df = pd.read_csv(fpath)

            fpath = glob.glob(f"{PREFIX}/*ep_matrix.mat")[0]
            data_dict = mat73.loadmat(fpath)

            temp_mat = data_dict["ep_sliced"]

            if df is None:
                time = data_dict["t_sliced"]
            else:
                assert (data_dict["t_sliced"] == time).all()

            temp_df[PARTICIPANT] = participant
            temp_df["subdir_pattern"] = subdir_pattern

            if df is None:
                df = temp_df.copy()
                mat = temp_mat
            else:
                df = pd.concat([df, temp_df], ignore_index=True).copy()
                mat = np.vstack((mat, temp_mat))

    df.reset_index(drop=True, inplace=True)
    return df, mat, time
