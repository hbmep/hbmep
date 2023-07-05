import glob
import tomllib
import logging
from pathlib import Path

import mat73
import numpy as np
import pandas as pd

from hbmep.data_access import DataClass
from hbmep.utils import timing
from hbmep.utils.constants import (
    PARTICIPANT,
    RESPONSE
)

logger = logging.getLogger(__name__)


@timing
def load_data(
    dir: Path,
    subdir_pattern: list[str] = ["*L_CIRC*"],
    participants: list[int] = range(1, 7),
):
    df = None

    for p in participants:
        participant = f"amap{p:02}"

        for pattern in subdir_pattern:
            PREFIX = f"{dir}/{participant}/{pattern}"

            subdirs = glob.glob(PREFIX)
            subdirs = sorted(subdirs)

            for subdir in subdirs:

                fpath = glob.glob(f"{subdir}/*auc_table.csv")[0]
                temp_df = pd.read_csv(fpath)

                fpath = glob.glob(f"{subdir}/*ep_matrix.mat")[0]
                data_dict = mat73.loadmat(fpath)

                temp_mat = data_dict["ep_sliced"]

                fpath = glob.glob(f"{subdir}/*cfg_proc.toml")[0]
                with open(fpath, "rb") as f:
                    cfg = tomllib.load(f)

                temp_df[PARTICIPANT] = participant
                temp_df["subdir_pattern"] = pattern

                if df is None:
                    df = temp_df.copy()
                    mat = temp_mat

                    time = data_dict["t_sliced"]
                    auc_window = cfg["auc"]["t_slice_minmax"]
                    continue

                assert (data_dict["t_sliced"] == time).all()
                assert cfg["auc"]["t_slice_minmax"] == auc_window

                df = pd.concat([df, temp_df], ignore_index=True).copy()
                mat = np.vstack((mat, temp_mat))

    response_ind = [int(response.split("_")[1]) - 1 for response in RESPONSE]
    mat = mat[..., response_ind]

    df.reset_index(drop=True, inplace=True)
    return df, mat, time, auc_window
