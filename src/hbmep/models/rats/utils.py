import glob
import tomllib
import logging
from pathlib import Path

import mat73
import numpy as np
import pandas as pd

from hbmep.dataset import MepDataset
from hbmep.utils import timing

logger = logging.getLogger(__name__)


@timing
def load_data(
    data: MepDataset,
    dir: Path,
    subdir_pattern: list[str] = ["*L_CIRC*"],
    subjects: list[int] = range(1, 7),
):
    df = None

    for p in subjects:
        subject = f"amap{p:02}"

        for pattern in subdir_pattern:
            PREFIX = f"{dir}/{subject}/{pattern}"

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

                temp_df[data.subject] = subject
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

    response_ind = [int(response.split("_")[1]) - 1 for response in data.response]
    mat = mat[..., response_ind]

    df.reset_index(drop=True, inplace=True)
    return df, mat, time, auc_window
